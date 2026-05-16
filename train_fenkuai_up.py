"""Training script for AdaptCLIP anomaly detection model.

Key changes vs. original:
  - Query images are NOT resized to a fixed square. The short edge is resized
    to `image_size`; the long edge is scaled proportionally.
  - After loading, each query image is split into tile_size × tile_size tiles
    along the long axis (minimal overlap, full coverage).
  - ALL tiles from ALL images in a dataloader batch are gathered and forwarded
    through the adapters in ONE call, so BatchNorm always sees batch_size ≥ 2.
  - GT masks share the same short-edge resize; tile GT regions are cropped from
    the resized mask at the same positions as the image tiles.
  - Prompt / reference images still use a fixed image_size × image_size resize.
  - stitch_anomaly_maps() is provided for inference-time reconstruction.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from PIL import Image as PILImage
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import (BinaryDiceLoss, FocalLoss, PQAdapter, TextualAdapter,
                          VisualSAdapter)
from dataset import Datasetfenkuai
from tools import get_logger, setup_seed


# ---------------------------------------------------------------------------
# CLIP normalisation constants
# ---------------------------------------------------------------------------
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ===========================================================================
# Transform utilities
# ===========================================================================

def _resize_short_edge(img: PILImage.Image, size: int) -> PILImage.Image:
    """Resize PIL image so its short edge == *size*; long edge is proportional.
    No cropping is performed."""
    w, h = img.size
    if h <= w:
        new_h, new_w = size, int(round(w * size / h))
    else:
        new_h, new_w = int(round(h * size / w)), size
    return img.resize((new_w, new_h), PILImage.BILINEAR)


def get_transform(image_size: int):
    """Return three transforms.

    preprocess       - query images : short-edge resize (variable long side)
    target_transform - GT masks     : same short-edge resize
    prompt_transform - prompt imgs  : fixed image_size x image_size square
    """
    preprocess = T.Compose([
        T.Lambda(lambda img: _resize_short_edge(img, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    target_transform = T.Compose([
        T.Lambda(lambda img: _resize_short_edge(img, image_size)),
        T.ToTensor(),
    ])
    prompt_transform = T.Compose([
        T.Resize((image_size, image_size),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    return preprocess, target_transform, prompt_transform


# ===========================================================================
# Tiling utilities
# ===========================================================================

def split_into_tiles(image: torch.Tensor, tile_size: int):
    """Split (B, C, H, W) into square tiles along the long axis.

    After short-edge resize the short axis == tile_size, so every tile is
    exactly tile_size x tile_size.

    Tiling strategy (example: W=1800, H=1000, tile_size=1000)
        n_tiles = ceil(1800 / 1000) = 2
        positions: x = [0, 800]  ->  tiles [0:1000] and [800:1800]
        overlap   = 200 px  (averaged when stitching)

    Returns
    -------
    tiles     : list[(B, C, tile_size, tile_size)]
    positions : list[(y1, x1, y2, x2)]
    orig_size : (H, W)
    """
    B, C, H, W = image.shape
    orig_size = (H, W)

    if H == tile_size and W == tile_size:
        return [image], [(0, 0, H, W)], orig_size

    positions = []
    if H > W:                           # tile vertically
        n = math.ceil(H / tile_size)
        for i in range(n):
            y1 = round(i * (H - tile_size) / (n - 1)) if n > 1 else 0
            if i == n - 1:
                y1 = H - tile_size
            positions.append((y1, 0, y1 + tile_size, W))
    else:                               # tile horizontally
        n = math.ceil(W / tile_size)
        for i in range(n):
            x1 = round(i * (W - tile_size) / (n - 1)) if n > 1 else 0
            if i == n - 1:
                x1 = W - tile_size
            positions.append((0, x1, H, x1 + tile_size))

    tiles = [image[:, :, y1:y2, x1:x2] for (y1, x1, y2, x2) in positions]
    return tiles, positions, orig_size


def stitch_anomaly_maps(
    tile_maps: list,
    positions: list,
    orig_size: tuple,
) -> torch.Tensor:
    """Stitch (B,C,th,tw) tile maps back to full size; overlaps are averaged.

    Parameters
    ----------
    tile_maps : list[(B, C, th, tw)]
    positions : list[(y1, x1, y2, x2)]
    orig_size : (H, W)

    Returns
    -------
    (B, C, H, W)
    """
    H, W = orig_size
    B, C = tile_maps[0].shape[:2]
    device, dtype = tile_maps[0].device, tile_maps[0].dtype

    result = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    count  = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    for tm, (y1, x1, y2, x2) in zip(tile_maps, positions):
        result[:, :, y1:y2, x1:x2] += tm
        count[:, :, y1:y2, x1:x2]  += 1

    return result / count.clamp(min=1)


# ===========================================================================
# Custom collate (variable-size query images)
# ===========================================================================

def collate_fn_variable_size(batch: list) -> dict:
    """img / img_mask returned as lists (variable H x W).
    All other fields are stacked / aggregated normally."""
    return {
        'img':        [item['img']      for item in batch],   # list[(C,H,W)]
        'img_mask':   [item['img_mask'] for item in batch],   # list[(1,H,W)]
        'anomaly':    torch.tensor([item['anomaly']  for item in batch]),
        'prompt_img': torch.stack([item['prompt_img'] for item in batch]),
        'cls_name':   [item['cls_name']  for item in batch],
        'view_id':    [item['view_id']   for item in batch],
        'sample_id':  [item['sample_id'] for item in batch],
        'cls_id':     torch.tensor([item['cls_id']   for item in batch]),
        'img_path':   [item['img_path']  for item in batch],
    }


# ===========================================================================
# Training
# ===========================================================================

def train(args):
    img_size      = args.image_size
    features_list = args.features_list
    save_path     = args.save_path
    dataset_name  = args.dataset
    batch_size    = args.batch_size
    k_shots       = args.k_shots
    seed          = args.seed
    vl_reduction  = args.vl_reduction
    pq_mid_dim    = args.pq_mid_dim
    pq_context    = args.pq_context

    log_file = f'{dataset_name}_{seed}seed_{k_shots}shot_train_log.txt'
    logger   = get_logger(args.save_path, log_file)
    logger.info('\n')
    logger.info(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.pretrained_model == 'ViT-L/14@336px':
        model, _ = adaptcliplib.load(args.pretrained_model, device=device)
        DPAM_layer, patch_size, input_dim = 20, 14, 768
    elif args.pretrained_model == 'VITB16_PLUS_240':
        model, _ = adaptcliplib.load(args.pretrained_model, device=device)
        DPAM_layer, patch_size, input_dim = 10, 16, 640
    elif args.pretrained_model == 'ViT-L-14-CLIPA-336':
        model, _ = adaptcliplib.load(args.pretrained_model, device=device)
        DPAM_layer, patch_size, input_dim = 20, 14, 768
    else:
        raise ValueError(f"Unknown pretrained_model: {args.pretrained_model}")

    model.visual.DAPM_replace(DPAM_layer=DPAM_layer)

    # ------------------------------------------------------------------
    # Adapters
    # ------------------------------------------------------------------
    textual_learner = TextualAdapter(model.to("cpu"), img_size, args.n_ctx)
    visual_learner = VisualSAdapter(
        img_size,
        patch_size,
        input_dim=input_dim,
        reduction=vl_reduction,
        decoder=args.up,          
        decoder_mid_dim=64,     
    )
    pq_learner      = PQAdapter(img_size, patch_size,
                                context=pq_context,
                                input_dim=input_dim,
                                mid_dim=pq_mid_dim,
                                layers_num=len(features_list))

    model.to(device);  model.eval()
    textual_learner.to(device);  textual_learner.train()
    visual_learner.to(device);   visual_learner.train()
    pq_learner.to(device);       pq_learner.train()

    tl_p = sum(p.numel() for p in textual_learner.parameters())
    vl_p = sum(p.numel() for p in visual_learner.parameters())
    pq_p = sum(p.numel() for p in pq_learner.parameters())
    learned = tl_p + vl_p + pq_p
    fixed   = sum(p.numel() for p in model.parameters())
    print(f"textual_learner params:{tl_p}  visual_learner:{vl_p/1e6:.1f}M  "
          f"pq_learner:{pq_p/1e6:.1f}M  learned:{learned/1e6:.1f}M  "
          f"fixed(CLIP):{fixed/1e6:.1f}M  all:{(learned+fixed)/1e6:.1f}M")

    # ------------------------------------------------------------------
    # Optimiser & losses
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        list(textual_learner.parameters()) +
        list(visual_learner.parameters()) +
        list(pq_learner.parameters()),
        lr=args.learning_rate, betas=(0.5, 0.999))

    loss_focal = FocalLoss()
    loss_dice  = BinaryDiceLoss()

    # ------------------------------------------------------------------
    # Transforms & dataset
    # ------------------------------------------------------------------
    preprocess, target_transform, prompt_transform = get_transform(img_size)

    train_data = Datasetfenkuai(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        prompt_transform=prompt_transform,
        dataset_name=dataset_name,
        k_shots=k_shots,
        save_dir=save_path,
        mode='train',
        seed=seed,
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_variable_size,
    )

    textual_learner.prepare_static_text_feature(model)

    # ==================================================================
    # Training loop
    # ==================================================================
    for epoch in tqdm(range(args.epoch)):
        epoch_local_loss  = []
        epoch_global_loss = []

        for items in tqdm(train_data_loader):
            # ── Prompt images (fixed size, one per image in batch) ──────────
            prompt_image = items['prompt_img'].to(device)   # (B, s, C, Hp, Wp)
            b, s, c, h_p, w_p = prompt_image.shape
            prompt_image = prompt_image.reshape(-1, c, h_p, w_p)   # (B*s, C, Hp, Wp)

            images  = items['img']       # list of (C, H_var, W_var), length B
            gts_raw = items['img_mask']  # list of (1, H_var, W_var), length B

            # ── Encode all prompt images once (frozen CLIP) ──────────────────
            with torch.no_grad():
                prompt_feats, prompt_patch_feats = model.encode_image(
                    prompt_image, features_list, DPAM_layer=DPAM_layer)
                # prompt_feats       : (B*s, d)
                # prompt_patch_feats : list of (B*s, L, d)
                prompt_feats = prompt_feats.reshape(b, s, -1)         # (B, s, d)
                for idx in range(len(features_list)):
                    prompt_patch_feats[idx] = rearrange(
                        prompt_patch_feats[idx],
                        '(b s) l d -> b s l d', b=b, s=s)
                # prompt_patch_feats[idx]: (B, s, L, d)

            # ── Collect ALL tiles from the entire batch ──────────────────────
            #
            #  Key fix: instead of processing tiles one-by-one (batch_size=1),
            #  we gather every tile from every image and forward them all at
            #  once.  BatchNorm then sees batch_size = total_tiles (>= 2).
            #
            all_tiles       = []    # each (1, C, S, S)
            all_gt_tiles    = []    # each (1, S, S)
            all_tile_labels = []    # each (1,) long tensor
            tile_to_img_idx = []    # which image each tile came from

            for img_idx in range(b):
                image_i = images[img_idx].unsqueeze(0).to(device)  # (1, C, H, W)
                gt_i    = gts_raw[img_idx].squeeze(0).to(device)   # (H, W)
                gt_i    = (gt_i > 0.5).float()

                tiles, positions, orig_size = split_into_tiles(image_i, img_size)

                for tile, (y1, x1, y2, x2) in zip(tiles, positions):
                    gt_tile    = gt_i[y1:y2, x1:x2].unsqueeze(0)          # (1, S, S)
                    tile_label = (gt_tile.max() > 0.5).long().view(1)      # (1,)

                    all_tiles.append(tile)
                    all_gt_tiles.append(gt_tile)
                    all_tile_labels.append(tile_label)
                    tile_to_img_idx.append(img_idx)

            # ── Stack into one big batch (N = total tiles across batch) ──────
            tile_batch  = torch.cat(all_tiles,       dim=0)            # (N, C, S, S)
            gt_batch    = torch.cat(all_gt_tiles,    dim=0)            # (N, S, S)
            label_batch = torch.cat(all_tile_labels).to(device)        # (N,)
            N = tile_batch.shape[0]

            # ── Encode all tiles with frozen CLIP ────────────────────────────
            with torch.no_grad():
                query_feats, query_patch_feats = model.encode_image(
                    tile_batch, features_list, DPAM_layer=DPAM_layer)
                # query_feats       : (N, d)
                # query_patch_feats : list of (N, L, d)

            # ── Build per-tile prompt features (replicate by image index) ────
            # PQAdapter needs prompt feats aligned with each tile.
            prompt_feats_tiled = torch.cat(
                [prompt_feats[tile_to_img_idx[t]:tile_to_img_idx[t]+1] for t in range(N)],
                dim=0)                                                  # (N, s, d)

            prompt_patch_feats_tiled = [
                torch.cat(
                    [prompt_patch_feats[fi][tile_to_img_idx[t]:tile_to_img_idx[t]+1]
                     for t in range(N)],
                    dim=0)                                              # (N, s, L, d)
                for fi in range(len(features_list))
            ]

            # ── Forward adapters + compute losses ────────────────────────────
            optimizer.zero_grad()
            local_loss  = 0
            global_loss = 0

            # VisualAdapter
            if args.visual_learner:
                static_text_features = textual_learner.static_text_features
                global_logit, local_score = visual_learner(
                    query_feats, query_patch_feats, static_text_features)
                # global_logit : (N, 2)   local_score : (N, 2, S, S)

                global_loss += F.cross_entropy(global_logit, label_batch)
                local_loss  += loss_focal(local_score, gt_batch)
                local_loss  += loss_dice(local_score[:, 1, :, :], gt_batch)
                local_loss  += loss_dice(local_score[:, 0, :, :], 1 - gt_batch)

            # TextualAdapter
            if args.textual_learner:
                learned_prompts, tokenized_prompts = textual_learner()
                learned_text_features = model.encode_text(
                    learned_prompts, tokenized_prompts).float()   # (2, d)

                global_logit, local_score = textual_learner.compute_global_local_score(
                    query_feats, query_patch_feats, learned_text_features)

                global_loss += F.cross_entropy(global_logit, label_batch)
                local_loss  += loss_focal(local_score, gt_batch)
                local_loss  += loss_dice(local_score[:, 1, :, :], gt_batch)
                local_loss  += loss_dice(local_score[:, 0, :, :], 1 - gt_batch)

            # PQAdapter
            if args.pq_learner and k_shots > 0:
                global_logit_list, local_score_list, _ = pq_learner(
                    query_feats, query_patch_feats,
                    prompt_feats_tiled, prompt_patch_feats_tiled)

                for lg in global_logit_list:
                    global_loss += F.cross_entropy(lg, label_batch)

                for ls in local_score_list:
                    local_loss += loss_focal(ls, gt_batch)
                    local_loss += loss_dice(ls[:, 1, :, :], gt_batch)
                    local_loss += loss_dice(ls[:, 0, :, :], 1 - gt_batch)

            # ── Single backward per dataloader batch ─────────────────────────
            (local_loss + global_loss).backward()
            optimizer.step()

            epoch_local_loss.append(
                local_loss.item()  if torch.is_tensor(local_loss)  else float(local_loss))
            epoch_global_loss.append(
                global_loss.item() if torch.is_tensor(global_loss) else float(global_loss))

        # ── Logging ───────────────────────────────────────────────────────────
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], global_loss:{:.4f}, local_loss:{:.4f}'.format(
                epoch + 1, args.epoch,
                np.mean(epoch_global_loss), np.mean(epoch_local_loss)))

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, f'epoch_{epoch + 1}.pth')
            torch.save({
                'textual_learner': textual_learner.state_dict(),
                'visual_learner':  visual_learner.state_dict(),
                'pq_learner':      pq_learner.state_dict(),
            }, ckp_path)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AdaptCLIP", add_help=True)
    parser.add_argument("--train_data_path",  type=str,   default="./data/visa")
    parser.add_argument("--save_path",        type=str,   default='./checkpoint')
    parser.add_argument("--dataset",          type=str,   default='mvtec')
    parser.add_argument("--pretrained_model", type=str,   default='ViT-L/14@336px')
    parser.add_argument("--n_ctx",            type=int,   default=12)
    parser.add_argument("--features_list",    type=int,   nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--epoch",            type=int,   default=15)
    parser.add_argument("--learning_rate",    type=float, default=0.001)
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--image_size",       type=int,   default=518)
    parser.add_argument("--print_freq",       type=int,   default=1)
    parser.add_argument("--save_freq",        type=int,   default=1)
    parser.add_argument("--seed",             type=int,   default=10)
    parser.add_argument("--k_shots",          type=int,   default=0)
    parser.add_argument("--visual_learner",   action="store_true")
    parser.add_argument("--textual_learner",  action="store_true")
    parser.add_argument("--pq_learner",       action="store_true")
    parser.add_argument("--vl_reduction",     type=int,   default=4)
    parser.add_argument("--pq_mid_dim",       type=int,   default=128)
    parser.add_argument("--pq_context",       action="store_true")
    parser.add_argument("--up",               type=str,   default='bilinear')
    # 新增：'fpn' | 'sim' | 'bilinear'
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)