import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import PQAdapter, TextualAdapter, VisualSAdapter, fusion_fun
from dataset import Datasetfenkuai, PromptDataset
from tools import Evaluator, get_logger, setup_seed, visualizer





# ---------------------------------------------------------------------------
# CLIP normalisation constants
# ---------------------------------------------------------------------------
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ===========================================================================
# Transform utilities
# ===========================================================================

def _resize_short_edge(img: PILImage.Image, size: int) -> PILImage.Image:
    """Resize so the short edge == *size*; long edge is proportional (no crop)."""
    w, h = img.size
    if h <= w:
        new_h, new_w = size, int(round(w * size / h))
    else:
        new_h, new_w = int(round(h * size / w)), size
    return img.resize((new_w, new_h), PILImage.BILINEAR)


def get_transform(image_size: int):
    """Return three transforms.

    preprocess       – query images : short-edge resize, variable long side
    target_transform – GT masks     : same short-edge resize
    prompt_transform – prompt imgs  : fixed image_size × image_size square
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
    exactly tile_size × tile_size (minimal overlap, full coverage).

    Example: W=1800, H=1000, tile_size=1000
        n = ceil(1800/1000) = 2
        x-positions: [0, 800]  → tiles [0:1000] and [800:1800]
        overlap = 200 px → averaged during stitch

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
    if H > W:                       # tile vertically
        n = math.ceil(H / tile_size)
        for i in range(n):
            y1 = round(i * (H - tile_size) / (n - 1)) if n > 1 else 0
            if i == n - 1:
                y1 = H - tile_size
            positions.append((y1, 0, y1 + tile_size, W))
    else:                           # tile horizontally
        n = math.ceil(W / tile_size)
        for i in range(n):
            x1 = round(i * (W - tile_size) / (n - 1)) if n > 1 else 0
            if i == n - 1:
                x1 = W - tile_size
            positions.append((0, x1, H, x1 + tile_size))

    tiles = [image[:, :, y1:y2, x1:x2] for (y1, x1, y2, x2) in positions]
    return tiles, positions, orig_size


def stitch_anomaly_maps(tile_maps: list,
                         positions: list,
                         orig_size: tuple) -> torch.Tensor:
    """Place tile maps at their pixel positions; overlaps are averaged.

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
# Variable-size collate function
# ===========================================================================

def collate_fn_variable_size(batch: list) -> dict:
    """img / img_mask returned as lists (variable H×W); rest stacked normally."""
    return {
        'img':        [item['img']      for item in batch],   # list[(C,H,W)]
        'img_mask':   [item['img_mask'] for item in batch],   # list[(1,H,W)]
        'anomaly':    torch.tensor([item['anomaly']  for item in batch]),
        'cls_name':   [item['cls_name']  for item in batch],
        'view_id':    [item['view_id']   for item in batch],
        'sample_id':  [item['sample_id'] for item in batch],
        'cls_id':     torch.tensor([item['cls_id']   for item in batch]),
        'img_path':   [item['img_path']  for item in batch],
    }


# ===========================================================================
# TIFF saving utilities  (from mvtec_ad2_inference.py)
# ===========================================================================

def save_tiff(array: np.ndarray, path: str):
    """Save a 2-D float32 array as a float16 TIFF file (lossless float)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, array.astype(np.float16))


def _parse_split(img_path: str, cls_name: str) -> str:
    """Determine the split key from the image path.

    Returns one of:
        'test_private'         → anomaly_images/<cls>/test_private/
        'test_private_mixed'   → anomaly_images/<cls>/test_private_mixed/
        'test_public/bad'      → test_public_predictions/<cls>/bad/
        'test_public/good'     → test_public_predictions/<cls>/good/
    """
    parts = img_path.replace('\\', '/').split('/')
    # Find the class folder index
    cls_idx = -1
    for i, p in enumerate(parts):
        if p == cls_name:
            cls_idx = i
            break
    if cls_idx == -1:
        return 'other'

    after_cls = parts[cls_idx + 1:]   # everything after <cls_name>/

    if not after_cls:
        return 'other'

    first = after_cls[0]
    if first == 'test_private_mixed':
        return 'test_private_mixed'
    if first == 'test_private':
        return 'test_private'
    if first == 'test_public' and len(after_cls) >= 2:
        label = after_cls[1]          # 'bad' or 'good'
        return f'test_public/{label}'
    return 'other'


# Fixed root for competition submission TIFFs
TIFF_OUTPUT_ROOT = './submission_folder/anomaly_images'


def _build_tiff_path(cls_name: str, split_key: str, idx: int) -> str:
    """Build the output TIFF path under the fixed competition layout.

    Output root is always:
        ./submission_folder/anomaly_images/

    Layout:
        <root>/<cls>/test_private/       NNN_regular.tiff
        <root>/<cls>/test_private_mixed/ NNN_mixed.tiff
    """
    if split_key == 'test_private':
        out_dir  = os.path.join(TIFF_OUTPUT_ROOT, cls_name, 'test_private')
        filename = f'{idx:03d}_regular.tiff'
    elif split_key == 'test_private_mixed':
        out_dir  = os.path.join(TIFF_OUTPUT_ROOT, cls_name, 'test_private_mixed')
        filename = f'{idx:03d}_mixed.tiff'
    else:
        raise ValueError(f"_build_tiff_path: unexpected split_key '{split_key}'")

    return os.path.join(out_dir, filename)


# ===========================================================================
# Direct file-system helpers  (used by --save_tiff mode)
# ===========================================================================

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _collect_images(folder: str) -> list:
    """Return sorted list of image paths directly inside *folder* (non-recursive)."""
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])


class FlatImageDataset(torch.utils.data.Dataset):
    """Minimal dataset that loads images from a flat file list."""
    def __init__(self, image_paths: list, transform):
        self.image_paths = image_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = PILImage.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img), self.image_paths[idx]


def run_tiff_inference(
    args, model,
    textual_learner, visual_learner, pq_learner,
    static_text_features, learned_text_features,
    DPAM_layer, device, logger,
):

    img_size      = args.image_size
    features_list = args.features_list
    k_shots       = args.k_shots

    # Short-edge transform (same as query images in normal eval)
    preprocess = T.Compose([
        T.Lambda(lambda img: _resize_short_edge(img, img_size)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    data_root = args.test_data_path

    # Discover classes: every sub-directory of data_root
    classes = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    logger.info(f'\n[save_tiff] Scanning {data_root}')
    logger.info(f'[save_tiff] Classes: {classes}')
    logger.info(f'[save_tiff] Output root: {TIFF_OUTPUT_ROOT}')

    split_counters: dict = defaultdict(int)   # (cls, split_key) -> count

    for cls_name in classes:
        for split_key in ('test_private', 'test_private_mixed'):
            split_dir   = os.path.join(data_root, cls_name, split_key)
            image_paths = _collect_images(split_dir)

            if not image_paths:
                logger.info(f'  {cls_name}/{split_key}: not found, skipping')
                continue

            logger.info(f'  {cls_name}/{split_key}: {len(image_paths)} images')

            dataset = FlatImageDataset(image_paths, preprocess)
            loader  = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=0)

            for imgs_batch, _ in tqdm(loader,
                                       desc=f'  {cls_name}/{split_key}',
                                       leave=False):
                imgs_batch = imgs_batch.to(device)   # (B, C, H, W)
                B_local    = imgs_batch.shape[0]

                # ── Collect tiles from all images in this mini-batch ───
                all_tiles         = []
                tile_to_img_idx   = []
                positions_per_img = []

                for img_idx in range(B_local):
                    image_i = imgs_batch[img_idx:img_idx + 1]
                    tiles, positions, orig_size = split_into_tiles(image_i, img_size)
                    positions_per_img.append((positions, orig_size))
                    for tile in tiles:
                        all_tiles.append(tile)
                        tile_to_img_idx.append(img_idx)

                tile_batch = torch.cat(all_tiles, dim=0)   # (N, C, S, S)

                # ── CLIP encode ────────────────────────────────────────
                with torch.no_grad():
                    query_feats, query_patch_feats = model.encode_image(
                        tile_batch, features_list, DPAM_layer=DPAM_layer)

                    # visual adapter
                    local_vl_map_all = None
                    if args.visual_learner:
                        _, _lv = visual_learner(
                            query_feats, query_patch_feats, static_text_features)
                        local_vl_map_all = _lv[:, 1:2]              # (N,1,S,S)

                    # textual adapter
                    local_tl_map_all = None
                    if args.textual_learner:
                        _, _lt = textual_learner.compute_global_local_score(
                            query_feats, query_patch_feats, learned_text_features)
                        local_tl_map_all = _lt[:, 1:2]              # (N,1,S,S)

                    # pq adapter (zero-shot mode has k_shots=0, skip)
                    # local_pq_map_all = None
                    # align_score_all  = None
                    # if args.pq_learner and k_shots > 0:
                    #     # NOTE: prompt memory not built in tiff-only mode;
                    #     # pq_learner is skipped silently when k_shots == 0.
                    #     pass

                # ── Per-image: stitch → gaussian filter → save TIFF ───
                tile_start = 0
                for img_idx in range(B_local):
                    positions, orig_size = positions_per_img[img_idx]
                    n_t      = len(positions)
                    tile_end = tile_start + n_t
                    t_range  = range(tile_start, tile_end)

                    maps_to_fuse = []

                    if local_vl_map_all is not None:
                        vl_tiles = [local_vl_map_all[t:t + 1] for t in t_range]
                        vl_full  = stitch_anomaly_maps(vl_tiles, positions, orig_size)
                        maps_to_fuse.append(vl_full[:, 0])           # (1, H, W)

                    if local_tl_map_all is not None:
                        tl_tiles = [local_tl_map_all[t:t + 1] for t in t_range]
                        tl_full  = stitch_anomaly_maps(tl_tiles, positions, orig_size)
                        maps_to_fuse.append(tl_full[:, 0])

                    if not maps_to_fuse:
                        tile_start = tile_end
                        continue

                    pixel_map = fusion_fun(maps_to_fuse,
                                          fusion_type=args.fusion_type)  # (1,H,W)

                    # Gaussian filter
                    pm_np = gaussian_filter(
                        pixel_map[0].cpu().numpy(), sigma=args.sigma)

                    # Save TIFF
                    counter_k = (cls_name, split_key)
                    tiff_idx  = split_counters[counter_k]
                    split_counters[counter_k] += 1
                    tiff_path = _build_tiff_path(cls_name, split_key, tiff_idx)
                    save_tiff(pm_np, tiff_path)

                    tile_start = tile_end

    total = sum(split_counters.values())
    logger.info(f'\n[save_tiff] Done. Saved {total} TIFF files to {TIFF_OUTPUT_ROOT}')
    for (cls, split), cnt in sorted(split_counters.items()):
        logger.info(f'  {cls}/{split}: {cnt} files')


# ===========================================================================
# Prompt memory helpers  (unchanged from original)
# ===========================================================================

def prompt_association(image_memory, patch_memory, target_class_name):
    patch_level_num = len(patch_memory[target_class_name[0]])
    retrive_image, retrive_patch = [], [[] for _ in range(patch_level_num)]
    for class_name in target_class_name:
        retrive_image.append(image_memory[class_name])
        for l in range(patch_level_num):
            retrive_patch[l].append(patch_memory[class_name][l])
    retrive_image = torch.stack(retrive_image)
    for l in range(patch_level_num):
        retrive_patch[l] = torch.stack(retrive_patch[l])
    return retrive_image, retrive_patch


def build_prompt_memory(model, prompt_dataloader, device, obj_list,
                        view_list, features_list, DPAM_layer):
    """Build few-shot prompt memory (unchanged)."""
    feats_scale_num = len(features_list)
    image_temp, patch_temp = [], [[] for _ in range(feats_scale_num)]
    cls_names_temp, view_ids_temp = [], []

    for idx, items in enumerate(tqdm(prompt_dataloader)):
        cls_name     = items['cls_name']
        prompt_image = items['img'].to(device)
        prompt_mask  = items['img_mask'].to(device)
        view_id      = items['view_id']

        with torch.no_grad():
            image_feat, patch_feat = model.encode_image(
                prompt_image, features_list, DPAM_layer=DPAM_layer)

        cls_names_temp.extend(cls_name)
        image_temp.append(image_feat)
        view_ids_temp.extend(view_id)
        for i in range(feats_scale_num):
            patch_temp[i].append(patch_feat[i])

    image_temp = torch.cat(image_temp, dim=0)
    for i in range(feats_scale_num):
        patch_temp[i] = torch.cat(patch_temp[i], dim=0)

    prompt_image_memory, prompt_patch_memory = {}, {}
    for obj in obj_list:
        if len(view_list) > 1:
            for view_id in view_list:
                indice   = (np.array(cls_names_temp) == obj) & (np.array(view_ids_temp) == view_id)
                obj_name = obj + '_' + view_id
                prompt_image_memory[obj_name] = image_temp[indice]
                prompt_patch_memory[obj_name] = [patch_temp[i][indice]
                                                  for i in range(feats_scale_num)]
        else:
            indice = (np.array(cls_names_temp) == obj)
            prompt_image_memory[obj] = image_temp[indice]
            prompt_patch_memory[obj] = [patch_temp[i][indice]
                                         for i in range(feats_scale_num)]

    return prompt_image_memory, prompt_patch_memory


# ===========================================================================
# Main test function
# ===========================================================================

def test(args):
    img_size      = args.image_size
    features_list = args.features_list
    dataset_dir   = args.test_data_path
    save_path     = args.save_path
    dataset_name  = args.dataset
    batch_size    = args.batch_size
    k_shots       = args.k_shots
    seed          = args.seed
    vl_reduction  = args.vl_reduction
    pq_mid_dim    = args.pq_mid_dim
    pq_context    = args.pq_context
    eval_metrics  = args.eval_metrics
    mode          = 'test'

    log_file = f'{dataset_name}_{seed}seed_{k_shots}shot_{mode}_log.txt'
    logger   = get_logger(save_path, log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.pretrained_model == 'ViT-L/14@336px':
        model, _ = adaptcliplib.load(args.pretrained_model, device=device)
        model.visual.DAPM_replace(DPAM_layer=20)
        patch_size, input_dim, DPAM_layer = 14, 768, 20
    if args.pretrained_model == 'VITB16_PLUS_240':
        model, _ = adaptcliplib.load(args.pretrained_model, device=device)
        model.visual.DAPM_replace(DPAM_layer=10)
        patch_size, input_dim, DPAM_layer = 16, 640, 10


    preprocess, target_transform, prompt_transform = get_transform(img_size)

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
    pq_learner      = PQAdapter(img_size, patch_size, context=pq_context,
                                input_dim=input_dim, mid_dim=pq_mid_dim,
                                layers_num=len(features_list))

    logger.info('\n' + 'loading model from: ' + args.checkpoint_path)
    checkpoint_adapter = torch.load(args.checkpoint_path)
    textual_learner.load_state_dict(checkpoint_adapter['textual_learner'])
    visual_learner.load_state_dict(checkpoint_adapter['visual_learner'])
    pq_learner.load_state_dict(checkpoint_adapter['pq_learner'])

    model.to(device);           model.eval()
    textual_learner.to(device); textual_learner.eval()
    visual_learner.to(device);  visual_learner.eval()
    pq_learner.to(device);      pq_learner.eval()

    tl_p = sum(p.numel() for p in textual_learner.parameters())
    vl_p = sum(p.numel() for p in visual_learner.parameters())
    pq_p = sum(p.numel() for p in pq_learner.parameters())
    learned = tl_p + vl_p + pq_p
    fixed   = sum(p.numel() for p in model.parameters())
    print(f"textual_learner params:{tl_p}  visual_learner:{vl_p/1e6:.1f}M  "
          f"pq_learner:{pq_p/1e6:.1f}M  learned:{learned/1e6:.1f}M  "
          f"fixed:{fixed/1e6:.1f}M  all:{(learned+fixed)/1e6:.1f}M")

    # ------------------------------------------------------------------
    # Text encoder (run once)
    # ------------------------------------------------------------------
    textual_learner.prepare_static_text_feature(model)
    static_text_features = textual_learner.static_text_features

    learned_prompts, tokenized_prompts = textual_learner()
    learned_text_features = model.encode_text_learn(
        learned_prompts, tokenized_prompts).float()

    # ------------------------------------------------------------------
    # --save_tiff mode: bypass Datasetfenkuai entirely.
    # Datasetfenkuai only loads test_public images (from meta.json).
    # test_private / test_private_mixed are NOT in meta.json, so we must
    # scan the filesystem directly via run_tiff_inference(), then return.
    # ------------------------------------------------------------------
    if args.save_tiff:
        run_tiff_inference(
            args, model,
            textual_learner, visual_learner, pq_learner,
            static_text_features, learned_text_features,
            DPAM_layer, device, logger,
        )
        return   # skip normal eval / ceshi entirely

    # ------------------------------------------------------------------
    # Datasets (only used when --save_tiff is False)
    # ------------------------------------------------------------------
    if dataset_name in ['Real-IAD-Variety', 'RealIAD']:
        sample_level = True
        prompt_data = PromptDataset(
            root=dataset_dir, transform=prompt_transform,
            target_transform=target_transform,
            dataset_name=dataset_name, k_shots=k_shots,
            save_dir=save_path, mode=mode, seed=seed,
            class_name=args.class_name)
        test_data = Datasetfenkuai(
            root=dataset_dir, transform=preprocess,
            target_transform=target_transform,
            dataset_name=dataset_name, k_shots=k_shots,
            save_dir=save_path, mode=mode, seed=seed,
            class_name=args.class_name)
    else:
        sample_level = False
        prompt_data = PromptDataset(
            root=dataset_dir, transform=prompt_transform,
            target_transform=target_transform,
            dataset_name=dataset_name, k_shots=k_shots,
            save_dir=save_path, mode=mode, seed=seed)
        test_data = Datasetfenkuai(
            root=dataset_dir, transform=preprocess,
            target_transform=target_transform,
            dataset_name=dataset_name, k_shots=k_shots,
            save_dir=save_path, mode=mode, seed=seed)

    # Prompt images are fixed-size → normal DataLoader collate is fine
    prompt_dataloader = torch.utils.data.DataLoader(
        prompt_data, batch_size=batch_size, shuffle=False)
    # Query images may have variable size → custom collate
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn_variable_size)

    obj_list  = test_data.obj_list
    view_list = test_data.view_list

    # ------------------------------------------------------------------
    # Evaluation setup
    # ------------------------------------------------------------------
    cpu_eva = False
    evaluator = Evaluator('cpu' if cpu_eva else device,
                          metrics=eval_metrics, sample_level=sample_level)

    # ------------------------------------------------------------------
    # Few-shot prompt memory
    # ------------------------------------------------------------------
    if k_shots > 0:
        prompt_image_memory, prompt_patch_memory = build_prompt_memory(
            model, prompt_dataloader, device,
            obj_list, view_list, features_list, DPAM_layer)

    # ==================================================================
    # Inference loop  –  collect all results then evaluate with ceshi
    # ==================================================================
    sample_ids, gt_masks, pr_masks = [], [], []
    cls_names_all, gt_anomalys, pr_anomalys, query_paths = [], [], [], []

    nums       = 0
    total_time = 0.0

    pbar = tqdm(test_dataloader, desc='Processing batches')
    for idx, items in enumerate(pbar):
        # items['img']      : list[(C, H_var, W_var)]  – variable size
        # items['img_mask'] : list[(1, H_var, W_var)]  – variable size
        images    = items['img']       # list, len = B
        gts_raw   = items['img_mask']  # list, len = B
        B         = len(images)

        gt_anomaly = items['anomaly'].to(device)
        cls_name   = items['cls_name']
        sample_id  = items['sample_id']
        query_path = items['img_path']

        # ── Prompt memory lookup ───────────────────────────────────────
        if k_shots > 0:
            target_cls = ([c + '_' + v
                           for c, v in zip(cls_name, items['view_id'])]
                          if len(view_list) > 1 else cls_name)
            prompt_feats_batch, prompt_patch_feats_batch = prompt_association(
                prompt_image_memory, prompt_patch_memory, target_cls)
            # prompt_feats_batch       : (B, s, d)
            # prompt_patch_feats_batch : list[(B, s, L, d)]

        # ── Start timing ───────────────────────────────────────────────
        torch.cuda.synchronize()
        start_time = time.time()

        # ── Collect ALL tiles from all images in this dataloader batch ─
        #
        #  Tiling strategy: after short-edge resize, short axis == img_size.
        #  Tiles along the long axis are img_size × img_size (minimal overlap).
        #  All tiles are forwarded through CLIP and the adapters in one call.
        #
        all_tiles       = []          # each (1, C, S, S)
        tile_to_img_idx = []          # which image in [0, B) each tile belongs to
        positions_per_img = []        # (positions, orig_size) for every image

        for img_idx in range(B):
            image_i = images[img_idx].unsqueeze(0).to(device)   # (1, C, H, W)
            tiles, positions, orig_size = split_into_tiles(image_i, img_size)
            positions_per_img.append((positions, orig_size))
            for tile in tiles:
                all_tiles.append(tile)
                tile_to_img_idx.append(img_idx)

        tile_batch = torch.cat(all_tiles, dim=0)    # (N, C, S, S)
        N = tile_batch.shape[0]

        # ── Encode all tiles with frozen CLIP ──────────────────────────
        with torch.no_grad():
            query_feats, query_patch_feats = model.encode_image(
                tile_batch, features_list, DPAM_layer=DPAM_layer)
            # query_feats       : (N, d)
            # query_patch_feats : list[(N, L, d)]

        # ── Build per-tile prompt features (replicate by image index) ──
        if k_shots > 0:
            prompt_feats_tiled = torch.cat(
                [prompt_feats_batch[tile_to_img_idx[t]:tile_to_img_idx[t]+1]
                 for t in range(N)], dim=0)                          # (N, s, d)
            prompt_patch_feats_tiled = [
                torch.cat(
                    [prompt_patch_feats_batch[fi][tile_to_img_idx[t]:tile_to_img_idx[t]+1]
                     for t in range(N)], dim=0)                      # (N, s, L, d)
                for fi in range(len(features_list))
            ]

        # ── Adapter forward on all tiles ───────────────────────────────
        with torch.no_grad():

            # visual adapter  →  (N, 2, S, S)  and  (N, 2)
            if args.visual_learner:
                global_vl_logit_all, local_vl_map_all = visual_learner(
                    query_feats, query_patch_feats, static_text_features)
                local_vl_map_all    = local_vl_map_all[:, 1:2].detach()     # (N,1,S,S)
                global_vl_score_all = global_vl_logit_all.softmax(-1)[:, 1].detach()  # (N,)

            # textual adapter
            if args.textual_learner:
                global_tl_logit_all, local_tl_map_all = \
                    textual_learner.compute_global_local_score(
                        query_feats, query_patch_feats, learned_text_features)
                local_tl_map_all    = local_tl_map_all[:, 1:2].detach()     # (N,1,S,S)
                global_tl_score_all = global_tl_logit_all.softmax(-1)[:, 1].detach()  # (N,)

            # pq adapter
            if args.pq_learner and k_shots > 0:
                global_pq_logit_all, local_pq_map_list_all, align_score_list_all = \
                    pq_learner(query_feats, query_patch_feats,
                               prompt_feats_tiled, prompt_patch_feats_tiled)

                # local pq map: average across layers  → (N, 1, S, S)
                local_pq_map_all = torch.stack(
                    [x[:, 1] for x in local_pq_map_list_all], dim=1
                ).mean(dim=1, keepdim=True).detach()

                # align score per tile: scalar  → (N,)
                align_score_all = fusion_fun(
                    align_score_list_all, fusion_type='harmonic_mean')[:, 0].detach()

                # global pq score per tile → (N,)
                if isinstance(global_pq_logit_all, list):
                    global_pq_score_all = torch.stack(
                        [x.softmax(-1)[:, 1] for x in global_pq_logit_all],
                        dim=1).mean(dim=1).detach()
                else:
                    global_pq_score_all = global_pq_logit_all.softmax(-1)[:, 1].detach()

        # ── Per-image: stitch → gaussian → resize → global score ──────
        pixel_anomaly_maps_batch  = []   # (1, S, S) per image, eval size
        image_anomaly_preds_batch = []   # scalar per image
        gt_masks_batch            = []   # (1, S, S) per image, eval size

        tile_start = 0
        for img_idx in range(B):
            positions, orig_size = positions_per_img[img_idx]
            H_orig, W_orig = orig_size
            n_t      = len(positions)
            tile_end = tile_start + n_t
            t_range  = range(tile_start, tile_end)

            # ── Stitch each adapter's pixel map back to full resolution ─
            if args.visual_learner:
                vl_tiles = [local_vl_map_all[t:t+1] for t in t_range]
                vl_full  = stitch_anomaly_maps(vl_tiles, positions, orig_size)  # (1,1,H,W)

            if args.textual_learner:
                tl_tiles = [local_tl_map_all[t:t+1] for t in t_range]
                tl_full  = stitch_anomaly_maps(tl_tiles, positions, orig_size)  # (1,1,H,W)

            if args.pq_learner and k_shots > 0:
                pq_tiles = [local_pq_map_all[t:t+1] for t in t_range]
                pq_full  = stitch_anomaly_maps(pq_tiles, positions, orig_size)  # (1,1,H,W)

                # align score: take max across tiles → broadcast to full map
                align_val  = align_score_all[tile_start:tile_end].max()          # scalar
                align_full = align_val.view(1, 1, 1, 1).expand(1, 1, H_orig, W_orig)

            # ── Fuse pixel-level maps at full resolution ────────────────
            if k_shots > 0:
                pixel_map = fusion_fun(
                    [vl_full[:, 0], tl_full[:, 0], pq_full[:, 0]],
                    fusion_type=args.fusion_type)                     # (1, H, W)
                pixel_map = fusion_fun(
                    [pixel_map, align_full[:, 0]],
                    fusion_type='harmonic_mean')                      # (1, H, W)
            else:
                pixel_map = fusion_fun(
                    [vl_full[:, 0], tl_full[:, 0]],
                    fusion_type=args.fusion_type)                     # (1, H, W)

            # ── Gaussian filter at full resolution (more accurate) ──────
            pm_np     = pixel_map[0].cpu().numpy()                   # (H, W)
            pm_np     = gaussian_filter(pm_np, sigma=args.sigma)
            pixel_map = torch.from_numpy(pm_np).unsqueeze(0).to(device)  # (1, H, W)

            # ── Resize to img_size × img_size for evaluation ────────────
            pixel_map_eval = F.interpolate(
                pixel_map.unsqueeze(0),                               # (1, 1, H, W)
                size=(img_size, img_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)                                              # (1, S, S)
            pixel_map_eval = torch.nan_to_num(
                pixel_map_eval, nan=0., posinf=0., neginf=0.)

            # ── GT mask: resize to img_size × img_size ──────────────────
            gt_i = gts_raw[img_idx].to(device)                       # (1, H, W)
            if gt_i.dim() == 2:
                gt_i = gt_i.unsqueeze(0)
            gt_i = (gt_i > 0.5).float()
            gt_i_eval = F.interpolate(
                gt_i.unsqueeze(0),                                    # (1, 1, H, W)
                size=(img_size, img_size),
                mode='nearest',
            ).squeeze(0)                                              # (1, S, S)
            # binarise after resize
            gt_i_eval = (gt_i_eval > 0.5).float()

            # ── Special resize for certain datasets ─────────────────────
            if dataset_name in ['Real-IAD-Variety', 'RealIAD', 'bmad-medical']:
                resize_mask = 256
                pixel_map_eval = F.interpolate(
                    pixel_map_eval.unsqueeze(0),
                    size=(resize_mask, resize_mask),
                    mode='bilinear', align_corners=False,
                ).squeeze(0)
                gt_i_eval = F.interpolate(
                    gt_i_eval.unsqueeze(0),
                    size=(resize_mask, resize_mask),
                    mode='nearest',
                ).squeeze(0).bool().float()

            # ── Aggregate tile global scores → image-level score ────────
            #    max-pool: anomalous if ANY tile is anomalous
            if args.visual_learner:
                vl_score_i = global_vl_score_all[tile_start:tile_end].max()
            if args.textual_learner:
                tl_score_i = global_tl_score_all[tile_start:tile_end].max()
            if args.pq_learner and k_shots > 0:
                pq_score_i = global_pq_score_all[tile_start:tile_end].max()

            pixel_max_i = pixel_map_eval.max()                        # scalar

            if k_shots > 0:
                img_pred = fusion_fun(
                    [vl_score_i.view(1), tl_score_i.view(1), pq_score_i.view(1)],
                    fusion_type=args.fusion_type)
                img_pred = fusion_fun(
                    [img_pred, pixel_max_i.view(1)],
                    fusion_type='harmonic_mean')
            else:
                img_pred = fusion_fun(
                    [vl_score_i.view(1), tl_score_i.view(1), pixel_max_i.view(1)],
                    fusion_type=args.fusion_type)

            img_pred = torch.nan_to_num(img_pred, nan=0., posinf=0., neginf=0.)

            pixel_anomaly_maps_batch.append(pixel_map_eval[0])        # (S, S)
            image_anomaly_preds_batch.append(img_pred)                 # (1,)
            gt_masks_batch.append(gt_i_eval[0])                       # (S, S)

            tile_start = tile_end

        # ── Stack results for this dataloader batch ────────────────────
        pixel_anomaly_map  = torch.stack(pixel_anomaly_maps_batch, dim=0)    # (B, S, S)
        image_anomaly_pred = torch.cat(image_anomaly_preds_batch, dim=0)      # (B,)
        gt_mask            = torch.stack(gt_masks_batch, dim=0).int()         # (B, S, S)

        # ── End timing ─────────────────────────────────────────────────
        torch.cuda.synchronize()
        total_time += time.time() - start_time
        nums       += B

        # ── Accumulate for ceshi evaluation ────────────────────────────
        sample_ids.extend(sample_id)
        cls_names_all.extend(cls_name)
        query_paths.extend(query_path)

        if cpu_eva:
            gt_masks.append(gt_mask.cpu())
            pr_masks.append(pixel_anomaly_map.cpu())
            gt_anomalys.append(gt_anomaly.int().cpu())
            pr_anomalys.append(image_anomaly_pred.cpu())
        else:
            gt_masks.append(gt_mask)
            pr_masks.append(pixel_anomaly_map)
            gt_anomalys.append(gt_anomaly.int())
            pr_anomalys.append(image_anomaly_pred)

    # ── Timing report ──────────────────────────────────────────────────────
    avg_time = total_time / nums if nums > 0 else 0.0
    print(f"Total samples: {nums}, Total time: {total_time:.2f}s, "
          f"Avg time per image: {avg_time * 1000:.1f}ms")
    logger.info(f"\nAvg time per image: {avg_time * 1000:.1f} ms")

    # ==================================================================
    # Build ceshi input format  (unchanged from original)
    # ==================================================================
    cls_names_arr      = np.array(cls_names_all)
    gt_masks_tensor    = torch.cat(gt_masks,    dim=0)              # (N, S, S)
    pr_masks_np        = torch.cat(pr_masks,    dim=0).cpu().numpy()  # (N, S, S)
    gt_anomalys_tensor = torch.cat(gt_anomalys, dim=0)
    pr_anomalys_tensor = torch.cat(pr_anomalys, dim=0)

    results_ceshi = {
        'cls_names':   cls_names_arr.tolist(),
        'imgs_masks':  [gt_masks_tensor[i]  for i in range(len(gt_masks_tensor))],
        'anomaly_maps':[pr_masks_np[i]      for i in range(len(pr_masks_np))],
        'gt_sp':       gt_anomalys_tensor.cpu().tolist(),
        'pr_sp':       pr_anomalys_tensor.cpu().tolist(),
    }

    # ==================================================================
    # Call ceshi  (unchanged from original)
    # ==================================================================
    table_ls = ceshi(results_ceshi)

    if isinstance(table_ls, list) and len(table_ls) > 0:
        headers      = table_ls[0]
        data         = table_ls[1:]
        results_table = tabulate(
            data, headers=headers, tablefmt='pipe',
            floatfmt='.1f', numalign='center', stralign='center')
    else:
        results_table = tabulate(table_ls, tablefmt='pipe')

    logger.info('\n' + results_table)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AdaptCLIP", add_help=True)
    parser.add_argument("--test_data_path",  type=str, default="./data/visa")
    parser.add_argument("--save_path",       type=str, default='./results/')
    parser.add_argument("--pretrained_model",type=str, default='ViT-L/14@336px')
    parser.add_argument("--checkpoint_path", type=str, default='./adaptclip_checkpoint/')
    parser.add_argument("--dataset",         type=str, default='mvtec')
    parser.add_argument("--features_list",   type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--batch_size",      type=int, default=8)
    parser.add_argument("--image_size",      type=int, default=518)
    parser.add_argument("--n_ctx",           type=int, default=12)
    parser.add_argument("--seed",            type=int, default=10)
    parser.add_argument("--sigma",           type=int, default=4)
    parser.add_argument("--k_shots",         type=int, default=1)
    parser.add_argument("--visual_learner",  action="store_true")
    parser.add_argument("--textual_learner", action="store_true")
    parser.add_argument("--pq_learner",      action="store_true")
    parser.add_argument("--eval_metrics",    type=str, nargs="+",
                        default=['I-AUROC', 'I-AP', 'I-F1max',
                                 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO'])
    parser.add_argument("--fusion_type",     type=str, default="average_mean")
    parser.add_argument("--vl_reduction",    type=int, default=4)
    parser.add_argument("--pq_mid_dim",      type=int, default=128)
    parser.add_argument("--pq_context",      action="store_true")
    parser.add_argument("--class_name",      type=str, default=None)
    parser.add_argument("--up",              type=str,   default='bilinear')
    # ── NEW: TIFF saving ──────────────────────────────────────────────────
    parser.add_argument(
        "--save_tiff", action="store_true",
        help=(
            "Save per-image anomaly maps as float16 TIFF files "
            "following the MVTec-AD2 competition layout. "
            "Files are written to <save_path>/anomaly_images/ and "
            "<save_path>/test_public_predictions/."
        ),
    )

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)