# CUDA_VISIBLE_DEVICES=1，这个是正确的
import os
import pickle
import numpy as np
import torch
from tabulate import tabulate
import sys
import argparse
from collections import defaultdict
import time


from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from skimage import measure



import torchmetrics
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
TORCHMETRICS_AVAILABLE = True
print(f"成功导入 torchmetrics {torchmetrics.__version__}计算方法")



class FastAUPro(Metric):

    def __init__(self, expect_fpr=0.3, max_step=100, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.expect_fpr = expect_fpr
        self.max_step = max_step
        

        self.add_state("all_masks", default=[], dist_reduce_fx="cat")
        self.add_state("all_amaps", default=[], dist_reduce_fx="cat")
        
    def update(self, masks: torch.Tensor, amaps: torch.Tensor):

        self.all_masks.append(masks.cpu())
        self.all_amaps.append(amaps.cpu())
    
    def compute(self):

        if not self.all_masks:
            return torch.tensor(0.0)
            
        masks = dim_zero_cat(self.all_masks)
        amaps = dim_zero_cat(self.all_amaps)
        
        return self._compute_fast_aupro(masks, amaps)
    
    def _compute_fast_aupro(self, masks, amaps):

        device = amaps.device
        min_th, max_th = amaps.min(), amaps.max()
        
        if min_th == max_th:
            return torch.tensor(0.0)
            
        delta = (max_th - min_th) / self.max_step
        thresholds = torch.arange(min_th, max_th, delta, device=device)
        
        pros, fprs = [], []
        
        
        regions_info = self._precompute_all_regions(masks.numpy())
        
        for th in thresholds:
            binary_amaps = (amaps > th).float()
            
            
            pro_value = self._compute_batch_pro(binary_amaps.numpy(), regions_info)
            
            
            inverse_masks = 1 - masks
            fp_pixels = (inverse_masks * binary_amaps).sum()
            fpr = fp_pixels / (inverse_masks.sum() + 1e-8)
            
            pros.append(pro_value)
            fprs.append(fpr.item())
        
        return self._calculate_auc(pros, fprs)
    
    def _precompute_all_regions(self, masks):

        regions_info = []
        
        for i in range(len(masks)):
            mask_np = masks[i]
            labeled_mask = measure.label(mask_np)
            regions = measure.regionprops(labeled_mask)
            regions_info.append(regions)
        
        return regions_info
    
    def _compute_batch_pro(self, binary_amaps, regions_info):

        all_pros = []
        
        for img_idx, regions in enumerate(regions_info):
            if not regions:
                continue
                
            binary_amap = binary_amaps[img_idx]
            
            for region in regions:
                coords = region.coords
                if len(coords) == 0:
                    continue
                    

                try:

                    valid_mask = (
                        (coords[:, 0] < binary_amap.shape[0]) & 
                        (coords[:, 1] < binary_amap.shape[1])
                    )
                    valid_coords = coords[valid_mask]
                    
                    if len(valid_coords) == 0:
                        continue
                        

                    tp_pixels = binary_amap[valid_coords[:, 0], valid_coords[:, 1]].sum()
                    region_pro = tp_pixels / region.area
                    all_pros.append(region_pro)
                    
                except IndexError as e:
                    print(f"坐标索引错误: {e}, 坐标范围: {coords.min(axis=0)} - {coords.max(axis=0)}, 图像形状: {binary_amap.shape}")
                    continue
        
        if not all_pros:
            return 0.0
            
        return np.mean(all_pros)
    
    def _calculate_auc(self, pros, fprs):

        pros, fprs = np.array(pros), np.array(fprs)
        idxes = fprs < self.expect_fpr
        
        if idxes.sum() == 0:
            return torch.tensor(0.0)
        
        fprs_filtered = fprs[idxes]
        pros_filtered = pros[idxes]
        
        if len(fprs_filtered) < 2 or fprs_filtered.max() == fprs_filtered.min():
            return torch.tensor(0.0)
            

        fprs_normalized = (fprs_filtered - fprs_filtered.min()) / (
            fprs_filtered.max() - fprs_filtered.min() + 1e-8
        )
        pro_auc = auc(fprs_normalized, pros_filtered)
        
        return torch.tensor(pro_auc)
def ceshi(results):
        table_ls = []
        auroc_sp_ls = []
        auroc_px_ls = []
        f1_sp_ls = []
        f1_px_ls = []
        aupro03_ls = []
        aupro005_ls = []
        ap_sp_ls = []
        ap_px_ls = []

        for obj in set(results['cls_names']):
            table = []
            gt_px = []
            pr_px = []
            gt_sp = []
            pr_sp = []
            pr_sp_tmp = []
            table.append(obj)
            aupro03_calculator = FastAUPro(expect_fpr=0.3, max_step=200)  # 与sklearn版本一致
            aupro005_calculator = FastAUPro(expect_fpr=0.05, max_step=200)
            for idx in range(len(results['cls_names'])):
                if results['cls_names'][idx] == obj:

                    mask = results['imgs_masks'][idx]
                    

                    if mask.ndim == 4:
                        mask = mask.squeeze()

                    elif mask.ndim == 3:
                        mask = mask.squeeze(0)
                    

                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    

                    if mask.ndim == 2:
                        gt_px.append(mask)
                    else:
                        raise ValueError(f"Unexpected mask shape after processing: {mask.shape}")

                    
                    pr_px.append(results['anomaly_maps'][idx])
                    pr_sp_tmp.append(np.max(results['anomaly_maps'][idx]))
                    pr_sp.append(results['pr_sp'][idx])
                    gt_sp.append(results['gt_sp'][idx])
                    
                    

            gt_px = np.array(gt_px)
            pr_px = np.array(pr_px)
            gt_sp = np.array(gt_sp)
            pr_sp = np.array(pr_sp)
            

            print(f"Processing {obj}:")
            print(f"gt_px shape: {gt_px.shape}, dtype: {gt_px.dtype}")
            print(f"pr_px shape: {pr_px.shape}, dtype: {pr_px.dtype}")

            auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            ap_sp = average_precision_score(gt_sp, pr_sp)
            ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())

            precisions, recalls, _ = precision_recall_curve(gt_sp, pr_sp)
            f1_sp = np.max((2 * precisions * recalls) / (precisions + recalls + 1e-6))

            precisions, recalls, _ = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
            f1_px = np.max((2 * precisions * recalls) / (precisions + recalls + 1e-6))

            if len(gt_px.shape) == 4:
                gt_px = gt_px.squeeze(1)
            if len(pr_px.shape) == 4:
                pr_px = pr_px.squeeze(1)
            print("gt_px.shape:", gt_px.shape)
            print("pr_px.shape:", pr_px.shape)
            gt_px_tensor = torch.from_numpy(gt_px).float()
            pr_px_tensor = torch.from_numpy(pr_px).float()
            

            aupro03_calculator.reset()
            aupro03_calculator.update(gt_px_tensor, pr_px_tensor)
            aupro03 = aupro03_calculator.compute().item()
            
            aupro005_calculator.reset()
            aupro005_calculator.update(gt_px_tensor, pr_px_tensor)
            aupro005 = aupro005_calculator.compute().item()

            table.append(str(np.round(auroc_px * 100, 1)))
            table.append(str(np.round(f1_px * 100, 1)))
            table.append(str(np.round(ap_px * 100, 1)))
            table.append(str(np.round(aupro03 * 100, 1)))
            table.append(str(np.round(aupro005 * 100, 1)))
            table.append(str(np.round(auroc_sp * 100, 1)))
            table.append(str(np.round(f1_sp * 100, 1)))
            table.append(str(np.round(ap_sp * 100, 1)))

            table_ls.append(table)
            auroc_sp_ls.append(auroc_sp)
            auroc_px_ls.append(auroc_px)
            f1_sp_ls.append(f1_sp)
            f1_px_ls.append(f1_px)
            aupro03_ls.append(aupro03)
            aupro005_ls.append(aupro005)
            ap_sp_ls.append(ap_sp)
            ap_px_ls.append(ap_px)


        mean_values = [
            'mean',
            str(np.round(np.mean(auroc_px_ls) * 100, 1)),
            str(np.round(np.mean(f1_px_ls) * 100, 1)),
            str(np.round(np.mean(ap_px_ls) * 100, 1)),
            str(np.round(np.mean(aupro03_ls) * 100, 1)),
            str(np.round(np.mean(aupro005_ls) * 100, 1)),
            str(np.round(np.mean(auroc_sp_ls) * 100, 1)),
            str(np.round(np.mean(f1_sp_ls) * 100, 1)),
            str(np.round(np.mean(ap_sp_ls) * 100, 1))
        ]
        table_ls.append(mean_values)
        return table_ls
