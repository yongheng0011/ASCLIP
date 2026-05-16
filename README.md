# ASCLIP

**ASCLIP** is our solution for the **Industrial Track — Zero-Shot** setting of the [VAND 4.0 @ CVPR 2026](https://sites.google.com/view/vand4-cvpr2026/challenge) challenge, evaluated on the [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) dataset.

> VAND 4.0 (Visual Anomaly and Novelty Detection, 4th Edition) is a workshop challenge held at CVPR 2026. The Industrial Track focuses on robust and generalizable anomaly detection under real-world conditions using the MVTec AD 2 benchmark, which features 8 challenging scenarios with distribution shifts caused by varying lighting and environmental conditions. The Zero-Shot setting requires models to detect anomalies **without any access to defect examples during training**.

---

## 1. Data Preparation

Download the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset and organize it as follows. Then download `meta.json` from the [`mvtec/`](./mvtec_ad/) folder in this repository and place it directly under your MVTec root directory.

```
mvtec_ad/
├── bottle/
├── cable/
├── capsule/
├── ...           # other categories
└── meta.json     # ← download from mvtec/meta.json in this repo
```

Download the [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) dataset and organize it as follows:

```
mvtec_ad_2/
├── can/
│   ├── train/
│   ├── test_public/
│   ├── test_private/
│   ├── test_private_mixed/
│   └── validation/
├── fabric/
│   └── ...
└── ...
```

---

## 2. Environment Setup

```bash
pip install -r requirements.txt
```

---

## 3. Model Weights

A pre-trained checkpoint is provided at:

```
./train_fenkuai_sim/use_mvtec/epoch_10.pth
```

You can use this checkpoint directly for inference, or retrain the model from scratch (see Section 4).

---

## 4. Training (Optional)

To retrain the model, run the script `train_asclip.sh`:

```bash
bash train_asclip.sh
```

which executes:

```bash
python train_fenkuai_up.py \
    --dataset mvtec \
    --train_data_path /data/MVTec-AD/mvtec \
    --save_path ./train_fenkuai_sim/use_mvtec \
    --k_shots 0 \
    --features_list 6 12 18 24 \
    --image_size 518 \
    --batch_size 8 \
    --print_freq 1 \
    --epoch 10 \
    --save_freq 1 \
    --n_ctx 12 \
    --vl_reduction 4 \
    --visual_learner \
    --textual_learner \
    --up sim
```

Modify `--train_data_path` to match your local setup before running.

---

## 5. Inference

### 5.1 Inference on MVTec AD 2 `test_public`

To run inference on the public test split (used for local evaluation):

```bash
bash test_asclip.sh
```

### 5.2 Inference on `test_private` and `test_private_mixed` — Save TIFF Files

The `test_asclip.sh` script also handles inference on the private test splits and saves per-image anomaly maps as float16 TIFF files following the MVTec AD 2 competition submission layout:

```bash
python test_new_fenkuai_up_vand4.0.py \
    --dataset mvtec_ad_2 \
    --test_data_path /data/MVTec-AD/mvtec_ad_2 \
    --seed 10 \
    --k_shots 0 \
    --checkpoint_path ./train_fenkuai_sim/use_mvtec/epoch_10.pth \
    --save_path ./results/use_mvtec_test_on_mvtec_ad_2_fenkuai_sim \
    --features_list 6 12 18 24 \
    --image_size 518 \
    --batch_size 8 \
    --n_ctx 12 \
    --vl_reduction 4 \
    --visual_learner \
    --textual_learner \
    --up sim \
    --save_tiff
```

Modify `--test_data_path` to point to your local MVTec AD 2 directory. TIFF files will be saved under `./submission_folder/anomaly_images/`.

### 6 Convert TIFF Files to PNG

To convert the saved TIFF anomaly maps to PNG images for visualization:

```bash
python tiff_trans_png.py ./submission_folder
```

---

## Acknowledgements

This work builds upon [AdaptCLIP](https://github.com/gaobb/AdaptCLIP). We thank the VAND 4.0 organizers and the MVTec team for providing the benchmark and challenge infrastructure.
