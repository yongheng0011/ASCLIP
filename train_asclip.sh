



CUDA_VISIBLE_DEVICES=0 python train_fenkuai_up.py --dataset mvtec --train_data_path /data/MVTec-AD/mvtec/ \
--save_path /workspace/project/AdaptCLIP/train_fenkuai_sim/use_mvtec \
--k_shots 0 \
--features_list 6 12 18 24 --image_size 518  --batch_size 8  --print_freq 1 \
--epoch 15 --save_freq 1  \
--n_ctx 12  --vl_reduction 4 \
--visual_learner --textual_learner  \
--up sim



