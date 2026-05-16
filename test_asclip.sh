




#Run inference on the test_private and test_private_mixed splits of MVTec AD 2, and save the results as TIFF files.
python test_new_fenkuai_up_vand4.0.py --dataset mvtec_ad_2  --test_data_path /data/MVTec-AD/mvtec_ad_2 \
        --seed 10 \
        --k_shots 0 \
        --checkpoint_path ./train_fenkuai_sim/use_mvtec/epoch_10.pth \
        --save_path ./results/use_mvtec_test_on_mvtec_ad_2_fenkuai_sim \
        --features_list 6 12 18 24 --image_size 518  --batch_size 8  \
        --n_ctx 12  --vl_reduction 4 \
        --visual_learner --textual_learner \
        --up sim --save_tiff

