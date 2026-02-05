DATA="stock"
python train_dual_vqvae.py \
--data ${DATA} \
--config "configs/train_vq_${DATA}.yaml" \
--device "cuda" \
--max_epochs 5000 \
--val_every 100 \
--save_dir "dual_vqvae_save_dir"