DATA="stock"
VQVAEDIR="../dual_vqvae_save_dir"
VQVAECONFIG="configs/train_vq_${DATA}.yaml"
VARCONFIG="configs/train_var_${DATA}.yaml"
VARDIR="../var_save_dir"
VQVAECKPT="/work/vb21/haochen/code/dual_vqvae_save_dir/vq_stock/checkpoints/latest.pt"

#CUDA_VISIBLE_DEVICES=0 python train_dual_vqvae.py \
#--data ${DATA} \
#--config  ${VQVAECONFIG} \
#--max_epochs 5000 \
#--val_every 100 \
#--save_dir ${VQVAEDIR}

CUDA_VISIBLE_DEVICES=0 python train_dual_vqvae.py \
--data ${DATA} \
--config  ${VQVAECONFIG} \
--max_epochs 500 \
--val_every 100 \
--save_dir ${VQVAEDIR}

CUDA_VISIBLE_DEVICES=0 python train_ar.py \
--data ${DATA} \
--vqvae_path ${VQVAECKPT} \
--config  ${VARCONFIG} \
--save_dir ${VARDIR}

#/playpen-shared/haochen/TimeMarReimplement/dual_vqvae_save_dir/vq_stock/checkpoints/best_epoch4200_fid0.000143.pt



