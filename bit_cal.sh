SECONDS=0

echo GPU $1

DATA_TYPES=(lego)
for DATA_TYPE in "${DATA_TYPES[@]}"
do
    echo =======================================================
    echo Render : "${DATA_TYPE}"

    CUDA_VISIBLE_DEVICES=$1  python bit_acc.py --experiment_path your_finetuned_model_path --gt_image_path ./gaussian_models/nerf_synthetic/lego/test/ours_30000 --epoch your_finetuning_epoch --decoder_att ./decoder/cfg_32_bce.json --seed 19701105
    echo DETACH
done

echo Total render execution time: $SECONDS seconds.