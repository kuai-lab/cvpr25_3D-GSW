SECONDS=0
echo GPU $1

DATASET_NAME=nerf_synthetic
DATA_PATH=./dataset/${DATASET_NAME}/
MODEL_PATH=./gaussian_models/${DATASET_NAME}/

DATA_TYPES=(lego  materials  mic ship chair  drums  ficus  hotdog) 

SEEDS=(19701105)

WM_LOSS=bce
ALPHA=4
EPOCHS=(8)
BITS=(32)

for SEED in "${SEEDS[@]}"
do
    for BIT in "${BITS[@]}"
    do
        for DATA_TYPE in "${DATA_TYPES[@]}"
        do
            EXP_NAME=${DATASET_NAME}_${BITS}_${SEED}

            for EPOCH in "${EPOCHS[@]}"
            do
                echo LOGFILE $RENDER_FILE
                CUDA_VISIBLE_DEVICES=$1  python metrics.py --wm_path ./output/${DATA_TYPE}/${BIT}bit/${EXP_NAME} --gt_image_path ${MODEL_PATH}${DATA_TYPE}/test/ours_30000/renders/ --epoch ${EPOCH} --decoder_att ./decoder/cfg_32_bce.json --seed ${SEED}

            done  
        done
    done
done