SECONDS=0
echo GPU $1

DATASET_NAME=nerf_synthetic
DATA_PATH=./dataset/${DATASET_NAME}/
MODEL_PATH=./gaussian_models/${DATASET_NAME}/

DATA_TYPES=(lego  materials  mic ship chair  drums  ficus  hotdog) 
DATA_TYPES=(lego) 

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
    
            CKPT_DIR=./log_finetuning
            mkdir -p $CKPT_DIR/$DATA_TYPE/${BIT}bit/$EXP_NAME
            RENDER_FILE=$CKPT_DIR/$DATA_TYPE/${BIT}bit/$EXP_NAME/log_render

            echo "======================================================="
            echo Render : "${DATA_TYPE}"
            for EPOCH in "${EPOCHS[@]}"
            do
                echo LOGFILE $RENDER_FILE
                CUDA_VISIBLE_DEVICES=$1 nohup python render.py \
                -s ${DATA_PATH}${DATA_TYPE} \
                -m ${MODEL_PATH}${DATA_TYPE} \
                --exp_name ${DATA_TYPE}/${BIT}bit/${EXP_NAME} --iteration ${EPOCH} --eval > ${RENDER_FILE} --wm 2>&1
            done  
        done
    done
done