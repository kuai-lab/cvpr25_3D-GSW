SECONDS=0
echo GPU $1

DATASET_NAME=nerf_synthetic
DATA_PATH=./dataset/${DATASET_NAME}/
MODEL_PATH=./gaussian_models/${DATASET_NAME}/
DECODER_CFG_PATH=./decoder/
POS_GRAD_INIT=0.0000016
POS_GRAD_FIN=0.00000016

DATA_TYPES=(lego  materials  mic ship chair  drums  ficus  hotdog) 

SEEDS=(19701105)
PATCH_SIZE=16
PROPORTION=0.001

LAMBDA_IS=(       1)
LAMBDA_DSSIMS=(  0.2)
LAMBDA_WMS=(     0.4)
LAMBDA_SUBBANDS=(0.3)
WM_LOSS=bce
ALPHA=4
EPOCHS=(2)
BITS=(32)

for SEED in "${SEEDS[@]}"
do
    for LAMBDA_I in "${LAMBDA_IS[@]}"
        do
        for LAMBDA_DSSIM in "${LAMBDA_DSSIMS[@]}"
            do
            for LAMBDA_WM in "${LAMBDA_WMS[@]}"
                do
                for LAMBDA_SUBBAND in "${LAMBDA_SUBBANDS[@]}"
                    do
                    for BIT in "${BITS[@]}"
                    do
                        for DATA_TYPE in "${DATA_TYPES[@]}"
                        do
                            EXP_NAME=${DATASET_NAME}_${LAMBDA_WM}
                 
                            CKPT_DIR=./log_finetuning
                            mkdir -p $CKPT_DIR/$DATA_TYPE/${BIT}bit/$EXP_NAME
                            NOHUP_FILE=$CKPT_DIR/$DATA_TYPE/${BIT}bit/$EXP_NAME/log
                            BIT_FILE=$CKPT_DIR/$DATA_TYPE/${BIT}bit/$EXP_NAME/log_bit
                            RENDER_FILE=$CKPT_DIR/$DATA_TYPE/${BIT}bit/$EXP_NAME/log_render
                            ROBUST_FILE=$CKPT_DIR/$DATA_TYPE/${BIT}bit/$EXP_NAME/log_robust
            
                            echo "======================================================="
                            echo Train : "${DATA_TYPE}"
                            echo Data_dir=$DATA_DIR
                            echo LOGFILE $NOHUP_FILE
                            for EPOCH in "${EPOCHS[@]}"
                            do
                            CUDA_VISIBLE_DEVICES=$1 nohup python -u finetuning_gaussian.py \
                            -s ${DATA_PATH}${DATA_TYPE}/ \
                            -m ${MODEL_PATH}${DATA_TYPE}/ \
                            --eval --data_name $DATA_TYPE --exp_name ${BIT}bit/$EXP_NAME \
                            --decoder_att ${DECODER_CFG_PATH}cfg_${BIT}_${WM_LOSS}.json \
                            --epochs ${EPOCH} --patch ${PATCH_SIZE} --proportion ${PROPORTION} --lambda_subband ${LAMBDA_SUBBAND} \
                            --lambda_i ${LAMBDA_I} --lambda_dssim ${LAMBDA_DSSIM} --lambda_wm ${LAMBDA_WM} \
                            --seed ${SEED} --alpha ${ALPHA} \
                            --position_lr_init ${POS_GRAD_INIT} --position_lr_final ${POS_GRAD_FIN}  > $NOHUP_FILE 2>&1
                        
                            echo LOGFILE $BIT_FILE
                            echo LOGFILE $RENDER_FILE
                            CUDA_VISIBLE_DEVICES=$1 nohup python render.py \
                            -s ${DATA_PATH}${DATA_TYPE} \
                            -m ${MODEL_PATH}${DATA_TYPE} \
                            --experiment_dir ${PROJ_ROOT_DIR}output/${DATA_TYPE} \
                            --exp_name ${BIT}bit/${EXP_NAME} --iteration ${EPOCH} --eval > ${RENDER_FILE} 2>&1
                            done  
                        done
                    done
                done
            done
        done
    done
done