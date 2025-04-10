model_name=AnomalyTransformer
data_path=./dataset/
data_name=PSM

python main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.datadir $data_path \
    DATASET.dataname $data_name \
    DEFAULT.exp_name ${model_name}_AD_${data_name} \