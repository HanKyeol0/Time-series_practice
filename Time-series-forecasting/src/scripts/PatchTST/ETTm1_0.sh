model_name=PatchTST
data_path=./dataset/ETT-small/
data_name=ETTm1.csv

python main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.datadir $data_path \
    DATASET.dataname $data_name \
    DEFAULT.exp_name ${model_name}_forecasting_${data_name} \