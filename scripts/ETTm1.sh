# Multivariate forecasting

python main.py --model_name informer --data ETTm1 --seq_len 672 --label_len 96 --pred_len 24 --factor 3 --mix --gpus 1
python main.py --model_name informer --data ETTm1 --seq_len 96 --label_len 48 --pred_len 48 --mix --gpus 1
python main.py --model_name informer --data ETTm1 --seq_len 384 --label_len 384 --pred_len 96 --mix --gpus 1
python main.py --model_name informer --data ETTm1 --seq_len 672 --label_len 288 --pred_len 288 --mix --gpus 1
python main.py --model_name informer --data ETTm1 --seq_len 672 --label_len 384 --pred_len 672 --mix --gpus 1

# Univariate forecasting

python main.py --model_name informer --data ETTm1 --variate u --seq_len 96 --label_len 48 --pred_len 24 --mix --gpus 1
python main.py --model_name informer --data ETTm1 --variate u --seq_len 96 --label_len 48 --pred_len 48 --mix --gpus 1
python main.py --model_name informer --data ETTm1 --variate u --seq_len 384 --label_len 384 --pred_len 96 --mix --gpus 1
python main.py --model_name informer --data ETTm1 --variate u --seq_len 384 --label_len 384 --pred_len 288 --mix --gpus 1
python main.py --model_name informer --data ETTm1 --variate u --seq_len 384 --label_len 384 --pred_len 672 --mix --gpus 1
