# Multivariate forecasting

python main.py --model_name informer --data ETTh1 --seq_len 48 --label_len 48 --pred_len 24 --factor 3 --mix --gpus 1
python main.py --model_name informer --data ETTh1 --seq_len 96 --label_len 48 --pred_len 48 --mix --gpus 1
python main.py --model_name informer --data ETTh1 --seq_len 168 --label_len 168 --pred_len 168 --mix --gpus 1
python main.py --model_name informer --data ETTh1 --seq_len 168 --label_len 168 --pred_len 336 --mix --gpus 1
python main.py --model_name informer --data ETTh1 --seq_len 336 --label_len 336 --pred_len 720 --mix --gpus 1

# Univariate forecasting

python main.py --model_name informer --data ETTh1 --variate u --seq_len 720 --label_len 168 --pred_len 24 --mix --gpus 1
python main.py --model_name informer --data ETTh1 --variate u --seq_len 720 --label_len 168 --pred_len 48 --mix --gpus 1
python main.py --model_name informer --data ETTh1 --variate u --seq_len 720 --label_len 336 --pred_len 168 --mix --gpus 1
python main.py --model_name informer --data ETTh1 --variate u --seq_len 720 --label_len 336 --pred_len 336 --mix --gpus 1
python main.py --model_name informer --data ETTh1 --variate u --seq_len 720 --label_len 336 --pred_len 720 --mix --gpus 1
