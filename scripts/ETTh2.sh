# Multivariate forecasting

python main.py --model_name informer --data ETTh2 --seq_len 48 --label_len 48 --pred_len 24 --gpus 1
python main.py --model_name informer --data ETTh2 --seq_len 96 --label_len 96 --pred_len 48 --gpus 1
python main.py --model_name informer --data ETTh2 --seq_len 336 --label_len 336 --num_encoder_layers 3 --num_decoder_layers 2 --pred_len 168 --gpus 1
python main.py --model_name informer --data ETTh2 --seq_len 336 --label_len 168 --num_encoder_layers 3 --num_decoder_layers 2 --pred_len 336 --gpus 1
python main.py --model_name informer --data ETTh2 --seq_len 720 --label_len 336 --num_encoder_layers 3 --num_decoder_layers 2 --pred_len 720 --gpus 1

# Univariate forecasting

python main.py --model_name informer --data ETTh2 --variate u --seq_len 48 --label_len 48 --pred_len 24 --gpus 1
python main.py --model_name informer --data ETTh2 --variate u --seq_len 96 --label_len 96 --pred_len 48 --gpus 1
python main.py --model_name informer --data ETTh2 --variate u --seq_len 336 --label_len 336 --pred_len 168 --gpus 1
python main.py --model_name informer --data ETTh2 --variate u --seq_len 336 --label_len 168 --pred_len 336 --num_encoder_layers 3 --num_decoder_layers 2 --gpus 1
python main.py --model_name informer --data ETTh2 --variate u --seq_len 336 --label_len 336 --pred_len 720 --num_encoder_layers 3 --num_decoder_layers 2 --gpus 1
