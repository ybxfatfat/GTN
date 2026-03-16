python3 main.py --model MPNRec --batch_size 128 --batches_per_check 1000  --require_improvement 6000 --debug --multi_gpu --device 0,1,2
nohup python3 main.py --model MPNRec --batch_size 128 --batches_per_check 1000  --require_improvement 6000 --multi_gpu 2> &1 &

python3 main.py --model wide_deep --text_encoding bert --batch_size 128 --batches_per_check 100 --require_improvement 1000 --agg_method pooling  --use_pretrain --debug


wide_deep
nohup python3 main.py --model wide_deep --text_encoding bert --batch_size 512 --batches_per_check 300 --require_improvement 3000  --use_pretrain --device 0 > wide_deep_bert_pretrain.log 2>&1 &
nohup python3 main.py --model wide_deep --text_encoding bert --batch_size 512 --batches_per_check 300 --require_improvement 3000  --device 0 > wide_deep_bert_no_pretrain.log 2>&1 &

DIN
nohup python3 main.py --model DIN --text_encoding bert --batch_size 512 --batches_per_check 300 --require_improvement 3000  --use_pretrain --device 0 > din_bert_pretrain.log 2>&1 &
nohup python3 main.py --model DIN --text_encoding bert --batch_size 512 --batches_per_check 300 --require_improvement 3000  --device 0 > din_bert_no_pretrain.log 2>&1 &


multi_gpu
nohup python3 main.py --model wide_deep --text_encoding w2v --batch_size 256 --batches_per_check 2000 --require_improvement 15000 --use_pretrain --multi_gpu --agg_method pooling --device 1,2 --text_len 128 --history_len 64 > wide_deep_w2v_pretrain_pooling.log 2>&1 &
nohup python3 main.py --model wide_deep --text_encoding w2v --batch_size 256 --batches_per_check 2000 --require_improvement 15000 --use_pretrain --multi_gpu --agg_method self_attention --device 1,2 --text_len 128 --history_len 64 > wide_deep_w2v_pretrain_self_att.log 2>&1 &
# nohup python3 main.py --model wide_deep --text_encoding w2v --batch_size 128 --batches_per_check 1500 --require_improvement 120000 --use_pretrain --multi_gpu --agg_method self_attention --device 1,2 > wide_deep_w2v_pretrain_self_att.log 2>&1 &


NRMS
nohup python3 main.py --model NRMS --text_encoding bert --batch_size 512 --batches_per_check 500 --require_improvement 6000 --use_pretrain --device 0,1,2 --multi_gpu --prefix 001_bs_384 > NRMS_001_bs_384.log 2>&1 &
nohup python3 main.py --model NRMS --text_encoding w2v --batch_size 128 --batches_per_check 500 --require_improvement 6000 --use_pretrain --device 0,1,2 --multi_gpu --prefix 002_w2v > NRMS_001_bs_384.log 2>&1 &


python3 main.py --model wide_deep --text_encoding w2v --batch_size 512 --batches_per_check 300 --require_improvement 3000 --agg_method pooling  --use_pretrain --debug --device 0


nohup python3 main.py --model MPNRec --batch_size 512 --batches_per_check 500 --require_improvement 8000 --multi_gpu --device 0,1,2 --sample_size 50 --lr 0.001 --prefix new_sample_006 > MPNRec006.log 2>&1 &

python3 main.py --model MPNRec --batch_size 768 --batches_per_check 300 --require_improvement 3000 --multi_gpu --device 0,1,2 --sample_size 30 --lr 0.001 --prefix new_sample  > MPNRec005.log 2>&1 &

python3 main.py --model MPNRec --batch_size 768 --batches_per_check 300 --require_improvement 3000 --multi_gpu --device 0,1,2 --sample_size 30 --lr 0.001 --prefix reinit_epoch --reinit_epoch  > MPNRec005.log 2>&1 &


nohup python3 main.py --model MPNRec --batch_size 768 --batches_per_check 300 --require_improvement 5000 --multi_gpu --device 0,1,2 --sample_size 10 --lr 0.001 --prefix 004 > MPNRec004.log 2>&1 &


nohup python3 main.py --model MPNRec --batch_size 768 --batches_per_check 300 --require_improvement 5000 --multi_gpu --device 0,1,2 --sample_size 30 --lr 0.001 --prefix new_sample_30 > MPNRec_new_sample_30.log 2>&1 &
nohup python3 main.py --model MPNRec --batch_size 768 --batches_per_check 300 --require_improvement 5000 --multi_gpu --device 0,1,2 --sample_size 50 --lr 0.001 --prefix new_sample_50 > MPNRec_new_sample_50.log 2>&1 &

nohup python3 main.py --model MPNRec --batch_size 768 --batches_per_check 300 --require_improvement 3000 --multi_gpu --device 0,1,2 --sample_size 30 --lr 0.001 --prefix 007 > MPNRec007.log 2>&1]

nohup python3 main.py --model wide_deep --text_encoding w2v --batch_size 300 --batches_per_check 500 --require_improvement 5000 --agg_method pooling --use_pretrain --device 0,1,2 --prefix 001_w2v_pool --multi_gpu > wide_deep_001_w2v_pool.log 2>&1 &
nohup python3 main.py --model wide_deep --text_encoding w2v --batch_size 300 --batches_per_check 1000 --require_improvement 10000 --agg_method self_attention --use_pretrain --device 0,1,2 --prefix 002_w2v_att --multi_gpu > wide_deep_002_w2v_att.log 2>&1 &

nohup python3 main.py --model DIN --text_encoding w2v --batch_size 300 --batches_per_check 1000 --require_improvement 10000 --agg_method pooling --use_pretrain --device 0,1,2 --prefix 005_w2v_pool --multi_gpu > DIN_005_w2v_pool.log 2>&1 &
nohup python3 main.py --model DIN --text_encoding w2v --batch_size 300 --batches_per_check 1000 --require_improvement 10000 --agg_method self_att --use_pretrain --device 0,1,2 --prefix 006_w2v_att --multi_gpu > DIN_006_w2v_att.log 2>&1 &