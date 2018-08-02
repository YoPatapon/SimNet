/data00/home/wupeihao/anaconda4/bin/python src/tf_simnet.py \
  --mode inference \
  --encoder_type cnn \
  --train_pos_file data/strong/train.pos \
  --infer_file data/strong/20180728_all.txt \
  --infer_out_file data/strong/strong_result_cnn_20180728_t0.3.txt \
  --config conf/train.conf \
