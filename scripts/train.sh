/data00/home/wupeihao/anaconda4/bin/python src/tf_simnet.py \
  --mode train \
  --encoder_type cnn \
  --train_pos_file data/abuse/train.pos.token \
  --train_neg_file data/abuse/train.neg.token \
  --dev_pos_file data/abuse/dev.pos.token \
  --dev_neg_file data/abuse/dev.neg.token \
  --config conf/train.conf \
