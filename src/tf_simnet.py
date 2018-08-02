import os
import sys
import json
import copy
import random
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from model import SimNet
from dataloader import train_data_loader, test_data_loader, inference_data_loader, batch_iter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help="'train', 'test' or 'inference'")
    parser.add_argument('--encoder_type', help="'cnn', or 'bow'")
    parser.add_argument('--train_pos_file', help="train positive file")
    parser.add_argument('--train_neg_file', help="train negative file")
    parser.add_argument('--dev_pos_file', help="validation positive file")
    parser.add_argument('--dev_neg_file', help="validation negative file")
    parser.add_argument('--test_pos_file', help="test positive file")
    parser.add_argument('--test_neg_file', help="test negative file")
    parser.add_argument('--infer_file', help="inference data file")
    parser.add_argument('--infer_out_file', help="inference output file")
    parser.add_argument('--config', help="config file path, check before train")

    args = parser.parse_args()
    config = json.load(open(args.config))
    return args, config

def train(args, config):
    seq_length = config['seq_length']

    query_train, pos_train, neg_train, query_dev, label_dev, train_pos_idx = train_data_loader(args, config)

    train_datasize = len(query_train)
    dev_datasize = len(query_dev)
    # train_iterator = train_dataset.make_one_shot_iterator()
    # dev_iterator = dev_dataset.make_one_shot_iterator()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    with tf.variable_scope('Model', reuse=None, initializer=init):
        model = SimNet(config, args.encoder_type)

    # train_init_op = train_iterator.make_initializer(train_dataset)
    # dev_init_op = dev_iterator.make_initializer(dev_dataset)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                allow_soft_placement=True,
                                log_device_placement=False)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        # sess.run(train_init_op)
        num_epoch = 0
        step = 0
        global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = float(config['learning_rate'])
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        grad_and_vars = optimizer.compute_gradients(model.pairwise_hinge_loss)
        train_op = optimizer.apply_gradients(grad_and_vars, global_step=global_step)

        checkpoint_dir = os.path.abspath(config['model_dir'])
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        train_batches = batch_iter(list(zip(query_train, pos_train, neg_train)),
                                   config['batch_size'],
                                   config['n_epochs'])

        val_best_acc = 0.0
        for batch in train_batches:
            query_batch, pos_batch, neg_batch = zip(*batch)
            feed_dict = {
                model.query: query_batch,
                model.pos_input: pos_batch,
                model.neg_input: neg_batch
            }
            fetches = {'train_op': train_op,
                       'train_loss': model.pairwise_hinge_loss,
                       'step': global_step}
            fetch_vals = sess.run(fetches, feed_dict)

            if fetch_vals['step'] % 1000 == 1:
                    print('step {}, loss {:g}.'.format(fetch_vals['step'],
                                                        fetch_vals['train_loss']))

            if fetch_vals['step'] % config['val_intervals'] == 0:
                print("Evaluating.")
                pos_fc_out = sess.run(model.query_encoder.fc_out,
                                      feed_dict={model.query: train_pos_idx})
                val_batches = batch_iter(list(zip(query_dev, label_dev)), config['batch_size'], 1, shuffle=False)
                val_corrects = 0
                val_step = 0
                for val_batch in tqdm(val_batches):
                    val_step += len(val_batch)
                    query_dev_batch, label_dev_batch = zip(*val_batch)
                    query_fc_out = sess.run(model.query_encoder.fc_out,
                                            feed_dict={model.query: query_dev_batch})
                    for i in range(len(val_batch)):
                        feed_dict = {
                                model.query_vec: np.repeat(np.expand_dims(query_fc_out[i], axis=0), len(train_pos_idx), axis=0),
                                model.pos_vec: pos_fc_out
                        }
                        val_pos_score = sess.run(model.pos_score, feed_dict)
                        if np.mean(val_pos_score) > config['threshold']:
                            pred_label = 1
                        else:
                            pred_label = 0
                        val_corrects = val_corrects + 1 if pred_label == label_dev_batch[i] else val_corrects
                val_acc = val_corrects * 1.0 / val_step
                # saver.save(sess, checkpoint_prefix, fetch_vals['step'])
                if val_acc >= val_best_acc:
                    val_best_acc = val_acc
                    saver.save(sess, os.path.join(checkpoint_dir, 'model-best'))
                    print("Better val accuray: {:.4g}%, saving at {}.".format(val_acc * 100, checkpoint_dir))
                else:
                    print('Val accuray: {:.4g}%'.format(val_acc * 100))
                sys.stdout.flush()

    sess.close()

def test(args, config):
    query_test, label_test, train_pos_idx, test_pos_num, test_neg_num = test_data_loader(args, config)

    test_datasize = len(query_test)

    with tf.variable_scope('Model', reuse=None):
        model = SimNet(config, args.encoder_type)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                allow_soft_placement=True,
                                log_device_placement=False)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        step = 0
        checkpoint_file = os.path.join(config['model_dir'], 'model-best')
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        test_batches = batch_iter(list(zip(query_test, label_test)), config['batch_size'], 1, shuffle=False)
        test_accuracy = 0.0
        test_precision = 0.0
        test_recall = 0.0
        test_f1 = 0.0

        test_corrects = 0
        test_true_pos = 0
        test_pred_pos = 0

        pos_fc_out = sess.run(model.query_encoder.fc_out,
                              feed_dict={model.query: train_pos_idx})

        for batch in tqdm(test_batches):
            step += len(batch)
            query_test_batch, label_test_batch = zip(*batch)
            query_fc_out = sess.run(model.query_encoder.fc_out,
                                    feed_dict={model.query: query_test_batch})
            for i in range(len(batch)):
                feed_dict = {
                        model.query_vec: np.repeat(np.expand_dims(query_fc_out[i], axis=0), len(train_pos_idx), axis=0),
                        model.pos_vec: pos_fc_out
                }
                test_pos_score = sess.run(model.pos_score, feed_dict)
                if np.mean(test_pos_score) > config['threshold']:
                    pred_label = 1
                else:
                    pred_label = 0
                test_corrects = test_corrects + 1 if pred_label == label_test_batch[i] else test_corrects
                test_true_pos = test_true_pos + 1 if pred_label == 1 and label_test_batch[i] == 1 else test_true_pos
                test_pred_pos = test_pred_pos + 1 if pred_label == 1 else test_pred_pos

        test_accuracy = test_corrects * 1.0 / step
        test_precision = test_true_pos * 1.0 / test_pred_pos
        test_recall = test_true_pos * 1.0 / test_pos_num
        test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall)
        print("Test accuracy : {:.4g}%\nprecision: {:.4g}%\nrecall: {:.4g}%\nf1: {:.4g}%".format(test_accuracy * 100, test_precision * 100, test_recall * 100, test_f1 * 100))

    sess.close()


def inference(args, config):
    infer_text, infer_session, infer_idx, train_pos_idx = inference_data_loader(args, config)
    if os.path.exists(args.infer_out_file):
        os.remove(args.infer_out_file)
    with tf.variable_scope('Model', reuse=None):
        model = SimNet(config, args.encoder_type)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                allow_soft_placement=True,
                                log_device_placement=False)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        step = 0
        batch_num = 0
        checkpoint_file = os.path.join(config['model_dir'], 'model-best')
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        infer_batches = batch_iter(infer_idx, config['batch_size'], 1, shuffle=False)

        pos_fc_out = sess.run(model.query_encoder.fc_out,
                              feed_dict={model.query: train_pos_idx})
        for infer_batch in tqdm(infer_batches):
            step += len(infer_batch)
            infer_fc_out = sess.run(model.query_encoder.fc_out,
                    feed_dict={model.query: infer_batch})
            for i in range(len(infer_batch)):
                feed_dict = {
                        model.query_vec: np.repeat(np.expand_dims(infer_fc_out[i], axis=0), len(train_pos_idx), axis=0),
                        model.pos_vec: pos_fc_out
                }
                infer_pos_score = sess.run(model.pos_score, feed_dict)
                if np.mean(infer_pos_score) > config['threshold']:
                    pred_label = 1
                else:
                    pred_label = 0
                if pred_label == 1:
                    sample_id = batch_num * config['batch_size'] + i
                    with open(args.infer_out_file, 'a') as f:
                        f.write('{}.json,{},{}\n'.format(infer_session[sample_id], ' '.join(infer_text[sample_id]), np.mean(infer_pos_score)))

            batch_num += 1

    sess.close()


def main(_):
    args, config = parse_args()
    if args.mode == 'train':
        train(args, config)
    elif args.mode == 'test':
        test(args, config)
    elif args.mode == 'inference':
        inference(args, config)
    else:
        raise ValueError('Mode is train/test/inferene')

if __name__ == '__main__':
    tf.app.run()

