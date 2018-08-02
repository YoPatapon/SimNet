import re
import os
import sys
import random
import itertools
import numpy as np
import tensorflow as tf
from collections import Counter

def data_to_idx(data, vocab, seq_length=1):
    out_data = []
    eos_id = vocab.char2id("</s>")
    for idx, line in enumerate(data):
        new_data = [None] * seq_length
        for pos in range(seq_length):
            new_data[pos] = vocab.char2id(line[pos]) if pos < len(line) else eos_id
        out_data.append(new_data)

    return out_data

class Vocab():
    """ this class is for vocab operations """

    def __init__(self, vocab_limits=-1):
        """ init for vocab """
        self._vocab_size = vocab_limits
        self.char2id_dict = dict()
        self.id2char_dict = dict()

    def create_vocab(self, data, vocab_limits=-1):
        """ read vocabs from files """
        print 'Start Vocab Create'
        sys.stdout.flush()
        total_data = []
        for words in data:
            total_data.extend(words)

        print 'Data Load End For Vocab Create'
        sys.stdout.flush()
        words = list(set(total_data))
        words.sort()
        words.insert(0, '<unk>')
        words.insert(0, '</s>')
        words.insert(0, '<s>')

        if vocab_limits == -1:
            self._vocab_size = len(words)
        else:
            self._vocab_size = min(vocab_limits, len(words))
        words = words[:self._vocab_size]

        print 'Vocabulary Size: %d' % self._vocab_size
        sys.stdout.flush()
        self.char2id_dict = {w: i for i, w in enumerate(words)}
        self.id2char_dict = {i: w for i, w in enumerate(words)}

    def vocab_size(self):
        """ return vocab size"""
        return self._vocab_size

    def char2id(self, c):
        """ get char id """
        if not self.char2id_dict.has_key(c):
            c = '<unk>'
        return self.char2id_dict[c]

    def id2char(self, idx):
        """ get char with idx """
        return self.id2char_dict[idx]

    def load_metadata(self, filename):
        """ load vocab meta from file """
        if not os.path.exists(filename):
            print 'Vocab Metadata {} does not exists'.format(filename)
            sys.exit(-1)
        self.char2id_dict = dict()
        self.id2char_dict = dict()

        cnt = 0
        for line in open(filename):
            try:
                idx, word = line.strip().split('\t')
                self.char2id_dict[word] = int(idx)
                self.id2char_dict[int(idx)] = word
            except:
                continue
            cnt += 1
            if cnt == self._vocab_size: break
        self._vocab_size = len(self.id2char_dict)
        print 'Loading Vocabulary Size:{}'.format(self._vocab_size)
        sys.stdout.flush()

    def save_metadata(self, filename):
        """ write vocab dict to file """
        with open(filename, 'w') as f:
            for i in range(self._vocab_size):
                c = self.id2char(i)
                f.write('{}\t{}\n'.format(i, c))

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(data_file=""):
    text = list(open(data_file, 'r').readlines()) if data_file != "" else []
    text = [t.strip() for t in text]

    text = [sent.split(' ') for sent in text]

    return text

def train_data_loader(args, config):
    print("loading data...")
    seq_length = int(config['seq_length'])
    train_pos_data = load_data(args.train_pos_file)
    train_neg_data = load_data(args.train_neg_file)

    dev_pos_data = load_data(args.dev_pos_file)
    dev_neg_data = load_data(args.dev_neg_file)
    total_data = train_pos_data + train_neg_data + dev_pos_data + dev_neg_data
    vocab = Vocab()
    vocab.create_vocab(total_data)
    vocab.save_metadata(config['metadata'])

    train_pos_idx = data_to_idx(train_pos_data, vocab, seq_length)
    train_neg_idx = data_to_idx(train_neg_data, vocab, seq_length)
    dev_pos_idx = data_to_idx(dev_pos_data, vocab, seq_length)
    dev_neg_idx = data_to_idx(dev_neg_data, vocab, seq_length)

    query_idx = train_pos_idx

    query_train = []
    pos_train = []
    neg_train = []
    pos_pointer = 0
    neg_pointer = 0

    pos_shuffle_indices = np.random.permutation(np.arange(len(train_pos_idx)))
    neg_shuffle_indices = np.random.permutation(np.arange(len(train_neg_idx)))

    for query in query_idx:
        times = 0
        while times < 1e3:
            times += 1
            query_train.append(query)
            pos_train.append(train_pos_idx[pos_shuffle_indices[pos_pointer]])
            pos_pointer += 1
            if pos_pointer >= len(train_pos_idx):
                np.random.shuffle(pos_shuffle_indices)
                pos_pointer = 0

            neg_train.append(train_neg_idx[neg_shuffle_indices[neg_pointer]])
            neg_pointer += 1
            if neg_pointer >= len(train_neg_idx):
                np.random.shuffle(neg_shuffle_indices)
                neg_pointer = 0
    '''
    train_dataset = tf.data.Dataset.from_tensor_slices({
        "query": np.array(query_train),
        "pos": np.array(pos_train),
        "neg": np.array(neg_train)})
    '''
    # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(config['batch_size']).repeat(config['n_epochs'])
    # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(config['batch_size'])

    query_dev = []
    label_dev = []

    for idx in dev_pos_idx:
        query_dev.append(idx)
        label_dev.append(1)
    for idx in dev_neg_idx:
        query_dev.append(idx)
        label_dev.append(0)
    '''
    dev_dataset = tf.data.Dataset.from_tensor_slices({
        "query": np.array(query_dev),
        "label": np.array(label_dev)})
    '''
    # dev_dataset = dev_dataset.batch(config['batch_size']).repeat(config['n_epochs'])
    # dev_dataset = dev_dataset.batch(1)

    return np.array(query_train), np.array(pos_train), np.array(neg_train), np.array(query_dev), np.array(label_dev),  np.array(train_pos_idx)

def test_data_loader(args, config):
    print("loading test data...")
    seq_length = int(config['seq_length'])
    train_pos_data = load_data(args.train_pos_file)
    test_pos_data = load_data(args.test_pos_file)
    test_neg_data = load_data(args.test_neg_file)

    vocab = Vocab()
    vocab.load_metadata(config['metadata'])

    train_pos_idx = data_to_idx(train_pos_data, vocab, seq_length)
    test_pos_idx = data_to_idx(test_pos_data, vocab, seq_length)
    test_neg_idx = data_to_idx(test_neg_data, vocab, seq_length)

    query_test = []
    label_test = []

    for idx in test_pos_idx:
        query_test.append(idx)
        label_test.append(1)
    for idx in test_neg_idx:
        query_test.append(idx)
        label_test.append(0)

    return np.array(query_test), np.array(label_test), np.array(train_pos_idx), len(test_pos_idx), len(test_neg_idx)


def load_inference_data(infer_file=""):
    session_text = []
    session_index = []
    if infer_file != "":
        for line in open(infer_file, 'r'):
            index, sentence = line.strip().split(',')
            session_text.append(sentence)
            session_index.append(int(index.strip('.json')))

        session_text = [sent.split(' ') for sent in session_text]
    return session_text, np.array(session_index)


def inference_data_loader(args, config):
    print("loading inference data...")
    seq_length = int(config['seq_length'])
    train_pos_data = load_data(args.train_pos_file)
    session_text, session_index = load_inference_data(args.infer_file)

    vocab = Vocab()
    vocab.load_metadata(config['metadata'])

    train_pos_idx = data_to_idx(train_pos_data, vocab, seq_length)
    infer_idx = data_to_idx(session_text, vocab, seq_length)
    return session_text, session_index, np.array(infer_idx), np.array(train_pos_idx)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

