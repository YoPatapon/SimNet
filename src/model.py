import inspect
import tensorflow as tf

class CNNEncoder():
    def __init__(self, input_x, config):
        self.embedding_size = config['embedding_size']
        self.vocab_size = config['vocab_size']
        self.seq_len = config['seq_length']
        self.kernel_size = int(config['kernel_size'])
        self.num_filters = int(config['num_filters'])
        self.hidden_size = int(config['hidden_size'])

        with tf.device('/cpu:0'), tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            emb_w = tf.get_variable("emb_w", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_uniform_initializer())
            embedded_chars = tf.nn.embedding_lookup(emb_w, input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # filter_sizes = [k for k in config['filter_sizes'].split(',')]

        with tf.variable_scope('conv-%s' % self.kernel_size, reuse=tf.AUTO_REUSE):

            # Conv layer
            filter_shape = [self.kernel_size, self.embedding_size, 1, self.num_filters]
            self.conv_w = tf.get_variable(name='conv_w', shape=filter_shape,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.conv_b = tf.get_variable(name='conv_b', shape=[self.num_filters],
                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    self.conv_w,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, self.conv_b), name='relu')
            pool = tf.nn.max_pool(
                    h,
                    ksize=[1, self.seq_len - self.kernel_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
            self.pool_flat = tf.reshape(pool, [-1, self.num_filters])

        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
            # FC layer
            self.fc_w = tf.get_variable(name='fc_w',
                    shape=[self.num_filters, self.hidden_size],
                    initializer=tf.random_normal_initializer(stddev=0.1))
            self.fc_b = tf.get_variable(name='fc_b', shape=[self.hidden_size],
                    initializer=tf.constant_initializer(0.1))
            self.fc_out_without_bias = tf.matmul(self.pool_flat, self.fc_w)
            self.fc_out = tf.nn.bias_add(self.fc_out_without_bias, self.fc_b)


class BOWEncoder():
    def __init__(self, input_x, config):
        self.embedding_size = config['embedding_size']
        self.vocab_size = config['vocab_size']
        self.seq_len = config['seq_length']
        self.hidden_size = config['hidden_size']

        with tf.device('/cpu:0'), tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            emb_w = tf.get_variable('emb_w', shape=[self.vocab_size, self.embedding_size], initializer=tf.random_uniform_initializer())
            embedded_chars = tf.nn.embedding_lookup(emb_w, input_x)
            # embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        with tf.variable_scope('pool', reuse=tf.AUTO_REUSE):
            self.seq_emb = tf.nn.softsign(tf.reduce_sum(embedded_chars, axis=1))

        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
            self.fc_w = tf.get_variable(name='fc_w',
                    shape=[self.embedding_size, self.hidden_size],
                    initializer=tf.random_normal_initializer(stddev=0.1))
            self.fc_b = tf.get_variable(name='fc_b', shape=[self.hidden_size],
                    initializer=tf.constant_initializer(0.1))
            self.fc_out_without_bias = tf.matmul(self.seq_emb, self.fc_w)
            self.fc_out = tf.nn.bias_add(self.fc_out_without_bias, self.fc_b)


class SimNet():

    def __init__(self, config, encoder_type='cnn'):
        self.query = tf.placeholder(tf.int32, [None, config['seq_length']], name='query')
        self.pos_input = tf.placeholder(tf.int32, [None, config['seq_length']], name='pos')
        self.neg_input = tf.placeholder(tf.int32, [None, config['seq_length']], name='neg')
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            if encoder_type == 'cnn':
                self.query_encoder = CNNEncoder(self.query, config)
                self.pos_encoder = CNNEncoder(self.pos_input, config)
                self.neg_encoder = CNNEncoder(self.neg_input, config)

            elif encoder_type == 'bow':
                self.query_encoder = BOWEncoder(self.query, config)
                self.pos_encoder = BOWEncoder(self.pos_input, config)
                self.neg_encoder = BOWEncoder(self.neg_input, config)

            else:
                raise ValueError("Encoder type is cnn/bow")

            self.query_vec = self.query_encoder.fc_out
            self.pos_vec = self.pos_encoder.fc_out
            self.neg_vec = self.neg_encoder.fc_out

        with tf.variable_scope('score', reuse=tf.AUTO_REUSE):
            norm_query = tf.nn.l2_normalize(self.query_vec, dim=1)
            norm_pos = tf.nn.l2_normalize(self.pos_vec, dim=1)
            norm_neg = tf.nn.l2_normalize(self.neg_vec, dim=1)

            self.pos_score = tf.expand_dims(tf.reduce_sum(tf.multiply(norm_query, norm_pos), 1), -1)
            self.neg_score = tf.expand_dims(tf.reduce_sum(tf.multiply(norm_query, norm_neg), 1), -1)

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            if encoder_type == 'cnn':
                self.pairwise_hinge_loss = tf.reduce_mean(tf.maximum(0.,
                        self.neg_score + float(config['margin']) - self.pos_score) +
                        0.1 * tf.nn.l2_loss(self.query_encoder.conv_w) + 0.01 * tf.nn.l2_loss(self.query_encoder.fc_w))
            elif encoder_type == 'bow':
                self.pairwise_hinge_loss = tf.reduce_mean(tf.maximum(0.,
                        self.neg_score + float(config['margin']) - self.pos_score) +
                        0.1 * tf.nn.l2_loss(self.query_encoder.fc_w))
            else:
                raise ValueError("Encoder type is cnn/bow")

