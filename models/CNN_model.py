# coding: utf8
from __future__ import print_function
import tensorflow as tf
from base.base_model import BaseModel
from utils import tf_utils


class CNNModel(BaseModel):
    def __init__(self, config, data):
        super(CNNModel, self).__init__(config, data)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.filter_sizes = self.config.filter_sizes
        self.num_filters = self.config.num_filters
        self.num_class = self.config.num_class
        self.embed_size = self.config.word_dim
        self.seq_len = self.config.max_sent_len
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.drop_keep_rate = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.we = tf.Variable(self.data.embed, name='emb')
        self.input_x = tf.placeholder(tf.int32, (None, self.seq_len))
        inputs = tf.nn.embedding_lookup(self.we, self.input_x)
        self.input_x_ = tf.expand_dims(inputs, -1)
        self.input_x_len = tf.placeholder(tf.int32, (None,))
        self.input_y = tf.placeholder(tf.int32, (None, self.num_class))
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(self.input_x_, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        h_pool = tf.concat(pooled_outputs, -1)
        num_filters_total = self.num_filters * len(self.filter_sizes)
        outputs = tf.reshape(h_pool, [-1, num_filters_total])
        if self.drop_keep_rate is not None:
            outputs = tf.nn.dropout(outputs, keep_prob=self.drop_keep_rate)
        logits = tf_utils.linear(outputs, self.num_class, bias=True, scope='softmax')
        predict_prob = tf.nn.softmax(logits, name='predict_prob')
        predict_label = tf.cast(tf.argmax(logits, 1), tf.int32)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.get_shape().ndims > 1])
            reg_loss = loss + self.config.lambda_l2 * l2_loss
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            if self.config.clipper:
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.clipper)
                train_step = optimizer.apply_gradients(list(zip(grads, tvars)))
            else:
                train_step = optimizer.minimize(loss, global_step=self.cur_epoch_tensor)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.predict_prob = predict_prob
            self.predict_label = predict_label
            self.logits = logits
            self.loss = loss
            self.reg_loss = reg_loss
            self.train_step = train_step
            self.accuracy = accuracy

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

