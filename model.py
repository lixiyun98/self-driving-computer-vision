import numpy as np
import tensorflow as tf
import os


class LeNet(object):
    def __init__(self, data, lr=0.001):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_placeholder = tf.placeholder(tf.float32, shape=[None, data.img_shape[0], data.img_shape[1]],
                                           name='input_x')
            tf.add_to_collection('x_placeholder', self.x_placeholder)
            self.y_placeholder = tf.placeholder(tf.int32, shape=[None], name='input_y')
            tf.add_to_collection('y_placeholder', self.y_placeholder)
            self.keep_p = tf.placeholder(tf.float32)

            input = tf.reshape(self.x_placeholder, (-1, data.img_shape[0], data.img_shape[1], 1))

            one_hot = tf.one_hot(self.y_placeholder, depth=data.n_classes)
            one_hot = tf.reshape(one_hot, (-1, data.n_classes))

            self.conv1 = self.conv2d(input, 'conv1', 5, 5, 1, 6)
            self.maxpool1 = self.maxpool2d(self.conv1, 'max_pool1')

            self.conv2 = self.conv2d(self.maxpool1, 'conv2', 5, 5, 6, 16)
            self.maxpool2 = self.maxpool2d(self.conv2, 'max_pool2')

            fc1 = tf.reshape(self.maxpool2, [-1, 5 * 5 * 16])
            fc1 = self.fc_layer(fc1, 'fc1', 5 * 5 * 16, 120, relu=True)
            # add dropout layer
            fc1_dropout = tf.nn.dropout(fc1, keep_prob=self.keep_p)

            fc2 = self.fc_layer(fc1_dropout, 'fc2', 120, 84, relu=True)
            fc2_dropout = tf.nn.dropout(fc2, keep_prob=self.keep_p)

            self.output = self.fc_layer(fc2_dropout, 'out', 84, data.n_classes, relu=False)

            with tf.name_scope('global_operation'):
                self.global_step = tf.Variable(0, trainable=False)

                softmax = tf.nn.softmax(self.output, name='softmax')
                self.top_k = tf.nn.top_k(softmax, k=5)

                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=self.output))
                tf.add_to_collection('loss', self.loss)
                tf.summary.scalar('loss', self.loss)

                correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(one_hot, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100
                tf.add_to_collection('accuracy', self.accuracy)
                tf.summary.scalar('accuracy', self.accuracy)

            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                self.train_op = optimizer.minimize(self.loss)

            self.saver = tf.train.Saver()

            self.merge = tf.summary.merge_all()
            log_dir = 'tf_log_dir'
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    def conv2d(self, x, name, filter_height, filter_width, num_input, num_output, strides=1, padding='VALID'):
        with tf.variable_scope(name):
            W = tf.get_variable('weight', [filter_height, filter_width, num_input, num_output],
                                tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.summary.histogram(name+'weight', W)

            b = tf.get_variable('bias', [num_output], tf.float32, initializer=tf.zeros_initializer())
            tf.summary.histogram(name+'bias', b)

            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

    def maxpool2d(self, x, name, k=2, padding='VALID'):
        with tf.name_scope(name):
            x = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)
            return x

    def fc_layer(self, x, name, num_input, num_output, relu=True):
        with tf.variable_scope(name):
            W = tf.get_variable('weight', [num_input, num_output], tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.summary.histogram(name + 'weight', W)
            b = tf.get_variable('bias', [num_output], tf.float32,
                                initializer=tf.zeros_initializer())
            tf.summary.histogram(name + 'bias', b)
            x = tf.nn.xw_plus_b(x, W, b)
            if relu:
                x = tf.nn.relu(x)
                return x
            else:
                return x