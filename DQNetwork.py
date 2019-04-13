import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


# class DQNetwork:
#     def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.learning_rate = learning_rate
#
#         with tf.variable_scope(name):
#             # We create the placeholders
#             # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
#             # [None, 84, 84, 4]
#             self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
#             self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
#
#             # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
#             self.target_Q = tf.placeholder(tf.float32, [None], name="target")
#
#             """
#             First convnet:
#             CNN
#             ELU
#             """
#             # Input is 110x84x4
#             self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
#                                           filters=32,
#                                           kernel_size=[8, 8],
#                                           strides=[4, 4],
#                                           padding="VALID",
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
#                                           name="conv1")
#
#             self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
#
#             """
#                         Second convnet:
#                         CNN
#                         ELU
#                         """
#             self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
#                                           filters=64,
#                                           kernel_size=[4, 4],
#                                           strides=[2, 2],
#                                           padding="VALID",
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
#                                           name="conv2")
#
#             self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
#
#             """
#                         Third convnet:
#                         CNN
#                         ELU
#                         """
#             self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
#                                           filters=64,
#                                           kernel_size=[3, 3],
#                                           strides=[2, 2],
#                                           padding="VALID",
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
#                                           name="conv3")
#
#             self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
#
#             self.flatten = tf.contrib.layers.flatten(self.conv3_out)
#
#             self.fc = tf.layers.dense(inputs=self.flatten,
#                                       units=512,
#                                       activation=tf.nn.elu,
#                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                       name="fc1")
#             self.output = tf.layers.dense(inputs=self.fc,
#                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                           units=self.action_size,
#                                           activation=None)
#
#             # Q is our predicted Q value.
#             self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
#
#             # The loss is the difference between our predicted Q_values and the Q_target
#             # Sum(Qtarget - Q)^2
#             self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
#
#             self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
# #

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            #
            # # # First layer:
            # self.W1 = tf.Variable(tf.zeros([self.state_size,128]),name="W1")
            # # self.W1 = tf.Variable(tf.truncated_normal([self.state_size,128], stddev=0.1), name="W1")
            # self.b1 = tf.Variable(tf.constant(0.1, shape=[128]),name="b1")
            # self.z1 = tf.nn.relu(tf.matmul(self.inputs_, self.W1) + self.b1,name="z1")

            # # Second layer:
            # self.W2 = tf.Variable(tf.truncated_normal([128,512], stddev=0.1), name="W2")
            # self.b2 = tf.Variable(tf.constant(0.1, shape=[512]), name="b2")
            # self.z2 = tf.nn.relu(tf.matmul(self.z1, self.W2) + self.b2,name="z2")
            #
            # # Third layer:
            # self.W3 = tf.Variable(tf.truncated_normal([512,512], stddev=0.1), name="W3")
            # self.b3 = tf.Variable(tf.constant(0.1, shape=[512]), name="b3")
            # self.z3 = tf.nn.relu(tf.matmul(self.z2, self.W3) + self.b3,name="z3")
            #
            # # Third layer:
            # self.W4 = tf.Variable(tf.truncated_normal([512,128], stddev=0.1), name="W4")
            # self.b4 = tf.Variable(tf.constant(0.1, shape=[128]), name="b4")
            # self.z4 = tf.nn.relu(tf.matmul(self.z3, self.W4) + self.b4, name="z4")



            self.flatten = tf.contrib.layers.flatten(self.inputs_)
            self.l2 = layers.fully_connected(self.flatten, num_outputs=256, activation_fn=tf.nn.relu)


            self.l3 = layers.fully_connected(self.l2, num_outputs=128, activation_fn=tf.nn.relu)
            self.l4 = layers.fully_connected(self.l3, num_outputs=64, activation_fn=tf.nn.relu)
            self.l5 = layers.fully_connected(self.l4, num_outputs=6, activation_fn=None)

            # self.W5 = tf.Variable(tf.truncated_normal([128, 3], stddev=0.1), name="W5")
            # self.b5 = tf.Variable(tf.constant(0.1, shape=[3]), name="b5")
            # self.z5 = tf.matmul(self.z4, self.W5) + self.b5

            # self.output = self.l5
            self.output = tf.nn.softmax(self.l5)
            # print(self.output.shape)
            # self.outputArgMax = np.argmax(self.output)


            # Q is our predicted Q value.
            # result = double
            self.Q = tf.reduce_sum(tf.multiply(self.output,self.actions_))
            # self.Q = tf.multiply(self.output,self.actions_)

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q-self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
