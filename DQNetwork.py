import tensorflow as tf
import tensorflow.contrib.layers as layers


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


            self.flatten = tf.contrib.layers.flatten(self.inputs_)

            # # First layer:
            self.W1 = tf.Variable(tf.contrib.layers.xavier_initializer()((16, 256)))
            self.b1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b1")
            self.z1 = tf.nn.relu(tf.matmul(self.flatten, self.W1) + self.b1, name="z1")

            # # Second layer:
            self.W2 = tf.Variable(tf.contrib.layers.xavier_initializer()((256, 6)))
            # self.W2 = tf.Variable(tf.truncated_normal([1024,6], stddev=0.1), name="W2")
            self.b2 = tf.Variable(tf.constant(0.1, shape=[6]), name="b2")
            self.z2 = tf.matmul(self.z1, self.W2) + self.b2

            self.output = self.z2

            # Q is our predicted Q value.
            # result = double
            self.Q = tf.reduce_sum(tf.multiply(self.output,self.actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q-self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
