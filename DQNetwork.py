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
            self.l2 = layers.fully_connected(self.flatten, num_outputs=256, activation_fn=tf.nn.relu)
            self.l3 = layers.fully_connected(self.l2, num_outputs=128, activation_fn=tf.nn.relu)
            self.l4 = layers.fully_connected(self.l3, num_outputs=64, activation_fn=tf.nn.relu)
            self.l5 = layers.fully_connected(self.l4, num_outputs=6, activation_fn=None)

            self.output = tf.nn.softmax(self.l5)

            # Q is our predicted Q value.
            # result = double
            self.Q = tf.reduce_sum(tf.multiply(self.output,self.actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q-self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
