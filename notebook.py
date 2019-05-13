##################
# Import Libraries
##################
# Ignore warning messages.
import warnings
import threading

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import time

import gym  # Game environment.
import numpy as np  # Handle matrices.
import pickle  # Save and restore data package.
from collections import deque  # For stacking states.

import tensorflow as tf  # Deep Learning library.
# import tensorflow.contrib.layers as layers


#######################
# Model hyperparameters
#######################
state_size = 4  # Our vector size.
original_state_size = (210, 160, 3)
action_size = 6  # Actions: [stay,stay,up,down,up,down]
stack_size = 4  # stack with 4 states.
stack_states_size = [stack_size, state_size]  # The size of the input to neural network.
batch_size = 64  # Mini batch size.

# possible_actions = # [[stay],[stay],[up],[down],[up],[down]]
possible_actions = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]]

learning_rate = 0.00001  # Alpha(learning rate).
gamma = 0.99  # Discounting rate.

total_episodes = 10000  # Total episodes for training.
saveEvery = 300  # Save the model every few games.

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.1  # minimum exploration probability
decay_rate = 0.00000001  # exponential decay rate for exploration prob

memory_size = 10000  # Number of experiences the Memory can keep

rewards_list = []  # list of all training rewards.

# MODIFY THIS TO FALSE IF IS NOT THE FIRST TRAINING EPISODE.
firstTraining = True
# firstTraining = False


################
# Neural Network
################
class DQNetwork:
    def __init__(self, name='DQNetwork'):
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            self.inputs_ = tf.placeholder(tf.float32, [None, *stack_states_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

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
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


####################
# Experiences memory
####################
class Memory():

    # Init deque for the memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    # Add experience to memory:
    def add(self, experience):
        self.buffer.append(experience)

    # Take random batch_size experiences from memory:
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)

        # Obtain random mini-batch from memory
        batch = [self.buffer[i] for i in index]
        states_mb = np.array([each[0] for each in batch], ndmin=2)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=2)
        dones_mb = np.array([each[4] for each in batch])

        return states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb

    # Get all experiences:
    def get_all_memory(self):
        return self.buffer

    # Get the size of the memory:
    def get_memory_size(self):
        return len(self.buffer)

    # Get max size of the memory:
    def get_capacity(self):
        return self.buffer.maxlen


###############
# My functions:
###############
# State to vector function:
# Argument: state - matrix of pixels.
# Return: vector of [P1,P2,xBall,yBall]
def state_to_vector(state):
    # [P1,P2,xBall,yBall]
    vector = [0, 0, 0, 0]

    # player1(left) position:
    for i in range(34, 194):
        if state[i][16][0] == 213:
            if i == 16:
                for i2 in range(34, 51):
                    if state[i2][16][0] == 213 and state[i2 + 1][16][0] == 144:
                        vector[0] = i2 - 16 + 1
            else:
                vector[0] = i
            break

    # # player2(right) position:
    for i in range(34, 194):
        if state[i][140][0] == 92:
            if i == 34:
                for i2 in range(34, 51):
                    if state[i2][140][0] == 92 and state[i2 + 1][140][0] == 144:
                        vector[1] = i2 - 16 + 1
            else:
                vector[1] = i
            break

    # Ball position:
    for i in range(34, 194):
        for j in range(0, 160):
            if state[i][j][0] == 236:
                vector[2] = i
                vector[3] = j
                break

    return vector


# Get time vector:
# Argument: counter of seconds from the starting training.
# Return: vector of: [DAYS,HOURS,MINUTES,SECONDS].
def get_time(counter):
    time_vector = []

    day = counter // (24 * 3600)
    time_vector.append(day)

    counter = counter % (24 * 3600)
    hour = counter // 3600
    time_vector.append(hour)

    counter %= 3600
    minutes = counter // 60
    time_vector.append(minutes)

    seconds = counter % 60
    time_vector.append(seconds)

    return time_vector


# Print the action(DOWN,UP,STAY):
# Argument: action - 0/1/2/3/4/5
def action_to_string(action):
    if action == 1 or action == 0:
        print("STAY")
    elif action == 2 or action == 4:
        print("UP")
    elif action == 3 or action == 5:
        print("DOWN")


# Return log file:
# If is the first training - create log file.
# Else - append to the old log file.
def get_log_file():
    if firstTraining:
        # Create log file:
        log = open("./saveData/log.txt", "w")
    else:
        log = open("./saveData/log.txt", "a")

    return log


################
# Initialization
################


class CreateGame:

    def __init__(self):
        # Create our environment:
        self.env = gym.make('Pong-v0')
        # Initialize deque with zero-vectors states.
        self.stacked_vectors = deque([np.zeros(state_size, dtype=np.float) for i in range(stack_size)], maxlen=4)
        # Instantiate the DQNetwork
        self.DQN = DQNetwork()
        # Instantiate memory
        self.memory = self.init_memory()

        # Create log file:
        self.logFile = get_log_file()

        # Tensor flow variables:
        # Episodes counter:
        self.episodeCounter = tf.Variable(1)
        self.step = tf.constant(1)
        self.update = tf.assign(self.episodeCounter, self.episodeCounter + self.step)
        # Time counter:
        self.secondsCounter = tf.Variable(.0)
        # Initialize the decay rate (that will use to reduce epsilon):
        self.decay_step = tf.Variable(0)
        self.decay_stepVar = 0
        self.min_decay_rate = False

        self.episode_render = False

        self.print_actions = False
        self.print_q_values = False

        self.test_next_game = False

        self.min_decay_rate = False

        # Saver will help us to save our model
        self.saver = tf.train.Saver()
        with tf.Session() as self.sess:
            self.sess = self.get_session()

    def get_session(self):
        with tf.Session() as self.sess:
            if not firstTraining:
                # Load the model and the variables
                self.saver.restore(self.sess, "./models/model.ckpt")
                return self.sess
            else:
                # Initialize the variables
                self.sess.run(tf.global_variables_initializer())
                return self.sess

    def init_memory(self):
        temp_memory = Memory(max_size=memory_size)

        if not firstTraining:
            # restore memory data:
            with open("./saveData/memory.dq", "rb") as fp:
                temp = pickle.load(fp)
            # Add to memory buffer:
            for i in temp:
                temp_memory.add(i)

        else:
            state = self.env.reset()
            # Init memory with states:
            for i in range(batch_size):
                # If it's the first step
                if i == 0:
                    state = self.env.reset()
                    state = self.stack_states(state, True)

                # Get the next_state, the rewards, done by taking a random action
                action = random.randint(1, len(possible_actions)) - 1
                next_state, reward, done, info = self.env.step(action)
                next_state = self.stack_states(next_state, False)

                # If the episode is finished (until we get 21)
                if done:
                    # We finished the episode
                    next_state = np.zeros(state.shape)

                    # Add experience to memory
                    temp_memory.add((state, possible_actions[action], reward, next_state, done))
                    # Start a new episode
                    state = self.env.reset()
                    state = self.stack_states(state, True)

                else:
                    # Add experience to memory
                    temp_memory.add((state, possible_actions[action], reward, next_state, done))
                    # Our new state is now the next_state
                    state = next_state

            self.env.close()

        return temp_memory

    # Predict action function: predict the next action:
    # Arguments: 1. sess - tensorflow session.
    #            2. DQNetwork2 - neural network model.
    #            3. explore_start - 1.0(const), for epsilon greedy strategy.
    #            4. explore_stop - 0.1(const), for epsilon greedy strategy.
    #            5. decay_rate - variable, for reducing the selection of a random step during the game.
    #            6. decay_step - variable, for reducing the selection of a random step during the game.
    #            7. state - matrix/vector of the current state.
    #            8. actions - possible actions.
    # Return: 1. action - the predicted action.
    #         2. explore_probability - the current probability for taking random action.
    def predict_action(self, state):
        # EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        # First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        if not self.min_decay_rate:
            # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(
                -decay_rate * self.sess.run(self.decay_step))

            if explore_probability<explore_stop+0.01:
                self.min_decay_rate = True
        else:
            explore_probability = explore_stop

        if explore_probability > exp_exp_tradeoff:
            # Make a random action (exploration)
            action = random.randint(1, len(possible_actions)) - 1
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # state = np.array(state)
            # print(state.shape)
            Qs = self.sess.run(self.DQN.output,
                               feed_dict={self.DQN.inputs_: state.reshape((1,*state.shape))})

            # print(Qs)
            # Take the biggest Q value (= the best action)
            action = np.argmax(Qs)

            if self.print_actions:
                action_to_string(action)
            if self.print_q_values:
                print(Qs)

        return action, explore_probability

    # stack_states function:
    # Arguments: 1. stacked_vectors - (deque) deque with 4 vectors.
    #            2. state - (matrix) vector of current state.
    #            3. is_new_episode - (boolean) check if we start an new episode.
    #            4. stack_size - (int).
    # Return: 1. stacked_state - (numpy stack).
    #         2. stacked_vectors - (deque)
    def stack_states(self, state, is_new_episode):
        # Preprocess frame
        state_vec = state_to_vector(state)

        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_vectors = deque([np.zeros(state_size, dtype=np.int) for i in range(stack_size)], maxlen=4)

            # Because we're in a new episode, copy the same state 4x
            self.stacked_vectors.append(state_vec)
            self.stacked_vectors.append(state_vec)
            self.stacked_vectors.append(state_vec)
            self.stacked_vectors.append(state_vec)

            # Stack the frames
            stacked_state = np.stack(self.stacked_vectors)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_vectors.append(state_vec)
            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_vectors)

        # return stacked_state, self.stacked_vectors
        return stacked_state

    def save_model(self):
        save_path = self.saver.save(self.sess, "./models/model.ckpt")
        print("Model Saved")

        # Save memory data:
        with open("./saveData/memory.dq", "wb") as fp:  # Pickling
            pickle.dump(self.memory.get_all_memory(), fp)

    def print_timer(self):
        # Get and print total training time:
        x = tf.Variable(1)
        time_vector = get_time(self.sess.run(self.secondsCounter))
        print("Ep: %d" % self.sess.run(self.episodeCounter), ",Total time: D:%d,H:%d,M:%d,S:%d" % (
            int(time_vector[0]), int(time_vector[1]), int(time_vector[2]), int(time_vector[3])))

    def get_game_summery(self, total_reward, explore_probability, loss):
        # Print episode summery:
        print('Episode: {}'.format(self.sess.run(self.episodeCounter)),
              'Total reward: {}'.format(total_reward),
              'Explore P: {:.4f}'.format(explore_probability),
              'Training Loss {}'.format(loss))
        # Send the summery to log file:
        str2 = "Episode: " + str(self.sess.run(self.episodeCounter)) + ", Total reward:" + str(
            total_reward) + ", Explore P: " + str(explore_probability) + ", loss: " + str(
            loss) + "\n"
        self.logFile.write(str2)

    def terminal_input(self):
        while True:
            cmd = input()
            if cmd == "info":
                print("Commands list:")
                print("\"r\" - enable/disable episode render.")
                print("\"a\" - enable/disable printing actions.")
                print("\"t\" - test the model in the next episode.")
                print("\"q\" - enable/disable printing q-values.")



            elif cmd == "r":
                self.episode_render = not self.episode_render
            elif cmd == "a":
                self.print_actions = not self.print_actions
            elif cmd == "t":
                self.test_next_game = True
            elif cmd == "q":
                self.print_q_values = not self.print_q_values

            time.sleep(1)

    ##########
    # Training
    ##########

    def training(self):
        with tf.Session() as self.sess:
            if not firstTraining:
                # Load the model and the variables
                self.saver.restore(self.sess, "./models/model.ckpt")
            else:
                # Initialize the variables
                self.sess.run(tf.global_variables_initializer())

            for episode in range(total_episodes):
                start_time_ep = time.time()  # Start episode time.
                # Print total training time:
                self.print_timer()
                # Set step to 0
                step = 0
                # Initialize the rewards of the episode
                episode_rewards = []

                # Make a new episode and observe the first state
                state = self.env.reset()
                state = self.stack_states(state, True)

                done = False

                while not done:
                    # Increase decay_step
                    self.decay_stepVar += 1

                    # Predict the next action:
                    action, explore_probability = self.predict_action(state)
                    # Perform the action and get the next_state, reward, and done information
                    next_state, reward, done, info = self.env.step(action)

                    # Game display:
                    if self.episode_render:
                        self.env.render()

                    # Add the reward to total reward
                    episode_rewards.append(reward)

                    # If the game is finished
                    if done:
                        # The episode ends so no next state
                        next_state = np.zeros(original_state_size, dtype=np.int)
                        next_state = self.stack_states(next_state, False)

                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        # print summery and write to log
                        self.get_game_summery(total_reward, explore_probability, loss)

                        # Add reward to total rewards list:
                        rewards_list.append((episode, total_reward))

                        # Store transition <st,at,rt+1,st+1> in memory D
                        self.memory.add((state, possible_actions[action], reward, next_state, done))

                    else:
                        next_state = self.stack_states(next_state, False)
                        # Add experience to memory
                        self.memory.add((state, possible_actions[action], reward, next_state, done))
                        # st+1 is now our current state
                        state = next_state

                    # LEARNING PART
                    # Obtain random mini-batch from memory
                    batch = self.memory.sample(batch_size)
                    states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = batch
                    target_Qs_batch = []

                    # Get Q values for next_state
                    Qs_next_state = self.sess.run(self.DQN.output,
                                                  feed_dict={self.DQN.inputs_: next_states_mb})

                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])

                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)

                    targets_mb = np.array([each for each in target_Qs_batch])

                    loss, _, Q = self.sess.run([self.DQN.loss, self.DQN.optimizer, self.DQN.Q],
                                               feed_dict={self.DQN.inputs_: states_mb,
                                                          self.DQN.target_Q: targets_mb,
                                                          self.DQN.actions_: actions_mb})

                # Update episode number:
                self.sess.run(self.update)

                # Time update:
                end_time_ep = time.time()
                time_update = tf.assign_add(self.secondsCounter, end_time_ep - start_time_ep)
                self.sess.run(time_update)

                # Decay update:
                decay_step_update = tf.assign_add(self.decay_step, self.decay_stepVar)
                self.sess.run(decay_step_update)

                # Save model every 100 episodes
                if episode % saveEvery == 0 or self.test_next_game:
                    self.save_model()
                    self.testing(1)
                    self.test_next_game = False

    #########
    # Testing
    #########

    def testing(self, games):

        # Testing mode:
        # with tf.Session() as sess:

        self.env = gym.wrappers.Monitor(self.env, "./vid", video_callable=lambda episode_id: True, force=True)

        total_test_rewards = []
        # Load the model
        # self.saver.restore(sess, "./models/model.ckpt")

        for episode in range(games):
            total_rewards = 0

            state = self.env.reset()
            state = self.stack_states(state, True)

            print("TEST EPISODE")

            while True:
                # state_arr = [state]
                # state_arr.append(state)

                # Get action from Q-network
                # Estimate the Qs values state
                # print("1: ",state_arr)
                # print("2: ",state.reshape(*state.shape))
                # print(state.shape)


                # state = np.array(state)
                Qs = self.sess.run(self.DQN.output,
                                   feed_dict={self.DQN.inputs_: state.reshape(1,*state.shape)})
                # Qs = self.sess.run(self.DQN.output, feed_dict={self.DQN.inputs_: state_arr})

                # Take the biggest Q value (= the best action)
                action = np.argmax(Qs[0])

                # if self.print_actions:
                #     action_to_string(action)
                # if self.print_q_values:
                #     print(Qs[0])

                # Perform the action and get the next_state, reward, and done information
                next_state, reward, done, _ = self.env.step(action)

                self.env.render()

                total_rewards += reward

                if done:
                    print("Score", total_rewards)
                    total_test_rewards.append(total_rewards)
                    break

                next_state = self.stack_states(next_state, False)
                state = next_state

        self.env.close()


######
# Main
######
if __name__ == "__main__":
    pong = CreateGame()

    get_terminal_input = threading.Thread(target=pong.terminal_input)
    get_terminal_input.daemon = True
    get_terminal_input.start()

    pong.training()
