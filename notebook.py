##################
# Import Libraries
##################
import threading
import random
import time
import gym  # Game environment.
import numpy as np  # Handle matrices.
import pickle  # Save and restore data package.
from collections import deque  # For stacking states.
import tensorflow as tf  # Deep Learning library.

# Ignore warning messages.
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#######################
# Model hyperparameters
#######################
state_size = 4  # Our vector size.
original_state_size = (210, 160, 3)
action_size = 6  # Actions: [stay,stay,up,down,up,down]
stack_size = 4  # stack with 4 states.
stack_states_size = [stack_size, state_size]  # The size of the input to neural network.
batch_size = 1024  # Mini batch size.

learning_rate = 0.0001  # Alpha(learning rate).
gamma = 0.9  # Discounting rate.

total_episodes = 50000  # Total episodes for training.
saveEvery = 100  # Save the model every few games.

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.1  # minimum exploration probability
decay_rate = 0.000000001  # 0.00000001 exponential decay rate for exploration prob

memory_size = 300000  # Number of experiences the Memory can keep

rewards_list = []  # list of all training rewards.

# MODIFY THIS TO FALSE IF IS NOT THE FIRST TRAINING EPISODE.
firstTraining = True
# firstTraining = False


################
# Neural Network
################
class DQNetwork:
    def __init__(self, name):
        self.graph = tf.Graph()

        # Create the placeholders
        self.inputs_ = tf.placeholder(tf.float32, [None, *stack_states_size], name="inputs_")
        self.target_ = tf.placeholder(tf.float32, [None, action_size], name="target_")

        # [?,4,4] -> [?,16]
        self.flatten = tf.contrib.layers.flatten(self.inputs_)
        # hidden layer:
        self.t1 = tf.layers.dense(self.flatten, 64, activation="relu")
        self.t2 = tf.layers.dense(self.t1, 128, activation="relu")

        # output layer:
        self.t3 = tf.layers.dense(self.t2, 6)
        self.output = self.t3

        self.loss = tf.reduce_mean(tf.pow(self.output - self.target_, 2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


####################
# Experiences memory
####################
class Memory:

    # Init deque for the memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    # Add experience to memory:
    def add(self, experience):
        self.buffer.append(experience)

    # Take random 'size' experiences from memory:
    def sample(self, size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=size, replace=False)

        # Obtain random mini-batch from memory
        batch = [self.buffer[i] for i in index]

        states_mb = np.array([each[0] for each in batch], ndmin=2)
        actions_mb = np.array([each[1] for each in batch])
        prev_rewards_mb = np.array([each[2] for each in batch])
        qs_mb = np.array([each[3] for each in batch], ndmin=2)
        next_states_mb = np.array([each[4] for each in batch])
        best_action_mb = np.array([each[5] for each in batch])
        rewards_mb = np.array([each[6] for each in batch])
        dones_mb = np.array([each[7] for each in batch])

        return states_mb, actions_mb, prev_rewards_mb, qs_mb, next_states_mb, best_action_mb, rewards_mb, dones_mb

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

    def __init__(self, check_point_number):
        # Create our environment:
        self.env = gym.make('Pong-v0')
        # Initialize deque with zero-vectors states:
        self.stacked_vectors = deque([np.zeros(state_size, dtype=np.float) for i in range(stack_size)], maxlen=4)
        # Instantiate the DQNetwork:
        self.graph_main = DQNetwork("main")
        # Instantiate memory:
        self.memory = self.init_memory()

        # Create log file:
        self.logFile = get_log_file()

        self.checkpoint_dir_name = 'checkpoints'

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
            prev_s = self.stack_states(state, True)
            prev_a = 0
            prev_qs = [.0, .0, .0, .0, .0, .0]
            prev_r = 0

            # Init memory with states:
            for i in range(batch_size):
                next_qs = [.0, .0, .0, .0, .0, .0]

                # Get the next_state, the rewards, done by taking a random action
                action = random.randint(1, action_size) - 1
                next_state, reward, done, info = self.env.step(action)
                next_state = self.stack_states(next_state, False)
                best_action = action

                # Add experience to memory
                temp_memory.add((prev_s, prev_a, prev_r, prev_qs, next_state, best_action, reward, done))

                # Our new state is now the next_state
                prev_s = next_state
                prev_a = best_action
                prev_r = reward
                prev_qs = next_qs

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
    #         3. qs - predicted Q-values.
    def predict_action(self, state):
        # EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        # First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        if not self.min_decay_rate:
            # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(
                -decay_rate * self.main_sess.run(self.decay_step))

            if explore_probability < explore_stop + 0.01:
                self.min_decay_rate = True
        else:
            explore_probability = explore_stop

        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        # state.reshape((1, *state.shape)): reshape from [4,4] to [1,4,4]
        # the input of the net is: [none,4,4].
        qs = self.main_sess.run(self.graph_main.output,
                                feed_dict={self.graph_main.inputs_: [state]})

        if explore_probability > exp_exp_tradeoff:
            # Make a random action (exploration)
            action = random.randint(1, action_size) - 1
        else:
            # Take the biggest Q value (= the best action)
            action = np.argmax(qs)

            if self.print_actions:
                action_to_string(action)
            if self.print_q_values:
                print(qs)

        return action, explore_probability, qs

    # stack_states function:
    # Arguments: 1. state - (matrix) vector of current state.
    #            2. is_new_episode - (boolean) check if we start an new episode.
    # Return: 1. stacked_state - (numpy stack).
    def stack_states(self, state, is_new_episode):
        # Preprocess frame
        state_vec = state_to_vector(state)
        state_vec = np.asarray(state_vec)

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

        return stacked_state

    def save_model(self):
        episode = self.main_sess.run(self.episodeCounter)
        filename = self.checkpoint_dir_name + '/' + str(episode) + '-model.ckpt'
        self.graph_main.saver.save(self.main_sess, filename)
        print("Model Saved")

        # Save memory data:
        with open("./saveData/memory.dq", "wb") as fp:  # Pickling
            pickle.dump(self.memory.get_all_memory(), fp)

    def print_timer(self):
        # Get and print total training time:
        time_vector = get_time(self.main_sess.run(self.secondsCounter))
        print("Ep: %d" % self.main_sess.run(self.episodeCounter), ",Total time: D:%d,H:%d,M:%d,S:%d" % (
            int(time_vector[0]), int(time_vector[1]), int(time_vector[2]), int(time_vector[3])))

    def get_game_summery(self, total_reward, explore_probability, loss):
        # Print episode summery:
        print('Episode: {}'.format(self.main_sess.run(self.episodeCounter)),
              'Total reward: {}'.format(total_reward),
              'Explore P: {:.4f}'.format(explore_probability),
              'Training Loss {}'.format(loss))
        # Send the summery to log file:
        str2 = "Episode: " + str(self.main_sess.run(self.episodeCounter)) + ", Total reward:" + str(
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

        self.main_sess = tf.Session()

        if firstTraining:
            self.main_sess.run(tf.global_variables_initializer())
        else:
            self.graph_main.saver.restore(self.main_sess, tf.train.latest_checkpoint('./checkpoints'))
            print(self.main_sess.run(self.secondsCounter))

        for episode in range(total_episodes):
            start_time_ep = time.time()  # Start episode time.
            # Print total training time:
            self.print_timer()

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            state = self.env.reset()
            state = self.stack_states(state, True)

            done = False

            prev_s = state
            prev_a = 0
            [prev_qs] = self.main_sess.run(self.graph_main.output,
                                           feed_dict={self.graph_main.inputs_: [prev_s]})
            prev_r = 0
            done = False

            while not done:
                # Increase decay_step
                self.decay_stepVar += 1

                # Predict the next action:
                action, explore_probability, [next_qs] = self.predict_action(state)
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
                    self.memory.add((prev_s, prev_a, prev_r, prev_qs, next_state, action, reward, done))


                else:
                    next_state = self.stack_states(next_state, False)
                    # Add experience to memory
                    self.memory.add((prev_s, prev_a, prev_r, prev_qs, next_state, action, reward, done))

                prev_s = next_state
                prev_a = action
                prev_r = reward
                prev_qs = next_qs

                # LEARNING PART
                # Obtain random mini-batch from memory
                batch = self.memory.sample(batch_size)
                states_mb, actions_mb, prev_rewards_mb, qs_mb, next_states_mb, best_action_mb, rewards_mb, dones_mb = batch
                target_qs_batch = []

                # Get Q values for next_state
                qs_next_state = self.main_sess.run(self.graph_main.output,
                                                     feed_dict={self.graph_main.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, batch_size):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target = np.zeros(action_size) + rewards_mb[i]
                        target_qs_batch.append(target)

                    else:
                        target = np.copy(qs_mb[i])
                        a = actions_mb[i]
                        r = prev_rewards_mb[i]
                        best = best_action_mb[i]
                        target[a] = r + gamma * qs_next_state[i][best]
                        target_qs_batch.append(target)

                targets_mb = np.array([each for each in target_qs_batch])

                loss, _ = self.main_sess.run([self.graph_main.loss, self.graph_main.optimizer],
                                             feed_dict={self.graph_main.inputs_: states_mb,
                                                        self.graph_main.target_: targets_mb})

            # Update episode number:
            self.main_sess.run(self.update)

            # Time update:
            end_time_ep = time.time()
            time_update = tf.assign_add(self.secondsCounter, end_time_ep - start_time_ep)
            self.main_sess.run(time_update)

            # Decay update:
            decay_step_update = tf.assign_add(self.decay_step, self.decay_stepVar)
            self.main_sess.run(decay_step_update)

            # Save model every 100 episodes
            if episode % saveEvery == 0 or self.test_next_game:
                # self.save_model()
                self.testing(1)
                self.test_next_game = False

    #########
    # Testing
    #########

    def testing(self, games):

        self.env = gym.wrappers.Monitor(self.env, "./vid", video_callable=lambda episode_id: True, force=True)

        total_test_rewards = []

        for episode in range(games):
            total_rewards = 0

            state = self.env.reset()
            state = self.stack_states(state, True)

            print("TEST EPISODE")

            while True:
                # Get action from Q-network
                # Estimate the Qs values state
                qs = self.main_sess.run(self.graph_main.output,
                                        feed_dict={self.graph_main.inputs_: [state]})

                # Take the biggest Q value (= the best action)
                action = np.argmax(qs[0])

                if self.print_actions:
                    action_to_string(action)
                if self.print_q_values:
                    print(qs[0])

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
    pong = CreateGame(2)
    # pong.get_session(1)

    get_terminal_input = threading.Thread(target=pong.terminal_input)
    get_terminal_input.daemon = True
    get_terminal_input.start()

    pong.training()
