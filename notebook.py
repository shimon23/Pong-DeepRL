##################
# Import Libraries
##################

# Ignore warning messages.
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import time
import glob

import gym # Game environment.
import numpy as np  # Handle matrices.
import pickle # Save and restore data package.
from collections import deque # For stacking states.

import tensorflow as tf  # Deep Learning library.
import tensorflow.contrib.layers as layers


# Import my functions and classes:
import DQNetwork as DNQ
import preFunctions as pre
import Memory as Mem


#######################
# Model hyperparameters
#######################
state_size = 4 # Our vector size.
original_state_size = (210, 160, 3)
action_size = 6  # Actions: [stay,stay,up,down,up,down]
stack_size = 4 # stack with 4 states.
stack_states_size = [stack_size,state_size] # The size of the input to neural network.
batch_size = 64  # Batch size.

learning_rate = 0.00001  # Alpha(learning rate).
gamma = 0.99  # Discounting rate.

total_episodes = 5000  # Total episodes for training.
max_steps = 50000  # Max possible steps in an episode.

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00000001  # exponential decay rate for exploration prob

pretrain_length = batch_size  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000  # Number of experiences the Memory can keep

rewards_list = [] # list of all training rewards.

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True
# training = False

### MODIFY THIS TO FALSE IF IS NOT THE FIRST TARINING EPISODE.
# firstTrain = True
firstTrain = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
# episode_render = True
episode_render = False

################
# Neural Network
################
class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
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
    def __init__(self,max_size):
        self.buffer = deque(maxlen= max_size)

    # Add experience to memory:
    def add(self, experience):
        self.buffer.append(experience)

    # Take random batch_size experiences from memory:
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),size=batch_size,replace=False)

        return [self.buffer[i] for i in index]

    # Get all experiences:
    def getAllMemory(self):
        return self.buffer

    # Get the size of the memory:
    def getMemorySize(self):
        return len(self.buffer)

    # Get max size of the memory:
    def getCapacity(self):
        return self.buffer.maxlen


###############
# My functions:
###############
# State to vector function:
# Argument: state - matrix of pixels.
# Reuturn: vector of [P1,P2,xBall,yBall,mX,mY,speed]
def stateToVector(state):
    # [P1,P2,xBall,yBall]
    vector = [0, 0, 0, 0]

    # player1(left) position:
    for i in range(34, 194):
        if (state[i][16][0] == 213):
            if (i == 16):
                for i2 in range(34, 51):
                    if (state[i2][16][0] == 213 and state[i2 + 1][16][0] == 144):
                        vector[0] = i2 - 16 + 1
            else:
                vector[0] = i
            break

    # # player2(right) position:
    for i in range(34, 194):
        if (state[i][140][0] == 92):
            if (i == 34):
                for i2 in range(34, 51):
                    if (state[i2][140][0] == 92 and state[i2 + 1][140][0] == 144):
                        vector[1] = i2 - 16 + 1
            else:
                vector[1] = i
            break

    # Ball position:
    for i in range(34, 194):
        for j in range(0, 160):
            if (state[i][j][0] == 236):
                # print("Ball: x=", i, ",y=", j)
                ball = (i, j)
                vector[2] = i
                vector[3] = j
                break

    return vector


# Get time vector:
# Argument: counter of secondes from the starting tarining.
# Return: vector of: [DAYS,HOURS,MINUTES,SECONDES].
def getTime(counter):
    time = []

    day = counter // (24 * 3600)
    time.append(day)

    counter = counter % (24 * 3600)
    hour = counter // 3600
    time.append(hour)

    counter %= 3600
    minutes = counter // 60
    time.append(minutes)

    seconds = counter % 60
    time.append(seconds)

    return time


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
def predict_action(sess, DQNetwork2, explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.randint(1, len(actions)) - 1
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        state = np.array(state)
        Qs = sess.run(DQNetwork2.output, feed_dict={DQNetwork2.inputs_: state.reshape((1, *state.shape))})
        # Take the biggest Q value (= the best action)
        action = np.argmax(Qs)
        # print(action)

    return action, explore_probability


# Print the action(DOWN,UP,STAY):
# Argument: action - 0/1/2/3/4/5
def actionToString(action):
    if (action == 1 or action == 0):
        print("STAY")
    elif (action == 2 or action == 4):
        print("UP")
    elif (action == 3 or action == 5):
        print("DOWN")


# stack_states function:
# Arguments: 1. stacked_vectors - (deque) deque with 4 vectors.
#            2. state - (matrix) vector of current state.
#            3. is_new_episode - (boolean) check if we start an new episode.
#            4. stack_size - (int).
# Return: 1. stacked_state - (numpy stack).
#         2. stacked_vectors - (deque)
def stack_states(stacked_vectors, state, is_new_episode, stack_size, state_size):
    # Preprocess frame
    stateVec = stateToVector(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_vectors = deque([np.zeros((state_size), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same state 4x
        stacked_vectors.append(stateVec)
        stacked_vectors.append(stateVec)
        stacked_vectors.append(stateVec)
        stacked_vectors.append(stateVec)

        # Stack the frames
        stacked_state = np.stack(stacked_vectors)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_vectors.append(stateVec)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_vectors)

    return stacked_state, stacked_vectors


################
# Initialization
################
# Create log file:
text_file = open("./saveData/log.txt", "a")

# Initialize deque with zero-vectors states.
stacked_vectors  =  deque([np.zeros((state_size), dtype=np.float) for i in range(stack_size)], maxlen=4)

# Instantiate the DQNetwork
DQNetwork2 = DNQ.DQNetwork(stack_states_size, action_size, learning_rate)

# Instantiate memory
memory = Mem.Memory(max_size=memory_size)


class CreateGame:

    def __init__(self):
        # Create our environment:
        env = gym.make('Pong-v0')


##########
# Training
##########
def training(self):
    print("test")

# If is our first training episode:
if(firstTrain):
    # Create log file:
    text_file = open("log.txt", "w")

    # Init memory with states:
    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            state = env.reset()
            state, stacked_vectors = pre.stack_states(stacked_vectors, state, True,stack_size,state_size)

        # Get the next_state, the rewards, done by taking a random action
        action = random.randint(1, len(possible_actions)) - 1
        # action = pre.actionAdapter(choice)
        next_state, reward, done, _ = env.step(action)
        next_state, stacked_vectors = pre.stack_states(stacked_vectors, next_state, False,stack_size,state_size)

        # If the episode is finished (until we get 21)
        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)

            # Add experience to memory
            memory.add((state, possible_actions[action], reward, next_state, done))
            # Start a new episode
            state = env.reset()
            state, stacked_vectors = pre.stack_states(stacked_vectors, state, True,stack_size,state_size)


        else:
            # append to log file:
            text_file = open("./saveData/log.txt", "a")

            # Add experience to memory
            memory.add((state, possible_actions[action], reward, next_state, done))
            # Our new state is now the next_state
            state = next_state
    env.close()


# If we continue with the training:
else:
    # restore memory data:
    with open("./saveData/memory.dq", "rb") as fp:
        temp = pickle.load(fp)

    # Add to memory buffer:
    for i in temp:
        memory.add(i)



# Tensorflow variables for save:

# Episodes counter:
episodeCounter = tf.Variable(1)
step = tf.constant(1)
update = tf.assign(episodeCounter, episodeCounter + step)

# TIme counter:
secondsCounter = tf.Variable(.0)

# Initialize the decay rate (that will use to reduce epsilon)
decay_step = tf.Variable(0)
decay_stepVar = 0

# Saver will help us to save our model
saver = tf.train.Saver()

# Training mode:
if training == True:
    env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)

    with tf.Session() as sess:

        if(firstTrain==False):
            # Load the model and the variables
            saver.restore(sess, "./models/model.ckpt")
        else:
            # Initialize the variables
            sess.run(tf.global_variables_initializer())

        for episode in range(total_episodes):
            startTimeEp = time.time() # Start episode time.

            # Get and print total training time:
            timeVector = pre.getTime(sess.run(secondsCounter))
            print("Ep: %d" %sess.run(episodeCounter),",Total time: D:%d,H:%d,M:%d,S:%d"%(int(timeVector[0]),int(timeVector[1]),int(timeVector[2]),int(timeVector[3])))

            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Record episodes:

            # Make a new episode and observe the first state
            state = env.reset()
            state, stacked_vectors = pre.stack_states(stacked_vectors, state, True,stack_size,state_size)

            while step < max_steps:
                # Increase decay_step
                decay_stepVar += 1

                # Predict the next action:
                action, explore_probability = pre.predict_action(sess, DQNetwork2, explore_start, explore_stop,
                                                                 decay_rate,
                                                                 sess.run(decay_step), state,
                                                                 possible_actions)
                # Perform the action and get the next_state, reward, and done information
                next_state, reward, done, _ = env.step(action)

                # Game display:
                if episode_render:
                    env.render()

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # The episode ends so no next state
                    next_state = np.zeros(original_state_size, dtype=np.int)
                    next_state, stacked_vectors = pre.stack_states(stacked_vectors, next_state, False,stack_size,state_size)

                    # Set step = max_steps to end the episode
                    step = max_steps
                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    # Print episode summery:
                    print('Episode: {}'.format(sess.run(episodeCounter)),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Training Loss {}'.format(loss))

                    # Send the summery to log file:
                    str2 = "Episode: " + str(sess.run(episodeCounter)) + ", Total reward:"+str(total_reward) + ", Explore P: "+ str(explore_probability)+ ", loss: "+str(loss) + "\n"
                    text_file.write(str2)

                    # Add reward to total rewards list:
                    rewards_list.append((episode, total_reward))

                    # Store transition <st,at,rt+1,st+1> in memory D
                    memory.add((state, possible_actions[action], reward, next_state, done))

                else:
                    next_state, stacked_vectors = pre.stack_states(stacked_vectors, next_state, False,stack_size,state_size)

                    # Add experience to memory
                    memory.add((state, possible_actions[action], reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state

                ### LEARNING PART
                #Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                # print(batch)
                states_mb = np.array([each[0] for each in batch],ndmin=2)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch],ndmin=2)
                dones_mb = np.array([each[4] for each in batch])
                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = sess.run(DQNetwork2.output, feed_dict={DQNetwork2.inputs_: next_states_mb})

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

                loss, _,Q= sess.run([DQNetwork2.loss, DQNetwork2.optimizer,DQNetwork2.Q],
                                   feed_dict={DQNetwork2.inputs_: states_mb,
                                              DQNetwork2.target_Q: targets_mb,
                                              DQNetwork2.actions_: actions_mb})

            # Update episode number:
            sess.run(update)

            # Time update:
            endTimeEp = time.time()
            timeUpdate = tf.assign_add(secondsCounter, endTimeEp - startTimeEp)
            sess.run(timeUpdate)

            # Decay update:
            decayStepUpdate = tf.assign_add(decay_step, decay_stepVar)
            sess.run(decayStepUpdate)


            # Save model every 10 episodes
            if episode % 10 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

                # Save memory data:
                with open("./saveData/memory.dq", "wb") as fp:  # Pickling
                    pickle.dump(memory.getAllMemory(), fp)


            # Test every 10 episodes:
            # if episode % 50 == 0:
                total_test_rewards = []

                total_rewards = 0

                state = env.reset()
                state, stacked_vectors = pre.stack_states(stacked_vectors, state, True, stack_size, state_size)


            

#########
# Testing
#########




######
# Main
######
if __name__ == "__main__":
    pong = CreateGame()