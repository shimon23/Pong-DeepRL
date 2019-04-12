# This ignore all the warning messages that are normally printed during the training because of skiimage
import warnings
warnings.filterwarnings('ignore')
# Cancels the warning message of tensorFlow:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import time

import gym # Game environment.
import tensorflow as tf  # Deep Learning library.
import numpy as np  # Handle matrices.
import pickle # Save and restore data package.

# Import my functions and classes:
import DQNetwork as DNQ
import preFunctions as pre
import Memory as Mem

# Create our environment:
env = gym.make('Pong-v0')
# Create log file:
text_file = open("./saveData/log.txt", "a")

# # possible_actions = # [[stay],[stay],[up],[down],[up],[down]]
possible_actions = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
# # possible_actions = # [[stay],[up],[down]]
# possible_actions = [[1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0]]



### MODEL HYPERPARAMETERS
state_size = 7  # Our vector size.
action_size = len(possible_actions)  # actions
learning_rate = 0.000000001  # Alpha(learning rate)
# test: 0.00000000000000000000001

### TRAINING HYPERPARAMETERS
total_episodes = 50000000  # Total episodes for training
max_steps = 50000  # Max possible steps in an episode
batch_size = 64  # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.5  # minimum exploration probability
decay_rate = 0.000001  # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99  # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 100000  # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True
# training = False

### MODIFY THIS TO FALSE IF IS NOT THE FIRST TARINING EPISODE.
firstTrain = True
# firstTrain = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True
# episode_render = False

rewards_list = [] # list of all training rewards.

# Instantiate the DQNetwork
DQNetwork2 = DNQ.DQNetwork(state_size, action_size, learning_rate)

# Instantiate memory
memory = Mem.Memory(max_size=memory_size)

# If is our first training episode:
if(firstTrain):
    # Create log file:
    text_file = open("log.txt", "w")

    # Init memory with states:
    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            state = env.reset()
            state = pre.stateToVector(state)

        # Get the next_state, the rewards, done by taking a random action
        action = random.randint(1, len(possible_actions)) - 1
        # action = pre.actionAdapter(choice)
        next_state, reward, done, _ = env.step(action)
        next_state = pre.stateToVector(next_state)

        # If the episode is finished (until we get 21)
        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)

            # Add experience to memory
            memory.add((state, possible_actions[action], reward, next_state, done))
            # Start a new episode
            state = env.reset()
            state = pre.stateToVector(state)


        else:
            # append to log file:
            text_file = open("./saveData/log.txt", "a")

            # Add experience to memory
            memory.add((state, possible_actions[action], reward, next_state, done))
            # Our new state is now the next_state
            state = next_state

# If we continue with the training:
else:
    # restore memory data:
    with open("memory.dq", "rb") as fp:
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

            # Make a new episode and observe the first state
            state = env.reset()
            state = pre.stateToVector(state)



            while step < max_steps:
                # Increase decay_step
                decay_stepVar += 1

                # Predict the next action:
                action, explore_probability = pre.predict_action(sess, DQNetwork2, explore_start, explore_stop,
                                                                 decay_rate,
                                                                 sess.run(decay_step), state,
                                                                 possible_actions)
                # Perform the action and get the next_state, reward, and done information
                # action = pre.actionAdapter(action)
                next_state, reward, done, _ = env.step(action)
                next_state = pre.stateToVector(next_state)

                # Game display:
                if episode_render:
                    env.render()

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # The episode ends so no next state
                    next_state = np.zeros(state_size, dtype=np.int)
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
                    # Add experience to memory
                    memory.add((state, possible_actions[action], reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state

                ### LEARNING PART
                #Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                # print(batch)
                states_mb = np.array([each[0] for each in batch])
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch])
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []


                # Get Q values for next_state
                Qs_next_state = sess.run(DQNetwork2.output, feed_dict={DQNetwork2.inputs_: next_states_mb})

                # state2 = np.reshape(state,(1,7))


                # Qs_next = sess.run(DQNetwork2.output, feed_dict={DQNetwork2.inputs_: state2})

                # print(np.argmax(Qs_next))
                # print(Qs_next)
                # break

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
                # print(targets_mb)

            # Update episode number:
            sess.run(update)

            # Time update:
            endTimeEp = time.time()
            timeUpdate = tf.assign_add(secondsCounter, endTimeEp - startTimeEp)
            sess.run(timeUpdate)

            # Decay update:
            decayStepUpdate = tf.assign_add(decay_step, decay_stepVar)
            sess.run(decayStepUpdate)


            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

                # Save memory data:
                with open("memory.dq", "wb") as fp:  # Pickling
                    pickle.dump(memory.getAllMemory(), fp)


# Testing mode:
with tf.Session() as sess:
    total_test_rewards = []
    # Load the model
    saver.restore(sess, "./models/model.ckpt")

    for episode in range(20):
        total_rewards = 0

        state = env.reset()
        state = pre.stateToVector(state)

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            # Convert to Numpy array and reshape the state
            state = np.array(state)
            state = state.reshape((1, state_size))

            # Get action from Q-network
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork2.output, feed_dict={DQNetwork2.inputs_: state})

            # Take the biggest Q value (= the best action)
            action = np.argmax(Qs[0])
            # time.sleep(0.1)
            print(Qs)
            print(action)


            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step(action)
            next_state = pre.stateToVector(next_state)

            env.render()

            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            state = next_state

    env.close()


text_file.close()

