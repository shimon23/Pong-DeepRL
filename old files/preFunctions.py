from collections import deque  # Ordered collection with ends
import tensorflow as tf
import numpy as np
import random


# # Init for global variables:
# oldX = 0
# oldY = 0


# State to vector function:
# Argument: state - matrix of pixels.
# Reuturn: vector of [P1,P2,xBall,yBall,mX,mY,speed]
def stateToVector(state):
    # [P1,P2,xBall,yBall]
    vector = [0, 0, 0, 0]

    # player1(left) position:
    for i in range(34, 194):
        if (state[i][16][0] == 213):
            if(i == 16):
                for i2 in range(34,51):
                    if(state[i2][16][0] == 213 and state[i2+1][16][0] == 144):
                        vector[0] = i2 - 16 + 1
            else:
                vector[0] = i
            break

    # # player2(right) position:
    for i in range(34, 194):
        if (state[i][140][0] == 92):
            if(i == 34):
                for i2 in range(34,51):
                    if(state[i2][140][0] == 92 and state[i2+1][140][0] == 144):
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


"""
This function will do the part
With ϵϵ select a random action a-t, otherwise select a-t=argmaxaQ(s-t,a)
"""
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
    if(action==1 or action==0):
        print("STAY")
    elif(action==2 or action==4):
        print("UP")
    elif(action== 3 or action==5):
        print("DOWN")



# stack_states function:
# Arguments: 1. stacked_vectors - (deque) deque with 4 vectors.
#            2. state - (matrix) vector of current state.
#            3. is_new_episode - (boolean) check if we start an new episode.
#            4. stack_size - (int).
# Return: 1. stacked_state - (numpy stack).
#         2. stacked_vectors - (deque)
def stack_states(stacked_vectors, state, is_new_episode,stack_size,state_size):
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


