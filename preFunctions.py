from skimage import transform  # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames

from collections import deque  # Ordered collection with ends

import numpy as np
import random


# Init for global variables:
oldX = 0
oldY = 0


# State to vector function:
# Argument: state - matrix of pixels.
# Reuturn: vector of [P1,P2,xBall,yBall,mX,mY,speed]
def stateToVector(state):
    # [P1,P2,xBall,yBall,mX,mY,speed]
    vector = [0, 0, 0, 0, 0, 0,0]

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

    global oldX
    global oldY



    # Set defulat values for some states:
    # If the state does not contain ball, define speed and m's to 0.
    if((vector[2]==0 and vector[3]==0) or (oldX==0 and oldY==0)):
        vector[4] = 0
        vector[5] = 0

        oldX = vector[2]
        oldY = vector[3]

        vector[6] = 0


    else:
        x1MinusX2 = vector[2] - oldX
        y1MinusY2 = vector[3] - oldY

        dist = np.sqrt(np.power(x1MinusX2, 2) + np.power(y1MinusY2, 2))
        vector[6] = dist / 2

        dirX = vector[2] - oldX
        dirY = vector[3] - oldY

        oldX = vector[2]
        oldY = vector[3]

        vector[4] = dirX
        vector[5] = dirY



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
        print(action)

    return action, explore_probability


# actions space consist 6 actions, but they are only 3 actions(stay,up,down) are possible.
def actionAdapter(action):
    if(action==1):
        return 0
    elif(action==2):
        return 4
    elif(action== 3):
        return 5
    else:
        return action
