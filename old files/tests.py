import tensorflow as tf      # Deep Learning library

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

from tensorflow.python.client import device_lib

# device_lib.list_local_devices()
print(tf.test.is_gpu_available())
device_lib.list_local_devices()


# #
# # play(gym.make("Pong-v0"))
# # gym.play(gym.make('Pong-v0'))
#
# env = gym.make('Pong-v0')
#
# #
# env.reset()
# # env.play()
#
# initState = 0
#
# for i in range(2000):
#
#     env.render()
#     # 0 - ?
#     # 1 - ?
#     # 2 - UP
#     # 3 - DOWN
#     # 4 - UP
#     # 5 - DOWN
#     next_state, reward, done, _ = env.step(5)
#     # print(next_state)
#     # if (i == 3):
#     #     initState = next_state
#     print(pre.stateToVector(next_state))
#     # sleep(0.3)
#
#
#
# # env = gym.make('Pong-v0')
# #
# # env.reset()
# # initState = 0
# #
# # for i in range(2000):
# #     env.render()
# #     next_state, reward, done, _ = env.step(random.randint(1,env.action_space.n) - 1)
# #     if (i == 3):
# #         initState = next_state
# #     print(pre.stateToVector(next_state))
# #     # sleep(0.3)
#
#
# env.close()