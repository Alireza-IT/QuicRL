
from locale import windows_locale
from pickle import TRUE
from gym import Env
from gym.spaces import Discrete,Box
import numpy as np
import random
import zmq
import time
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from hashlib import new
# import socket
# import threading
# from pubsub import pub
# import zmq
# import time
from aioquic.quic import recovery
# from examplesimport http3_client
# from pyparsing import FollowedBy
from aioquic.h3.events import (
    DataReceived,
    H3Event,
    HeadersReceived,
    PushPromiseReceived,
)


# ----------Define the constant-------------

# HEADER = 64
# SERVER = socket.gethostbyname(socket.gethostname())
# ADDR = (SERVER, PORT)
# FORMAT = 'utf-8'
# DISCONNECT_MESSAGE = "!DISCONNECT"
# ----------Creating and binding the server----------

# server = socket.socket(socket.AF_INET , socket.SOCK_STREAM)
# server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# server.bind(ADDR)


HEADER = 64
PORT = 2000
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
context = zmq.Context()
server = context.socket(zmq.SUB)
server.connect(f"tcp://{SERVER}:{PORT}")
server.setsockopt_string(zmq.SUBSCRIBE, '')

# ----------handling client-------------

# def listener_client(arg):
#     print('receive data from Quick ', arg['result'])

# pub.subscribe(listener_client,"throughput")

# def handle_client(conn, addr):
#     print(f"[NEW CONNECTION] {addr} connected")

#     connected = True

#     while connected:
#         msg_length = conn.recv(HEADER).decode(FORMAT)

#         if msg_length:
#             msg_length = int(msg_length)
#             msg = conn.recv(msg_length).decode(FORMAT)
#             # print(f"[{addr}] {msg}")
#             # apply action
#             # 0 -1 = -1 temp
#             # 1 -1 = 0 stay
#             # 2 -1 = 1 temp
#             new_value = 0
#             if msg == 0:
#                 new_value = recovery.K_LOSS_REDUCTION_FACTOR - 0.1
#                 print(f"[Decreasing reduction factor] {new_value}")
#             elif msg == 1:
#                 new_value = recovery.K_LOSS_REDUCTION_FACTOR
#                 print(f"[No change] {new_value}")
#             elif msg == 2:
#                 new_value = recovery.K_LOSS_REDUCTION_FACTOR + 0.1
#                 print(f"[Increasing reduction factor] {new_value}")

#             # else msg == DISCONNECT_MESSAGE:
#             #     connected = False

#             # send from server to client
#             octets = 0
#             for http_event in http_events:
#                 if isinstance(http_event, DataReceived):
#                     octets += len(http_event.data)
#                 # throughput  = (octets * 8 / elapsed / 1000000)
#                     print(octets)
#             conn.send({octets}.encode(FORMAT))
#     conn.close()


# ---------START CONNECTION---------
# def start():
#     server.listen()
#     print(f"[LISTENING] Server is listening on {SERVER}")
#     while True:
#         conn , addr = server.accept()
#         thread = threading.Thread(target=handle_client, args=(conn, addr))
#         thread.start()
#         print(f"[ACTIVE CONNECTION] {threading.activeCount() - 1}")
#         time.sleep(0.5)


# print("[STARTING] Server is starting .....")

# qTop = threading.Thread(target=quicToProxy, args=())
# qTop.start()
# pTorl = threading.Thread(target=start, args=())
# pTorl.start()
# qTop.join()
# pTorl.join()

###################STARTING REINFORCEMENT LEARNING#######################

# print(random.randint(-1,1) / 10)

class QuicRL(Env):
    #the __init__ method is called when you first
    # create a class, this as a jump start for all the stuff
    # you want to do at the start when creating your environment
    def __init__(self) :
        #Actions we can take down stay up
        self.action_space = Discrete(2)
        #Throughput Array
        self.observation_space = Box(low=np.array([0.0]),high=np.array([10.0])) #continue of values
        #set start throughput
        self.state = 1.0 + random.randint(-.1,.1)
        #set throughput length
        self.test_length = 60

    def get_data():
        while True:
            data = server.recv_pyobj()
            print(f"receiving data from quic server is {data}")
            time.sleep(0.1)
    

    def step(self,action):
        observation = get_data()
        self.test_length -= 1
        #apply action
        # 0 -1 = -1 
        # 1 -1 = 0
        # 2 -1 = 1 
        self.state += action - 1
        #calculate the reward
        if self.state >= 4 and self.state <=5:
            reward  = 1 
        else : 
            reward = -1
        #check if the test is done
        if self.test_length <= 0 : 
            done = True
        else : 
            done = False
        #return step information
        info= {}
        return self.state,reward,done, info
        
    def render(self):
        # implement VIZ
        pass
    def reset(self):
        self.state = 38 + random.randint(-1,1)
        self.test_length = 60
        return self.state

    
env = QuicRL()

print(env.action_space.sample())
print(env.observation_space.sample())


episodes = 50
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 

    while not done :
        #env.render()
        action = env.action_space.sample()
        n_state,reward,done,info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode,score))


###############Create a Deep Learning Model with Keras ###############
states = env.observation_space.shape
actions = env.action_space.n

print(actions)


def build_model(states,actions):
    model = Sequential()
    model.add(Dense(24,activation="relu",input_shape=states)) # pass value to the deep learning model 
    model.add(Dense(24,activation="relu"))
    model.add(Dense(actions,activation='linear'))
    return model


model = build_model(states,actions)
print(model.summary())
# del model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))