from env import RL
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.vec_env import dummy_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time
import os 

models_dir = f"model-PPO-{int(time.time())}"
logdir = f"logs-PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.mkdir(models_dir)
if not os.path.exists(logdir):
    os.mkdir(logdir)

env = RL()


episodes = 10
for episode in range(1, episodes+1 ):
    obs = env.reset()
    done = False
    score = 0 

    while not done:
        action = env.action_space.sample()
        obs ,reward, done ,info  = env.step(action)
        score += reward
    print(f'Episode: {episode}, Score: {score}')
env.close()



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