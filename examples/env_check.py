from env import RL
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.vec_env import dummy_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time
import os 

models_dir = f"model-PPO-{int(time.time())}"
logdir = f"logs-PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

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



model = PPO("Mlpolicy",env, verbose=1,tensorboard_log=logdir)

TIMESTEPS= 10000
for i in range(1,100)
    model.learn(total_timesteps=TIMESTEPS ,reset_num_timesteps=False, tb_log_name="PPO")
    # model.learn(total_timesteps=TIMESTEPS ,reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")