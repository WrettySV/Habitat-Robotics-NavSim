import numpy as np
import gym
from gym import spaces
import cv2
from environments.object_hunt_env import ObjectHuntEnv

class ObjectHuntTrainer:
    def __init__(self, env, model_class, model_kwargs):
        self.env = env
        self.model = model_class("MlpPolicy", env, **model_kwargs)

    def train(self, timesteps=100000):
        self.model.learn(total_timesteps=timesteps)
        self.model.save("models/ppo_object_hunt_weights")

if __name__ == "__main__":
    env = ObjectHuntEnv(max_steps=30000)
    trainer = ObjectHuntTrainer(env, PPO, model_kwargs={"learning_rate" : 0.00005, "ent_coef" : 0.01, "verbose": 1})
    trainer.train(timesteps=100000)

