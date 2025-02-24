import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from environments.object_hunt_env import ObjectHuntEnv
from graph.graph_builder import GraphBuilder
from neptune_logging.neptune_utils import init_neptune
from neptune_logging.base_callback import NeptuneEpisodeLoggingCallback


class ObjectHuntTrainer:
    def __init__(self, env, run):
        self.env = env
        self.model = PPO("MultiInputPolicy", env)

        self.run = run
        
    def train(self, timesteps, param_filename):
        callback = NeptuneEpisodeLoggingCallback(self.run)
        self.model.learn(total_timesteps=timesteps,callback=callback)
        self.run["train/final_reward"].append(self.env.total_reward)

        self.model.save(f"models/ppo_object_hunt_weights_{param_filename}")
        GraphBuilder.save_graph(file_path=f"graph/train_knowledge_graph_{param_filename}.png")

    