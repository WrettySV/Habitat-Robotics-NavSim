import numpy as np
import gym
from gym import spaces
from environments.object_hunt_env import ObjectHuntEnv
from graph.graph_builder import GraphBuilder


class ObjectHuntTrainer:
    def __init__(self, env, model_class, model_kwargs):
        self.env = env
        self.model = model_class("MlpPolicy", env, **model_kwargs)

    def train(self, timesteps=100000):
        self.model.learn(total_timesteps=timesteps)
        self.model.save("models/ppo_object_hunt_weights")
        GraphBuilder.save_graph(file_path=f'graph/train_knowledge_graph.png')