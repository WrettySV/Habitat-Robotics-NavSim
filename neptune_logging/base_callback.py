import neptune
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

class NeptuneEpisodeLoggingCallback(BaseCallback):
    def __init__(self, run, verbose=0):
        super().__init__(verbose)
        self.run = run
        self.episode_losses = [] 
        self.episode_value_losses = []

    def _on_step(self):

        if len(self.logger.name_to_value) > 0:
            for key, value in self.logger.name_to_value.items():
                self.run[key].append(value)
                if (key == "train/value_loss"):
                    self.episode_value_losses.append(value)
                if (key == "train/loss"):
                    self.episode_losses.append(value)

        if "dones" in self.locals and np.any(self.locals["dones"]):
            # Calculate the average loss for the episode
            avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0
            avg_value_loss = np.mean(self.episode_value_losses) if self.episode_value_losses else 0
            self.run["train/avg_episode_loss"].append(avg_loss)
            self.run["train/avg_episode_value_loss"].append(avg_value_loss)

            # Reset the episode losses after logging
            self.episode_losses = []
            self.episode_value_losses = []


        return True
