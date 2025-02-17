import argparse
from scripts.train import ObjectHuntTrainer
from scripts.evaluate import ObjectHuntEvaluator
from environments.object_hunt_env import ObjectHuntEnv
from stable_baselines3 import PPO
from neptune_logging.neptune_utils import init_neptune
import itertools
from utils.param_filename import get_param_filename


reward_params_grid = {"new_object": [100, 1000], "tot_ep_exploration": [0, 100]}
penalty_params_grid = {"same_position": [0, -1], "obstacle": [0, -1]}
ppo_hyperparams_grid = {"learning_rate": [0.0001], "batch_size": [64], "n_epochs": [4], "ent_coef": [0.01], "gamma": [0.9999]}
max_steps_grid = [1000, 10000]
timesteps_grid = [200000]
seeds = [23]


def train_agent(run, reward_params, penalty_params, max_steps, ppo_params, timesteps, seed):
    env = ObjectHuntEnv(mode="train", run=run, max_steps=max_steps, reward_params=reward_params, penalty_params=penalty_params, seed=seed)
    param_filename = get_param_filename(env, max_steps)
    trainer = ObjectHuntTrainer(env, run, model_kwargs=ppo_params)
    trainer.train(timesteps=timesteps, param_filename=param_filename)


def evaluate_agent(run, reward_params, penalty_params, max_steps, ppo_params, timesteps, seed):
    env = ObjectHuntEnv(mode="eval", run=run, max_steps=2000, reward_params=reward_params, penalty_params=penalty_params, seed=seed)
    param_filename = get_param_filename(env, max_steps)
    evaluator = ObjectHuntEvaluator(env, run, model_path=f"models/ppo_object_hunt_weights_{param_filename}")
    evaluator.evaluate(param_filename=param_filename)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", choices=["train", "evaluate"], required=True, help="Choose to train or evaluate the model")

    #args = parser.parse_args()

    for (
        (new_object, tot_ep_exploration), 
        (same_position, obstacle), 
        (learning_rate, batch_size, n_epochs, ent_coef, gamma), 
        max_steps, timesteps, seed
    ) in itertools.product(
        itertools.product(reward_params_grid["new_object"], reward_params_grid["tot_ep_exploration"]),
        itertools.product(penalty_params_grid["same_position"], penalty_params_grid["obstacle"]),
        itertools.product(ppo_hyperparams_grid["learning_rate"], ppo_hyperparams_grid["batch_size"], ppo_hyperparams_grid["n_epochs"], ppo_hyperparams_grid["ent_coef"], ppo_hyperparams_grid["gamma"]),
        max_steps_grid,
        timesteps_grid,
        seeds
    ):
        reward_params = {"new_object": new_object, "tot_ep_exploration": tot_ep_exploration}
        penalty_params = {"same_position": same_position, "obstacle": obstacle}
        ppo_params = {"learning_rate": learning_rate, "batch_size" : batch_size, "n_epochs" : n_epochs, "ent_coef": ent_coef, "gamma": gamma}

        run = init_neptune()
        run["params/reward_params"] = reward_params
        run["params/penalty_params"] = penalty_params
        run["params/ppo_params"] = ppo_params
        run["params/max_steps"] = max_steps
        run["params/timesteps"] = timesteps
        run["params/seed"] = seed

        train_agent(run, reward_params, penalty_params, max_steps, ppo_params, timesteps, seed)
        evaluate_agent(run, reward_params, penalty_params, max_steps, ppo_params, timesteps, seed)

        run.stop()