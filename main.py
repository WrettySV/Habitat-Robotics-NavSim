import argparse
import itertools
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.train import ObjectHuntTrainer
from scripts.evaluate import ObjectHuntEvaluator
from environments.object_hunt_env import ObjectHuntEnv
from neptune_logging.neptune_utils import init_neptune
from utils.param_filename import get_param_filename
from graph.graph_builder import GraphBuilder

def make_env(mode, run, max_steps, reward_params, penalty_params, seed):
    return lambda: ObjectHuntEnv(mode=mode, run=run, max_steps=max_steps, reward_params=reward_params, penalty_params=penalty_params, seed=seed)

def train_and_evaluate(params):
    reward_params, penalty_params, max_steps, timesteps, seed = params
    param_filename = get_param_filename(reward_params,penalty_params,seed,max_steps,timesteps)

    run = init_neptune()
    run["sys/name"] = param_filename
    run["params/reward_params"] = reward_params
    run["params/penalty_params"] = penalty_params
    run["params/max_steps"] = max_steps
    run["params/timesteps"] = timesteps
    run["params/seed"] = seed

    num_envs = 4 #multiprocessing.cpu_count()
    #envs = SubprocVecEnv([make_env("train", run, max_steps, reward_params, penalty_params, seed) for _ in range(num_envs)])
    env = ObjectHuntEnv(mode="train", run=run, max_steps=max_steps, reward_params=reward_params, penalty_params=penalty_params, seed=seed, param_filename=param_filename)
    trainer = ObjectHuntTrainer(env, run)
    trainer.train(timesteps=timesteps, param_filename=param_filename)
    #envs.close()
    env.close()

    GraphBuilder.reset_graph()
    
    # Create a single evaluation environment
    eval_env = ObjectHuntEnv(mode="eval", run=run, max_steps=20000, reward_params=reward_params, penalty_params=penalty_params, seed=seed, param_filename=param_filename)
    evaluator = ObjectHuntEvaluator(eval_env, run, model_path=f"models/ppo_object_hunt_weights_{param_filename}")
    evaluator.evaluate(param_filename=param_filename)
    eval_env.close()
    
    run.stop()

if __name__ == "__main__":
    reward_params_grid = {"new_object": [100], "tot_ep_exploration": [500]}
    penalty_params_grid = {"same_position": [0], "obstacle": [500]}
    #ppo_hyperparams_grid = {"learning_rate": [0.0003, 2.5e-4], "batch_size": [64], "n_epochs": [4], "ent_coef": [0.01], "gamma": [0.9999]}
    max_steps_grid = [2000]
    timesteps_grid = [2000000]
    seeds = [56,57]



    param_combinations = list(itertools.product(
        itertools.product(reward_params_grid["new_object"], reward_params_grid["tot_ep_exploration"]),
        itertools.product(penalty_params_grid["same_position"], penalty_params_grid["obstacle"]),
        #itertools.product(ppo_hyperparams_grid["learning_rate"], ppo_hyperparams_grid["batch_size"], 
        #                  ppo_hyperparams_grid["n_epochs"], ppo_hyperparams_grid["ent_coef"], ppo_hyperparams_grid["gamma"]),
        max_steps_grid,
        timesteps_grid,
        seeds
    ))

    param_list = []
    for (new_object, tot_ep_exploration), (same_position, obstacle), max_steps, timesteps, seed in param_combinations:
        reward_params = {"new_object": new_object, "tot_ep_exploration": tot_ep_exploration}
        penalty_params = {"same_position": same_position, "obstacle": obstacle}
        #ppo_params = {"learning_rate": learning_rate, "batch_size": batch_size, "n_epochs": n_epochs, "ent_coef": ent_coef, "gamma": gamma}
        param_list.append((reward_params, penalty_params, max_steps, timesteps, seed))

    with multiprocessing.Pool(processes=2) as pool:
        pool.map(train_and_evaluate, param_list)
