import argparse
from scripts.train import ObjectHuntTrainer
from scripts.evaluate import ObjectHuntEvaluator
from environments.object_hunt_env import ObjectHuntEnv
from stable_baselines3 import PPO

def train_agent(seed):
    env = ObjectHuntEnv(max_steps=10000, seed=seed)
    trainer = ObjectHuntTrainer(env, PPO, model_kwargs={"learning_rate" : 0.00005, "ent_coef" : 0.01, "verbose": 1})
    trainer.train(timesteps=100000)

def evaluate_agent(seed):
    evaluator = ObjectHuntEvaluator(model_path="models/ppo_object_hunt_weights", seed=seed)
    evaluator.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True, help="Choose to train or evaluate the model")
    args = parser.parse_args()
    seed = 23

    if args.mode == "train":
        train_agent(seed=seed)
    elif args.mode == "evaluate":
        evaluate_agent(seed=seed)
