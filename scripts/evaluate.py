import cv2
from stable_baselines3 import PPO
from environments.object_hunt_env import ObjectHuntEnv
from graph.graph_builder import GraphBuilder
from neptune_logging.neptune_utils import init_neptune
from utils.param_filename import get_param_filename


class ObjectHuntEvaluator:
    def __init__(self, env, run, model_path, save_videos=False):

        """Initializes the evaluator."""
        self.env = env
        self.model = PPO.load(model_path)
        self.save_videos = save_videos
        self.run = run 

    def evaluate(self, param_filename):
        obs = self.env.reset()
        frames = []
        done = False

        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)

            # frame = self.env.render()
            # cv2.putText(frame, f"Obj count: {len(self.env.collected_objects)}", (5, 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.putText(frame, f"Steps: {self.env.steps}", (5, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # frames.append(frame)
        self.run["eval/final_reward"].append(self.env.total_reward)


        if self.save_videos:
            self.save_video(frames, param_filename)

        GraphBuilder.save_graph(file_path=f"graph/eval_knowledge_graph_{param_filename}.png")
        self.run.stop()

    
    def save_video(self, frames, param_filename):

        filename = f"video/evaluate/eval_ppo_{param_filename}.mp4"
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Saved video: {filename}")

    
