import cv2
from stable_baselines3 import PPO
from environments.object_hunt_env import ObjectHuntEnv
from graph.graph_builder import GraphBuilder

class ObjectHuntEvaluator:
    def __init__(self, model_path, num_episodes=3, max_steps=30000, seed=None, save_videos=True):
        """Initializes the evaluator."""
        self.model = PPO.load(model_path)
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.save_videos = save_videos
        self.seed = seed

    def evaluate(self):
        """Runs multiple evaluation episodes and records videos."""
        for episode in range(1, self.num_episodes + 1):
            env = ObjectHuntEnv(max_steps=self.max_steps, seed = self.seed)
            obs = env.reset()
            frames = []
            done = False

            while not done:
                action, _states = self.model.predict(obs)
                obs, reward, done, _ = env.step(action)

                frame = env.render()
                cv2.putText(frame, f"Episode: {episode}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Obj count: {len(env.collected_objects)}", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Steps: {env.steps}", (5, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                frames.append(frame)

            if self.save_videos:
                self.save_video(frames, episode)

            GraphBuilder.save_graph(file_path=f'graph/evaluate_knowledge_graph_{episode}.png')

    def save_video(self, frames, episode):
        """Saves frames as a video file."""
        height, width, _ = frames[0].shape
        filename = f'ppo_object_hunt_ep{episode}.mp4'
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Saved video: {filename}")

if __name__ == "__main__":
    evaluator = ObjectHuntEvaluator(model_path="models/ppo_object_hunt_weights", num_episodes=3)
    evaluator.evaluate()

