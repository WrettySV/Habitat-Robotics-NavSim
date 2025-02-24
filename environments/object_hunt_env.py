import numpy as np
import gym
from gym import spaces
import cv2
from maps.map_generator import HabitatMapGenerator
import networkx as nx

class ObjectHuntEnv(gym.Env):
    def __init__(self, mode, run, reward_params, penalty_params, seed, param_filename, max_steps, map_size=(200, 200), num_objects=10, view_distance=20, view_angle=60, agent_size=(10, 10)):
        super().__init__()

        self.mode = mode
        self.run = run
        self.map_size = map_size
        self.num_objects = num_objects
        self.max_steps = max_steps
        self.view_distance = view_distance
        self.view_angle = view_angle
        self.agent_size = agent_size
        
        self.instance_map = HabitatMapGenerator.generate(map_size=map_size, num_objects = num_objects, seed=seed)
        
        self.agent_pos = self.set_agent_start_position()
        self.agent_dir = np.random.randint(0, 360)
        self.collected_objects = set()
        self.cmlt_collected_objects = set() #do not reset

        self.visited_positions = set()
        self.steps = 0             
        self.steps_since_last_object = 0
        self.steps_between_objects = []

        self.episode_reward = 0
        self.episode_cumulative_reward = 0 #do not reset


        self.total_reward = 0 #do not reset

        self.episode_newobj_reward = 0
        
        # self.ep_avg_steps_between_objects = [] #do not rest
        # self.ep_steps = [] #do not rest

        self.action_space = spaces.Discrete(6)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(view_distance, view_distance, 1), dtype=np.uint8)

        self.observation_space = spaces.Dict({
        "map": spaces.Box(low=0.0, high=1.0, shape=(view_distance, view_distance, 1), dtype=np.float32), #normalized, 1 channel for NN
        "collected_objects": spaces.Box(low=0, high=num_objects, shape=(1,), dtype=np.int32),
        "new_objects": spaces.Box(low=0, high=num_objects, shape=(1,), dtype=np.int32),
        })


        self.reward_params = reward_params 
        self.penalty_params = penalty_params 
        self.seed = seed
        self.param_filename = param_filename
    
    def step(self, action):

        prev_pos = self.agent_pos.copy()
        height, width = self.agent_size
        reward = 0

        # Define movement logic considering agent size; #+1 at the and for include the upper bound
        if (action == 0 and prev_pos[0] - height // 2 - 1 >= 0 and 
            np.all(self.instance_map[prev_pos[0] - height // 2 - 1 : prev_pos[0] + height // 2 - 1 + 1, prev_pos[1] - width // 2 : prev_pos[1] + width // 2 + 1] == 0)):
            self.agent_pos[0] -= 1  # Move up
            #self.steps += 1
        elif (action == 1 and 
              prev_pos[0] + height // 2 + 1 < self.map_size[0] and 
              np.all(self.instance_map[prev_pos[0] - height // 2 + 1 : prev_pos[0] + height // 2 + 1 + 1, prev_pos[1] - width // 2 : prev_pos[1] + width // 2 + 1] == 0)):
            self.agent_pos[0] += 1  # Move down
            #self.steps += 1
        elif (action == 2 and prev_pos[1] - width // 2 - 1 >= 0 and 
              np.all(self.instance_map[prev_pos[0] - height // 2 : prev_pos[0] + height // 2 + 1, prev_pos[1] - width // 2 - 1 : prev_pos[1] + width // 2 - 1 + 1] == 0)):
            self.agent_pos[1] -= 1  # Move left
            #self.steps += 1
        elif (action == 3 and prev_pos[1] + width // 2 + 1 < self.map_size[1] and 
              np.all(self.instance_map[prev_pos[0] - height // 2 : prev_pos[0] + height // 2 + 1, prev_pos[1] - width // 2 + 1 : prev_pos[1] + width // 2 + 1 + 1] == 0)):
            self.agent_pos[1] += 1  # Move right
            #self.steps += 1
        elif action == 4:
            self.agent_dir = (self.agent_dir - 15) % 360  # Rotate left
        elif action == 5:
            self.agent_dir = (self.agent_dir + 15) % 360  # Rotate right

        else:
            reward -= self.penalty_params["obstacle"]/self.max_steps  # Penalize for hitting an obstacle

        # if np.array_equal(self.agent_pos, prev_pos):  
        #     reward -= self.penalty_params["prev_position"] # Penalize for returning to the same position

        visible_objects = self.get_visible_objects()
        from graph.graph_builder import GraphBuilder
        GraphBuilder.update_graph(visible_objects)
        new_objects = visible_objects - self.collected_objects
        self.collected_objects.update(new_objects)
        self.cmlt_collected_objects.update(new_objects)

        if new_objects:
            reward += self.reward_params["new_object"] * (len(self.collected_objects) + 1) * len(new_objects)
            self.steps_between_objects.append(self.steps_since_last_object)  # Log steps
            self.steps_since_last_object = 0
            self.episode_newobj_reward += self.reward_params["new_object"] * (len(self.collected_objects) + 1) * len(new_objects)
            if (len(new_objects) >=3):
                self.render()

        
        if tuple(self.agent_pos) not in self.visited_positions:
            self.visited_positions.add(tuple(self.agent_pos))
            reward += self.reward_params["tot_ep_exploration"]/self.max_steps
        else:
            reward -= self.penalty_params["same_position"]

        self.steps += 1
        self.steps_since_last_object += 1

        if (self.mode == "train"):
            self.run["train/reward"].append(reward)
        else:
            self.run["eval/reward"].append(reward)

        self.episode_reward += reward
        self.total_reward += reward

        success = len(self.collected_objects) == self.num_objects

        done = self.steps >= self.max_steps or success
        if done:
            extra_term_to_reward = 0
            if (len(self.collected_objects) >= 6):
                extra_term_to_reward = 1000 * len(self.collected_objects) 
                reward += extra_term_to_reward
                self.total_reward += extra_term_to_reward
                self.episode_reward += extra_term_to_reward
            else: 
                extra_term_to_reward = (self.num_objects - len(self.collected_objects))*100
                reward -= extra_term_to_reward
                self.total_reward -= extra_term_to_reward
                self.episode_reward -= extra_term_to_reward

            # reward += extra_term_to_reward
            # self.total_reward += extra_term_to_reward
            # self.episode_reward += extra_term_to_reward
            self.episode_cumulative_reward += self.episode_reward

            if (self.mode == "train"):
                self.run["train/reward"].append(reward)
            else:
                self.run["eval/reward"].append(reward)
            
            if (self.mode == "train"):
                avg_steps_between_objects = (sum(self.steps_between_objects) / len(self.steps_between_objects)) if self.steps_between_objects else 0
                self.run["train/episode_avg_steps_between_objects"].append(avg_steps_between_objects)
                self.run["train/episode_steps"].append(self.steps)
                self.run["train/episode_reward"].append(self.episode_reward)
                self.run["train/episode_cumulative_reward"].append(self.episode_cumulative_reward)
                self.run["train/episode_num_collected_objects"].append(len(self.collected_objects))
                self.run["train/episode_cumulative_num_collected_objects"].append(len(self.cmlt_collected_objects))
                self.run["train/new_object_reward"].append(self.episode_newobj_reward)


            
            if (self.mode == "eval"):
                avg_steps_between_objects = (sum(self.steps_between_objects) / len(self.steps_between_objects)) if self.steps_between_objects else 0
                self.run["eval/episode_avg_steps_between_objects"].append(avg_steps_between_objects)
                self.run["eval/episode_steps"].append(self.steps)
                self.run["eval/episode_reward"].append(self.episode_reward)
                self.run["eval/episode_cumulative_reward"].append(self.episode_cumulative_reward)
                self.run["eval/episode_num_collected_objects"].append(len(self.collected_objects))
                self.run["eval/episode_cumulative_num_collected_objects"].append(len(self.cmlt_collected_objects))

                self.run["eval/new_object_reward"].append(self.episode_newobj_reward)

        

        return self.get_observation(len(new_objects)), reward, done, {}


    def get_visible_objects(self):
        visible_objects = set()
        cx, cy = self.agent_pos  

        for i in range(-self.view_distance, self.view_distance + 1):
            for j in range(-self.view_distance, self.view_distance + 1):
                x, y = cx + i, cy + j
                if not (0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]): #the position is within map bounds
                    continue
                distance = np.sqrt(i**2 + j**2)
                if distance > self.view_distance:
                    continue  # Ignore objects beyond the view distance
                angle_to_object = (np.degrees(np.arctan2(j, i)) + 360) % 360 #relative angle to the object (in degrees)

                # Compute the agent's field of view boundaries
                left_bound = (self.agent_dir - self.view_angle / 2) % 360
                right_bound = (self.agent_dir + self.view_angle / 2) % 360

                if left_bound < right_bound:
                    in_view = left_bound <= angle_to_object <= right_bound
                else:
                    in_view = angle_to_object >= left_bound or angle_to_object <= right_bound

                if in_view:
                    obj_id = self.instance_map[x, y]
                    if obj_id > 0:
                        visible_objects.add(obj_id)

        return visible_objects
    

    def get_observation(self, num_new_obj):
        obs = np.zeros((self.view_distance, self.view_distance), dtype=np.uint8)
        cx, cy = self.agent_pos

        for i in range(-self.view_distance // 2, self.view_distance // 2):
            for j in range(-self.view_distance // 2, self.view_distance // 2):
                x, y = cx + i, cy + j
                if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                    angle = np.degrees(np.arctan2(j, i)) % 360
                    if abs((angle - self.agent_dir + 180) % 360 - 180) <= self.view_angle / 2:
                        obs[i + self.view_distance // 2, j + self.view_distance // 2] = self.instance_map[x, y]
        
        # Normalize map values to [0,1] for PPO
        obs = obs.astype(np.float32) / self.num_objects  

        return {
            "map": obs[..., np.newaxis],  # Add channel dimension (H, W, 1)
            "collected_objects": np.array([len(self.collected_objects)], dtype=np.int32),
            "new_objects": np.array([num_new_obj], dtype=np.int32),
        }



    def reset(self):
      
      self.agent_pos = self.set_agent_start_position()
      self.agent_dir = np.random.randint(0, 360)
      self.collected_objects = set()
      self.visited_positions = set()

      self.steps = 0
      self.steps_since_last_object = 0
      self.steps_between_objects = []
      self.episode_reward = 0
      self.episode_newobj_reward = 0


      return self.get_observation(0)
    

    def set_agent_start_position(self):
        while True:
            x = np.random.randint(self.agent_size[0] // 2, self.map_size[0] - self.agent_size[0] // 2)
            y = np.random.randint(self.agent_size[1] // 2, self.map_size[1] - self.agent_size[1] // 2)

            if np.all(self.instance_map[x - self.agent_size[0] // 2: x + self.agent_size[0] // 2,
                                        y - self.agent_size[1] // 2: y + self.agent_size[1] // 2] == 0):
                return np.array([x, y])


    def render(self, mode='human'):
      img = np.copy(self.instance_map)
      img = (img / img.max() * 255).astype(np.uint8)
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

      # Draw agent's position
      #cv2.circle(img, (self.agent_pos[1], self.agent_pos[0]), self.agent_size[0]//2, (0, 0, 255), -1)
      cv2.rectangle(img, (self.agent_pos[1] - self.agent_size[1] // 2, self.agent_pos[0] - self.agent_size[0] // 2),  # Top-left corner
                    (self.agent_pos[1] + self.agent_size[1] // 2, self.agent_pos[0] + self.agent_size[0] // 2),  # Bottom-right corner
                    (0, 0, 255), -1)
      # Draw direction of view
      angle_rad = np.radians(self.agent_dir)
      dx = int(self.view_distance * np.sin(angle_rad))
      dy = int(self.view_distance * np.cos(angle_rad))

      cv2.arrowedLine(img, (self.agent_pos[1], self.agent_pos[0]), (self.agent_pos[1] + dx, self.agent_pos[0] + dy), (255, 0, 0), thickness=1, tipLength=0.15)

      # Calculate view angle boundaries
      left_angle = np.radians(self.agent_dir - self.view_angle / 2)
      right_angle = np.radians(self.agent_dir + self.view_angle / 2)

      # Calculate end points of the view angle lines
      left_dx = int(self.view_distance * np.sin(left_angle))
      left_dy = int(self.view_distance * np.cos(left_angle))
      right_dx = int(self.view_distance * np.sin(right_angle))
      right_dy = int(self.view_distance * np.cos(right_angle))

      # Draw view angle lines
      cv2.line(img, (self.agent_pos[1], self.agent_pos[0]), (self.agent_pos[1] + left_dx, self.agent_pos[0] + left_dy), (0, 255, 0), 1)
      cv2.line(img, (self.agent_pos[1], self.agent_pos[0]), (self.agent_pos[1] + right_dx, self.agent_pos[0] + right_dy), (0, 255, 0), 1)

      visible_objects = self.get_visible_objects()


      for obj_id in self.collected_objects:
          bbox = self.get_object_bbox(obj_id)
          if bbox is not None:
              cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)

      for obj_id in visible_objects:
          bbox = self.get_object_bbox(obj_id)  
          if bbox is not None:
              cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)  


      frame_name = f"frames/frame_{self.param_filename}_cnt_{len(visible_objects)}.png"
      cv2.imwrite(frame_name, img)


      return img


    def get_object_bbox(self, obj_id):
      object_indices = np.argwhere(self.instance_map == obj_id)
      if object_indices.size > 0:
          x_min, y_min = object_indices.min(axis=0)
          x_max, y_max = object_indices.max(axis=0)
          return (y_min, x_min, y_max, x_max)  # (y_min, x_min) to (y_max, x_max) for rectangle
      return None
