import numpy as np
import cv2
import os


class HabitatMapGenerator:
    @staticmethod
    def generate(map_size=(200, 200), num_objects=10, seed=None):
        if seed is not None:
            np.random.seed(seed)

        habitat_map = np.zeros(map_size, dtype=np.uint8)
        for object_id in range(1, num_objects + 1):
            shape = np.random.choice(["rectangle", "circle"])
            if shape == "rectangle":
                w, h = np.random.randint(10, 50, size=2)
                x, y = np.random.randint(0, map_size[1] - w), np.random.randint(0, map_size[0] - h)
                habitat_map[y:y+h, x:x+w] = object_id
            else:
                r = np.random.randint(3, 10)
                cx, cy = np.random.randint(r, map_size[1] - r), np.random.randint(r, map_size[0] - r)
                for i in range(map_size[1]):
                    for j in range(map_size[0]):
                        if (i - cy) ** 2 + (j - cx) ** 2 <= r ** 2:
                            habitat_map[i, j] = object_id

        # Adding object ID labels to the map image

        habitat_map_image = np.zeros((habitat_map.shape[0], habitat_map.shape[1], 3), dtype=np.uint8) 
        for object_id in range(1, 11):
            # Find the coordinates of the object
            indices = np.where(habitat_map == object_id)
            if indices[0].size > 0:  
            
                habitat_map_image[habitat_map == object_id] = [object_id * 25.5, object_id * 25.5, object_id * 25.5]

                y_center, x_center = np.mean(indices, axis=1).astype(int)

                label = str(object_id)
                cv2.putText(habitat_map_image, label, (x_center+10, y_center+7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        HabitatMapGenerator.save_map(habitat_map, habitat_map_image)  
        return habitat_map
    
    @staticmethod
    def save_map(habitat_map, habitat_map_image):
        os.makedirs("maps", exist_ok=True)
        np.save("maps/habitat_map.npy", habitat_map)
        cv2.imwrite("maps/habitat_map.png", (habitat_map_image / np.max(habitat_map_image) * 255))

