import numpy as np

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
        return habitat_map
