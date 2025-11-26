import numpy as np
data = np.load("logs/seed_0/eval/evaluations.npz")
print(data["timesteps"].shape)  # expect something like (20,) not (1,)
print(data["timesteps"])
print(data["results"].shape)