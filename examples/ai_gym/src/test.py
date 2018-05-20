import numpy as np
import throttle_env

cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
memory = [1, 2, 3, 4, 5]


for col, value in zip(cols, memory):
    col.append(value)

print("done")