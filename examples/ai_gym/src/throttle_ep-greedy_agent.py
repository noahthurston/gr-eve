import numpy as np
import matplotlib.pyplot as plt
import gym
import throttle_env
import epsilon_greedy

NUM_EPISODES = 100

#create environment
env = throttle_env.ThrottleEnv()

#def __init__(self, epsilon, num_arms, counts, average_rewards):
model = epsilon_greedy.epsilon_greedy_model(0.5, 4, np.zeros(4), np.zeros(4))

average_rewards_overtime = [[] for x in range(4)]
counts_overtime = np.zeros((NUM_EPISODES,4))

for episode in range(NUM_EPISODES):
    curr_arm = model.select_arm()
    ob, reward, episode_over = env._step(curr_arm)
    model.update_model(curr_arm, reward)


    for arm in range(model.num_arms):
        average_rewards_overtime[arm].append(model.average_rewards[arm])

    counts_overtime[episode] = model.counts

marker_colors = ['r-', 'y-', 'b-', 'g-']

plt.figure(1)

for arm in range(model.num_arms):
    plt.plot(range(len(average_rewards_overtime[0])), average_rewards_overtime[arm], marker_colors[arm], label=str(arm))


plt.legend(loc='lower right')
plt.grid(True)
plt.show()

plt.figure(2)
counts_overtime = np.array(counts_overtime).T
print(counts_overtime)
for arm in range(model.num_arms):
    plt.plot(range(len(counts_overtime[0])), counts_overtime[arm], marker_colors[arm], label=str(arm))

plt.legend(loc='upper left')
plt.grid(True)
plt.show()