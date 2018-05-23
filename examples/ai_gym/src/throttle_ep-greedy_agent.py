# This agent uses an epsilon greedy algorithm to choose the best jamming power

import numpy as np
import matplotlib.pyplot as plt
# import gym
import throttle_env
import epsilon_greedy

num_episodes = 10000

#create environment
env = throttle_env.ThrottleEnv()

# create epsilon greedy model
model = epsilon_greedy.epsilon_greedy_model(0.25, 4, np.zeros(4), np.zeros(4))

# lists to hold values for graphing later
average_rewards_overtime = [[] for x in range(5)]
counts_overtime = np.zeros((num_episodes,4))
mod_overtime = []


for episode in range(num_episodes):
    curr_arm = model.select_arm()
    ob, reward, episode_over = env._step(curr_arm)
    model.update_model(curr_arm, reward)

    # record data for graphing
    for arm in range(model.num_arms):
        average_rewards_overtime[arm].append(model.average_rewards[arm])
    average_rewards_overtime[4] = np.append(average_rewards_overtime[4], model.average_reward_earned)
    counts_overtime[episode] = model.counts
    mod_overtime = mod_overtime + [ob]


# graphing code
marker_colors = ['r-', 'y-', 'b-', 'g-', 'c-']

plt.figure(0)
plt.title("Mod vs Episode")
plt.plot(range(len(mod_overtime)), mod_overtime)
plt.grid(True)


plt.figure(1)
plt.title("Jamming Power Avg Rewards vs Episode")
for arm in range(model.num_arms):
    plt.plot(range(len(average_rewards_overtime[0])), average_rewards_overtime[arm], marker_colors[arm], label=str(arm))
plt.plot(range(len(average_rewards_overtime[0])), average_rewards_overtime[4], marker_colors[4], label='average')
plt.legend(loc='lower right')
plt.grid(True)


plt.figure(2)
plt.title("Jamming Power Counts vs Episode")
counts_overtime = np.array(counts_overtime).T
print(counts_overtime)
for arm in range(model.num_arms):
    plt.plot(range(len(counts_overtime[0])), counts_overtime[arm], marker_colors[arm], label=str(arm))

plt.legend(loc='upper left')
plt.grid(True)
plt.show()