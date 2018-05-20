import numpy as np
import random as random
import matplotlib.pyplot as plt


class epsilon_greedy_model():
    def __init__(self, epsilon, num_arms, counts, average_rewards):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.counts = counts
        self.average_rewards = average_rewards
        self.average_reward_earned = 0

    def initalize_lists(self, num_arms):
        self.counts = np.zeros(num_arms)
        self.average_rewards = np.zeros(num_arms)

    def find_max_index(self, list_x):
        # m = max(list_x)
        # return list_x.index(m)
        return list_x.index(max(list_x))

    def select_arm(self):
        if random.random() > self.epsilon:
            return self.average_rewards.argmax()
        else:
            return random.randrange(len(self.counts))

    def update_model(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = float(self.counts[chosen_arm])

        total_counts = np.sum(self.counts)
        self.average_reward_earned = ((total_counts - 1) / total_counts) * self.average_reward_earned + reward / total_counts

        avg_reward = self.average_rewards[chosen_arm]
        new_avg_reward = ((n - 1) / n) * avg_reward + reward / n
        self.average_rewards[chosen_arm] = new_avg_reward

    # test function for debugging: the higher index the arm, the larger the values it can return
    # should teach algorithm to settle on the highest arm
    def get_debug_arm_results(self, chosen_arm):
        return random.randrange(chosen_arm + 1)

def test_epsilon_greedy():
    epsilon = 0.75
    num_arms = 4
    counts = []
    average_rewards = []

    average_rewards_overtime = [[] for x in range(num_arms)]

    model = epsilon_greedy_model(epsilon, num_arms, counts, average_rewards)
    model.initalize_lists(4)

    for trial in range(1000):
        chosen_arm = model.select_arm()
        reward_rec = model.get_debug_arm_results(chosen_arm)
        model.update_model(chosen_arm, reward_rec)
        # print("trial #%d, arm chosen:%d, reward:%d" % (trial, chosen_arm, reward_rec))

        for arm in range(model.num_arms):
            average_rewards_overtime[arm].append(model.average_rewards[arm])
    # print(average_rewards_overtime)
    # print("\n")

    plt.plot(range(len(average_rewards_overtime[0])), average_rewards_overtime[0], 'r-')
    plt.plot(range(len(average_rewards_overtime[0])), average_rewards_overtime[1], 'y-')
    plt.plot(range(len(average_rewards_overtime[0])), average_rewards_overtime[2], 'b-')
    plt.plot(range(len(average_rewards_overtime[0])), average_rewards_overtime[3], 'g-')

    plt.show()

#test_epsilon_greedy()