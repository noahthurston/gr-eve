import numpy as np

"""

init create structure:
	-2d array of dbs arms at each timesteps
	-2d array of value of arms at each timesteps
	-2d array number of times each arm has been chosen

	-2d array of artifical rewards

run episode:
	-for every timestep
		-pick random arm
		-add to list of arm choices

	-calcualte reward



"""



import numpy as np
import random as random
import matplotlib.pyplot as plt

class monte_carlo_model():
	def __init__(self, arms, arm_counts, arm_rewards, artificial_rewards, artificial_best_pattern):
		self.arms = arms
		self.arm_counts = arm_counts
		self.arm_rewards = arm_rewards

		self.curr_episode = 0

		#artificial rewards by timesteps that the algorithm is going to try to model
		self.artificial_rewards = artificial_rewards
		self.artificial_best_pattern = artificial_best_pattern

		#current pattern and result used when running episodes
		self.curr_pattern = np.zeros((len(arm_counts)))
		self.pattern_result = 0
		self.episode_reward = 0

		self.average_reward_per_arm = np.zeros((len(self.arm_counts),len(self.arm_counts[0])))
		self.best_pattern = np.zeros((len(self.arm_counts)))

		self.pattern_mistakes_over_time = np.array([len(self.arm_counts)])

	def train_model(self, num_episodes):
		
		print(self.artificial_rewards)

		while self.curr_episode < num_episodes:
			self.run_episode()

			if(self.curr_episode % 10 == 0):
				self.update_graph_data()
			self.curr_episode = self.curr_episode + 1

		self.calculate_average_reward_per_arm()
		print(self.average_reward_per_arm)

		self.calculate_best_pattern()
		print(self.best_pattern)

		self.graph_pattern_mistakes_over_time()


	#runs through whole episode process
	def run_episode(self):
		self.generate_random_pattern()
		#self.run_through_model()
		self.calculate_reward()
		self.update_model()

		#print("Pattern " + str(self.curr_pattern))
		#print("Reward: " + str(self.episode_reward))



	#generates random pattern
	def generate_random_pattern(self):
		for time, curr_arms in enumerate(self.arms):
			self.curr_pattern[time] = random.randint(0,len(curr_arms)-1)

	#finds the best pattern given array of values for each arm in each timestep
	def find_best_pattern(self):
		print()

	#using generated pattern, runs episode through the model
	def run_through_model(self):
		#right now this done nothing, but it will be used when sending signals through GNU radio
		print("running pattern through model")

	#looks at the results of the episodes, what arems 
	def calculate_reward(self):
		
		#reward should be calculated depending on the output of the model
		#here we just sum the rewards from each arm chosen, as decided by the artifical_rewars array
		self.episode_reward = 0
		for timestep, arm_chosen in enumerate(self.curr_pattern):
			self.episode_reward = self.episode_reward + self.artificial_rewards[int(timestep)][int(arm_chosen)]


	def calculate_average_reward_per_arm(self):
		for timestep, arms in enumerate(self.arm_rewards):
			for arm_index, arm_reward in enumerate(arms):
				self.average_reward_per_arm[timestep][arm_index] = float(arm_reward) / float(self.arm_counts[timestep][arm_index])


	def calculate_best_pattern(self):
		for timestep in range(len(self.best_pattern)):
			self.best_pattern[timestep] = self.average_reward_per_arm[timestep].argmax(axis=0)


	def update_model(self):
		#distribute reward evenly among all arms taken within the pattern
		num_timesteps = len(self.arms)
		reward_per_arm = self.episode_reward / num_timesteps

		for timestep in range(num_timesteps):
			chosen_arm_index = self.curr_pattern[timestep]
			self.arm_rewards[timestep][int(chosen_arm_index)]  = self.arm_rewards[timestep][int(chosen_arm_index)] + reward_per_arm
			self.arm_counts[timestep][int(chosen_arm_index)] = self.arm_counts[timestep][int(chosen_arm_index)] + 1


	def update_graph_data(self):

		#saving data to graph number of mistakes in best calculation pattern vs time
		pattern_mistakes = 0
		self.calculate_best_pattern()
		for index in range(len(self.best_pattern)):
			if self.artificial_best_pattern[index] - self.best_pattern[index] != 0:
				pattern_mistakes = pattern_mistakes + 1
		
		self.pattern_mistakes_over_time = np.append(self.pattern_mistakes_over_time, [pattern_mistakes])
		#print(self.pattern_mistakes_over_time)

	def graph_pattern_mistakes_over_time(self):
		x = np.array(range(len(self.pattern_mistakes_over_time)))
		y = self.pattern_mistakes_over_time

		plt.plot(x,y)
		plt.show()


"""
For this test, there will be 5 timesteps, and 2 choices at each timestep; high and low
the best pattern should be: high, low, low, high, high


"""

arms = np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
arm_counts = np.zeros((10,2))
average_rewards = np.zeros((10,2))
artificial_rewards = np.array([[0,5],[5,0],[5,0],[0,5],[0,5],[0,5],[5,0],[5,0],[0,5],[0,5]])
artificial_best_pattern = np.array([1,0,0,1,1,1,0,0,1,1])


test_model = monte_carlo_model(arms, arm_counts, average_rewards, artificial_rewards, artificial_best_pattern)

test_model.train_model(100)