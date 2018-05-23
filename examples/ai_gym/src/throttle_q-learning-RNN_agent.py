# This agent uses a RNN and is fed the past states to choose the best jamming power
# The agent learns that it only has to jam once every 4 actions, but cannot remember its last actions
# The result is that it gains an average reward greater than 3, but does not approach the ideal 3.5

import numpy as np
import tensorflow as tf
from collections import deque
import datetime
import matplotlib.pyplot as plt
# import gym
import throttle_env

# create environment
env = throttle_env.ThrottleEnv()

# network hyper-parameters
num_inputs = 1
num_hidden = 16
num_timesteps = 5
hidden_activation = tf.nn.relu
num_outputs = 4  # equal to action space
initializer = tf.contrib.layers.variance_scaling_initializer()

# training parameters
learning_rate = 0.005
num_steps = 5*1000
training_start = 1000
training_interval = 3
save_steps = 100
copy_steps = 50
discount_rate = 0.95
batch_size = 50
checkpoint_path = "../models/"

# test interval triggers test_model function to measure current average reward
test_interval = 250
# the more test episodes conducted, the more accurate the average reward
test_episodes = 2000

# replay memory for training actor/critic
replay_memory_size = 10*1000
replay_memory = deque([], maxlen=replay_memory_size)

# epsilon value starts at eps_max and decays rationally to eps_min after eps_decay_steps
eps_min = 0.01
eps_max = 1.0
eps_decay_steps = 5*1000


# epsilon greedy algorithm
# chooses whether to pick the highest q-value or explore another option
def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(num_outputs)
    else:
        return np.argmax(q_values)


# builds RNN for predicting q-values based on last num_timesteps of state
# returns the outputs placeholder and the trainable variables (needed for copying critic to actor)
def q_network(X_state, scope):
    with tf.variable_scope(scope) as scope:

        # RNN cell with num_hidden nodes
        cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        rnn_output, last_state = tf.nn.dynamic_rnn(cell, X_state, dtype=tf.float32)

        # transpose the output and use tf.gather to get ONLY THE LAST TIMESTEP of output
        rnn_output_T = tf.transpose(rnn_output, [1, 0, 2])
        rnn_last_output = tf.gather(rnn_output_T, int(rnn_output_T.get_shape()[0]) - 1)

        # create dense layer to reduce dimensions from num_hidden to num_outputs
        weights = tf.Variable(tf.truncated_normal([num_hidden, num_outputs]))
        biases = tf.Variable(tf.constant(0.1, shape=[num_outputs]))
        outputs = tf.matmul(rnn_last_output, weights) + biases

    # grab all trainable parameters and create dictionary of their names
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}

    return outputs, trainable_vars_by_name


# sample memories from replay memory
# returns lists of state, action, reward, next_state and continue
def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []]  # state, action, reward, next_state, continue

    for index in indices:
        memory = replay_memory[index]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


# function is called during training to evaluate the average reward earned by the actor
def test_model(actor_q_values, X_state, test_episodes=10000):
    reward_overtime = []
    next_state = np.zeros((num_timesteps, num_inputs))+3
    state = next_state

    for episode in range(test_episodes):
        q_values = actor_q_values.eval(feed_dict={X_state: np.array(state).reshape(-1, num_timesteps, num_inputs)})
        action = np.argmax(q_values)

        obs, reward, done = env._step(action)

        state = np.append(state, obs)[-num_timesteps:]

        reward_overtime.append(reward)

    return np.average(reward_overtime)


def train():
    # placeholder for feeding in state
    X_state = tf.placeholder(tf.float32, shape=[None, num_timesteps, num_inputs])

    # create networks
    actor_q_values, actor_vars = q_network(X_state, scope='q_networks/actor')
    critic_q_values, critic_vars = q_network(X_state, scope='q_networks/critic')

    # define copy operation
    copy_ops = [actor_var.assign(critic_vars[var_name]) for var_name, actor_var in actor_vars.items()]
    copy_critic_to_actor = tf.group(*copy_ops)

    X_action = tf.placeholder(tf.int32, shape=[None])
    q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, num_outputs), axis=1, keepdims=True)

    # target q value
    y = tf.placeholder(tf.float32, shape=[None, 1])

    # training operations
    cost = tf.reduce_mean(tf.square(y - q_value))
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(cost, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # initialize past state observations to 3
    next_state = np.zeros(num_timesteps)+3
    state = next_state

    # list for tracking average reward
    avg_reward_overtime = []

    with tf.Session() as sess:
        init.run()
        saver = tf.train.Saver(max_to_keep=100)
        episode = 0
        while True:
            step = global_step.eval()
            if step > num_steps:
                break
            episode += 1

            # actor decides what to do
            q_values = actor_q_values.eval(feed_dict={X_state: np.array(state).reshape(-1, num_timesteps, num_inputs)})  #### num_inputs not tested here
            action = epsilon_greedy(q_values, step)

            # actor plays
            obs, reward, done = env._step(action)

            # append the new state observed and drop the oldest state
            next_state = np.append(next_state, obs)[-num_timesteps:]

            # recording data for replay memory
            replay_memory.append(
                (state, action, reward, next_state, 1 - done))  # state, action, reward, next_state, continue
            state = next_state

            # only train critic on training intervals
            # ex: play 3 episodes, then take 1 training step
            if episode % training_interval != 0:
                continue

            # sample memories from the replay_memory
            X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))

            # actor calculates the max future q-value from each state
            next_q_values = actor_q_values.eval(
                feed_dict={X_state: np.array(X_next_state_val).reshape(-1, num_timesteps, num_inputs)})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)

            # the "actual" q-values for each state can now be calculated in retrospect
            # note these values are not perfect because actor still must predict future value to add to reward
            y_val = rewards + continues * discount_rate * max_next_q_values

            # train the critic using the critics predicted q-value and the actors earned reward + predicted future value
            training_op.run(feed_dict={X_state: np.array(X_state_val).reshape(-1, num_timesteps, num_inputs),
                                       X_action: X_action_val, y: y_val})

            # copy critic to actor
            if step % copy_steps == 0:
                print("STEP #" + str(step) + ": copying critic to actor")
                copy_critic_to_actor.run()

            # save model
            if step % save_steps == 0:
                checkpoint_save_name = checkpoint_path + "QNN_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
                print("STEP #" + str(step) + ": Saving to model to: " + checkpoint_save_name)
                saver.save(sess, checkpoint_save_name)

            # test the model (useful for graphing)
            if step % test_interval == 0:
                avg_reward = test_model(actor_q_values, X_state, test_episodes=test_episodes)
                print("avg_reward: "+ str(avg_reward))
                avg_reward_overtime.append(avg_reward)

        # plot average reward vs training steps
        print("Average reward over time: " + str(avg_reward_overtime))
        plt.figure(1)
        plt.plot(np.array(range(len(avg_reward_overtime)))*test_interval, avg_reward_overtime)
        plt.title("Average Reward vs Training Episode")

        plt.xlabel("Training Episode")
        plt.ylabel("Average Reward")

        plt.ylim(0,4)
        plt.grid(True)
        plt.show()


# function can load a saved model and evaluate the average reward it earns
def validate(saved_filename):
    X_state = tf.placeholder(tf.float32, shape=[None, num_timesteps, num_inputs])

    # create networks
    actor_q_values, actor_vars = q_network(X_state, scope='q_networks/actor')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # initialize past state observations to 3
    next_state = np.zeros(num_timesteps)+3
    state = next_state

    num_episodes = 40

    # lists for graphing variables over time
    state_overtime = []
    reward_overtime = []
    action_overtime = []

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, "../models/" + saved_filename)

        # create environment
        env = throttle_env.ThrottleEnv()

        for episode in range(num_episodes):
            q_values = actor_q_values.eval(feed_dict={X_state: np.array(state).reshape(-1, num_timesteps, num_inputs)})
            action = np.argmax(q_values)

            obs, reward, done = env._step(action)

            state_overtime.append(state)
            action_overtime.append(action)
            reward_overtime.append(reward)

            # append the new state observed and drop the oldest state
            state = np.append(state, obs)[-num_timesteps:]

        print("state_overtime: " + str(np.array(state_overtime, dtype=int).T[-1].tolist()))
        print("action_overtime: " + str(action_overtime))
        print("reward_overtime: " + str(reward_overtime))
        print("average_reward: " + str(np.average(reward_overtime)))


train()
# validate("QNN_05-21--15-36")
