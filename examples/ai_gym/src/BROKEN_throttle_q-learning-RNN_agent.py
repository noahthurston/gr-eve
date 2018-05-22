import numpy as np
import tensorflow as tf
from collections import deque
import datetime
import os
import matplotlib.pyplot as plt
import gym
import throttle_env

# create environment
env = throttle_env.ThrottleEnv()

# variables
num_inputs = 1
num_hidden = 16
num_timesteps = 4
num_layers = 4
hidden_activation = tf.nn.tanh
num_outputs = 4  # equal to action space
initializer = tf.contrib.layers.variance_scaling_initializer()

# training
learning_rate = 0.01
num_steps = 20*1000
training_start = 1000
training_interval = 3
save_steps = 100
copy_steps = 50
discount_rate = 0.95
skip_start = 0
batch_size = 50
checkpoint_path = "../models/"
checkpoint_to_load = "../models/blahblah.ckpt"

replay_memory_size = 10*1000
replay_memory = deque([], maxlen=replay_memory_size)

eps_min = 0.20
eps_max = 1.0
eps_decay_steps = 50*1000

# epsilon greedy for exploring game
def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(num_outputs)
    else:
        return np.argmax(q_values)

# builds NN for predicting q-values based on state
def q_network(X_state, scope):
    with tf.variable_scope(scope) as scope:


        hidden_layer = tf.contrib.layers.fully_connected(X_state, num_hidden, activation_fn=hidden_activation,
                                                         weights_initializer=initializer)
        output = tf.contrib.layers.fully_connected(hidden_layer, num_outputs, activation_fn=None,
                                                    weights_initializer=initializer)



        """
        cells = [tf.contrib.rnn.BasicRNNCell(num_units=num_outputs) for layer in range(num_layers)]
        stacked_cells = tf.contrib.rnn.MultiRNNCell(cells)
        hidden_output, states = tf.nn.dynamic_rnn(stacked_cells, X_state, dtype=tf.float32)
        output = hidden_output[0, -1, :]
        """

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}

    print(output)

    return output, trainable_vars_by_name

# sample memories from replay memory
def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []]  # state, action, reward, next_state, continue

    for index in indices:
        memory = replay_memory[index]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]

    return(cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))

"""
def extract_axis_1(data):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, num_outputs-1], axis=1)
    res = tf.gather_nd(data, indices)
    return res
"""

def train():
    X_state = tf.placeholder(tf.float32, shape=[None, num_timesteps, num_inputs])

    # create networks
    actor_q_values, actor_vars = q_network(X_state, scope='q_networks/actor')
    critic_q_values, critic_vars = q_network(X_state, scope='q_networks/critic')

    # copy operation
    copy_ops = [actor_var.assign(critic_vars[var_name]) for var_name, actor_var in actor_vars.items()]
    copy_critic_to_actor = tf.group(*copy_ops)

    X_action = tf.placeholder(tf.int32, shape=[None])
    print(X_action)
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

    # initialize observation to 2
    state = np.zeros(num_timesteps)+2
    next_state = np.zeros(num_timesteps)+2

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
            q_values = actor_q_values.eval(feed_dict={X_state: np.array(state).reshape(-1,num_timesteps,1)})

            action = epsilon_greedy(q_values, step)

            # actor plays
            obs, reward, done = env._step(action)
            next_state = np.append(next_state, obs)[-num_timesteps:]


            # recording for replay memory
            replay_memory.append((state, action, reward, next_state, 1-done))  # state, action, reward, next_state, continue
            state = next_state

            # only train critic on training intervals
            if episode % training_interval != 0:
                continue

            X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))

            next_q_values = actor_q_values.eval(feed_dict={X_state: np.array(X_next_state_val).reshape(-1, num_timesteps, 1)}).reshape(-1, num_outputs)
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)

            y_val = rewards + continues * discount_rate * max_next_q_values
            training_op.run(feed_dict={X_state: np.array(X_state_val).reshape(-1,num_timesteps,1), X_action: X_action_val, y:y_val})

            # copy critic to actor
            if step % copy_steps == 0:
                print("STEP #" + str(step) + ": copying critic to actor")
                copy_critic_to_actor.run()

            # save
            if step % save_steps == 0:
                checkpoint_save_name = checkpoint_path + "QNN_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
                print("STEP #" + str(step) + ": Saving to model to: " + checkpoint_save_name)
                saver.save(sess, checkpoint_save_name)
                #saver.save(sess, checkpoint_save_name)

def validate(saved_filename):


    X_state = tf.placeholder(tf.float32, shape=[None, num_timesteps, num_inputs])

    # create networks
    actor_q_values, actor_vars = q_network(X_state, scope='q_networks/actor')
    critic_q_values, critic_vars = q_network(X_state, scope='q_networks/critic')


    """
    # copy operation
    copy_ops = [actor_var.assign(critic_vars[var_name]) for var_name, actor_var in actor_vars.items()]
    copy_critic_to_actor = tf.group(*copy_ops)
    """

    X_action = tf.placeholder(tf.int32, shape=[None])
    q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, num_outputs), axis=1, keepdims=True)

    # target q value
    y = tf.placeholder(tf.float32, shape=[None, 1])

    """
    # training operations
    cost = tf.reduce_mean(tf.square(y - q_value))
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(cost, global_step=global_step)
    """

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # initialize state to 2
    state = np.zeros(num_timesteps)+2
    next_state = np.zeros(num_timesteps)+2

    num_episodes = 100

    state_overtime = []
    reward_overtime = []
    action_overtime = []

    with tf.Session() as sess:

        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, "../models/" + saved_filename)


        print("loaded!")

        # create environment
        env = throttle_env.ThrottleEnv()

        for episode in range(num_episodes):
            q_values = actor_q_values.eval(feed_dict={X_state: np.array(state).reshape(-1, num_timesteps, 1)}).reshape(4)
            action = np.argmax(q_values)

            obs, reward, done = env._step(action)

            state_overtime.append(state)
            action_overtime.append(action)
            reward_overtime.append(reward)

            state = np.append(state, obs)[-num_timesteps:]

        print("state_overtime: " + str(np.array(state_overtime, dtype=int).T[-1].tolist()))
        print("action_overtime: " + str(action_overtime))
        print("reward_overtime: " + str(reward_overtime))
        print("average_reward: " + str(np.average(reward_overtime)))

train()
#validate("QNN_05-20--23-34")