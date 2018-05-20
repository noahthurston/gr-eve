import numpy as np
import tensorflow as tf
from collections import deque
#import random
import os
import matplotlib.pyplot as plt
import gym
import throttle_env

NUM_EPISODES = 10000

# create environment
env = throttle_env.ThrottleEnv()

# variables
num_inputs = 1
num_hidden = 16
hidden_activation = tf.nn.relu
num_outputs = 4  # equal to action space
initializer = tf.contrib.layers.variance_scaling_initializer()

# training
learning_rate = 0.001
num_steps = 1000
training_start = 1000
training_interval = 3
save_steps = 100
copy_steps = 50
discount_rate = 0.95
skip_start = 0
batch_size = 50
checkpoint_path = "./my_dqn.ckpt"

replay_memory_size = 10*1000
replay_memory = deque([], maxlen=replay_memory_size)

eps_min = 0.05
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
        outputs = tf.contrib.layers.fully_connected(hidden_layer, num_outputs, activation_fn=None,
                                                    weights_initializer=initializer)

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}

    return outputs, trainable_vars_by_name

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

def train():

    X_state = tf.placeholder(tf.float32, shape=[None, num_inputs])

    # create networks
    actor_q_values, actor_vars = q_network(X_state, scope='q_networks/actor')
    critic_q_values, critic_vars = q_network(X_state, scope='q_networks/critic')

    # copy operation
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

    # initialize observation to 2
    state = 2

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_path):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()
        iteration = 0
        while True:
            step = global_step.eval()
            if step >= num_steps:
                break
            iteration += 1

            #if done statement... our game never ends

            # actor decides what to do
            q_values = actor_q_values.eval(feed_dict={X_state: np.array(state).reshape(-1,1)})
            action = epsilon_greedy(q_values, step)

            # actor plays
            obs, reward, done = env._step(action)
            next_state = obs

            # recording for replay memory
            replay_memory.append((state, action, reward, next_state, 1-done))  # state, action, reward, next_state, continue
            state = next_state

            # only train critic on training intervals
            if iteration % training_interval != 0:
                continue
            X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
            next_q_values = actor_q_values.eval(feed_dict={X_state: np.array(X_next_state_val).reshape(-1,1)})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues * discount_rate * max_next_q_values
            training_op.run(feed_dict={X_state: np.array(X_state_val).reshape(-1,1), X_action: X_action_val, y:y_val})

            # copy critic to actor
            if step % copy_steps == 0:
                copy_critic_to_actor.run()

            # save
            if step % save_steps == 0:
                saver.save(sess, checkpoint_path)

train()
