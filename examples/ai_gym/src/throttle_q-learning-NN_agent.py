# This agent uses a neural network to choose the best jamming power
# The agent learns to always jam using 2 in state 1 and not jam in state 0
# Best average reward earned is 3

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
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(num_outputs)
    else:
        return np.argmax(q_values)


# builds NN for predicting q-values based on state
# returns the outputs placeholder and the trainable variables (needed for copying critic to actor)
def q_network(X_state, scope):
    with tf.variable_scope(scope) as scope:
        # hidden layer of size num_hidden
        hidden_layer = tf.contrib.layers.fully_connected(X_state, num_hidden, activation_fn=hidden_activation,
                                                         weights_initializer=initializer)
        # output layer of size num_outputs
        outputs = tf.contrib.layers.fully_connected(hidden_layer, num_outputs, activation_fn=None,
                                                    weights_initializer=initializer)

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
    next_state = np.zeros((num_inputs))
    state = next_state

    for episode in range(test_episodes):
        q_values = actor_q_values.eval(feed_dict={X_state: np.array(state).reshape(-1, 1)})
        action = np.argmax(q_values)

        obs, reward, done = env._step(action)
        state = obs
        reward_overtime.append(reward)

    return np.average(reward_overtime)


def train():
    # placeholder for feeding in state
    X_state = tf.placeholder(tf.float32, shape=[None, num_inputs])

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

    # initialize observation of state to 3
    state = 3

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
            q_values = actor_q_values.eval(feed_dict={X_state: np.array(state).reshape(-1,1)})
            action = epsilon_greedy(q_values, step)

            # actor plays
            obs, reward, done = env._step(action)
            next_state = obs

            # recording data for replay memory
            replay_memory.append((state, action, reward, next_state, 1-done))  # state, action, reward, next_state, continue
            state = next_state

            # only train critic on training intervals
            # ex: play 3 episodes, then take 1 training step
            if episode % training_interval != 0:
                continue

            # sample memories from the replay_memory
            X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))

            # actor calculates the max future q-value from each state
            next_q_values = actor_q_values.eval(feed_dict={X_state: np.array(X_next_state_val).reshape(-1,1)})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)

            # the "actual" q-value can now be calculated in retrospect
            # note these values are not perfect because actor still must predict future value to add to reward
            y_val = rewards + continues * discount_rate * max_next_q_values

            # train the critic using the critics predicted q-value and the actors earned reward + predicted future value
            training_op.run(feed_dict={X_state: np.array(X_state_val).reshape(-1,1), X_action: X_action_val, y:y_val})

            # copy critic to actor
            if step % copy_steps == 0:
                print("STEP #" + str(step) + ": copying critic to actor")
                copy_critic_to_actor.run()

            # save model
            if step % save_steps == 0:
                checkpoint_save_name = checkpoint_path + "QNN_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
                print("STEP #" + str(step) + ": Saving to model to: " + checkpoint_save_name)
                saver.save(sess, checkpoint_save_name)
                #saver.save(sess, checkpoint_save_name)

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
    X_state = tf.placeholder(tf.float32, shape=[None, num_inputs])

    # create networks
    actor_q_values, actor_vars = q_network(X_state, scope='q_networks/actor')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # initialize state to 2
    state = 2

    num_episodes = 20

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
            q_values = actor_q_values.eval(feed_dict={X_state: np.array(state).reshape(-1, 1)})
            action = np.argmax(q_values)

            obs, reward, done = env._step(action)

            state_overtime.append(state)
            action_overtime.append(action)
            reward_overtime.append(reward)

            state = obs

        print("state_overtime: " + str(state_overtime))
        print("action_overtime: " + str(action_overtime))
        print("reward_overtime: " + str(reward_overtime))
        print("average_reward: " + str(np.average(reward_overtime)))

train()
# validate("QNN_05-20--16-49")
