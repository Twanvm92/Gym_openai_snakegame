import numpy as np
import time
import gym_foo.envs.foo_env as fenv
from DQNagent import DQNAgent
import seaborn as sns
import matplotlib.pyplot as plt


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1,
                     line_kws={'color': 'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()


env = fenv.FooEnv()
state_size = 11
action_size = env.action_space.n
score_plot = []
counter_plot = []
weights = "weights.hdf5"
# weights = None


if __name__ == '__main__':

    # initialize gym environment and the agent
    episodes = 300
    # start training a model or just let a model play based on if the variable weights is empty or not
    if weights is None:
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        # Iterate the game
        for e in range(episodes):
            # reset state in the beginning of each game
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            while True:
                # turn this on if you want to render
                env.render()
                # Decide action
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                # Remember the previous state, action, reward, and done
                agent.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}"
                          .format(e, episodes))
                    break
            # train the agent with the experience of the episode
            replay_len = len(agent.memory)
            if len(agent.memory) > 600:
                replay_len = 600

            agent.replay(replay_len)
            # agent.epsilon = 80 - e
            print("agent epsilon: {}".format(agent.epsilon))
            score_plot.append(env.score)
            counter_plot.append(e)

        agent.model.save_weights('weights3.hdf5')
        plot_seaborn(counter_plot, score_plot)

    else:
        agent = DQNAgent(state_size=state_size, action_size=action_size, weights=weights)
        for e in range(episodes):
            # reset state in the beginning of each game
            state = env.reset()
            # state = np.ndarray.flatten(state)
            state = np.reshape(state, [1, state_size])
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            while True:
                # turn this on if you want to render
                env.render()
                # Decide action
                action = agent.predict(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}"
                          .format(e, episodes))
                    break
                time.sleep(0.1)

            score_plot.append(env.score)
            counter_plot.append(e)

        plot_seaborn(counter_plot, score_plot)
