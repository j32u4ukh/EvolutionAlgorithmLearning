import time

import numpy as np
import pandas as pd

np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:
        # act greedy
        # replace argmax to idxmax as argmax means a different function in newer version of pandas
        action_name = state_actions.idxmax()

    return action_name


def get_env_feedback(s, a):
    # This is how agent will interact with the environment
    if a == 'right':    # move right
        if s == N_STATES - 2:   # terminate
            s_ = 'terminal'
            r = 1
        else:
            s_ = s + 1
            r = 0
    else:   # move left
        r = 0
        if s == 0:
            s_ = s  # reach the wall
        else:
            s_ = s - 1
    return s_, r


def update_env(s, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if s == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        s = 0
        is_terminated = False
        update_env(s, episode, step_counter)
        while not is_terminated:

            a = choose_action(s, q_table)
            s_, r = get_env_feedback(s, a)  # take action & get next state and reward
            q_predict = q_table.loc[s, a]
            if s_ != 'terminal':
                q_target = r + GAMMA * q_table.iloc[s_, :].max()   # next state is not terminal
            else:
                q_target = r     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[s, a] += ALPHA * (q_target - q_predict)  # update
            s = s_  # move to next state

            update_env(s, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
