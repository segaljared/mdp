from mdp import MDP, MDPState
import random
import numpy
import time


class QLearning:
    # Largely based off of this implementation: https://github.com/ankonzoid/LearningX/blob/master/classical_RL/gridworld/gridworld.py

    def __init__(self, mdp: MDP, epsilon_start, epsilon_minimum, gamma, on_update=None, on_episode=None, initializer=None, start_at_0=False, alpha=None, is_done=None, every_n_episode=10, print_n_episodes=100):
        self.mdp = mdp
        self.epsilon_start = epsilon_start
        self.epsilon_minimum = epsilon_minimum
        self.gamma = gamma
        self.Q = {}
        self.on_update = on_update
        self.on_episode = on_episode
        self.initializer = initializer
        self.start_at_0 = start_at_0
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.is_done = is_done
        self.every_n_episode = every_n_episode
        self.print_n_episodes = print_n_episodes

    def run(self, n_episodes):
        for state in self.mdp.get_states():
            if self.initializer is None:
                self.Q[state] = [0] * len(state.get_actions())
            else:
                self.Q[state] = self.initializer(state)
        non_terminal_states = []
        for state in self.mdp.get_states():
            if len(state.get_actions()) > 0:
                non_terminal_states.append(state)
        total_start = time.time()
        for episode in range(0, n_episodes):
            start = time.time()
            done = False
            if self.start_at_0 or random.random() < 0.25 + 0.75 * episode / n_episodes:
                state = self.mdp.get_states()[0]
            else:
                state = numpy.random.choice(non_terminal_states)
            if self.alpha is None:
                alpha = 1.0 / (episode + 1)
            else:
                alpha = self.alpha
            states = []
            while not done:
                action, action_i = self.epsilon_greedy_next_action(state)
                next_state = self.action_step(state, action)

                if len(next_state.get_actions()) == 0 or (self.is_done is not None and self.is_done(state, action_i, next_state)):
                    done = True
                
                states.append((state, action_i, next_state))
                self.update_q(alpha, state, action_i, next_state, next_state.get_reward(state, action_i))

                if self.on_update is not None:
                    self.on_update(state, action, next_state, self)
                else:
                    print('.', end='', flush=True)
                state = next_state
            # update Q
            # for state, action_i, next_state in reversed(states):
            #     self.update_q(alpha, state, action_i, next_state, next_state.get_reward(state, action_i))

            self.epsilon = self.epsilon_minimum + (self.epsilon_start - self.epsilon_minimum) / (2 * numpy.log2(episode + 1) + 1)
            if episode % self.print_n_episodes == 0 or episode % self.every_n_episode == 0 or episode == n_episodes - 1:
                q = [[] for _ in range(0, 5)]
                for s in self.Q.values():
                    for i in range(0, 5):
                        if i < len(s):
                            q[i].append(s[i])
                        else:
                            q[i].append(numpy.nan)
                print('\nEpisode[{}]: average Q: {}, alpha: {}, epsilon: {}, time elapsed: {}'.format(episode, numpy.nanmean(numpy.nanmax(q, axis=0)), alpha, self.epsilon, time.time() - total_start))
                if self.on_episode is not None and (episode % self.every_n_episode == 0 or episode == n_episodes - 1):
                    self.on_episode(episode, time.time() - start, self, q)

    def get_policy(self):
        policy = []
        for state in self.mdp.get_states():
            q_s = self.Q[state]
            if len(q_s) > 0:
                policy.append(numpy.argmax(q_s))
            else:
                policy.append(0)
        return policy
                
    def update_q(self, alpha, state, action_i, next_state, reward):
        if len(self.Q[next_state]) == 0:
            update = reward
        else:
            update = reward + self.gamma * max(self.Q[next_state])
        self.Q[state][action_i] = (1 - alpha) * self.Q[state][action_i] + alpha * update

    def epsilon_greedy_next_action(self, state: MDPState):
        actions = state.get_actions()
        if random.uniform(0, 1) < self.epsilon:
            action_i = numpy.random.randint(0, len(actions))
        else:
            q_s = self.Q[state]
            indices = numpy.flatnonzero(q_s == numpy.max(q_s))
            action_i = numpy.random.choice(indices)
        return actions[action_i], action_i

    def action_step(self, state, action):
        next_states = []
        probabilities = []
        for transition in action[1]:
            if transition[1] is None:
                next_states.append(state)
            else:
                next_states.append(transition[1])
            probabilities.append(transition[0])
        return numpy.random.choice(next_states, p=probabilities)
