from mdp import MDP, MDPState
from grid_world import GridWorldTile
import time

class ValueIteration:

    def __init__(self, mdp: MDP, gamma, epsilon):
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon

    def solve(self):
        states = self.mdp.get_states()
        for state in states:
            if abs(state.get_reward()) > 1:
                state.update_value(state.get_reward(), 0)
            else:
                state.update_value(0, 0)
            state.old_value = state.value
        at_convergence = False
        iteration = 1
        start = time.time()
        while not at_convergence:
            diffs = []
            at_convergence = True
            for state in states:
                ns = state.get_neighbors()
                for n in ns:
                    if n.tile_type == GridWorldTile.GOAL:
                        pass
                qs = []
                for action in state.get_actions():
                    qs.append(self.q(state, action, iteration - 1))
                if len(qs) > 0:
                    if state.x == 122 and state.y == 27 and state.t_hash == 1:
                        print(qs)
                    state.update_value(max(qs), iteration)
                    diff = abs(state.value[1] - state.old_value[1])
                    at_convergence &= diff < self.epsilon
                    diffs.append(diff)
            self.mdp.print(print, y=26, x=119, w=7, h=3, t=[True, False])
            #self.mdp.print(y=24, x=168, w=6, h=6, t=[True, True])
            self.mdp.print(print, y=26, x=119, w=7, h=3, t=[True, True])
            #self.mdp.print(y=40, x=50, w=11, h=6)
            #self.mdp.print(y=41, x=1, w=11, h=5, t=[True, True])
            # self.mdp.print(y=6, x=167, w=8, h=5)
            # self.mdp.print(y=6, x=167, w=8, h=5, t=[True, True])
            #print('Avg. diff: {}, max: {}'.format(sum(diffs) / len(diffs), max(diffs)))
            if iteration % 5 == 0:
                print('Iter[{}] Time elapsed: {}'.format(iteration, time.time() - start))
            iteration += 1
        print('Total time: {}'.format(time.time() - start))
        return iteration - 1
    
    def q(self, state, action, iteration):
        action_value = 0
        for transition in action[1]:
            if transition[1] is None:
                t_state = state
            else:
                t_state = transition[1]
            action_value += transition[0] * t_state.get_value(iteration)
        return state.get_reward() + self.gamma * action_value