import mdptoolbox.example
from mdp import MDP, MDPState
import numpy

class WrappedForest(MDP):

    def __init__(self, S, r1=4, r2=2, p=0.1, is_sparse=False):
        super().__init__()
        self.transitions, self.rewards = mdptoolbox.example.forest(S, r1, r2, p, is_sparse)
        self.states = []
        for s in range(0, S):
            self.states.append(WrappedForestState(s, self))

    def get_states(self):
        return self.states

    def get_transition_matrix(self):
        return self.transitions

    def get_reward_matrix(self):
        return self.rewards

    def print(self, printline, **kwargs):
        pass

    def print_policy(self, printline, policy):
        ages = ['Age   ']
        policies = ['Action']
        for i in range(0, len(self.states)):
            ages.append('{:2}'.format(i))
            if policy[i] == 0:
                policies.append(' W')
            else:
                policies.append(' C')
        printline(' '.join(ages))
        printline(' '.join(policies))
    

class WrappedForestState(MDPState):

    def __init__(self, state_num, forest: WrappedForest):
        super().__init__(0)
        self.state_num = state_num
        self.forest = forest
    
    def __hash__(self):
        return self.state_num

    def __eq__(self, other):
        if isinstance(other, WrappedForestState):
            return self.state_num == other.state_num
        return NotImplemented

    def get_neighbors(self):
        neighbors = []
        neighbors.append(self.forest.get_states()[0])
        if self.state_num < len(self.forest.get_states()) - 1:
            neighbors.append(self.forest.get_states()[self.state_num + 1])
        return neighbors

    def get_actions(self):
        cut = ('cut', [(1.0, self.forest.get_states()[0])])
        wait_indices = numpy.nonzero(self.forest.transitions[0][self.state_num])
        wait_results = []
        for index in wait_indices[0]:
            prob = self.forest.transitions[0][self.state_num][index]
            next_state = self.forest.get_states()[index]
            wait_results.append((prob, next_state))
        wait = ('wait', wait_results)
        return [wait, cut]

    def get_reward(self, prev_state=None, action=None):
        return self.forest.rewards[prev_state.state_num][action]
