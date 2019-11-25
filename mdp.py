

class MDP:

    def __init__(self):
        pass

    def get_states(self):
        return []

    def get_transition_matrix(self):
        return []

    def get_reward_matrix(self):
        return []

    def print(self, printline, **kwargs):
        pass

    def print_policy(self, printline, policy):
        pass

class MDPState:

    def __init__(self, value):
        self.value = (0, value)
        self.old_value = (0, value)
        self.iteration = 0
        self.reward = 0

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return False

    def get_neighbors(self):
        return []

    def get_actions(self):
        return []

    def get_reward(self, prev_state=None, action=None):
        return self.reward

    def get_value(self, iteration):
        if self.value[0] == iteration:
            return self.value[1]
        return self.old_value[1]

    def update_value(self, new_value, iteration):
        self.old_value = self.value
        self.value = (iteration, new_value)
        