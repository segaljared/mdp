from iteration_stats import IterationStats
from grid_world import GridWorld
import functools
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
import numpy


def first_equal_iteration(world: GridWorld, stats: IterationStats, transitions, policy_to_compare):
    equal_iterations = []
    policy_to_compare = [int(p) for p in policy_to_compare]

    def compare_policy(iteration_number, iteration_time, iteration_value, values):
            Q = numpy.empty((5, len(world.all_states) + 1))
            for aa in range(0, 5):
                Q[aa] = transitions[aa].dot(values)
            policy = Q.argmax(axis=0)
            diffs = numpy.sum(policy != policy_to_compare)
            if diffs == 0:
                equal_iterations.append(iteration_number)

    stats.load_and_run_analysis(compare_policy)
    return equal_iterations


def create_iteration_value_graph(stats: IterationStats, value_name, name, basefolder):
    iteration_values = []
    times = []

    def add_variation(iteration_number, iteration_time, iteration_value, values):
        iteration_values.append(iteration_value)
        times.append(iteration_time)

    stats.load_and_run_analysis(add_variation)

    plot.figure(1, clear=True)
    plot.title(name)
    plot.ylabel(value_name)
    plot.xlabel('iteration')
    plot.plot(range(1, len(iteration_values) + 1), iteration_values)
    plot.pause(0.001)
    plot.savefig('{}/{}'.format(basefolder, name.replace(' ', '_').lower()))
    plot.close()
    return times


def create_grid_world_state_graphs_value_iteration(world: GridWorld, stats: IterationStats, transitions, name, base_folder, iterations=None):
    
    def get_policy(w: GridWorld, t, V):
        Q = numpy.empty((5, len(w.all_states) + 1))
        for aa in range(0, 5):
            Q[aa] = t[aa].dot(V)
        return Q.argmax(axis=0)

    on_iteration = functools.partial(__create_world_graph__, world, transitions, iterations, name, base_folder, get_policy, None)

    stats.load_and_run_analysis(on_iteration)


def get_vi_policy(world: GridWorld, stats: IterationStats, transitions, iteration):
    policy = []
    def get_policy(iteration_number, iteration_time, iteration_value, values):
        if iteration_number == iteration:
            Q = numpy.empty((5, len(world.all_states) + 1))
            for aa in range(0, 5):
                Q[aa] = transitions[aa].dot(values)
            policy.append(Q.argmax(axis=0))
    
    stats.load_and_run_analysis(get_policy)

    return policy[0]
    
def create_grid_world_state_graphs_policy_iteration(world: GridWorld, stats: IterationStats, transitions, name, base_folder, iterations=None):

    def get_policy(w: GridWorld, t, p):
        return p

    def get_value(v, i):
        return 0.0
    
    on_iteration = functools.partial(__create_world_graph__, world, transitions, iterations, name, base_folder, get_policy, get_value)

    stats.load_and_run_analysis(on_iteration)

def get_pi_policy(world: GridWorld, stats: IterationStats, transitions, iteration):
    policy = []
    def get_policy(iteration_number, iteration_time, iteration_value, values):
        if iteration_number == iteration:
            policy.append(values)
    
    stats.load_and_run_analysis(get_policy)

    return policy[0]


def create_grid_world_state_graphs_q_learning(world: GridWorld, stats: IterationStats, name, base_folder, iterations=None):
    
    def get_policy(w: GridWorld, t, V):
        policy = []
        Q = numpy.array(V)
        for i, state in enumerate(world.get_states()):
            if len(state.get_actions()) > 0:
                policy.append(numpy.nanargmax(Q[:,i]))
            else:
                policy.append(0)
        return policy

    def get_value(Q, i):
        q_values = []
        for a in range(0, len(Q)):
            q_values.append(Q[a][i])
        if numpy.sum(numpy.isnan(q_values)) == len(q_values):
            return 0.0
        return numpy.nanmax(q_values)

    on_iteration = functools.partial(__create_world_graph__, world, None, iterations, name, base_folder, get_policy, get_value)

    stats.load_and_run_analysis(on_iteration)

def compare_policies(world: GridWorld, policy_a, policy_b, base_folder, name):
    differences = {}
    policy_layers = {}
    other_policy_layers = {}
    height = len(world.grid[0])
    width = len(world.grid[0][0])
    for layer_hash in world.grid:
        diff_layer = numpy.zeros((height, width))
        differences[layer_hash] = diff_layer
        policy_layers[layer_hash] = [[' ' for w in range(0, width)] for h in range(0, height)]
        other_policy_layers[layer_hash] = [[' ' for w in range(0, width)] for h in range(0, height)]
    
    for i in range(0, len(world.all_states)):
        state = world.all_states[i]
        policy_layers[state.t_hash][state.y][state.x] = __policy_to_text__(state, policy_a[i])
        if policy_a[i] != policy_b[i]:
            other_policy_layers[state.t_hash][state.y][state.x] = __policy_to_text__(state, policy_b[i])
            differences[state.t_hash][state.y][state.x] = 1.0

    colors = numpy.ones((2,4))
    colors[1] = numpy.array([180/256.0, 180/256.0, 1.0, 1.0])
    colormap = ListedColormap(colors)
    for layer_hash in world.grid:
        plot.figure(num=1, figsize=(width / 7.0, height / 5.0), clear=True)
        plot.title(name)
        plot.pcolormesh(numpy.flip(differences[layer_hash], axis=0), cmap=colormap)
        policies = numpy.flip(policy_layers[layer_hash], axis=0)
        other_policies = numpy.flip(other_policy_layers[layer_hash], axis=0)
        for y in range(0, height):
            for x in range(0, width):
                plot.text(x + 0.5, y + 0.5, policies[y][x], ha='center', va='center', color='black')
                plot.text(x + 0.5, y + 0.5, other_policies[y][x], ha='center', va='center', color='red')
        plot.pause(0.001)
        plot.savefig('{}/{}_T{}'.format(base_folder, name.replace(' ', '_').lower(), layer_hash))


def __create_world_graph__(world, transitions, iterations, name, base_folder, get_policy, get_value, iteration_number, iteration_time, iteration_value, values):
    if iterations is not None and iteration_number not in iterations:
        return
    value_layers = {}
    policy_layers = {}
    height = len(world.grid[0])
    width = len(world.grid[0][0])
    for layer_hash in world.grid:
        value_layer = numpy.zeros((height, width))
        value_layers[layer_hash] = value_layer
        policy_layers[layer_hash] = [[' ' for w in range(0, width)] for h in range(0, height)]

    policy = get_policy(world, transitions, values)

    for i in range(0, len(world.all_states)):
        state = world.all_states[i]
        if get_value is None:
            value = values[i]
        else:
            value = get_value(values, i)
        pol = policy[i]
        if len(state.get_actions()) == 0:
            action = ' '
        elif pol == 0:
            action = '↑'
        elif pol == 1:
            action = '→'
        elif pol == 2:
            action = '↓'
        elif pol == 3:
            action = '←'
        else:
            action = '©'
        value_layers[state.t_hash][state.y][state.x] = value
        policy_layers[state.t_hash][state.y][state.x] = action
    
    min_v = numpy.nanmin(values)
    max_v = numpy.nanmax(values)
    for layer_hash in world.grid:
        plot.figure(num=1, figsize=(width / 7.0, height / 5.0), clear=True)
        plot.title(name)
        value_map = plot.pcolormesh(numpy.flip(value_layers[layer_hash], axis=0), vmin=min_v, vmax=max_v)
        cmap = value_map.get_cmap()
        policies = numpy.flip(policy_layers[layer_hash], axis=0)
        for y in range(0, height):
            for x in range(0, width):
                plot.text(x + 0.5, y + 0.5, policies[y][x], ha='center', va='center', color='white')
        plot.colorbar(value_map)
        plot.pause(0.001)
        plot.savefig('{}/{}_{}_T{}'.format(base_folder, name.replace(' ', '_').lower(), iteration_number, layer_hash))


def __policy_to_text__(state, policy):
    if len(state.get_actions()) == 0:
        return ' '
    elif policy == 0:
        return '↑'
    elif policy == 1:
        return '→'
    elif policy == 2:
        return '↓'
    elif policy == 3:
        return '←'
    else:
        return '©'
