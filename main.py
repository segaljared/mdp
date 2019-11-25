from grid_world import GridWorld, GridWorldTile
from iteration_stats import IterationStats
from q_learning import QLearning
import numpy
import cmdptoolbox.mdp
import mdptoolbox.example
import time
import analysis
from forest import WrappedForest
import os

def main():
    if not os.path.exists('grid_world_results'):
        os.makedirs('grid_world_results')
    if not os.path.exists('grid_world_results/vi'):
        os.makedirs('grid_world_results/vi')
    if not os.path.exists('grid_world_results/pi'):
        os.makedirs('grid_world_results/pi')
    if not os.path.exists('grid_world_results/ql'):
        os.makedirs('grid_world_results/ql')
    if not os.path.exists('forest_results'):
        os.makedirs('forest_results')
    if not os.path.exists('forest_results/vi'):
        os.makedirs('forest_results/vi')
    if not os.path.exists('stats'):
        os.makedirs('stats')
    
    run_forest()
    run_grid_world()

def run_forest():
    run_value_iteration_forest(50, 80, 40)
    run_policy_iteration_forest(50, 80, 40)
    get_graphs_and_time_stats_forest_mdp()
    run_q_learning_forest(50, 80, 40)

def run_value_iteration_forest(S, r1, r2):
    forest = WrappedForest(S, r1, r2)
    transitions, reward = mdptoolbox.example.forest(S, r1, r2)
    for gamma in numpy.linspace(0.5, 1.0, num=11):
        if gamma == 1.0:
            gamma = 0.99
        stats = IterationStats('stats/vi_forest_{}.csv'.format('{:.2f}'.format(gamma).replace('.', '-')))

        def write_vi_stats(vi_iter, iteration, time, variation):
            stats.save_iteration(iteration, time, variation, vi_iter.V)
            print('[{}]:\t{}'.format(iteration, variation))

        print('gamma={}'.format(gamma))
        vi = cmdptoolbox.mdp.ValueIteration(transitions, reward, gamma, epsilon=0.0001, max_iter=10000, skip_check=True)
        vi.setVerbose()
        vi.setPrint(write_vi_stats)
        stats.start_writing()
        vi.run()
        stats.done_writing()
        print('found in {} iterations'.format(vi.iter))
        print('took {}'.format(vi.time))
        forest.print_policy(print, vi.policy)

def run_policy_iteration_forest(S, r1, r2):
    forest = WrappedForest(S, r1, r2)
    transitions, reward = mdptoolbox.example.forest(S, r1, r2)
    
    stats = IterationStats('stats/pi_forest.csv')

    def write_pi_stats(pi_iter, iteration, time, variation):
        stats.save_iteration(iteration, time, variation, pi_iter.policy)
        print('[{}]:\t{}'.format(iteration, variation))

    pi = cmdptoolbox.mdp.PolicyIteration(transitions, reward, 0.9, max_iter=1000)

    pi.setVerbose()
    pi.setPrint(write_pi_stats)
    stats.start_writing()
    pi.run()
    stats.done_writing()
    print('found in {} iterations'.format(pi.iter))
    print('took {}'.format(pi.time))
    forest.print_policy(print, pi.policy)

def get_graphs_and_time_stats_forest_mdp():
    stats = IterationStats('stats/vi_forest_0-90.csv')
    time_taken = analysis.create_iteration_value_graph(stats, 'variation', 'Variation for Forest Value Iteration', 'forest_results/vi')
    print(sum(time_taken) * 1000)
    stats = IterationStats('stats/pi_forest.csv')
    time_taken = analysis.create_iteration_value_graph(stats, 'variation', 'Changed Elements for Forest Policy Iteration', 'forest_results')
    print(sum(time_taken) * 1000)

def run_q_learning_forest(S, r1, r2):
    forest = WrappedForest(S, r1, r2)
    n_episodes = 10000
    how_often = n_episodes / 100

    stats = IterationStats('stats/ql_forest.csv', dims=2)
    
    def on_episode(episode, time, q_learner, q):
        forest.print_policy(print, q_learner.get_policy())
        stats.save_iteration(episode, time, numpy.nanmean(numpy.nanmax(q, axis=0)), q)

    def is_done(state, action, next_state):
        if next_state.state_num == 0:
            return True
        return False
    
    gamma = 0.99
    start = time.time()
    numpy.random.seed(5263228)
    q_l = QLearning(forest, 0.5, 0.2, gamma, on_episode=on_episode, start_at_0=True, alpha=0.1, is_done=is_done, every_n_episode=how_often)
    stats.start_writing()
    q_l.run(n_episodes)
    stats.done_writing()
    forest.print_policy(print, q_l.get_policy())
    print('took {} s'.format(time.time() - start))

    stats = IterationStats('stats/ql_forest.csv', dims=2)
    analysis.create_iteration_value_graph(stats, 'average Q', 'Average Q for each iteration on Forest Q Learning', 'forest_results')

def run_grid_world():
    world = GridWorld('simple_grid.txt', -0.01, include_treasure=True)
    print('# of states: {}'.format(len(world.all_states)))

    # uncomment this after the transition matrix has been saved
    #transitions = GridWorld.read_transition_matrix_file('simple_grid_t_matrix.csv')
    transitions = world.get_transition_matrix(save_to='simple_grid_t_matrix.csv')
    reward = world.get_reward_matrix()

    run_value_iteration_grid_world(world, transitions, reward)
    run_policy_iteration_grid_world(world, transitions, reward)
    compare_different_gamma_policies(world, transitions, reward)
    get_graphs_and_time_stats_grid_world_mdp(world, transitions)
    find_converged_policy(world, transitions)

    run_q_learning_grid_world()
    get_graph_q_learning()

def run_value_iteration_grid_world(world, transitions, reward):
    
    def write_vi_stats(vi_iter, iteration, time, variation):
        stats.save_iteration(iteration, time, variation, vi_iter.V)
        print('[{}]:\t{}'.format(iteration, variation))

    for gamma in numpy.linspace(0.8, 0.99, num=20):
        stats = IterationStats('stats/vi_simple_grid_{}.csv'.format(str(gamma).replace('.', '-')))

        vi = cmdptoolbox.mdp.ValueIteration(transitions, reward, gamma, epsilon=0.0001, max_iter=10000, skip_check=True)
        vi.setVerbose()
        vi.setPrint(write_vi_stats)
        stats.start_writing()
        vi.run()
        stats.done_writing()
        print('found in {} iterations'.format(vi.iter))
        print('took {}'.format(vi.time))
        world.print_policy(print, vi.policy)
        last_policy = vi.policy
    return last_policy

def run_policy_iteration_grid_world(world, transitions, reward):

    def write_pi_stats(pi_iter, iteration, time, variation):
        stats.save_iteration(iteration, time, variation, pi_iter.policy)
        print('[{}]:\t{}'.format(iteration, variation))

    for gamma in numpy.linspace(0.8, 1.0, num=5):
        if gamma == 1.0:
            gamma = 0.99
        stats = IterationStats('stats/pi_simple_grid_{}.csv'.format(str(gamma).replace('.', '-')))
        pi = cmdptoolbox.mdp.PolicyIteration(transitions, reward, gamma, max_iter=1000, skip_check=True)

        print("set up before run")
        pi.setVerbose()
        pi.setPrint(write_pi_stats)
        stats.start_writing()
        pi.run()
        stats.done_writing()
        print('found in {} iterations'.format(pi.iter))
        print('took {}'.format(pi.time))
        world.print_policy(print, pi.policy)
        last_policy = pi.policy
    return last_policy

def run_q_learning_grid_world():
    world = GridWorld('simple_grid.txt', -0.01, include_treasure=False)
    n_episodes = 500000
    how_often = n_episodes / 500

    stats = IterationStats('stats/ql_simple_grid.csv', dims=5)

    def on_update(state, action, next_state, q_learner):
        #print('[{},{}] - {} -> [{},{}]'.format(state.x, state.y, action[0], next_state.x, next_state.y))
        pass

    def on_episode(episode, time, q_learner, q):
        world.print_policy(print, q_learner.get_policy())
        stats.save_iteration(episode, time, numpy.nanmean(numpy.nanmax(q, axis=0)), q)
        #time.sleep(1)

    for state in world.get_states():
        if state.tile_type == GridWorldTile.GOAL:
            goal_state = state
            break

    def initialize_toward_goal(state: GridWorldTile):
        actions = state.get_actions()
        if len(actions) == 0:
            return []
        diff_x = goal_state.x - state.x
        diff_y = goal_state.y - state.y
        best_value = 0.1
        if len(actions) == 5 and actions[4][0].startswith('get treasure'):
            best_action = actions[4][0]
        elif abs(diff_x) >= abs(diff_y):
            if diff_x > 0:
                best_action = 'move east'
            else:
                best_action = 'move west'
        else:
            if diff_y < 0:
                best_action = 'move north'
            else:
                best_action = 'move south'
        values = [-0.1] * len(actions)
        for i, action in enumerate(actions):
            if action[0] == best_action:
                values[i] = best_value
        return values

    gamma = 0.99
    q_l = QLearning(world, 0.5, 0.05, gamma, on_update=on_update, on_episode=on_episode, initializer=initialize_toward_goal, start_at_0=True, alpha=0.1, every_n_episode=how_often)
    stats.start_writing()
    q_l.run(n_episodes)
    stats.done_writing()
    world.print_policy(print, q_l.get_policy())

def compare_different_gamma_policies(world, transitions, reward):
    stats = IterationStats('stats/pi_simple_grid_0-99.csv')
    analysis.create_grid_world_state_graphs_policy_iteration(world, stats, transitions, 'PI state graph', 'grid_world_results/pi', iterations=[8])
    stats = IterationStats('stats/vi_simple_grid_0-99.csv')
    vi_policy = analysis.get_vi_policy(world, stats, transitions, 252)
    stats = IterationStats('stats/pi_simple_grid_0-99.csv')
    pi_policy = analysis.get_pi_policy(world, stats, transitions, 26)
    analysis.compare_policies(world, vi_policy, pi_policy, 'grid_world_results', 'comparison_pi_vi')
    stats = IterationStats('stats/vi_simple_grid_0-8.csv')
    vi_policy = analysis.get_vi_policy(world, stats, transitions, 55)
    analysis.compare_policies(world, vi_policy, pi_policy, 'grid_world_results', 'comparison pi vi low gamma')

def get_graphs_and_time_stats_grid_world_mdp(world, transitions):
    stats = IterationStats('stats/vi_simple_grid_0-99.csv')
    analysis.create_grid_world_state_graphs_value_iteration(world, stats, transitions, 'vi_graph', 'grid_world_results/vi', iterations=[252])
    times = analysis.create_iteration_value_graph(stats, 'variation', 'Variation for Grid World Value Iteration', 'grid_world_results/vi')
    print('vi total time: {} ms'.format(numpy.sum(times) * 1000.0))
    stats = IterationStats('stats/pi_simple_grid_0-99.csv')
    times = analysis.create_iteration_value_graph(stats, '# of elements changed', 'Changed Elements for Grid World Policy Iteration', 'grid_world_results/pi')
    print('pi total time: {} ms'.format(numpy.sum(times) * 1000.0))

def find_converged_policy(world, transitions):
    stats = IterationStats('stats/pi_simple_grid_0-99.csv')
    pi_policy = analysis.get_pi_policy(world, stats, transitions, 26)
    stats = IterationStats('stats/vi_simple_grid_0-99.csv')
    equal_iters = analysis.first_equal_iteration(world, stats, transitions, pi_policy)
    print(equal_iters)

def get_graph_q_learning():
    stats = IterationStats('stats/ql_simple_grid.csv', dims=5)
    analysis.create_iteration_value_graph(stats, 'average Q', 'Average Q for each iteration on Grid World Q Learning', 'grid_world_results/ql')

if __name__ == "__main__":
    main()
