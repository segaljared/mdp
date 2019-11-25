from mdp import MDP, MDPState
import copy
import csv
import numpy
import scipy.sparse


class GridWorld(MDP):

    def __init__(self, grid_filename, normal_reward, goal_reward=10, trap_reward=-10, pit_reward=-5, include_treasure=True):
        super().__init__()
        self.grid = {}
        self.all_states = []
        self.width = 0
        self.height = 0
        self.__read_grid_file__(grid_filename, normal_reward, goal_reward, trap_reward, pit_reward, include_treasure)

    def print(self, printline, **kwargs):
        if 'x' in kwargs:
            start_x = kwargs['x']
        else:
            start_x = 0
        if 'y' in kwargs:
            start_y = kwargs['y']
        else:
            start_y = 0
        if 'w' in kwargs:
            width = kwargs['w']
        elif 'width' in kwargs:
            width = kwargs['width']
        else:
            width = self.width - start_x
        if 'h' in kwargs:
            height = kwargs['h']
        elif 'height' in kwargs:
            height = kwargs['height']
        else:
            height = self.height - start_y
        if 't' in kwargs:
            t_state = kwargs['t']
            t_hash = __get_treasure_hash__(t_state)
        else:
            t_hash = 0
        printline('=' * (width * 7))
        for y in range(start_y, min(start_y + height, self.height)):
            line_values = []
            for x in range(start_x, min(start_x + width, self.width)):
                line_values.append('{:6.2f}'.format(self.grid[t_hash][y][x].value[1]))
            printline(' '.join(line_values))
        printline('=' * (width * 7))

    def print_policy(self, printline, policy):
        reverse_tile_dict = {}
        for tile_char in GridWorldTile.TILE_TYPE:
            reverse_tile_dict[GridWorldTile.TILE_TYPE[tile_char]] = tile_char
        def get_reverse(tile_num):
            if tile_num in reverse_tile_dict:
                return reverse_tile_dict[tile_num]
            return 'T'

        gridlines = {}
        for grid in self.grid:
            g = []
            for row in self.grid[grid]:
                grid_row = []
                for tile in row:
                    grid_row.append(get_reverse(tile.tile_type))
                g.append(grid_row)
            gridlines[grid] = g
        for i in range(0, len(self.all_states)):
            state = self.all_states[i]
            pol = policy[i]
            if len(state.get_actions()) == 0:
                 continue
            elif pol == 0:
                action = '↑'
            elif pol == 1:
                action = '→'
            elif pol == 2:
                action = '↓'
            elif pol == 3:
                action = '←'
            else:
                action = '@'
            gridlines[state.t_hash][state.y][state.x] = action
        for t_hash in gridlines:
            printline('=' * self.width)
            printline('=' * self.width)
            grid = gridlines[t_hash]
            for row in grid:
                printline(''.join(row))
    
    def get_states(self):
        return self.all_states

    def get_transition_matrix(self, save_to=None):
        transitions = []
        prob_all = []
        row_ind_all = []
        col_ind_all = []
        terminal = len(self.all_states)
        for a in range(0, 5):
            probs = []
            row_ind = []
            col_ind = []
            for i, state in enumerate(self.all_states):
                actions = state.get_actions()
                if a < len(actions):
                    action = actions[a]
                    for transition in action[1]:
                        if transition[1] is None:
                            t_index = i
                        else:
                            t_index = self.all_states.index(transition[1])
                        probs.append(transition[0])
                        row_ind.append(i)
                        col_ind.append(t_index)
                elif len(actions) == 0:
                    probs.append(1.0)
                    row_ind.append(i)
                    col_ind.append(terminal)
                else:
                    probs.append(1.0)
                    row_ind.append(i)
                    col_ind.append(i)
            #create terminal state
            probs.append(1.0)
            row_ind.append(terminal)
            col_ind.append(terminal)

            prob_all.append(probs)
            row_ind_all.append(row_ind)
            col_ind_all.append(col_ind)
            transitions.append(scipy.sparse.csr_matrix((probs, (row_ind, col_ind)), shape=(len(self.all_states) + 1, len(self.all_states) + 1)))
        if save_to is not None:
            with open(save_to, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerow([len(self.all_states) + 1])
                for a in range(0, 5):
                    writer.writerow(prob_all[a])
                    writer.writerow(row_ind_all[a])
                    writer.writerow(col_ind_all[a])
        return transitions

    @staticmethod
    def read_transition_matrix_file(filename):
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            size_row = next(reader)
            size = int(size_row[0])
            transitions = []
            for _ in range(0, 5):
                probs_s = next(reader)
                probs = [float(p) for p in probs_s]
                row_ind_s = next(reader)
                row_ind = [int(r) for r in row_ind_s]
                col_ind_s = next(reader)
                col_ind = [int(c) for c in col_ind_s]
                transitions.append(scipy.sparse.csr_matrix((probs, (row_ind, col_ind)), shape=(size, size)))
            return transitions

    def get_reward_matrix(self):
        rewards = numpy.zeros((len(self.all_states) + 1,))
        for i, state in enumerate(self.all_states):
            rewards[i] = state.get_reward()
        rewards[len(self.all_states)] = 0.0
        return rewards

    def __read_grid_file__(self, grid_filename, normal_reward, goal_reward, trap_reward, pit_reward, include_treasure):
        with open(grid_filename, mode="r") as f:
            size_pair = f.readline()
            sizes = size_pair.split(',')
            self.width = int(sizes[0])
            self.height = int(sizes[1])
            temp_grid = []
            treasure_count = 0
            for _ in range(0, self.height):
                line = f.readline()
                grid_line = []
                for x in range(0, self.width):
                    grid_line.append(line[x])
                    if line[x] == 'T':
                        treasure_count += 1
                temp_grid.append(grid_line)
            treasure_state = []
            for _ in range(0, treasure_count):
                treasure_state.append(False)
            treasure_states = __enumerate_states__(treasure_state)
            for treasure_state in treasure_states:
                captured_treasure = treasure_state.count(True)
                treasure_hash = __get_treasure_hash__(treasure_state)
                this_grid = []
                treasure_current = 0
                for y in range(0, self.height):
                    row = []
                    for x in range(0, self.width):
                        tile_char = temp_grid[y][x]
                        tile_type = GridWorldTile.TILE_TYPE[tile_char]
                        if tile_type == GridWorldTile.TREASURE:
                            tile_type += treasure_current
                            treasure_current += 1
                        tile = GridWorldTile(x, y, tile_type, treasure_state, self, include_treasure)
                        if tile_type == GridWorldTile.GOAL:
                            tile.reward = goal_reward + goal_reward * captured_treasure
                        elif tile_type == GridWorldTile.TRAP:
                            tile.reward = trap_reward
                        elif tile_type == GridWorldTile.PIT:
                            tile.reward = pit_reward
                        else:
                            tile.reward = normal_reward
                        row.append(tile)
                        if tile.tile_type == GridWorldTile.START and treasure_hash == 0:
                            self.all_states.insert(0, tile)
                        elif tile.tile_type != GridWorldTile.IMPASSABLE:
                            self.all_states.append(tile)
                    this_grid.append(row)
                self.grid[treasure_hash] = this_grid

class GridWorldTile(MDPState):

    IMPASSABLE = 0
    NORMAL_TILE = 1
    GOAL = 2
    TRAP = 3
    PIT = 4
    START = 5
    WEST_LIGHT_SLOPE = 6
    WEST_STEEP_SLOPE = 7
    EAST_LIGHT_SLOPE = 8
    EAST_STEEP_SLOPE = 9
    NORTH_LIGHT_SLOPE = 10
    NORTH_STEEP_SLOPE = 11
    SOUTH_LIGHT_SLOPE = 12
    SOUTH_STEEP_SLOPE = 13
    TREASURE = 14

    TILE_TYPE = {'#': IMPASSABLE,
                 'O': NORMAL_TILE,
                 'G': GOAL,
                 'U': TRAP,
                 'P': PIT,
                 'S': START,
                 '(': WEST_LIGHT_SLOPE,
                 '<': WEST_STEEP_SLOPE,
                 ')': EAST_LIGHT_SLOPE,
                 '>': EAST_STEEP_SLOPE,
                 '^': NORTH_LIGHT_SLOPE,
                 'A': NORTH_STEEP_SLOPE,
                 'v': SOUTH_LIGHT_SLOPE,
                 'V': SOUTH_STEEP_SLOPE,
                 'T': TREASURE}

    NORTH_PROBS = {IMPASSABLE: [0.0,0.0,0.0,0.0],
                   GOAL: [0.0,0.0,0.0,0.0],
                   TRAP: [0.0,0.0,0.0,0.0],
                   PIT: [0.0,0.0,0.0,0.0],
                   NORMAL_TILE: [0.8, 0.1, 0.1, 0.0],
                   START: [0.8, 0.1, 0.1, 0.0],
                   WEST_LIGHT_SLOPE: [0.75, 0.2 , 0.05, 0.0],
                   WEST_STEEP_SLOPE: [0.4, 0.6 , 0.0, 0.0],
                   EAST_LIGHT_SLOPE: [0.75, 0.05 , 0.2, 0.0],
                   EAST_STEEP_SLOPE: [0.4, 0.0 , 0.6, 0.0],
                   NORTH_LIGHT_SLOPE: [0.98, 0.01 , 0.01, 0.0],
                   NORTH_STEEP_SLOPE: [1.0, 0.0 , 0.0, 0.0],
                   SOUTH_LIGHT_SLOPE: [0.6, 0.15 , 0.15, 0.1],
                   SOUTH_STEEP_SLOPE: [0.1, 0.1 , 0.1, 0.7]}

    SOUTH_PROBS = {IMPASSABLE: [0.0,0.0,0.0,0.0],
                   GOAL: [0.0,0.0,0.0,0.0],
                   TRAP: [0.0,0.0,0.0,0.0],
                   PIT: [0.0,0.0,0.0,0.0],
                   NORMAL_TILE: [0.8, 0.1, 0.1, 0.0],
                   START: [0.8, 0.1, 0.1, 0.0],
                   EAST_LIGHT_SLOPE: [0.75, 0.2 , 0.05, 0.0],
                   EAST_STEEP_SLOPE: [0.4, 0.6 , 0.0, 0.0],
                   WEST_LIGHT_SLOPE: [0.75, 0.05 , 0.2, 0.0],
                   WEST_STEEP_SLOPE: [0.4, 0.0 , 0.6, 0.0],
                   SOUTH_LIGHT_SLOPE: [0.98, 0.01 , 0.01, 0.0],
                   SOUTH_STEEP_SLOPE: [1.0, 0.0 , 0.0, 0.0],
                   NORTH_LIGHT_SLOPE: [0.6, 0.15 , 0.15, 0.1],
                   NORTH_STEEP_SLOPE: [0.1, 0.1 , 0.1, 0.7]}

    WEST_PROBS = {IMPASSABLE: [0.0,0.0,0.0,0.0],
                  GOAL: [0.0,0.0,0.0,0.0],
                  TRAP: [0.0,0.0,0.0,0.0],
                  PIT: [0.0,0.0,0.0,0.0],
                  NORMAL_TILE: [0.8, 0.1, 0.1, 0.0],
                  START: [0.8, 0.1, 0.1, 0.0],
                  SOUTH_LIGHT_SLOPE: [0.75, 0.2 , 0.05, 0.0],
                  SOUTH_STEEP_SLOPE: [0.4, 0.6 , 0.0, 0.0],
                  NORTH_LIGHT_SLOPE: [0.75, 0.05 , 0.2, 0.0],
                  NORTH_STEEP_SLOPE: [0.4, 0.0 , 0.6, 0.0],
                  WEST_LIGHT_SLOPE: [0.98, 0.01 , 0.01, 0.0],
                  WEST_STEEP_SLOPE: [1.0, 0.0 , 0.0, 0.0],
                  EAST_LIGHT_SLOPE: [0.6, 0.15 , 0.15, 0.1],
                  EAST_STEEP_SLOPE: [0.1, 0.1 , 0.1, 0.7]}

    EAST_PROBS = {IMPASSABLE: [0.0,0.0,0.0,0.0],
                  GOAL: [0.0,0.0,0.0,0.0],
                  TRAP: [0.0,0.0,0.0,0.0],
                  PIT: [0.0,0.0,0.0,0.0],
                  NORMAL_TILE: [0.8, 0.1, 0.1, 0.0],
                  START: [0.8, 0.1, 0.1, 0.0],
                  NORTH_LIGHT_SLOPE: [0.75, 0.2 , 0.05, 0.0],
                  NORTH_STEEP_SLOPE: [0.4, 0.6 , 0.0, 0.0],
                  SOUTH_LIGHT_SLOPE: [0.75, 0.05 , 0.2, 0.0],
                  SOUTH_STEEP_SLOPE: [0.4, 0.0 , 0.6, 0.0],
                  EAST_LIGHT_SLOPE: [0.98, 0.01 , 0.01, 0.0],
                  EAST_STEEP_SLOPE: [1.0, 0.0 , 0.0, 0.0],
                  WEST_LIGHT_SLOPE: [0.6, 0.15 , 0.15, 0.1],
                  WEST_STEEP_SLOPE: [0.1, 0.1 , 0.1, 0.7]}
    
    def __init__(self, x, y, tile_type, treasure_state, grid_world, include_treasure):
        super().__init__(0)
        self.x = x
        self.y = y
        self.tile_type = tile_type
        self.treasure_state = treasure_state
        self.grid_world = grid_world
        self.neighbors = None
        self.actions = None

        self.__treasure_hash = __get_treasure_hash__(self.treasure_state)
        self.t_hash = self.__treasure_hash
        self.include_treasure = include_treasure

    def __hash__(self):
        return hash((self.x, self.y, self.tile_type, self.__treasure_hash))

    def __eq__(self, other):
        if isinstance(other, GridWorldTile):
            return self.x == other.x and self.y == other.y and self.tile_type == other.tile_type and self.__treasure_hash == other.__treasure_hash
        return NotImplemented
    
    def get_neighbors(self):
        if self.neighbors is None:
            self.__setup_neighbors_actions__()
        return self.neighbors

    def get_actions(self):
        if self.actions is None:
            self.__setup_neighbors_actions__()
        return self.actions

    def __setup_neighbors_actions__(self):
        self.neighbors = []
        self.actions = []
        if (self.tile_type == GridWorldTile.IMPASSABLE 
            or self.tile_type == GridWorldTile.GOAL
            or self.tile_type == GridWorldTile.TRAP
            or self.tile_type == GridWorldTile.PIT):
            return

        if self.y > 0:
            north_neighbor = self.grid_world.grid[self.__treasure_hash][self.y - 1][self.x]
            if north_neighbor.tile_type == GridWorldTile.IMPASSABLE:
                north_neighbor = None
            else:
                self.neighbors.append(north_neighbor)
        else:
            north_neighbor = None
        if self.y < self.grid_world.height - 1:
            south_neighbor = self.grid_world.grid[self.__treasure_hash][self.y + 1][self.x]
            if south_neighbor.tile_type == GridWorldTile.IMPASSABLE:
                south_neighbor = None
            else:
                self.neighbors.append(south_neighbor)
        else:
            south_neighbor = None
        if self.x > 0:
            west_neighbor = self.grid_world.grid[self.__treasure_hash][self.y][self.x - 1]
            if west_neighbor.tile_type == GridWorldTile.IMPASSABLE:
                west_neighbor = None
            else:
                self.neighbors.append(west_neighbor)
        else:
            west_neighbor = None
        if self.x < self.grid_world.width - 1:
            east_neighbor = self.grid_world.grid[self.__treasure_hash][self.y][self.x + 1]
            if east_neighbor.tile_type == GridWorldTile.IMPASSABLE:
                east_neighbor = None
            else:
                self.neighbors.append(east_neighbor)
        else:
            east_neighbor = None
        
        if self.tile_type >= GridWorldTile.TREASURE:
            tile_type = GridWorldTile.NORMAL_TILE
        else:
            tile_type = self.tile_type
        self.actions.append(GridWorldTile.__create_north_action__(tile_type, north_neighbor, west_neighbor, east_neighbor, south_neighbor))
        self.actions.append(GridWorldTile.__create_east_action__(tile_type, east_neighbor, north_neighbor, south_neighbor, west_neighbor))
        self.actions.append(GridWorldTile.__create_south_action__(tile_type, south_neighbor, east_neighbor, west_neighbor, north_neighbor))
        self.actions.append(GridWorldTile.__create_west_action__(tile_type, west_neighbor, south_neighbor, north_neighbor, east_neighbor))
        add_fake = True
        if self.tile_type >= GridWorldTile.TREASURE:
            treasure_id = self.tile_type - GridWorldTile.TREASURE
            if not self.treasure_state[treasure_id]:
                add_fake = False
                new_treasure_state = list(self.treasure_state)
                new_treasure_state[treasure_id] = True
                new_treasure_hash = __get_treasure_hash__(new_treasure_state)
                self.actions.append(('get treasure {}'.format(treasure_id), [(0.8, self.grid_world.grid[new_treasure_hash][self.y][self.x]), (0.2, self)]))
        if add_fake and self.include_treasure:
            self.actions.append(GridWorldTile.__create_no_treasure_action__(tile_type, north_neighbor, east_neighbor, south_neighbor, west_neighbor, self))

    @staticmethod
    def __create_north_action__(tile_type, north_neighbor, west_neighbor, east_neighbor, south_neighbor):
        probs = GridWorldTile.NORTH_PROBS[tile_type]
        return GridWorldTile.__create_action__(probs, 'move north', north_neighbor, west_neighbor, east_neighbor, south_neighbor)

    @staticmethod
    def __create_south_action__(tile_type, south_neighbor, east_neighbor, west_neighbor, north_neighbor):
        probs = GridWorldTile.SOUTH_PROBS[tile_type]
        return GridWorldTile.__create_action__(probs, 'move south', south_neighbor, east_neighbor, west_neighbor, north_neighbor)

    @staticmethod
    def __create_west_action__(tile_type, west_neighbor, south_neighbor, north_neighbor, east_neighbor):
        probs = GridWorldTile.WEST_PROBS[tile_type]
        return GridWorldTile.__create_action__(probs, 'move west', west_neighbor, south_neighbor, north_neighbor, east_neighbor)

    @staticmethod
    def __create_east_action__(tile_type, east_neighbor, north_neighbor, south_neighbor, west_neighbor):
        probs = GridWorldTile.EAST_PROBS[tile_type]
        return GridWorldTile.__create_action__(probs, 'move east', east_neighbor, north_neighbor, south_neighbor, west_neighbor)

    @staticmethod
    def __create_action__(probs, name, success, left, right, back):
        return (name, [(probs[0], success), (probs[1], left), (probs[2], right), (probs[3], back)])
        
    @staticmethod
    def __create_no_treasure_action__(tile_type, north, east, south, west, own):
        name = 'try pick up treasure'
        if tile_type == GridWorldTile.NORTH_STEEP_SLOPE:
            return (name, [(1.0, north)])
        if tile_type == GridWorldTile.EAST_STEEP_SLOPE:
            return (name, [(1.0, east)])
        if tile_type == GridWorldTile.SOUTH_STEEP_SLOPE:
            return (name, [(1.0, south)])
        if tile_type == GridWorldTile.WEST_STEEP_SLOPE:
            return (name, [(1.0, west)])
        if tile_type == GridWorldTile.NORTH_LIGHT_SLOPE:
            return (name, [(0.2, north), (0.8, own)])
        if tile_type == GridWorldTile.EAST_LIGHT_SLOPE:
            return (name, [(0.2, east), (0.8, own)])
        if tile_type == GridWorldTile.SOUTH_LIGHT_SLOPE:
            return (name, [(0.2, south), (0.8, own)])
        if tile_type == GridWorldTile.WEST_LIGHT_SLOPE:
            return (name, [(0.2, west), (0.8, own)])
        return (name, [(0.8, own), (0.05, north), (0.05, east), (0.05, south), (0.05, west)])

def __get_treasure_hash__(treasure_state):
    treasure_hash = 0
    for i, s in enumerate(treasure_state):
        treasure_hash += (2**i if s else 0)
    return treasure_hash

def __enumerate_states__(state, i=0):
    if i >= len(state):
        if len(state) == 0:
            return [[False]]
        return [state]
    other_states = __enumerate_states__(state, i + 1)
    states_to_return = []
    for other in other_states:
        o = copy.deepcopy(other)
        o[i] = False
        states_to_return.append(o)
        other[i] = True
        states_to_return.append(other)
    return states_to_return
