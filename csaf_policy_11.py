import gym
import time
import math
import random
import numpy as np

TIMESTEP = .1
PHEROMONE_INIT_VALUE = 1
OBS_CENTER = (5, 5)
TROUBLESHOOTING = False

from enum import Enum

from MAF_gym.envs.entity import *
from MAF_gym.envs.MAF_env import MAF_gym

"""
No movement: 0
North (-1,0): 1
East (0,1): 2
South (1,0): 3
West (0,-1): 4
Join queue: 5
"""

"""
Observation
0: EMPTY agents
1: FULL agents
2: obstacles
3: rqep
4: cqep
5: rq agents
6: cq_agents
7: phero map & highways
"""

"""
Pheromones
positive: brown pheromones
negative: yellow pheromones
0: black (default)
"""

class Actions(Enum):
    AT_HOME_1 = 1
    LOOK_FOR_FOOD = 2
    CHOOSE_NEXT_PATCH = 3
    PICK_FOOD = 4
    REMOVE_TRAIL = 6
    CLIMB = 7
    RETURN_TO_NEST = 8
    RETURN_AND_COLOR = 9
    AT_HOME_2 = 10
    WIPEOUT = 11
    DROP_FOOD = 13
    LOOK_FOR_TRAIL_1 = 14
    LOOK_FOR_TRAIL_2 = 15
    LOOK_FOR_TRAIL_3 = 16
    LOOK_FOR_TRAIL_4 = 17 
    STRAIGHT_TO_RESOURCE_1 = 18
    STRAIGHT_TO_RESOURCE_2 = 19
        
def get_coords_ahead(curr_coords, prev_action, world_size=128):
    if prev_action == 1:
        return (max(0, curr_coords[0] - 1), curr_coords[1])
    elif prev_action == 2:
        return (curr_coords[0], min(world_size -1,curr_coords[1] + 1))
    elif prev_action == 3:
        return (min(world_size -1,curr_coords[0] + 1), curr_coords[1])
    elif prev_action == 4:
        return (curr_coords[0], max(0,curr_coords[1] - 1))
    else:
        return (-1, -1)

def get_coords_right(curr_coords, prev_action, world_size=128, num_spaces_over=1):
    if prev_action == 1:
        return (curr_coords[0], min(world_size -1,curr_coords[1] + num_spaces_over))
    elif prev_action == 2:
        return (min(world_size -1,curr_coords[0] + num_spaces_over), curr_coords[1])
    elif prev_action == 3:
        return (curr_coords[0], max(0,curr_coords[1] - num_spaces_over))
    elif prev_action == 4:
        return (max(0,curr_coords[0] - num_spaces_over), curr_coords[1])
    else:
        return (-1, -1)

def get_coords_left(curr_coords, prev_action, world_size=128, num_spaces_over=1):
    if prev_action == 1:
        return (curr_coords[0], max(0,curr_coords[1] - num_spaces_over))
    elif prev_action == 2:
        return (max(0,curr_coords[0] - num_spaces_over), curr_coords[1])
    elif prev_action == 3:
        return (curr_coords[0], min(world_size - 1,curr_coords[1] + num_spaces_over))
    else: # prev_action == 4:
        return (min(world_size -1,curr_coords[0] + num_spaces_over), curr_coords[1])

def get_coords_behind(curr_coords, prev_action, world_size=128):
    if prev_action == 1:
        return (min(world_size -1,curr_coords[0] + 1), curr_coords[1])
    elif prev_action == 2:
        return (curr_coords[0], max(0,curr_coords[1] - 1))
    elif prev_action == 3:
        return (max(0,curr_coords[0] - 1), curr_coords[1])
    elif prev_action == 4:
        return (curr_coords[0], min(world_size -1,curr_coords[1] + 1))
    else:
        return (-1, -1)


def pheromone_to_right(observation, prev_action):
    # get coords of square to the right
    if prev_action not in range(1, 5):
        return False
    right_pos = get_coords_right(OBS_CENTER, prev_action)

    # if it's an obstacle/out of bounds, return false
    if observation[2][right_pos[0]][right_pos[1]] > 0:
        return False
    # otherwise, check if it has pheromones
    return observation[7][right_pos[0]][right_pos[1]]

def get_value_ahead_obs(layer, observation, prev_action):
    ahead_pos = get_coords_ahead(OBS_CENTER, prev_action)
    if prev_action not in range(1, 5):
        return 0
    return observation[layer][ahead_pos[0]][ahead_pos[1]]

def is_rc_ahead(env, prev_action, agentID):
    curr_pos = env.agents[agentID - 1].getLocation()
    ahead_pos = get_coords_ahead(curr_pos, prev_action)
    for rc in env.RCs:
        if rc.row == ahead_pos[0] and rc.col == ahead_pos[1]:
            return True
    return False

def is_cache(coords):
    for rc in env.RCs:
        if rc.row == coords[0] and rc.col == coords[1]:
            return True
    return False
    
def detect_and_adjust_heading(env, observation, prev_action, agentID, cache_vector):
    # if pheromone_value in right_cell = 0:
    # added: if there's not an obstacle to the right

    right_pos = get_coords_right(OBS_CENTER, prev_action)
    ahead_pos = get_coords_ahead(OBS_CENTER, prev_action)
    left_pos = get_coords_left(OBS_CENTER, prev_action)
    if (pheromone_to_right(observation, prev_action) == 0 \
    and no_obstacle_at(observation, right_pos)):
        return move_right(prev_action)
    elif observation[7][ahead_pos[0]][ahead_pos[1]] == 0 \
    and no_obstacle_at(observation, ahead_pos):
        return prev_action
    elif observation[7][left_pos[0]][left_pos[1]] == 0 \
    and no_obstacle_at(observation, left_pos):
        return move_left(prev_action)
    else:
        # surrounded by pheromones, so following gradient
        return get_upward_gradient_direction(observation, prev_action, observation[7], cache_vector)
    #     behind_pos_env = get_coords_behind(env.agents[agentID - 1].getLocation(), prev_action)
    #     env.phero_stack_loaded[behind_pos_env[0], behind_pos_env[1], :] = 0 
    #     return move_backward(prev_action)



def orbit_nest(observation, prev_action):
    right_pos = get_coords_right(OBS_CENTER, prev_action)

    if (pheromone_to_right(observation, prev_action) < 1) \
    and observation[0][right_pos[0]][right_pos[1]] == 0 \
    and observation[1][right_pos[0]][right_pos[1]] == 0:
        return prev_action % 4 + 1
    else:
        return prev_action
    

    

# check for yellow pheromones
def check_for_trail(pheromones, agent_pos):
    if pheromones[agent_pos[0] - 1][agent_pos[1]] > 0 \
    or pheromones[agent_pos[0]][agent_pos[1] + 1] > 0 \
    or pheromones[agent_pos[0] + 1][agent_pos[1]] > 0 \
    or pheromones[agent_pos[0]][agent_pos[1] - 1] > 0:
        return True
    else:
        return False

def get_downward_gradient_direction(obs, prev_action, pheromones, cache_vector, curr_coords=OBS_CENTER):

    # phero_value = 0
    phero_value = pheromones[curr_coords[0]][curr_coords[1]]
    if TROUBLESHOOTING:
        print('GET_DOWNWARD_TRAIL')

    phero_values = np.array([
        pheromones[curr_coords[0] - 1][curr_coords[1]],
        pheromones[curr_coords[0]][curr_coords[1] + 1],
        pheromones[curr_coords[0] + 1][curr_coords[1]],
        pheromones[curr_coords[0]][curr_coords[1] - 1]])

    if TROUBLESHOOTING:
        print('PHERO VALUE: ' + str(phero_value))
        print('DOWNWARD TRAIL DIRECTION OPTIONS:')
        print(phero_values)

    if 1.05 in phero_values:
        return -1 * (np.where(phero_values == 1.05)[0][0] + 1)

    phero_indices = np.argsort(phero_values)

    locations = {
        0: (OBS_CENTER[0] - 1, OBS_CENTER[1]),
        1: (OBS_CENTER[0], OBS_CENTER[1] + 1),    
        2: (OBS_CENTER[0] + 1, OBS_CENTER[1]),
        3: (OBS_CENTER[0], OBS_CENTER[1] - 1)
    }

    possible_moves = []
    
    for i in range(len(phero_indices)):
        # and phero_values[phero_indices[i]] <= phero_value -> sometimes may need to go to next value that isn't strictly <=
        if phero_values[phero_indices[i]] > 0 \
        and prev_action != ((phero_indices[i] + 2) % 4) + 1:
            possible_moves.append(phero_indices[i])
        i += 1

    if TROUBLESHOOTING:
        print('POSSIBLE MOVES: ')
        print(possible_moves)

    better_possible_moves = []

    for move in possible_moves:
        if no_obstacle_at(obs, locations[move]): 
            better_possible_moves.append(move)
    
    if len(better_possible_moves) > 0:
        if len(better_possible_moves) > 1 \
        and phero_values[better_possible_moves[0]] == phero_values[better_possible_moves[1]]:
            if TROUBLESHOOTING:
                print('USING CACHE VECTOR')
            return selectAction(cache_vector[0], cache_vector[1])
        else:
            if TROUBLESHOOTING:
                print('BEST MOVE: ' + str(better_possible_moves[0] + 1))
            return better_possible_moves[0] + 1

    # going to be invalid (blocked), but returning zero from this function would
    # indicate that we're done with this trail - better to no-op then confuse the agent
    # (and since no other move is feasible, we can't do anything anyway)
    if len(possible_moves) > 0:
        if TROUBLESHOOTING:
            print('DOWNWARD_TRAIL_DIRECTION DECISION (not real move): ' + str(possible_moves[0] + 1))
        return possible_moves[0] + 1
    else:
        if TROUBLESHOOTING:
            print('DOWNWARD_TRAIL_DIRECTION DECISION: 0 (no path moves left)')
        return 0
    return 0

def get_upward_trail_direction(obs, prev_action, pheromones, curr_coords=OBS_CENTER, world_size=128):
    phero_value = pheromones[curr_coords[0]][curr_coords[1]]
    action = 0

    if TROUBLESHOOTING:
        print([
            pheromones[curr_coords[0] - 1][curr_coords[1]],
            pheromones[curr_coords[0]][curr_coords[1] + 1],
            pheromones[curr_coords[0] + 1][curr_coords[1]],
            pheromones[curr_coords[0]][curr_coords[1] - 1]
        ])

    if pheromones[max(0,curr_coords[0] - 1)][curr_coords[1]] >= phero_value \
    and prev_action != 3:
        action = 1
        phero_value = pheromones[max(0,curr_coords[0] - 1)][curr_coords[1]]
    if pheromones[curr_coords[0]][min(world_size-1,curr_coords[1] + 1)] >= phero_value \
    and prev_action != 4:
        action = 2
        phero_value = pheromones[curr_coords[0]][min(world_size-1,curr_coords[1] + 1)]
    if pheromones[min(world_size-1,curr_coords[0] + 1)][curr_coords[1]] >= phero_value \
    and prev_action != 1:
        action = 3
        phero_value = pheromones[min(world_size-1,curr_coords[0] + 1)][curr_coords[1]]
    if pheromones[curr_coords[0]][max(0,curr_coords[1] - 1)] >= phero_value \
    and prev_action != 2:
        action = 4
        phero_value = pheromones[curr_coords[0]][max(0,curr_coords[1] - 1)]
    return action

# choose the action best aligned with unit vector [ur, uc]
def selectAction(ur,uc): 
    
    eps = 0.01
    if np.linalg.norm([ur,uc]) < eps:
        return 0

    return np.argmax([np.dot([ur,uc],[r,c]) for r,c in [(-1,0), (0,1), (1,0), (0,-1)]]) + 1

def get_upward_gradient_direction(obs, prev_action, pheromones, cache_vector, curr_coords=OBS_CENTER, world_size=128):

    phero_value = pheromones[curr_coords[0]][curr_coords[1]]
    if TROUBLESHOOTING:
        print('GET_UPWARD_GRADIENT')

    phero_values = np.array([
        pheromones[max(0,curr_coords[0] - 1)][curr_coords[1]],
        pheromones[curr_coords[0]][min(world_size-1,curr_coords[1] + 1)],
        pheromones[min(world_size-1,curr_coords[0] + 1)][curr_coords[1]],
        pheromones[curr_coords[0]][max(0,curr_coords[1] - 1)]])

    locations = {
        0: (OBS_CENTER[0] - 1, OBS_CENTER[1]),
        1: (OBS_CENTER[0], OBS_CENTER[1] + 1),    
        2: (OBS_CENTER[0] + 1, OBS_CENTER[1]),
        3: (OBS_CENTER[0], OBS_CENTER[1] - 1)
    }

    if TROUBLESHOOTING:
        print('PHERO VALUE: ' + str(phero_value))
        print('UPWARD GRADIENT DIRECTION OPTIONS:')
        print(phero_values)

    # climbing the gradient -> we'd prefer to get out of pheromones if we can
    if curr_coords == OBS_CENTER:
        if 0 in phero_values:
            zero_values = np.where(phero_values == 0)[0]
            for i in range(len(zero_values)):
                if obs[2][locations[zero_values[i]][0]][locations[zero_values[i]][1]] == 0:
                    if TROUBLESHOOTING:
                        print('FOUND ZERO: ' + str(np.where(phero_values == 0)[0][0] + 1))
                    return (zero_values[i] + 1)

    phero_indices = np.argsort(phero_values)

    possible_moves = []
    
    for i in range(len(phero_indices)):
        # and phero_values[phero_indices[i]] <= phero_value -> sometimes may need to go to next value that isn't strictly <=
        if phero_values[phero_indices[i]] > phero_value \
        and prev_action != ((phero_indices[i] + 2) % 4) + 1 \
        and phero_values[phero_indices[i]] != 1.05:
            possible_moves.append(phero_indices[i])

    possible_moves.reverse()
    if TROUBLESHOOTING:
        print('POSSIBLE MOVES: ')
        print(possible_moves)

    for move in possible_moves:
        if no_obstacle_at(obs, locations[move]):
            if TROUBLESHOOTING:
                print('DOWNWARD_TRAIL_DIRECTION DECISION: ' + str(move + 1))
            return move + 1

    # going to be invalid (blocked), but returning zero from this function would
    # indicate that we're done with this trail - better to no-op then confuse the agent
    # (and since no other move is feasible, we can't do anything anyway)
    if len(possible_moves) > 0:
        if TROUBLESHOOTING:
            print('DOWNWARD_TRAIL_DIRECTION DECISION (not real move): ' + str(possible_moves[0] + 1))
        return possible_moves[0] + 1
    else:
        if TROUBLESHOOTING:
            print('DOWNWARD_TRAIL_DIRECTION DECISION: 0 (no path moves left)')
        return 0
    return 0

def no_obstacle_at(obs, loc):
    return obs[0][loc[0]][loc[1]] == 0 \
    and obs[1][loc[0]][loc[1]] == 0 \
    and obs[2][loc[0]][loc[1]] == 0

def is_obstacle_at(obs, loc):
    return (obs[0][loc[0]][loc[1]] > 0 \
    or obs[1][loc[0]][loc[1]] > 0 \
    or obs[2][loc[0]][loc[1]] > 0)

def get_biggest_pheromone_direction(obs):
    phero_value = obs[7][OBS_CENTER[0]][OBS_CENTER[1]]
    action = 0

    if obs[7][OBS_CENTER[0] - 1][OBS_CENTER[1]] > phero_value:
        action = 1
        phero_value = obs[7][OBS_CENTER[0] - 1][OBS_CENTER[1]]
    if obs[7][OBS_CENTER[0]][OBS_CENTER[1] + 1] > phero_value:
        action = 2
        phero_value = obs[7][OBS_CENTER[0]][OBS_CENTER[1] + 1]
    if obs[7][OBS_CENTER[0] + 1][OBS_CENTER[1]] > phero_value:
        action = 3
        phero_value = obs[7][OBS_CENTER[0] + 1][OBS_CENTER[1]]
    if obs[7][OBS_CENTER[0]][OBS_CENTER[1] - 1] > phero_value:
        action = 4
        phero_value = obs[7][OBS_CENTER[0]][OBS_CENTER[1] - 1]
    return action

def get_cache_direction(obs):
    if obs[2][OBS_CENTER[0] - 1][OBS_CENTER[1]] == 1:
        return 1
    elif obs[2][OBS_CENTER[0]][OBS_CENTER[1] + 1] == 1:
        return 2
    elif obs[2][OBS_CENTER[0] + 1][OBS_CENTER[1]] == 1:
        return 3
    elif obs[2][OBS_CENTER[0]][OBS_CENTER[1] - 1] == 1:
        return 4
    pass

def build_trail_direction(obs, prev_action):
    phero_value = obs[7][OBS_CENTER[0]][OBS_CENTER[1]]
    action = 0
    if TROUBLESHOOTING:
        print('PHERO VALUE: ' + str(phero_value))
        print('BUILD TRAIL DIRECTION OPTIONS:')
        print(obs[7][OBS_CENTER[0] - 1][OBS_CENTER[1]])
        print(obs[7][OBS_CENTER[0]][OBS_CENTER[1] + 1])
        print(obs[7][OBS_CENTER[0] + 1][OBS_CENTER[1]])
        print(obs[7][OBS_CENTER[0]][OBS_CENTER[1] - 1])


    if obs[7][OBS_CENTER[0] - 1][OBS_CENTER[1]] <= phero_value \
    and obs[7][OBS_CENTER[0] - 1][OBS_CENTER[1]] > 0 \
    and obs[2][OBS_CENTER[0] - 1][OBS_CENTER[1]] == 0 \
    and prev_action != 3:
        action = 1
        phero_value = obs[7][OBS_CENTER[0] - 1][OBS_CENTER[1]]
    if obs[7][OBS_CENTER[0]][OBS_CENTER[1] + 1] <= phero_value \
    and obs[7][OBS_CENTER[0]][OBS_CENTER[1] + 1] > 0 \
    and obs[2][OBS_CENTER[0]][OBS_CENTER[1] + 1] == 0 \
    and prev_action != 4:
        action = 2
        phero_value = obs[7][OBS_CENTER[0]][OBS_CENTER[1] + 1]
    if obs[7][OBS_CENTER[0] + 1][OBS_CENTER[1]] <= phero_value \
    and obs[7][OBS_CENTER[0] + 1][OBS_CENTER[1]] > 0 \
    and obs[2][OBS_CENTER[0] + 1][OBS_CENTER[1]] == 0 \
    and prev_action != 1:
        action = 3
        phero_value = obs[7][OBS_CENTER[0] + 1][OBS_CENTER[1]]
    if obs[7][OBS_CENTER[0]][OBS_CENTER[1] - 1] <= phero_value \
    and obs[7][OBS_CENTER[0]][OBS_CENTER[1] - 1] > 0 \
    and obs[2][OBS_CENTER[0]][OBS_CENTER[1] - 1] == 0 \
    and prev_action != 2:
        action = 4
        phero_value = obs[7][OBS_CENTER[0]][OBS_CENTER[1] - 1]
    if TROUBLESHOOTING:
        print('BUILD_TRAIL_DIRECTION DECISION: ' + str(action))
        print('\n\n\n\n')
    return action

def diffuse_pheromones(obs, env, agent_env_pos, agentID):
    curr_coords = OBS_CENTER
    pheromones = obs[7]

    phero_values = np.array([
        pheromones[curr_coords[0] - 1][curr_coords[1]],
        pheromones[curr_coords[0]][curr_coords[1] + 1],
        pheromones[curr_coords[0] + 1][curr_coords[1]],
        pheromones[curr_coords[0]][curr_coords[1] - 1]])

    if np.nonzero(phero_values)[0].shape[0] > 1:
        return

    nonzero_index = np.argsort(phero_values)[-1]

    # index represents direction 2 or 4
    if (nonzero_index + 1) % 2 == 0:
        if no_obstacle_at(obs, (OBS_CENTER[0] - 1, OBS_CENTER[1])):
            drop_pheromones(env, (agent_env_pos[0] - 1, agent_env_pos[1]), agentID)
        if no_obstacle_at(obs, (OBS_CENTER[0] + 1, OBS_CENTER[1])):   
            drop_pheromones(env, (agent_env_pos[0] + 1, agent_env_pos[1]), agentID)
    else:
        if no_obstacle_at(obs, (OBS_CENTER[0], OBS_CENTER[1] - 1)):
            drop_pheromones(env, (agent_env_pos[0], agent_env_pos[1] - 1), agentID)
        if no_obstacle_at(obs, (OBS_CENTER[0], OBS_CENTER[1] + 1)):
            drop_pheromones(env, (agent_env_pos[0], agent_env_pos[1] + 1), agentID)

def get_resource_id(env, resource_entry_loc):
    if TROUBLESHOOTING:
        print('CURRENT_LOCATION: ' + str(resource_entry_loc))
    for i in range(4, len(env.RCs)):
        if TROUBLESHOOTING:
            print('RESOURCE # ' + str(i) + ': ' + str(env.RCs[i].queue.getQueueEntry()))
        if env.RCs[i].queue.getQueueEntry() == (resource_entry_loc[0], resource_entry_loc[1]):
            if TROUBLESHOOTING:
                print('RESOURCE ID: ' + str(i))
            return i
    return -1

def drop_pheromones(env, agent_position, agentID, amount=None):
    agent = env.agents[agentID - 1]
    if amount == None: 
        env.phero_stack_loaded[agent_position[0]][agent_position[1]][agentID - 1] = (env.phero_map_loaded_intensity * agent.pheroIntensity) / env.pheroTimeDecay
    else:
        env.phero_stack_loaded[agent_position[0]][agent_position[1]][agentID - 1] = amount
def move_right(prev_action):
    return (prev_action % 4) + 1

def move_left(prev_action):
    return ((prev_action - 2) % 4 ) + 1

def move_backward(prev_action):
    return ((prev_action + 1) % 4) + 1

def get_loc_by_move(move, curr_coords, world_size=128):
    locations = {
        0: (max(0, curr_coords[0] - 1), curr_coords[1]),
        1: (curr_coords[0], min(world_size-1, curr_coords[1] + 1)),
        2: (min(world_size-1, curr_coords[0] + 1), curr_coords[1]),
        3: (curr_coords[0], max(0, curr_coords[1] - 1))
    }
    return locations[move - 1]

def check_for_wipeout(env, curr_coords):
    move1 = get_loc_by_move(1, curr_coords) 
    move2 = get_loc_by_move(2, curr_coords)
    move3 = get_loc_by_move(3, curr_coords)
    move4 = get_loc_by_move(4, curr_coords)

    if (move1[0] not in range(env.shape[0]) or move1[1] not in range(env.shape[1]) or env.phero_map_loaded[move1[0], move1[1]] == 0) \
    and (move2[0] not in range(env.shape[0]) or move2[1] not in range(env.shape[1]) or env.phero_map_loaded[move2[0], move2[1]] == 0) \
    and (move3[0] not in range(env.shape[0]) or move3[1] not in range(env.shape[1]) or env.phero_map_loaded[move3[0], move3[1]] == 0) \
    and (move4[0] not in range(env.shape[0]) or move4[1] not in range(env.shape[1]) or env.phero_map_loaded[move4[0], move4[1]] == 0):
        if TROUBLESHOOTING:
            print('WIPEOUT')
        return True
    else:
        return False

def timeMS():
        return int(round(time.time() * 1000))

class CSAFController():
    def __init__(self, env, agent_placement):
        self.currentStatusList = [Actions.CHOOSE_NEXT_PATCH] * env.numAgents
        self.previousExecutedActionList = [np.nan] * env.numAgents
        self.currentResourceList = [-1] * env.numAgents
        for agent in env.agents:
            agent.pheroIntensity = 1.0
        env.phero_auto_update = False
        for rc in env.RCs:
            rc.queue.length = 1

        preset_pheromones = np.zeros(env.shapes)
        preset_pheromones[env.shapes[0] // 2 - 2][env.shapes[1] // 2 - 2 : env.shapes[1] // 2 + 2] = 1.05
        preset_pheromones[env.shapes[0] // 2 + 1][env.shapes[1] // 2 - 2 : env.shapes[1] // 2 + 2] = 1.05
        preset_pheromones[env.shapes[0] // 2 - 1][env.shapes[1] // 2 - 2] = 1.05
        preset_pheromones[env.shapes[0] // 2 - 0][env.shapes[1] // 2 - 2] = 1.05
        preset_pheromones[env.shapes[0] // 2 - 1][env.shapes[1] // 2 + 1] = 1.05
        preset_pheromones[env.shapes[0] // 2 - 0][env.shapes[1] // 2 + 1] = 1.05
        env.phero_highway = preset_pheromones

        if agent_placement in ["evenCorners", "xFormation"]:
            self.previousActionList = []
            for i in range(len(env.agents)):
                self.previousActionList.append(i % 4 + 1)
        elif agent_placement == "singleCorner":
            self.previousActionList = [1] * env.numAgents
        else:
            # must use one of these two initializations
            assert(0 == 1)


    def csaf_step(self, env, observation, agentID, curr_status, prev_action, prev_executed_action, curr_resource, world_size=128):
        # this policy only uses the vector observation values
        cache_vector = observation[0][1]
        observation = observation[0][0]
        agent = env.agents[agentID - 1]
        right_pos = get_coords_right(OBS_CENTER, prev_action)
        ahead_pos = get_coords_ahead(OBS_CENTER, prev_action)
        left_pos = get_coords_left(OBS_CENTER, prev_action)

        agent_pos_env = agent.getLocation()

        action = -1

        if not np.isnan(prev_executed_action) and check_for_wipeout(env, agent_pos_env):
            curr_status = Actions.WIPEOUT
        while action < 0:
            if curr_status == Actions.AT_HOME_1:
                if prev_executed_action == 0:
                    curr_status = Actions.DROP_FOOD
                else:
                    curr_status = Actions.AT_HOME_2
            elif curr_status == Actions.AT_HOME_2:
                if TROUBLESHOOTING:
                    print('AT_HOME')
                # move away from cache
                env.agents[agentID - 1].pheroIntensity = 1.0
                behind_pos = get_coords_behind(OBS_CENTER, prev_action)
                if observation[0][behind_pos[0]][behind_pos[1]] > 0 \
                or observation[1][behind_pos[0]][behind_pos[1]] > 0:
                    # going to no-op, then try again
                    action = get_cache_direction(observation)
                else: 
                    action = ((get_cache_direction(observation) + 1) % 4) + 1
                    curr_status = Actions.LOOK_FOR_TRAIL_1
            elif curr_status == Actions.WIPEOUT:
                if observation[7][ahead_pos[0]][ahead_pos[1]] > 0:
                    action = move_left(prev_action)
                    curr_status = Actions.CHOOSE_NEXT_PATCH
                elif observation[7][right_pos[0]][right_pos[1]] > 0:
                    curr_status = Actions.CHOOSE_NEXT_PATCH
                elif observation[7][left_pos[0]][left_pos[1]] > 0:
                    action = move_backward(prev_action)
                    curr_status = Actions.CHOOSE_NEXT_PATCH
                else:
                    if agent.numResources > 0:
#                        # let's find the closest nest entry
#                        dists_nest_entries = [ (env.RCs[i].queue.getQueueEntryNO()[0] - agent.row)**2 + \
#                                               (env.RCs[i].queue.getQueueEntryNO()[1] - agent.col)**2 \
#                                               for i in range(4) ] # nest entry points are the first 4 items in env.RCs
#                        # and its ID in the RCs list
#                        nne = np.argmin( dists_nest_entries )
#                        if dists_nest_entries[nne] <= 1: # close enough to enter?

                        # entry nests: [Left, Bottom, Top, Right]
                        if abs(cache_vector[0]) > abs(cache_vector[1]):
                            # Left or Right
                            nne = (0 if cache_vector[0] > 0 else 3)
                        else:
                            # Top or Bottom
                            nne = (2 if cache_vector[1] > 0 else 1)
                        nest_entry = env.RCs[nne].queue.getQueueEntry()
                        nest_entry_vector = [ (nest_entry[0] - agent.row)**2, (nest_entry[1] - agent.col)**2 ]
                        if nest_entry_vector[0]+nest_entry_vector[1] <= 1:
                            action = 5
                        else:
#                            nest_entry_vector = [ (env.RCs[nne].queue.getQueueEntryNO()[0] - agent.row), \
#                                                  (env.RCs[nne].queue.getQueueEntryNO()[1] - agent.col) ]
                            action = selectAction( nest_entry_vector[0], nest_entry_vector[1] )
                    else:
                        action = selectAction(cache_vector[0], cache_vector[1])
                # if ahead is pheromones, turn left
                # if right is pheromones, do nothing
                # if left is pheromones, go backward
                # otherwise, keep following cache vector
            elif curr_status == Actions.LOOK_FOR_TRAIL_1 or curr_status == Actions.LOOK_FOR_TRAIL_2 \
            or curr_status == Actions.LOOK_FOR_TRAIL_3 or curr_status == Actions.LOOK_FOR_TRAIL_4:
                if TROUBLESHOOTING:
                    print('LOOK_FOR_TRAIL')

                # if ahead is beginning of yellow phero trail

                if env.phero_map_unloaded[agent_pos_env[0]][agent_pos_env[1]] > 0:
                    if TROUBLESHOOTING:
                        print('FOUND YELLOW TRAIL\n')
                        print(np.where(env.phero_stack_unloaded[agent_pos_env[0], agent_pos_env[1], :] > 0))
                    action = get_upward_trail_direction(observation, prev_action, env.phero_map_unloaded, agent_pos_env)
                    # action = get_upward_gradient_direction(observation, prev_action, env.phero_map_unloaded, cache_vector, agent_pos_env)
                    curr_status = Actions.CLIMB

                #if ahead is queue entry point
                elif observation[4][ahead_pos[0]][ahead_pos[1]] > 0:
                    if TROUBLESHOOTING:
                        print('FOUND NEXT QUEUE ENTRANCE\n')
                        print(observation[4])
                    if curr_status == Actions.LOOK_FOR_TRAIL_1:
                        curr_status = Actions.LOOK_FOR_TRAIL_2
                        action = prev_action
                    elif curr_status == Actions.LOOK_FOR_TRAIL_2:
                        curr_status = Actions.LOOK_FOR_TRAIL_3
                        action = prev_action
                    elif curr_status == Actions.LOOK_FOR_TRAIL_3:
                        curr_status = Actions.LOOK_FOR_TRAIL_4
                        action = prev_action
                    else:
                        curr_status = Actions.LOOK_FOR_FOOD
                        # move out of inner ring
                        action = move_left(prev_action)
                else:
                    action = orbit_nest(observation, prev_action)
            elif curr_status == Actions.LOOK_FOR_FOOD:
                if TROUBLESHOOTING:
                    print('LOOK_FOR_FOOD')

                # if current square is resource queue entry point: 
                if observation[3][OBS_CENTER[0]][OBS_CENTER[1]] == 1:
                    drop_pheromones(env, agent_pos_env, agentID)
                    curr_status = Actions.PICK_FOOD
                    action = 5

                else:
                    curr_status = Actions.CHOOSE_NEXT_PATCH
            elif curr_status == Actions.CHOOSE_NEXT_PATCH:
                if TROUBLESHOOTING:
                    print('CHOOSE_NEXT_PATCH')
                # if obstacle detected, avoid_Obstacle()

                # if brown pheromones here and brown pheromones to 
                # the right (relative to direction last traveled in):
                pheromones_here = observation[7][OBS_CENTER[0]][OBS_CENTER[1]] > 0
                pheromones_right = pheromone_to_right(observation, prev_action) > 0
                if (pheromones_here and pheromones_right):
                    if TROUBLESHOOTING:
                        print('pheromones here and to right')
                    # Diffuse pheromones to left cell
                    # if it doesn't already have pheromones
                    if observation[7][left_pos[0]][left_pos[1]] == 0:
                        if TROUBLESHOOTING:
                            print('diffused to left cell')
                        left_pos_env = get_coords_left(agent_pos_env, prev_action)
                        env.phero_stack_loaded[left_pos_env[0]][left_pos_env[1]][agentID - 1] = np.sum(env.phero_stack_loaded[agent_pos_env[0]][left_pos_env[1]][:])

                    # move to food location using brown cells -> interpret as go toward least pheromone value
                    action = get_upward_gradient_direction(observation, prev_action, observation[7], cache_vector)

                # elif brown pheromones here and no brown pheromones in the right cell:
                elif pheromones_here:
                    if TROUBLESHOOTING:
                        print('pheromones here but not to right')
                    # remove brown trail
                    env.phero_stack_loaded[agent_pos_env[0], agent_pos_env[1], :] = 0
                    env.phero_map_loaded = np.sum(env.phero_stack_loaded, axis = 2)
                    observation = env.extractObservation(env.agents[agentID - 1])
                    action = get_upward_gradient_direction(observation, prev_action, observation[7], cache_vector)

                else:
                    if TROUBLESHOOTING:
                        print('no pheromones here')
                    
                    found_obstacle = False
                    for i in range(1, 6):
                        new_pos_env = get_coords_right(agent_pos_env, prev_action, num_spaces_over=i)
                        new_pos_obs = get_coords_right(OBS_CENTER, prev_action, num_spaces_over=i)
                        if (not found_obstacle) and get_resource_id(env, new_pos_env) != -1:
                            if TROUBLESHOOTING:
                                print('FOUND A RESOURCE TO RIGHT')
                            action = move_right(prev_action)
                            curr_status = Actions.STRAIGHT_TO_RESOURCE_1
                        elif observation[2][new_pos_obs[0]][new_pos_obs[1]] > 0:
                            found_obstacle = True
                    # found_obstacle = False
                    # if action < 0:
                    #     for i in range(1, 6):
                    #         new_pos_env = get_coords_left(agent_pos_env, prev_action, num_spaces_over=i)
                    #         new_pos_obs = get_coords_left(OBS_CENTER, prev_action, num_spaces_over=i)
                    #         if (not found_obstacle) and get_resource_id(env, new_pos_env) != -1:
                    #             if TROUBLESHOOTING:
                    #                 print('FOUND A RESOURCE TO LEFT')
                    #             action = move_left(prev_action)
                    #             curr_status = Actions.STRAIGHT_TO_RESOURCE_1
                    #             continue
                    #         elif observation[2][new_pos_obs[0]][new_pos_obs[1]] > 0:
                    #             found_obstacle = True
                            
                    # Lay limited amount of (pheromones)
                    # Detect_And_Adjust_Heading(pheromones)
                    if action < 0:
                        action = detect_and_adjust_heading(env, observation, prev_action, agentID, cache_vector)
                    if (observation[7][OBS_CENTER[0] - 1][OBS_CENTER[1]] > 0 or observation[2][OBS_CENTER[0] - 1][OBS_CENTER[1]] > 0) \
                    and (observation[7][OBS_CENTER[0]][OBS_CENTER[1] + 1] > 0 or observation[2][OBS_CENTER[0]][OBS_CENTER[1] + 1] > 0) \
                    and (observation[7][OBS_CENTER[0] + 1][OBS_CENTER[1]] > 0 or observation[2][OBS_CENTER[0] + 1][OBS_CENTER[1]] > 0) \
                    and (observation[7][OBS_CENTER[0]][OBS_CENTER[1] - 1] > 0 or observation[2][OBS_CENTER[0]][OBS_CENTER[1] - 1] > 0):
                        pheros = np.array([
                            observation[7][OBS_CENTER[0] - 1][OBS_CENTER[1]],
                            observation[7][OBS_CENTER[0]][OBS_CENTER[1] + 1],
                            observation[7][OBS_CENTER[0] + 1][OBS_CENTER[1]],
                            observation[7][OBS_CENTER[0]][OBS_CENTER[1] - 1]
                        ])
                        drop_pheromones(env, agent_pos_env, agentID, np.median(pheros))
                    else:
                        drop_pheromones(env, agent_pos_env, agentID)
                if curr_status != Actions.STRAIGHT_TO_RESOURCE_1:
                    curr_status = Actions.LOOK_FOR_FOOD
            elif curr_status == Actions.STRAIGHT_TO_RESOURCE_1:
                if observation[3][OBS_CENTER[0]][OBS_CENTER[1]] > 0:
                    if TROUBLESHOOTING:
                        print('exiting STR')
                    action = 5
                    curr_status = Actions.PICK_FOOD
                else:
                    if TROUBLESHOOTING:
                        print('in STR')
                    action = prev_action
                if observation[7][OBS_CENTER[0]][OBS_CENTER[1]] == 0:
                    drop_pheromones(env, agent_pos_env, agentID)
            elif curr_status == Actions.PICK_FOOD:
                if TROUBLESHOOTING:
                    print('PICK_FOOD') 

                if prev_executed_action == 0:
                    if observation[3][OBS_CENTER[0]][OBS_CENTER[1]] > 0:
                        action = 5
                    elif env.phero_map_unloaded[agent_pos_env[0]][agent_pos_env[1]] > 0:
                        curr_status = Actions.CLIMB
                    else:
                        curr_status = Actions.LOOK_FOR_FOOD
                else:
                    # Diffuse pheromones
                    diffuse_pheromones(observation, env, agent_pos_env, agentID)
                    drop_pheromones(env, agent_pos_env, agentID)
                    env.phero_map_loaded = np.sum(env.phero_stack_loaded, axis = 2)
                    observation = env.extractObservation(env.agents[agentID - 1])

                    if check_for_trail(env.phero_map_unloaded, agent_pos_env):
                        curr_status = Actions.RETURN_TO_NEST
                    else:
                        curr_status = Actions.RETURN_AND_COLOR
                        curr_resource = get_resource_id(env, agent_pos_env) - 4
                        if TROUBLESHOOTING:
                            print(observation[3])
                            print('CURRENT RESOURCE: ' + str(curr_resource))
            elif curr_status == Actions.REMOVE_TRAIL: # remove trail
                if TROUBLESHOOTING:
                    print('REMOVE_TRAIL')
                trail_values = np.array([
                    env.phero_stack_unloaded[max(0, agent_pos_env[0] - 1), agent_pos_env[1], curr_resource],
                    env.phero_stack_unloaded[agent_pos_env[0], min(world_size-1, agent_pos_env[1] + 1), curr_resource],
                    env.phero_stack_unloaded[min(world_size-1,agent_pos_env[0] + 1), agent_pos_env[1], curr_resource],
                    env.phero_stack_unloaded[agent_pos_env[0], max(0, agent_pos_env[1] - 1), curr_resource]])

                if TROUBLESHOOTING:
                    print('NUM VALUES IN TRAIL: ' + str(np.sum(trail_values > 0)))
                env.phero_stack_unloaded[agent_pos_env[0], agent_pos_env[1], curr_resource] = 0
                if np.sum(trail_values > 0) == 0:
                    curr_status = Actions.LOOK_FOR_FOOD
                else:
                    action = (np.where(trail_values > 0)[0][0] + 1)
            elif curr_status == Actions.CLIMB:
                if TROUBLESHOOTING:
                    print('CLIMB')
                a = get_upward_trail_direction(observation, prev_action, env.phero_map_unloaded, agent_pos_env)
                # a = get_upward_gradient_direction(observation, prev_action, env.phero_map_unloaded, cache_vector, agent_pos_env)
                if observation[3][OBS_CENTER[0]][OBS_CENTER[1]] == 1:
                    curr_status = Actions.PICK_FOOD
                    action = 5
                elif a == 0:
                    if TROUBLESHOOTING:
                        print(np.where(env.phero_stack_unloaded[agent_pos_env[0], agent_pos_env[1], :] > 0))
                    resource = np.where(env.phero_stack_unloaded[agent_pos_env[0], agent_pos_env[1], :] > 0)[0][0]
                    curr_status = Actions.REMOVE_TRAIL
                    curr_resource = resource
                else:
                    new_pos = get_loc_by_move(a, OBS_CENTER)
                    if observation[0][new_pos[0]][new_pos[1]] == 0:
                        action = a
                    else:
                        for m in range(1, 5):
                            if m != move_backward(prev_action) and m != a \
                            and no_obstacle_at(observation, get_loc_by_move(m, OBS_CENTER)):
                                action = m
                        if action < 0:
                            action = move_backward(prev_action)
            elif curr_status == Actions.RETURN_TO_NEST:
                if TROUBLESHOOTING:
                    print('RETURN_TO_NEST')

                a = get_downward_gradient_direction(observation, prev_action, observation[7], cache_vector)

                if observation[4][OBS_CENTER[0]][OBS_CENTER[1]] == 1:
                    action = 5
                    curr_status = Actions.AT_HOME_1
                elif a < 0:
                    prev_action = move_left(a * -1)
                    curr_status = Actions.DROP_FOOD
                else:
                    action = a
            elif curr_status == Actions.RETURN_AND_COLOR:
                if TROUBLESHOOTING:
                    print('RETURN_AND_COLOR')
                
                if TROUBLESHOOTING:
                    print('curr_resource: ' + str(curr_resource))
                env.phero_stack_unloaded[agent_pos_env[0], agent_pos_env[1], curr_resource] = np.sum(env.phero_stack_loaded[agent_pos_env[0], agent_pos_env[1], :])
                if TROUBLESHOOTING:
                    print(env.phero_stack_unloaded[agent_pos_env[0], agent_pos_env[1], :])
                a = get_downward_gradient_direction(observation, prev_action, observation[7], cache_vector)
                # a = build_trail_direction(observation, prev_action)

                #if standing on/near a cqep
                if observation[4][OBS_CENTER[0]][OBS_CENTER[1]] == 1:
                    action = 5
                    curr_status = Actions.AT_HOME_1
                    if TROUBLESHOOTING:
                        print('FOUND CACHE ENTRY IN RETURN AND COLOR')
                        print(observation[4])
                elif a < 0:
                    curr_status = Actions.DROP_FOOD
                    # turn to left
                    prev_action = move_left(a * -1)
                    if TROUBLESHOOTING:
                        print("DROP FOOD TAKING ACTION: " + str(action))
                else:
                    action = a 
            else: # curr_status == Actions.DROP_FOOD
                if TROUBLESHOOTING:
                    print('DROP_FOOD')
                if observation[4][OBS_CENTER[0]][OBS_CENTER[1]] == 1 \
                and observation[0][right_pos[0]][right_pos[1]] == 0 \
                and observation[1][right_pos[0]][right_pos[1]] == 0:
                    if TROUBLESHOOTING:
                        print('FOUND CACHE ENTRY IN DROP FOOD\n')
                    action = 5
                    curr_status = Actions.AT_HOME_1
                else: 
                    action = orbit_nest(observation, prev_action)


        return action, curr_status, curr_resource


    def step_all_agents(self, env):

        for agentID in range(1, len(env.agents) + 1):
            if not env.isAgentQueued(agentID):
                observation = env.observe(agentID, unloadedPheromones=True)
                currentStatus = self.currentStatusList[agentID - 1]
                previousAction = self.previousActionList[agentID - 1]
                previousExecutedAction = self.previousExecutedActionList[agentID - 1]
                currentResource = self.currentResourceList[agentID - 1]
                    
                action, currentStatus, currentResource = self.csaf_step(env, observation, agentID, currentStatus, previousAction, previousExecutedAction, currentResource)

                if TROUBLESHOOTING:
                    print('AGENT ' + str(agentID))
                    print('ACTION: ' + str(action))

                actualAction = env.step(agentID, action)

                # don't want no-op as previous action
                self.previousActionList[agentID - 1] = action
                self.currentStatusList[agentID - 1] = currentStatus
                self.currentResourceList[agentID - 1] = currentResource
                self.previousExecutedActionList[agentID - 1] = actualAction
                    
                if TROUBLESHOOTING:
                    print('ACTUAL: ' + str(actualAction))
                    print('\n\n\n\n\n')

                env.phero_map_loaded = np.sum(env.phero_stack_loaded, axis = 2)
                env.phero_map_unloaded = np.sum(env.phero_stack_unloaded, axis = 2)
        
        env.advanceQueues()
        if TROUBLESHOOTING:
            print(env.state[4:-4, 4:-4])
        # print(env.state[20:-20, 20:-20])
      
if __name__ == "__main__":
    
    env_shape = (64,64) 
    env_numAgents = 12
    env_freeAgentPlacement = "evenCorners"

    env = MAF_gym(shape=env_shape, numAgents=env_numAgents, no_agents_queued=0, pheroActionDecay=1, pheroTimeDecay=0.925, episodeNumber=20000, 
        freeAgentPlacement=env_freeAgentPlacement, freeAgentFull=0.0, pheroAutoUpdate=False)

    env.reappearingResources = True
    for rc in env.RCs:
        rc.infiniteFood = False
        rc.refreshRate = 0.0
        rc.miningRate = 0.1
        rc.dropOffRate = 1.0

    
    num_resources = len(env.RCs) - 4
    env.phero_stack_unloaded = np.zeros((env.shape[0], env.shape[1], num_resources))
    for a in env.agents:
        a.pheroIntensity = 1.0
    
    csaf = CSAFController(env, env_freeAgentPlacement)

    env.render(unloadedPheromones=True)

    # time.sleep(5)
    for i in range(512):
        csaf.step_all_agents(env)
        env.render()
        time.sleep(TIMESTEP)
        # sleepTime = np.max([TIMESTEP - ((timeMS() - startTime) / 1000.0), 0])
        #         time.sleep(sleepTime)

        # advance queues, decay pheromones
        env.advanceQueues()

    env.close()
