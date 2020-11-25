import gym
from gym import spaces
import math
import numpy as np

from colorsys import *

from MAF_gym.envs.draw import *
from MAF_gym.envs.entity import *
from MAF_gym.envs.setupState import *


from threading import Lock


RESOURCE_COLOR = (0, 0.620, 0.914) # blue
CACHE_COLOR = (0.396, 0.263, 0.129) # brown
OBST_COLOR = (.7, .7, .7)
AGENT_LOADED = (0, 0, 0)
AGENT_UNLOADED = (0.4, 0.4, 0.4)

OBSERVATION_SIZE = 11
OPPOSITE_ACTIONS = {1: 3, 2: 4, 3: 1, 4: 2, 5: 5}

from gym.envs.classic_control import rendering


class MAF_gym(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}


    def __init__(self, state0=None, numAgents=32, shape=(20,20), observationSize=11, pheroActionDecay=0.98, pheroTimeDecay=0.99, \
                 episodeNumber=0, no_agents_queued=32, freeAgentPlacement="nearCache", freeAgentFull=0.0, pheroAutoUpdate=True, obstacleRatio=[0.0, 0.0], tempResources=False, donutSpacing=False):
        """
        NOTES:
        """
        super(MAF_gym, self).__init__()
        
        self.observationSize = observationSize
        self.pheroActionDecay = pheroActionDecay
        self.pheroTimeDecay = pheroTimeDecay
        self.rewards = np.zeros((numAgents,))
        self.viewer = None
        self.state = None
        self.phero_map_loaded = None
        self.phero_map_unloaded = None
        self.phero_stack_loaded = None
        self.phero_stack_unloaded = None
        self.phero_highway = None
        self.phero_map_loaded_intensity = None
        self.phero_map_unloaded_intensity = None
        self.phero_highway_intensity = None
        self.phero_auto_update = pheroAutoUpdate
        self.numAgents = numAgents
        self.shapes = shape
        self.agents = []
        self.RCs = []
        self.no_agents_queued = no_agents_queued
        self.freeAgentPlacement = freeAgentPlacement
        self.freeAgentFull = freeAgentFull
        self.obstacleRatio = obstacleRatio # 2 values, (min,max) ratios
        self.resourceBuffer = None

        self.reappearingResources = tempResources
        self.donutSpacing = donutSpacing
    
        self.mutex = Lock()

        self.reset(episodeNumber, state0)


    def copyTestData(self, env):
        for key in env.__dict__:
            setattr(self, key, env.__dict__[key])


    # Collect observations for all agents
    #   should be used once everyone has moved and queues are updated
    #   observations should be a preallocated array of None
    def observe(self, agentID, unloadedPheromones=False):
        assert(agentID > 0)

        agent = self.agents[agentID-1]
        posVec, posNorm  = agent.getCacheVector(self.shape)
        scalarObs = np.array([posVec[0], posVec[1], posNorm])
        vectorObs = self.extractObservation(self.agents[agentID-1])

        if unloadedPheromones == False:
            # If you don't want unloaded pheromones, remove last matrix.
            # assertion is to help make sure last matrix is indeed unloaded pheromones
            assert(vectorObs.shape == (9,11,11))
            vectorObs = vectorObs[:-1]
        
        return ([vectorObs, scalarObs], self.rewards[agentID - 1], self.isAgentQueued(agentID))


    # Execute one time step within the environment
    def step(self, agentID, action):
        assert(agentID > 0)

        with self.mutex:
 
            actualAction = action

            if self.isAgentQueued(agentID):
                self.rewards[agentID - 1] = 0
            else:
                actualAction = self.executeAction(self.agents[agentID-1], action)

            return actualAction

    def reset(self, episodeNumber, world0=None):
        curriculumStart = 5000
        curriculumEnd =  10000
        minSide = (self.shapes[0] // 2) * 2
        maxSide = self.shapes[1]
        side = (np.random.randint(minSide, maxSide + 1) // 2) * 2

        self.resourceBuffer = 0
        if episodeNumber < curriculumStart:
            self.phero_map_loaded_intensity = 0.0
        else:
            self.phero_map_loaded_intensity =  min(1.0, (episodeNumber - curriculumStart) /
                                                           (curriculumEnd - curriculumStart))
            if self.donutSpacing == True:
                maxBuffer = (side // 2) - (side // 4)
                self.resourceBuffer = int(maxBuffer * min(1.0, (episodeNumber - curriculumStart) /
                                                          (curriculumEnd - curriculumStart)))
            
            
        self.phero_highway_intensity = 1.0 - self.phero_map_loaded_intensity
        self.phero_map_unloaded_intensity = 1.0
        
        # Initialize highways.
        self.phero_highway = None
        self.phero_stack_loaded = None
        self.phero_map_loaded = None
        while self.phero_highway is None:
            self.setWorld(episodeNumber, side, world0)
            if self.phero_highway_intensity == 0.0:
                self.phero_highway = np.zeros(self.state.shape)
            else:
                self.phero_highway = makeAllHighways(self.state, maxPheromone=self.phero_highway_intensity,
                                                 pheroActionDecay=self.pheroActionDecay, pheroTimeDecay=self.pheroTimeDecay, RCs=self.RCs)

        for rc in self.RCs:
            rc.dropOffRate = envSetup.TRAIN_dropOffRate.value
            rc.miningRate = envSetup.TRAIN_miningRate.value
            rc.numResources = rc.maxResources
            if self.reappearingResources == True:
                rc.infiniteFood = False
                rc.refreshRate = 0.0
            
        if self.viewer is not None:
            self.viewer = None


    def calculateScore(self):
        totalScore = 0
        for rc in self.RCs:
            if rc.entityType == entity.CACHE.value:
                totalScore += rc.numResources
        return totalScore


    def render(self, rcsList=[], mode='human', close=False, screenWidth=1000, screenHeight=1000, unloadedPheromones=False):

    # Render the environment to the screen
        if self.viewer == None:
            self.viewer = rendering.Viewer(screenWidth, screenHeight)


            
        rows = self.shape[0]
        cols = self.shape[1]

        rcsList = []
        cacheQueuedAgents = []
        resourceQueuedAgents = []
        for rc in self.RCs:
            if rc.entityType == entity.RESOURCE.value:
                resourceQueuedAgents.extend([agent.ID for agent in rc.queue.agents])
            else:
                cacheQueuedAgents.extend([agent.ID for agent in rc.queue.agents])

        entryPointMatrix = np.zeros(self.state.shape)
        for rc in self.RCs:
            entryRow, entryCol = rc.queue.getQueueEntry()
            if entryRow is -2 and (entryCol is -2): #queue is full
                continue
            entryPointMatrix[entryRow][entryCol] = rc.entityType
        entryPointMatrix = np.rot90(entryPointMatrix, k=3)
        world = np.rot90(self.state, k=3)
        phero_loaded = np.rot90(self.phero_map_loaded + self.phero_highway, k=3)
        phero_unloaded = np.rot90(self.phero_map_unloaded, k=3)

        size = screenWidth/max(rows, cols)
        for i in range(cols):
            for j in range(rows):
                if unloadedPheromones == True:
                    sat_scale = 6
                    hue = 0.333333 * phero_loaded[i][j]/(phero_loaded[i][j] + phero_unloaded[i][j])
                    sat = np.min([1.0, (phero_loaded[i][j] + phero_unloaded[i][j])])
                else:
                    sat_scale = 12
                    hue = 0.333333
                    sat = np.min([1.0, phero_loaded[i][j]])
                phero_rgb = hsv_to_rgb(hue, sat, 1.0)

                if world[i,j] == -1:
                    self.viewer.add_onetime(create_rectangle(i * size, j * size,
                                                             size, size, OBST_COLOR, False))
                elif world[i,j] == -2:
                    r = self.findRC(row=(self.shape[0] - 1) - j, col=i)
                    fill = (r.numResources / r.maxResources) * size
                    self.viewer.add_onetime(create_rectangle(i * size, j * size,
                                                             size, fill, RESOURCE_COLOR, False))
                    self.viewer.add_onetime(create_rectangle(i * size, j * size,
                                                             size, size, RESOURCE_COLOR, True))
                elif world[i,j] == -3:
                    self.viewer.add_onetime(create_rectangle(i * size, j * size,
                                                             size, size, CACHE_COLOR, False))
                elif world[i,j] > 0:
                    agent = self.agents[world[i,j] - 1]
                    if world[i,j] in resourceQueuedAgents:
                        color = RESOURCE_COLOR
                    elif world[i,j] in cacheQueuedAgents:
                        color = CACHE_COLOR
                    else:

                        if agent.numResources == agent.maxResources:
                            color = AGENT_LOADED
                        else:
                            color = AGENT_UNLOADED

                    thickness = .75
                    self.viewer.add_onetime(create_rectangle(i * size, j * size,
                                                             size, size, phero_rgb, False))
                    self.viewer.add_onetime(create_circle(i * size, j * size,
                                                          size, size, AGENT_LOADED))
                    self.viewer.add_onetime(create_circle(i * size, j * size,
                                                          size*thickness, size, (1,1,1)))

                    self.viewer.add_onetime(create_semifilled_circle(i*size, j*size,
                                                                     size*(thickness + .03), size, RESOURCE_COLOR,
                                                                     AGENT_UNLOADED,
                                                                     agent.numResources /
                                                                     agent.maxResources))
     
                    
                
                else: # empty space, let's show the pheromones level instead
                    self.viewer.add_onetime(create_rectangle(i * size, j * size,
                                                             size, size, phero_rgb, False))

                # Entry points are not stored on state, render separately
                if entryPointMatrix[i,j] == entity.RESOURCE.value:
                    self.viewer.add_onetime(create_rectangle(i * size, j * size,
                                                                 size, size, RESOURCE_COLOR, True))
                elif entryPointMatrix[i,j] == entity.CACHE.value:
                    self.viewer.add_onetime(create_rectangle(i * size, j * size,
                                                                 size, size, CACHE_COLOR, True))

                    
        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer is not None:
            self.viewer.close()


    def setWorld(self, episodeNumber, side=None, state0=None, pheromones0=None):
        queueLength = 5
        if state0 is None:
            if side is None:
                print("if not loading state need to give min max side")
                assert(1 == 0)
 
            self.shape = (side, side)
            self.state = np.zeros(self.shape)
            mean_resources = int(side * side * 10 / (48 * 48))
            minResources = max(10, 0.5 * mean_resources)
            maxResources = max(10, 1.5 * mean_resources)
            numResources = np.random.randint(minResources, maxResources+1)
            numCaches = 4 # 4 central ones

            # 1. add caches and resources, and spawn queued up agents
            self.state, self.RCs = addRCs(queueLength, numResources, numCaches, self.shape, episodeNumber, resourceBuffer=self.resourceBuffer)

            obstRatio = self.obstacleRatio[0] + np.random.rand() * ( self.obstacleRatio[1] - self.obstacleRatio[0] )
            self.state = addObstacles(self.state, obstRatio)
            self.state = clearReservedSpace(self.state)
            self.agents = []


            # All agents start in the middle now
            numSpots = numCaches * queueLength
            if self.no_agents_queued > 0:
                resourceCacheRatio = 0. # all agents at nest initially
                

                # We have a limited number of cache queue spots. Make sure we don't overfill them
                self.no_agents_queued = min((self.no_agents_queued * (1 - resourceCacheRatio),
                                             4 * self.RCs[0].queue.length)

                self.state, self.RCs, self.agents = spawnAgentsAtRCs(self.state, self.RCs, self.no_agents_queued,
                                                                      resourceCacheRatio=resourceCacheRatio)

            # 2. Spawn the non-queued-up agents
            self.state, self.agents = addAgents(self.state, self.agents, self.numAgents, self.shape, self.freeAgentPlacement, self.freeAgentFull, self.no_agents_queued+1)

        else:
            self.agents, self.RCs = parseState(state0)

            self.numAgents = len(self.agents)
            self.shape = (state0.shape[0], state0.shape[1])
            self.state = state0
            self.state[self.state == -4] = 0

        if pheromones0 is None:
            self.phero_map_loaded = np.zeros(self.state.shape)
            self.phero_map_unloaded = np.zeros(self.state.shape)
        else:
            self.phero_map_loaded = pheromones0
            self.phero_map_unloaded = pheromones0

        self.phero_stack_loaded = np.zeros((self.shape[0],self.shape[1],self.numAgents))
        self.phero_stack_unloaded = np.zeros((self.shape[0],self.shape[1],self.numAgents))

    def extractObservation(self, agent):
        offsets = [(1,0),(0,1),(-1,0),(0,-1)] # unused for now

        # num of rows and columns to shift values
        transform_row = self.observationSize // 2 - agent.row
        transform_col = self.observationSize // 2 - agent.col

        min_row = max((agent.row - self.observationSize // 2), 0)
        max_row = min((agent.row + self.observationSize // 2 + 1), self.shape[0])
        min_col = max((agent.col - self.observationSize // 2), 0)
        max_col = min((agent.col + self.observationSize // 2 + 1), self.shape[1])

        observation = np.full((self.observationSize, self.observationSize), -1)
        phero_map_loaded   = np.zeros((self.observationSize, self.observationSize))
        phero_map_unloaded   = np.zeros((self.observationSize, self.observationSize))
        phero_highway = np.zeros((self.observationSize, self.observationSize))
        for row in range(min_row, max_row):
            observation[(min_row + transform_row):(max_row + transform_row), (min_col + transform_col):(max_col + transform_col)] = self.state[min_row:max_row, min_col:max_col]
            phero_map_loaded[(min_row + transform_row):(max_row + transform_row), (min_col + transform_col):(max_col + transform_col)] = self.phero_map_loaded[min_row:max_row, min_col:max_col]
            phero_map_unloaded[(min_row + transform_row):(max_row + transform_row), (min_col + transform_col):(max_col + transform_col)] = self.phero_map_unloaded[min_row:max_row, min_col:max_col]
            phero_highway[(min_row + transform_row):(max_row + transform_row), (min_col + transform_col):(max_col + transform_col)] = self.phero_highway[min_row:max_row, min_col:max_col]

            
            
        observation_layers = np.zeros((9, self.observationSize, self.observationSize))

        # resource queue entry points, agents
        rqep = np.zeros((self.observationSize, self.observationSize))
        rq_agents = np.zeros((self.observationSize, self.observationSize))
        # cache queue entry points, agents
        cqep = np.zeros((self.observationSize, self.observationSize))
        cq_agents = np.zeros((self.observationSize, self.observationSize))
        for rc in self.RCs: 
            if rc.entityType == entity.RESOURCE.value:
                if rc.numResources == 0.0:
                    continue
                entry_points = rqep
                queue_agents = rq_agents
                num_food = (rc.numResources - len(rc.queue.agents)) / rc.maxResources
            else:
                entry_points = cqep
                queue_agents = cq_agents
                num_food = 1

            qe = rc.queue.getQueueEntry()

            if qe[0] in range(min_row, max_row) and qe[1] in range(min_col, max_col):
                entry_points[qe[0] + transform_row][qe[1] + transform_col] = num_food # add entry point
                for dr, dc in offsets:
                    # compute where the join action might be available
                    j_r = qe[0] + dr
                    j_c = qe[1] + dc

                    if j_r in range(min_row, max_row) and j_c in range(min_col, max_col):
                        # if spot is free, add it
                        if self.state[j_r, j_c] == 0:
                            entry_points[j_r + transform_row][j_c + transform_col] = 1

                        # if spot it taken, but not by a queued agent, add it
                        elif self.state[j_r, j_c] > 0:
                            agentID = self.state[j_r, j_c]
                            if not self.isAgentQueued(agentID):
                                entry_points[j_r + transform_row][j_c + transform_col] = 1

                for a in rc.queue.agents:
                    if a.row in range(min_row, max_row) and a.col in range(min_col, max_col):
                        queue_agents[a.row + transform_row][a.col + transform_col] = 1

        # EMPTY agents
        observation_layers[0] = observation > 0
        # FULL agents
        observation_layers[1] = observation > 0

        for i in range(self.observationSize):
            for j in range(self.observationSize):
                if observation[i][j] > 0:
                    if self.agents[ observation[i][j] - 1 ].numResources > 0:
                        # agent full, remove it from empty agents layer
                        observation_layers[0][i][j] = 0
                    else:
                        # agent empty, remove it from full agents layer
                        observation_layers[1][i][j] = 0

        # obstacles
        observation_layers[2] = np.logical_or( observation == -1, np.logical_or( observation == -2, observation == -3 ) )
        # resource queue entry points
        observation_layers[3] = rqep
        # cache queue entry points
        observation_layers[4] = cqep
        # agents queued for resources for which entry point is in observation
        observation_layers[5] = rq_agents
        # agents queued for caches for which entry point is in observation
        observation_layers[6] = cq_agents
        # pheromones for loaded agents within FoV
        observation_layers[7] = phero_map_loaded + phero_highway

        if not self.phero_auto_update:
            # Only return the first 9 channels
            return observation_layers[:8]
        else:
            # pheromones for unloaded (not full) agents within FoV
            observation_layers[8] = phero_map_unloaded

            return observation_layers

        
    def listNextValidActions(self, agent_id, prev_action=0):
        """
        No movement: 0
        North (-1,0): 1
        East (0,1): 2
        South (1,0): 3
        West (0,-1): 4
        Join queue: 5
        """
        available_actions = []

        # Get current agent
        agent = self.agents[agent_id-1]

        MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)] # off by one index

        for i in range(4):
            out_of_bounds = agent.row+MOVES[i][0]>=self.state.shape[0] or agent.row+MOVES[i][0]<0 or agent.col+MOVES[i][1]>=self.state.shape[1] or agent.col+MOVES[i][1]<0

            if (not out_of_bounds) and self.state[agent.row+MOVES[i][0], agent.col+MOVES[i][1]] == 0 \
                and (self.agents[agent_id-1].numResources > 0 or not(prev_action == OPPOSITE_ACTIONS[i + 1])):

                    available_actions.append(i + 1)

        if self.agents[agent_id-1].numResources > 0:
            posVec, _  = agent.getCacheVector(self.shape)
            movePriorities = np.argsort([np.dot(posVec,[r,c]) for r,c in MOVES]) # ranked worst to best
            for action in movePriorities:
                if len(available_actions) <= 2:
                    break
                elif (action+1) in available_actions:
                    available_actions.remove(action + 1)

        if self.locateNearbyQueue(agent.row, agent.col) is not None and prev_action != OPPOSITE_ACTIONS[5]:
            available_actions.append(5)

        return available_actions


    def isAgentQueued(self, agentID):
        return self.agents[agentID-1].queueID != None   


    def locateNearbyQueue(self, row, col):
        pairs = [[row+1, col], [row, col+1],
                 [row-1, col], [row, col-1],
                 [row, col]]
        for rc in self.RCs:
            for i, j in pairs:
                if rc.queue.getQueueEntry() == (i, j):
                    if rc.entityType == entity.CACHE.value:
                        return rc
                    elif  (rc.numResources - len(rc.queue.agents)) > 0:
                        return rc

        return None


    def executeAction(self, agent, action):
        """
        No movement: 0
        North (-1,0): 1
        East (0,1): 2
        South (1,0): 3
        West (0,-1): 4
        Join queue: 5
        """
        origLoc = agent.getLocation()
        
        #Move N,E,S,W
        if (action >= 1) and (action <= 4):
            agent.move(action)
            row, col = agent.getLocation()

            # If the move is not valid, roll it back
            if ((row < 0) or (col < 0) or (row >= self.shape[0]) or (col >= self.shape[1]) or
                (self.state[row][col] != 0)):
                agent.reverseMove(action)

                self.rewards[agent.ID - 1] = reward.COLLISION.value
                action = 0

            # move is valid
            else:

                newLoc = agent.getLocation()
                self.state[origLoc[0]][origLoc[1]] = 0
                self.state[newLoc[0]][newLoc[1]] = agent.ID #FLAG
                self.rewards[agent.ID - 1] = reward.MOVE.value
                
        elif action == 5:
            row, col = agent.getLocation()
            rc = self.locateNearbyQueue(row, col)
            if rc is None:

                self.rewards[agent.ID - 1] = reward.CANNOTFINDRC.value
                action = 0

            # There is a queue nearby
            else:

                entryRow, entryCol = rc.queue.getQueueEntry()
                if self.state[entryRow][entryCol] > 0 and (self.state[entryRow][entryCol] != agent.ID):
                    self.rewards[agent.ID - 1] = reward.COLLISION.value
                    action = 0

                # Joined the correct type of queue
                elif (((rc.entityType == entity.RESOURCE.value) and (agent.numResources < agent.maxResources)) or
                    ((rc.entityType == entity.CACHE.value) and (agent.numResources > 0))):

                    self.rewards[agent.ID - 1] = reward.ENTEREDGOODRC.value
                    agent.setLocation(entryRow, entryCol)
                    rc.queue.addAgent(agent)
                    newLoc = agent.getLocation()
                    self.state[origLoc[0]][origLoc[1]] = 0
                    self.state[newLoc[0]][newLoc[1]] = agent.ID
                    action = 5
            
                else:
                    self.rewards[agent.ID - 1] = reward.INCORRECTRC.value
                    action = 0
        
        elif action == 0:
            self.rewards[agent.ID - 1] = reward.NOMOVE.value

        else:
            print("INVALID ACTION: {}".format(action))
            sys.exit()

        # Extra penalty for not adhering to strict social distancing
        obs = self.extractObservation(agent)
        if agent.queueID == None and agent.numResources < agent.maxResources and np.sum(obs[4]) == 0:
            self.rewards[agent.ID - 1] += reward.PER_AGENT.value * (np.sum(obs[0]) - np.sum(obs[6]))

        moveList = [None, [-1,0], [0,1], [1,0], [0,-1]]

        if self.phero_auto_update:
            if agent.hasFood():
                self.phero_stack_loaded[origLoc[0]][origLoc[1]][agent.ID - 1] = \
                    (self.phero_map_loaded_intensity * agent.pheroIntensity) / self.pheroTimeDecay
                agent.pheroIntensity *= self.pheroActionDecay
            else:
                self.phero_stack_unloaded[origLoc[0]][origLoc[1]][agent.ID-1] = self.phero_map_unloaded_intensity



        return action


    def advanceQueues(self):
        # Update the pheromones for everyone, very easy
        # Step 1: decay phermones
        self.phero_stack_loaded *= self.pheroTimeDecay
        self.phero_stack_unloaded *= self.pheroTimeDecay

        # Step 2: sum the stack on the third axis to make a phermone map
        self.phero_map_loaded = np.max(self.phero_stack_loaded, axis = 2)
        self.phero_map_unloaded = np.max(self.phero_stack_unloaded, axis = 2)

        depletedRCs = []
        exitedQueue = []

        for k, rc in enumerate(self.RCs):
            if len(rc.queue.agents) == 0:
                continue

            firstAgentLoc = rc.queue.agents[0].getLocation()
            firstSpot = rc.queue.queueSpots[0]

            #If the first agent is in the first spot, do stuff "normally"
            if firstAgentLoc[0] == firstSpot[0] and (firstAgentLoc[1] == firstSpot[1]):
                # give QUEUEDSTEP rewards to everyone in the queue (might be changed for front agent)
                for agent in rc.queue.agents:
                    self.rewards[agent.ID - 1] = reward.QUEUEDSTEP.value

                # Check that the agent is actually in the correct queue
                if rc.entityType == entity.RESOURCE.value:
                    completedTask = rc.mineResource()
                elif rc.entityType == entity.CACHE.value:
                    completedTask = rc.dropOffResource()

                # completedTask, 0 = not complete, 1 = complete, 2 = went into wrong type of queue
                #                3 = correct queue, but it was completely exhausted
                if completedTask == 1:
                    freeAgent = rc.queue.freeFirstAgent()
                    exitedQueue.append(freeAgent.ID)

                    if freeAgent.numResources > 0: #completed mining
                        self.rewards[freeAgent.ID - 1] = \
                            (freeAgent.numResources / freeAgent.maxResources) \
                            * reward.LEFTGOODRES.value \
                            * np.exp(-0.0115 * (np.abs(rc.row - self.shape[0]/2) + \
                                              np.abs(rc.col - self.shape[0]/2)))


                    else: #completed dropping off at cache
                        self.rewards[freeAgent.ID - 1] = reward.LEFTGOODCACHE.value

                elif completedTask == 2:
                    freeAgent = rc.queue.freeFirstAgent()
                    exitedQueue.append(freeAgent.ID)
                    self.rewards[freeAgent.ID - 1] = reward.INCORRECTRC.value
                elif completedTask == 3:
                    freeAgent = rc.queue.freeFirstAgent()
                    exitedQueue.append(freeAgent.ID)
                    self.rewards[freeAgent.ID - 1] = reward.EMPTYRC.value
                    assert(0 == 1)

                if len(rc.queue.agents) == 0:
                    rc.queue.freeFront = False

            #If they aren't, see if you can advance, and if so do it.
            else:
                row, col = rc.queue.queueSpots[0]
                if self.state[row][col] == 0:
                    for agent in rc.queue.agents:
                        row, col = agent.getLocation()
                        self.state[row][col] = 0

                    rc.queue.advance()

                    for agent in rc.queue.agents:
                        row, col = agent.getLocation()
                        self.state[row][col] = agent.ID
                        self.rewards[agent.ID - 1] = reward.QUEUEDSTEP.value

            if rc.numResources == 0:
                if self.reappearingResources == True:
                    self.state[rc.row, rc.col] = 0
                    for agent in rc.queue.agents:
                        exitedQueue.append(agent.ID)
                        agent.queueID = None
                        print("hello")
                    self.RCs.pop(k)
                    self.state, newRc = addSingleRC(self.state, length=4, resourceBuffer=self.resourceBuffer)
                    newRc.refreshRate = 0
                    newRc.infiniteFood = False
                    self.RCs.insert(k, newRc)
                    depletedRCs.append(k)


        if np.sum(self.state > 0) != len(self.agents):
            for a in self.agents:
                if self.state[a.row][a.col] != a.ID:
                    print('Fixing position of agent {}'.format(a.ID))
                    assert( self.state[a.row][a.col] == 0 )
                    self.state[a.row][a.col] = a.ID
            assert(np.sum(self.state > 0) == len(self.agents))

        return exitedQueue, depletedRCs


    def findRC(self, row, col):
        for rc in self.RCs:
            if (rc.row == row) and (rc.col == col):
                return rc


class CsafEnv(MAF_gym):
    def __init__(self, origEnv):
        super().__init__()
        self.copyTestData(origEnv)
        self.phero_stack_unloaded = np.zeros((origEnv.shape[0], origEnv.shape[1], len(origEnv.RCs)))

        
    def executeAction(self, agent, action):
        """
        No movement: 0
        North (-1,0): 1
        East (0,1): 2
        South (1,0): 3
        West (0,-1): 4
        Join queue: 5
        """
        origLoc = agent.getLocation()
        
        #Move N,E,S,W
        if (action >= 1) and (action <= 4):
            agent.move(action)
            row, col = agent.getLocation()

            # If the move is not valid, roll it back
            if ((row < 0) or (col < 0) or (row >= self.shape[0]) or (col >= self.shape[1]) or
                (self.state[row][col] != 0)):
                agent.reverseMove(action)

                self.rewards[agent.ID - 1] = reward.COLLISION.value
                action = 0

            # move is valid
            else:
                newLoc = agent.getLocation()
                self.state[origLoc[0]][origLoc[1]] = 0
                self.state[newLoc[0]][newLoc[1]] = agent.ID #FLAG
                self.rewards[agent.ID - 1] = reward.MOVE.value

                # action already set to correct move

        elif action == 5:
            row, col = agent.getLocation()
            rc = self.locateNearbyQueue(row, col)
            if rc is None:

                self.rewards[agent.ID - 1] = reward.CANNOTFINDRC.value
                action = 0

            # There is a queue nearby
            else:
                entryRow, entryCol = rc.queue.getQueueEntry()
                
                if self.state[entryRow][entryCol] > 0 and (self.state[entryRow][entryCol] != agent.ID):
                    self.rewards[agent.ID - 1] = reward.COLLISION.value
                    action = 0

                # Joined the correct type of queue
                elif (((rc.entityType == entity.RESOURCE.value) and (agent.numResources < agent.maxResources)) or
                    ((rc.entityType == entity.CACHE.value) and (agent.numResources > 0))):

                    self.rewards[agent.ID - 1] = reward.ENTEREDGOODRC.value
                    agent.setLocation(entryRow, entryCol)
                    rc.queue.addAgent(agent)
                    newLoc = agent.getLocation()
                    self.state[origLoc[0]][origLoc[1]] = 0
                    self.state[newLoc[0]][newLoc[1]] = agent.ID

                    action = 5
            
                else:
                    self.rewards[agent.ID - 1] = reward.INCORRECTRC.value

                    action = 0

        elif action == 0:
            self.rewards[agent.ID - 1] = reward.NOMOVE.value

        else:
            print("INVALID ACTION: {}".format(action))
            sys.exit()

        obs = self.extractObservation(agent)
        if agent.queueID == None and agent.numResources < agent.maxResources and np.sum(obs[4]) == 0:
            min_row = max((agent.row - self.observationSize // 2), 0)
            max_row = max((agent.row + self.observationSize // 2 + 1), self.shape[0])
            min_col = max((agent.col - self.observationSize // 2), 0)
            max_col = max((agent.col + self.observationSize // 2 + 1), self.shape[1])

            
            self.rewards[agent.ID - 1] += reward.PER_AGENT.value * (np.sum(obs[0]) - np.sum(obs[6]))
            min_row                     = max((agent.row - self.observationSize // 2), 0)
            max_row                     = min((agent.row + self.observationSize // 2 + 1), self.shape[0])
            min_col                     = max((agent.col - self.observationSize // 2), 0)
            max_col                     = min((agent.col + self.observationSize // 2 + 1), self.shape[1])

        moveList = [None, [-1,0], [0,1], [1,0], [0,-1]]
        if self.phero_auto_update: 
            if agent.hasFood():
                self.phero_stack_loaded[origLoc[0]][origLoc[1]][agent.ID - 1] = \
                        (self.phero_map_loaded_intensity * agent.pheroIntensity) / self.pheroTimeDecay

                agent.pheroIntensity *= self.pheroActionDecay
            else:
                self.phero_stack_unloaded[origLoc[0]][origLoc[1]][agent.ID-1] = self.phero_map_unloaded_intensity

        return action
