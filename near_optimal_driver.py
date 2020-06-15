import gym
import time
import math
import numpy as np
from operator import itemgetter
from collections import deque

TIMESTEP = .1

from MAF_gym.envs.entity import *
from MAF_gym.envs.MAF_env import MAF_gym
from astar2d import *

"""
No movement: 0
North (-1,0): 1
East (0,1): 2
South (1,0): 3
West (0,-1): 4
Join queue: 5
"""

moves = [(-1,0), (0,1), (1,0), (0,-1)]

# choose the action best aligned with vector [ur, uc]
def selectAction(ur,uc): 
    
    eps = 0.01
    if np.linalg.norm([ur,uc]) < eps:
        return 0

    return np.argmax([np.dot([ur,uc],[r,c]) for r,c in moves]) + 1


def actionPreference(ur,uc):
    return np.flip(np.argsort([np.dot([ur,uc],[r,c]) for r,c in moves])) + 1

def unitVec(r, c, norm_thresh = 0.01):
    
    ur = 0
    uc = 0
    norm = np.linalg.norm([r, c])

    if norm > norm_thresh:
        ur = r / norm_thresh
        uc = c / norm_thresh

    return ur, uc

def timeMS():
    return int(round(time.time() * 1000))


def findClosestAgent(agents, agentIDs, targetRow, targetCol):

    minManDist = np.inf
    closestAgentID = 0

    for agentID in agentIDs:
        agent = agents[agentID - 1]

        manDist = np.abs(targetRow - agent.row) + np.abs(targetCol - agent.col)

        if manDist < minManDist:
            minManDist = manDist
            closestAgentID = agentID

    return closestAgentID


class NOController():
    def __init__(self, env):
        self.unassignedEmpty = set([i + 1 for i in range(len(env.agents))])
        self.unassignedFull  = set()
        self.inboundAgents   = [deque() for i in range(len(env.RCs))]
        self.agentTargets    = [[0,0,0] for i in range(len(env.agents))]
        self.prevCapacity    = [agent.numResources for agent in env.agents]
        self.prevQueued      = [(agent.queueID is not None) for agent in env.agents]
        self.exitedQueue     = [] 
        self.depletedRCIDs   = []
        self.hasRandomTarget = [False] * len(env.agents)

    def rankRCs(self, env):
        rcs = env.RCs
        size = env.shapes[0]

        resourceRankings = []
        cacheRankings = []

        for i in range(len(rcs)):
            rc = rcs[i]

            row, col = rc.queue.getQueueEntryNO()

            if rc.entityType == entity.RESOURCE.value:

                distPenalty = np.abs(row - size/2) + np.abs(col - size/2)

                closestAgentID = findClosestAgent(env.agents, self.unassignedEmpty, row, col)
                agent = env.agents[closestAgentID - 1]
                agentDist = np.abs(agent.row - row) + np.abs(agent.col - col)

                nQueued = len(rc.queue.agents)
                processTime = 0
                resourcePenalty = 0
                if nQueued > 0:
                    processTime = (nQueued - rc.queue.agents[0].numResources)/rc.miningRate
                    if (nQueued - rc.queue.agents[0].numResources) + len(self.inboundAgents[i]) > rc.numResources:
                        resourcePenalty = np.inf
            
                for inboundAgentID in self.inboundAgents[i]:
                    inboundAgent = env.agents[inboundAgentID - 1]
                    inboundAgentDist = np.abs(inboundAgent.row - row) + np.abs(inboundAgent.col - col)
                    processTime = np.max([inboundAgentDist, processTime]) + 1/rc.miningRate


                queuePenalty = np.max([processTime - agentDist, 0])

                penalty = distPenalty + queuePenalty + resourcePenalty

                resourceRankings.append((penalty, i, (row, col)))
            else:
                closestAgentID = findClosestAgent(env.agents, self.unassignedFull, row, col)
                agent = env.agents[closestAgentID - 1]
                agentDist = np.abs(agent.row - row) + np.abs(agent.col - col)

                nQueued = len(rc.queue.agents)
                processTime = 0
                if nQueued > 0:
                    processTime = (nQueued - rc.queue.agents[0].numResources)/rc.dropOffRate
            
                for inboundAgentID in self.inboundAgents[i]:
                    inboundAgent = env.agents[inboundAgentID - 1]
                    inboundAgentDist = np.abs(inboundAgent.row - row) + np.abs(inboundAgent.col - col)
                    processTime = np.max([inboundAgentDist, processTime]) + 1/rc.dropOffRate


                queuePenalty = np.max([processTime - agentDist, 0])

                cacheRankings.append((queuePenalty, i, (row, col)))

        resourceRankings = sorted(resourceRankings,key=itemgetter(0))
        cacheRankings = sorted(cacheRankings,key=itemgetter(0))

        return resourceRankings, cacheRankings


    def step_all_agents(self, env):
        # 1a. check if agents are finished, mark them for assignments
        for agentID in self.exitedQueue:
            agent = env.agents[agentID - 1]

            if agent.numResources < agent.maxResources:
                self.unassignedEmpty.add(agentID)
            else:
                self.unassignedFull.add(agentID)

        self.exitedQueue = []
        actions = [np.nan for _ in range(len(env.agents))]

        # 1b. collect agents from depleted RCs
        for rcID in self.depletedRCIDs:
            for agentID in self.inboundAgents[rcID]:
                self.unassignedEmpty.add(agentID)
            self.inboundAgents[rcID] = deque()  

        self.depletedRCIDs = []

        # 1c. collect agents with random targets
        for i, randomTarget in enumerate(self.hasRandomTarget):
            if randomTarget:
                agentID = i + 1
                self.unassignedEmpty.add(agentID)

        # 2. assign agents
        while self.unassignedEmpty:

            resourceRankings, _ = self.rankRCs(env)
            bestResourceScore = resourceRankings[0][0]
            if not(np.isinf(bestResourceScore)):
                bestResourceIndex = resourceRankings[0][1]
                targetRow, targetCol = resourceRankings[0][2]
                closestAgentID = findClosestAgent(env.agents, self.unassignedEmpty, targetRow, targetCol)
                self.agentTargets[closestAgentID - 1][0] = targetRow
                self.agentTargets[closestAgentID - 1][1] = targetCol
                self.agentTargets[closestAgentID - 1][2] = bestResourceIndex

                #print(bestResourceIndex)
                self.inboundAgents[bestResourceIndex].append(closestAgentID)
                self.unassignedEmpty.discard(closestAgentID)
                self.hasRandomTarget[closestAgentID - 1] = False
            else:
                world = env.state.copy()
                world[world != 0] = 1
                for agentID in self.unassignedEmpty:
                    if not(self.hasRandomTarget[agentID - 1]): #don't give a new random target if has one already
                        targ = np.random.randint(env.shapes[0], size=(2,1))
                        while world[targ[0], targ[1]] == 1:
                            targ = np.random.randint(env.shapes[0], size=(2,1))

                        print("Giving agent",agentID,"randomtarg",targ[0], targ[1])

                        self.agentTargets[agentID - 1][0] = targ[0]
                        self.agentTargets[agentID - 1][1] = targ[1]
                        self.agentTargets[agentID - 1][2] = None
                        world[targ[0], targ[1]] = 1 #prevent someone else from getting same rand targ

                        self.hasRandomTarget[agentID - 1] = True
                self.unassignedEmpty = set()

        while self.unassignedFull:

            _, cacheRankings = self.rankRCs(env)
            bestCacheIndex = cacheRankings[0][1]
            targetRow, targetCol = cacheRankings[0][2]
            closestAgentID = findClosestAgent(env.agents, self.unassignedFull, targetRow, targetCol)
            self.agentTargets[closestAgentID - 1][0] = targetRow
            self.agentTargets[closestAgentID - 1][1] = targetCol
            self.agentTargets[closestAgentID - 1][2] = bestCacheIndex

            #print(bestResourceIndex)
            self.inboundAgents[bestCacheIndex].append(closestAgentID)
            self.unassignedFull.discard(closestAgentID)

        # 2a. Update targets (if queue entry pos changed since assingment)
        for i in range(len(self.agentTargets)):
            _,_,indexRC = self.agentTargets[i]
            if indexRC is not None: #if random target don't check for entry point...
                row, col = env.RCs[indexRC].queue.getQueueEntryNO()
                self.agentTargets[i][0] = row
                self.agentTargets[i][1] = col

        # 3. A* moves to targets
        for agentID in range(1, len(env.agents) + 1):
    
            observation = env.observe(agentID)
            [vectorObs, scalarObs] = observation[0]
            agent = env.agents[agentID - 1]
            action = 0

            empty_agent_FOV = vectorObs[0]
            obstacles = vectorObs[2]

            cacheR_unit = scalarObs[0]
            cacheC_unit = scalarObs[1]
            cache_norm = scalarObs[2]

            rqep_FOV = vectorObs[3]
            cqep_FOV = vectorObs[4]
            phero_FOV = vectorObs[7]

            agentIsEmpty = vectorObs[0][5,5] > 0 # Empty agent channel
            agentIsQueued = not(agent.queueID == None) #vectorObs[5][5,5] > 0 or vectorObs[6][5,5] > 0 # Agent in a resource queue or a cache queue

            if not agentIsQueued:
    
                # if on an entry point, join the cache
                #if cqep_FOV[5,5] > 0 or rqep_FOV[5,5] > 0:
                #    action = 5

                targetRow = self.agentTargets[agentID - 1][0]
                targetCol = self.agentTargets[agentID - 1][1]

                world = env.state.copy()
                world[world != 0] = 1
                if agent.numResources > 0:
                    world[world.shape[0]//2-2:world.shape[0]//2+2,world.shape[1]//2-2:world.shape[1]//2+2] = 1
                world[targetRow, targetCol] = 0

                path = astar2d(world, (agent.row,agent.col), (targetRow,targetCol))

                if path != False:

                    if len(path) <= 2:
                        action = 5
        
                    else:

                        dr = path[1][0] - agent.row
                        dc = path[1][1] - agent.col
          
                        ur, uc = unitVec(dr, dc)
                        action = selectAction(ur, uc)

                    actualAction = env.step(agentID, action)
                    actions[agentID - 1] = action

                    if action == 5 and actualAction == 5: # successful join...
                        bestResourceIndex = self.agentTargets[agentID - 1][2]
                        if bestResourceIndex is not None:
                            self.inboundAgents[bestResourceIndex].remove(agentID)

        # advance queues
        self.exitedQueue, self.depletedRCIDs = env.advanceQueues()
        '''
        # update self.exitedQueue
        for agentID in range(1, len(env.agents) + 1):
            # Agent deposited at nest
            if (env.agents[agentID - 1].numResources == 0 and self.prevCapacity[agentID - 1] > 0):
                self.exitedQueue.append(agentID)
            # Agent successfully mined, or was kicked out of the queue upon resource depletion
            elif (agent.queueID is None) and self.prevQueued[agentID - 1] or \
               (env.agents[agentID - 1].numResources == 1 and self.prevCapacity[agentID - 1] < 1):
                self.exitedQueue.append(agentID)
            # Agent's resource was depleted while on its way to it
            elif abs( self.agentTargets[agentID - 1][0] - env.RCs[self.agentTargets[agentID - 1][2]].row ) + \
                 abs( self.agentTargets[agentID - 1][1] - env.RCs[self.agentTargets[agentID - 1][2]].col ) > \
                  env.RCs[self.agentTargets[agentID - 1][2]].queue.length+1:
                self.exitedQueue.append(agentID)

        self.prevCapacity = [agent.numResources for agent in env.agents]
        self.prevQueued   = [(agent.queueID is not None) for agent in env.agents]
        '''
        return env, actions


def observe_all(env):

    observations = []

    for agentID in range(1, len(env.agents) + 1):
        observations.append(env.observe(agentID))

    return observations


if __name__ == "__main__":
    
    env_shape = (64,64)
    env_numAgents = 32
    env_freeAgentPlacement = "nearCache" #"random" 

    env = MAF_gym(shape=env_shape, numAgents=env_numAgents, no_agents_queued=0,
                  episodeNumber=20000, freeAgentPlacement=env_freeAgentPlacement, freeAgentFull=0.0) #, resourceBuffer=0)

    env.reappearingResources = True
    for rc in env.RCs:
        rc.infiniteFood = False
        rc.refreshRate = 0.0
        rc.miningRate = 0.1
        rc.dropOffRate = 1.0

    env.render()

#    time.sleep(TIMESTEP)

    controller = NOController(env)

    for i in range(512):

#        startTime = timeMS()

        print(env.calculateScore())

        controller.step_all_agents(env)
        
        # render
        env.render()

#        sleepTime = np.max([TIMESTEP - ((timeMS() - startTime) / 1000.0), 0])
        #print(sleepTime)
#        time.sleep(sleepTime)

    env.close()
