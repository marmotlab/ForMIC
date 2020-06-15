import math
import numpy as np
import sys
import random

from MAF_gym.envs.entity import *
from collections import deque
np.set_printoptions(threshold=sys.maxsize)

def addRCs(length, num_resources, num_caches, shape, episodeNumber, resourceBuffer=0):
    # 1s in the mask are 'reserved spaces'
    # 2s in the mask are resource locations
    # 3s in the mask are cache locations

    while True:
        
        MAX_VALUE = 255
        mask = np.ones((shape[0]+(2*length),shape[1]+(2*length)), dtype=np.int64)
        mask[length+1: (shape[0]+length)-1,
            length+1: (shape[1]+length)-1] = 0

        if resourceBuffer != 0:
            mask[length+1 + (int(shape[0]/2) - (resourceBuffer+1)):  length+1 +(int(shape[0]/2) + (resourceBuffer-1)),
                 length+1 + (int(shape[1]/2) - (resourceBuffer+1)):  length+1 +(int(shape[1]/2) + (resourceBuffer-1))] = 1
            
        RCs = []
        numPlacedRCs = 0
        numTries = 0
        while len(RCs) < (num_resources + num_caches) and numTries != 10000:
            numTries += 1
            row = np.random.randint(0, shape[0]-1) + length
            col = np.random.randint(0, shape[1]-1) + length
            tempMask = np.zeros((shape[0]+(2*length), shape[1]+(2*length)), dtype=np.int64)

            if len(RCs) < num_caches: # place cache(s) first
                entityType = entity.CACHE.value
                rc = RC(entityType, shape[0]//2-1, shape[1]//2-1,
                        entity.QUEUE_ID_OFFSET.value - numPlacedRCs, 'straightQueue', length, direction=3)
                tempMask[rc.row + length, rc.col + length] = 1
                for s in rc.reservedSpace:
                    tempMask[s[0] + length, s[1] + length] = 1
                    mask = np.maximum(mask, tempMask)
                RCs.append(rc)
                numPlacedRCs += 1


                rc = RC(entityType, shape[0]//2-1, shape[1]//2,
                        entity.QUEUE_ID_OFFSET.value - numPlacedRCs, 'straightQueue', length, direction=2)
                tempMask[rc.row + length, rc.col + length] = 1
                for s in rc.reservedSpace:
                    tempMask[s[0] + length, s[1] + length] = 1
                    mask = np.maximum(mask, tempMask)
                RCs.append(rc)
                numPlacedRCs += 1

                rc = RC(entityType, shape[0]//2, shape[1]//2-1,
                        entity.QUEUE_ID_OFFSET.value - numPlacedRCs, 'straightQueue', length, direction=0)
                tempMask[rc.row + length, rc.col + length] = 1
                for s in rc.reservedSpace:
                    tempMask[s[0] + length, s[1] + length] = 1
                    mask = np.maximum(mask, tempMask)
                RCs.append(rc)
                numPlacedRCs += 1

                rc = RC(entityType, shape[0]//2, shape[1]//2,
                        entity.QUEUE_ID_OFFSET.value - numPlacedRCs, 'straightQueue', length, direction=1)
                tempMask[rc.row + length, rc.col + length] = 1
                for s in rc.reservedSpace:
                    tempMask[s[0] + length, s[1] + length] = 1
                    mask = np.maximum(mask, tempMask)
                RCs.append(rc)
                numPlacedRCs += 1

            else:
                entityType = entity.RESOURCE.value
                rc = RC(entityType, row - length, col - length,
                        entity.QUEUE_ID_OFFSET.value - numPlacedRCs, 'straightQueue', length)

                tempMask[rc.row + length, rc.col + length] = 1
                for s in rc.reservedSpace:
                    tempMask[s[0] + length, s[1] + length] = 1

                #If there is no overlap (ie valid placement)
                if np.sum(np.logical_and(mask, tempMask)) == 0:
                    mask = np.maximum(mask, tempMask)
                    RCs.append(rc)
                    numPlacedRCs += 1

        if numTries != 10000:
            #Generate a shell to return the outer edge to 0s
            shell = np.full((shape[0]+(2*length),shape[1]+(2*length)),
                        fill_value = MAX_VALUE, dtype=np.int64)
            shell[length: shape[0]+length, length: shape[1]+length] = 0
            shell[length+1: (shape[0]+length)-1, length+1: (shape[1]+length)-1] = MAX_VALUE

            mask = np.bitwise_and(mask, shell).astype(np.int64)
            mask = mask[length: shape[0]+length,
                length: shape[1]+length]

            for rc in RCs:
                mask[rc.row][rc.col] = rc.entityType

            return mask, RCs


def addObstacles(state, portion):
    assert(portion >= 0)
    assert(portion <  1.0)

    numCells = state.shape[0] * state.shape[1]
    numObstacles = int(numCells * portion)

    for i in range(numObstacles):
        while True:
            row = np.random.randint(state.shape[0])
            col = np.random.randint(state.shape[1])

            if state[row][col] == 0:
                break

        state[row][col] = -1.0
        
    return state

        
def addSingleRC(state, length, resourceBuffer):
    shape = state.shape
    # TODO - This is currently very conservative, it will not place an entry point trail
    # if there are agent in the way
    def bad(r, c):
        if r < 0 or (r >= shape[0]):
            return True
        if c < 0 or (c >= shape[1]):
            return True
        if state[r, c] < 0:
            return True
        return False

    middle = (shape[0] // 2, shape[1] // 2)
    while True:
        row = np.random.randint(shape[0])
        col = np.random.randint(shape[1])

        if (np.abs(middle[0] - row) < resourceBuffer) or \
           (np.abs(middle[1] - col) < resourceBuffer):
            continue
            
        
        direction = np.random.randint(4)
        goodPos = True
        if state[row, col] != 0:
            continue
        for i in range(length+2):
            if direction == 0:
                if bad(row + i, col):
                    goodPos = False
            elif direction == 1:
                if bad(row, col + i):
                    goodPos = False
            elif direction == 2:
                if bad(row - i, col):
                    goodPos = False
            elif direction == 3:
                if bad(row, col - i):
                    goodPos = False

        if goodPos == True:
            break

    ID = np.min(state) - 1 # TODO, the ID number will explode, is this used for any obs?
    rc = RC(entity.RESOURCE.value, row, col, ID, 'straightQueue', length, direction=direction)


    state[row][col] = rc.entityType
    return state, rc

        
def addAgents(state, agents, numAgents, shape, freeAgentPlacement, freeAgentFull, startID=1):
    currAgentID = startID
    if freeAgentPlacement == "nearCache":
        #radius = int(np.sqrt(numAgents) / 2)
        radius = 1
        while True:
            availableSpaces =  np.sum(state[shape[0]//2 - radius: shape[0]//2 + radius,
                                       shape[1]//2 - radius: shape[1]//2 + radius] == 0)
            if availableSpaces > numAgents:
                break
            radius += 1        
        mask = np.ones(shape)
        mask[shape[0]//2 - radius: shape[0]//2 + radius, shape[1]//2 - radius: shape[1]//2 + radius] = 0

    elif freeAgentPlacement == "evenCorners":
        # assure enough locations to place all agents
        assert (numAgents <= shape[0] - 4 + shape[1] - 4)
        num_from_cache = 0
        first_pos_ul = (shape[0] // 2 - 3, shape[1] // 2 - 2)
        first_pos_ur = (shape[0] // 2 - 2, shape[1] // 2 + 2)
        first_pos_lr = (shape[0] // 2 + 2, shape[1] // 2 + 1)
        first_pos_ll = (shape[0] // 2 + 1, shape[1] // 2 - 3)
        while currAgentID <= numAgents:
            if currAgentID % 4 == 1: 
                row = first_pos_ul[0] - (num_from_cache)
                col = first_pos_ul[1]
            elif currAgentID % 4 == 2:
                row = first_pos_ur[0]
                col = first_pos_ur[1] + (num_from_cache)
            elif currAgentID % 4 == 3: 
                row = first_pos_lr[0] + (num_from_cache)
                col = first_pos_lr[1]
            else: 
                row = first_pos_ll[0]
                col = first_pos_ll[1] - (num_from_cache)
                num_from_cache += 1
            agent = Agent(ID=currAgentID, row=row, col=col)
            state[row][col] = currAgentID
            agents.append(agent)
            currAgentID += 1

    elif freeAgentPlacement == "singleCorner":
        agent_locs = np.arange(numAgents)
        np.random.shuffle(agent_locs)
        assert(numAgents <= shape[0] // 2 - 2)
        first_pos = (shape[0] // 2 - 3, shape[1] // 2 - 2)
        while currAgentID <= numAgents:
            row = first_pos[0] - agent_locs[currAgentID - startID]
            col = first_pos[1]
            agent = Agent(ID=currAgentID, row=row, col=col)
            state[row][col] = currAgentID
            agents.append(agent)
            currAgentID += 1
    elif freeAgentPlacement == "random":
        mask = np.zeros(state.shape)    
    else:
        print("did not find placement option")
        mask = np.zeros(state.shape)    


    while currAgentID <= numAgents:
        row = np.random.randint(0, shape[0]-1)
        col = np.random.randint(0, shape[1]-1)
        
        if (state[row][col] == 0) and (mask[row][col] == 0):
            agent = Agent(ID=currAgentID, row=row, col=col)
            # randomly spawn agents carrying resources already
            if np.random.rand() < freeAgentFull:
                agent.numResources = agent.maxResources
            state[row][col] = currAgentID
            agents.append(agent)
            currAgentID += 1
        #else:
            #print("invalid")
    return state, agents


def clearReservedSpace(state, reservedValue=1):
    state[state == reservedValue] = 0
    return state


def parseState(state0):
        def findQueue(state0, accumQueue, row, col):
            potentialSpots = [[row+1, col], [row,col+1],
                              [row-1, col], [row, col-1]]
            for i,j in potentialSpots:
                if state0[i][j] == -4:
                    state0[i][j] = 0
                    accumQueue.append((i,j))
                    return findQueue(state0, accumQueue, i, j)
            return accumQueue

        numAgents = 0
        for i in range(state0.shape[0]):
            for j in range(state0.shape[0]):
                if state0[i][j] > 0:
                    numAgents += 1

        agents = [None] * numAgents
        RCs = []
        for i in range(state0.shape[0]):
            for j in range(state0.shape[1]):
                if state0[i][j] > 0:
                    if agents[state0[i][j]-1] != None:
                        print("GAVE TWO AGENTS THE SAME ID")
                        sys.exit()
                    agents[state0[i][j] - 1] = Agent(state0[i][j], i, j)
                elif (state0[i][j] == entity.RESOURCE.value or
                      (state0[i][j] == entity.CACHE.value)):
                    rcID = entity.QUEUE_ID_OFFSET.value - len(RCs)
                    rc = RC(state0[i][j], i, j, rcID, "GUI")
                    queueSpots = findQueue(state0, [], i, j)
                    rc.setQueue(queueSpots)
                    RCs.append(rc)

        return agents, RCs


def findClosestRC(RCs, row, col, entityType, maxDistance=10000):
    assert((entityType == entity.RESOURCE.value) or
           (entityType == entity.CACHE.value))

    # This distance should always be greater than any potential distance on the map
    closestDist = maxDistance
    closestLoc = (-1, -1)

    for rc in RCs:
        if rc.entityType == entityType:
            entryRow, entryCol = rc.queue.queueSpots[0]
            curDist = distance([row, col], (entryRow, entryCol))
            if closestDist > curDist:
                closestDist = curDist
                closestRC = rc

    return closestRC

def distance(x1, x2):
    assert(len(x1) == len(x2))
    total = 0
    for i in range(len(x1)):
        total += (x1[i] - x2[i])**2

    return math.sqrt(total)


def makeAllHighways(state, maxPheromone, pheroActionDecay, pheroTimeDecay, RCs):
    allHighways = np.zeros(state.shape)
    for i, rc in enumerate(RCs):
        if rc.entityType == entity.RESOURCE.value:
            row, col = rc.queue.queueSpots[0]
            closestCache = findClosestRC(RCs, row, col, entity.CACHE.value)
            cRow, cCol = closestCache.queue.queueSpots[0]
            phermoneMask = phermoneHighway(state, maxPheromone, pheroActionDecay, pheroTimeDecay, row, col, cRow, cCol)
            if phermoneMask is None:
                return None

            allHighways = np.max(np.stack([allHighways, phermoneMask]), axis = 0)

    return allHighways


def phermoneHighway(state, maxPheromone, pheroActionDecay, pheroTimeDecay, row, col, cRow, cCol):
    
    phermoneMask = np.zeros(state.shape)
    moves = [(-1,0), (0,1), (1,0), (0,-1)]

    # find path from resource to cache
    i = 0
    while (row != cRow) or (col != cCol):

        # compute cache vector
        vector = [cRow - row, cCol - col]
        norm = np.sqrt(vector[0]**2 + vector[1]**2)
        normVec = [vector[0] / norm, vector[1] / norm]

        # make move that best follows cache vector
        sortedMoves = np.flip(np.argsort([np.dot([normVec[0],normVec[1]],[r,c]) for r,c in moves]))
        foundGoodMove = False
        for moveIndex in sortedMoves:
            move = moves[moveIndex]
            row += move[0]; col += move[1]
            if state[row][col] >= 0:

                # decay all the existing pheromones
                phermoneMask *= pheroTimeDecay 

                # lay down new pheromone...
                phermoneMask[row][col] = maxPheromone * (pheroActionDecay ** i)

                foundGoodMove = True
                break
            row -= move[0]; col -= move[1]

            
        if not foundGoodMove:
            print("There was no valid move available when generating the pheromone highway")
            return None

        if i > state.shape[0]**2:
            print("Took too many steps to generate the pheromone highway")
            return None

        i += 1 

    return phermoneMask


# Puts the first "numAgents" of the agents list into a random RC
# resourceCacheRatio of 0 means "numAgents" go into caches
# resourceCacheRatio of 1 means "numAgents" go into resources
def spawnAgentsAtRCs(state, RCs, numAgents, resourceCacheRatio=0.0):
    numAgentsToR = int(numAgents * resourceCacheRatio)
    resourceIndices = []
    cacheIndices = []
    for i in range(len(RCs)):
        if RCs[i].entityType == entity.RESOURCE.value:
            resourceIndices.append(i)
        else:
            cacheIndices.append(i)

    agents = []
    for i in range(numAgentsToR):
        #Put the agents into a random resource queue
        while True:
            rIndex = random.choice(resourceIndices)
            if len(RCs[rIndex].queue.agents) < RCs[rIndex].queue.length:
                break

        row, col = RCs[rIndex].queue.getQueueEntry()
        agent = Agent(ID=i+1, row=row, col=col)
        agent.numResources = 0
        agents.append(agent)
        RCs[rIndex].queue.addAgent(agent)
        if len(RCs[rIndex].queue.agents) >= RCs[rIndex].queue.length:
            resourceIndices.remove(rIndex)
        state[row][col] = agent.ID

    for i in range(numAgentsToR, numAgents):
        #Put the agents into a random cache queue
        while True:
            cIndex = random.choice(cacheIndices)
            if len(RCs[cIndex].queue.agents) < RCs[cIndex].queue.length:
                break

        row, col = RCs[cIndex].queue.getQueueEntry()
        agent = Agent(ID=i+1, row=row, col=col)
        agent.numResources = agent.maxResources
        agents.append(agent)
        RCs[cIndex].queue.addAgent(agent)
        if len(RCs[cIndex].queue.agents) >= RCs[cIndex].queue.length:
            cacheIndices.remove(cIndex)
        state[row][col] = agent.ID

    return state, RCs, agents


    
# Puts the first "numAgents" of the agents list into a random RC
# resourceCacheRatio of 0 means "numAgents" go into caches
# resourceCacheRatio of 1 means "numAgents" go into resources
def assignAgentsToRCs(state, RCs, agents, resourceCacheRatio=0.0):
    numAgentsToR = int(len(agents) * resourceCacheRatio)
    resourceIndices = []
    cacheIndices = []
    for i in range(len(RCs)):
        if RCs[i].entityType == entity.RESOURCE.value:
            resourceIndices.append(i)
        else:
            cacheIndices.append(i)

    for i in range(numAgentsToR):
        #Put the agents into a random resource queue
        while True:
            rIndex = random.choice(resourceIndices)
            if len(RCs[rIndex].queue.agents) < RCs[rIndex].queue.length:
                break

        agents[i].numResources = 0
        if len(RCs[rIndex].queue.agents) == 0: # front agent
            agents[i].numResources = np.random.randint(1, 2+ 1//RCs[rIndex].miningRate) * RCs[rIndex].miningRate
        origLoc = agents[i].getLocation()
        state[origLoc[0]][origLoc[1]] = 0
        RCs[rIndex].queue.addAgent(agents[i])
        newLoc = agents[i].getLocation()
        state[newLoc[0]][newLoc[1]] = agents[i].ID

    for i in range(numAgentsToR, len(agents)):
        #Put the agents into a random cache queue
        while True:
            cIndex = random.choice(cacheIndices)
            if len(RCs[cIndex].queue.agents) < RCs[cIndex].queue.length:
                break

        agents[i].numResources = agents[i].maxResources
        if len(RCs[cIndex].queue.agents) == 0: # front agent
            agents[i].numResources = np.random.randint(1, 2+ 1//RCs[cIndex].dropOffRate) * RCs[cIndex].dropOffRate
        origLoc = agents[i].getLocation()
        state[origLoc[0]][origLoc[1]] = 0
        RCs[cIndex].queue.addAgent(agents[i])
        newLoc = agents[i].getLocation()
        state[newLoc[0]][newLoc[1]] = agents[i].ID

    return state, RCs, agents


    
