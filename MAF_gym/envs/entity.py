import numpy as np
import math
import sys

from enum import Enum

class entity(Enum):
    RESOURCE = -2
    CACHE = -3
    QUEUE_ID_OFFSET = -4


class reward(Enum):
    MOVE          =  -0.05
    QUEUEDSTEP    =  -0.05
    COLLISION     =   0.0
    NOMOVE        =  -0.5
    LEFTGOODRES   =  10.0
    LEFTGOODCACHE =   5.0
    ENTEREDGOODRC =   0.0
    INCORRECTRC   =  -1.0
    EMPTYRC       =  -1.0
    CANNOTFINDRC  =   0.0
    PER_AGENT     = - 0.0
    

class envSetup(Enum):
    INIT_miningRate = 0.1
    INIT_dropOffRate = 0.1

    TRAIN_miningRate = 0.1
    TRAIN_dropOffRate = 0.5

    
class RC():
    def __init__(self, entityType, row, col, ID, queueStyle='straightQueue', queueLength=4,
                 direction=None, maxResources=8):
        self.entityType = entityType
        self.row = row
        self.col = col
        self.queue = None
        self.reservedSpace = None
        self.queueStyle = queueStyle
        if entityType == entity.RESOURCE.value:
            self.maxResources = maxResources #np.random.randint(5,  15)
            self.numResources = np.random.randint(maxResources//2, maxResources)
            queueBuffer = 0
        else:
            self.numResources = 0
            self.maxResources = 0
            queueBuffer = 1
        self.miningRate = envSetup.INIT_miningRate.value
        self.dropOffRate = envSetup.INIT_dropOffRate.value
        self.refreshRate = self.miningRate / 2
        self.infiniteFood = True
        self.ID = ID

        if self.queueStyle == "straightQueue":
            self.straightQueue(queueLength, ID, direction=direction, queueBuffer=queueBuffer)
        elif self.queueStyle == "GUI":
            return
        else:
            print("INVALID QUEUE STYLE")
            sys.exit()


    def getLocation(self):
        return [self.row, self.col]


    def validEjectionActions(self, state):
        gatherRow, gatherCol = self.queue.queueSpots[0]
        validActions = []
        
        diff = [self.row - gatherRow, self.col - gatherCol]

        if (diff == [0, 1]) or (diff == [0, -1]):
            if gatherRow-1 >= 0 and (state[gatherRow-1][gatherCol] == 0):
                validActions.append(1)
            if (gatherRow+1 < state.shape[0]) and (state[gatherRow+1][gatherCol] == 0):
                validActions.append(3)
                
        elif (diff == [1,0]) or (diff == [-1, 0]):
            if (gatherCol+1 < state.shape[0]) and (state[gatherRow][gatherCol+1] == 0):
                validActions.append(2)
                
            if gatherCol-1 >= 0 and (state[gatherRow][gatherCol-1] == 0):
                validActions.append(4)

        return validActions


    def setQueue(self, queue):
        self.queue = Queue(queue, len(queue), self.ID)
        
    #queueBuffer - The number of squares reserved on each side
    #              of the queueSpots
    def straightQueue(self, queueLength, queueID, direction=None, queueBuffer=1):
        queueSpots = []
        reservedSpace = []
        if direction is None:
            direction = np.random.randint(0,3)
        for i in range(queueLength+1):
            for j in range(-1*queueBuffer, queueBuffer+1):
                if direction == 0:
                    reservedSpace.append((self.row+(i+1), self.col+j))
                elif direction == 1:
                    reservedSpace.append((self.row+j, self.col+(i+1)))
                elif direction == 2:
                    reservedSpace.append((self.row-(i+1), self.col+j))
                elif direction == 3:
                    reservedSpace.append((self.row+j, self.col-(i+1)))
            if direction == 0:
                queueSpots.append((self.row+(i+1), self.col))
            elif direction == 1:
                queueSpots.append((self.row, self.col+(i+1)))
            elif direction == 2:
                queueSpots.append((self.row-(i+1), self.col))
            elif direction == 3:
                queueSpots.append((self.row, self.col-(i+1)))

        # We delete the last queueSpot as it was only used to
        # generate an appropriately sized reservedSpace
        self.queue = Queue(queueSpots[:-1], queueLength, queueID)

        self.reservedSpace = reservedSpace


    def mineResource(self):
        if len(self.queue.agents) == 0:
            return 0
        if self.infiniteFood == True:
            self.numResources = self.maxResources

        # If the resource is empty, signal that. Otherwise mine.
        if self.numResources == 0:
            return 3
        elif self.numResources > 0:
            agent = self.queue.agents[0]
            if agent.numResources >= agent.maxResources:
                return 2

            # Resource will be depleted on this 'mine'
            if self.numResources < self.miningRate:
                agent.numResources = min(agent.numResources + self.numResources,agent.maxResources)
                self.numResources = 0
            else: #Resource has plenty of food
                agent.numResources = min(agent.numResources + self.miningRate, agent.maxResources)
                self.numResources -= self.miningRate

            if agent.numResources < agent.maxResources:
                if self.numResources == 0:
                    return 1
                return 0

        return 1

    
    def dropOffResource(self):
        if len(self.queue.agents) == 0:
            return 0
        agent = self.queue.agents[0]
        if agent.numResources == 0:
            return 2
        else:
            agent.numResources -= self.dropOffRate
            agent.numResources = max(agent.numResources, 0)
            
            self.numResources += self.dropOffRate

            if agent.numResources > 0:
                return 0
            else:
                agent.numResources = 0
                return 1

        
# TODO - Queue/Dequeue logic instead of list slicing
class Queue():
    def __init__(self, queueSpots, length, queueID):
        self.agents = []
        self.length = length
        self.queueSpots = queueSpots
        self.ID = queueID

        self.lastFive = [None] * 5
        self.lastFiveCounter = 0

        self.freeFront = False


    def updateLastFive(self, agentID):
        self.lastFive[self.lastFiveCounter] = agentID
        self.lastFiveCounter = (self.lastFiveCounter + 1) % 5
        
    def freeFirstAgent(self):
        if self.freeFront == False:
            self.freeFront = True
        else:
            print("AGENT IN FRONT IS ALREADY FREE.")
            print(len(self.agents))
            sys.exit()

        if len(self.agents) == 0:
            return None

        freeAgent = self.agents[0]
        self.agents = self.agents[1:]
        if freeAgent.queueID < -7: #check if its a R or a C
            freeAgent.pheroIntensity = 1.0 #resource
        else:
            freeAgent.pheroIntensity = 0.0 #cache
        freeAgent.queueID = None
        return freeAgent


    def advance(self):
        if self.freeFront == True:
            for agent, newLocation in zip(self.agents, self.queueSpots):
                agent.setLocation(newLocation[0], newLocation[1])
            self.freeFront = False
        else:
            print("AGENT IN FRONT NOT FREED. CAN'T ADVANCE")
            sys.exit()


    def addAgent(self, agent):
        newLocation = self.getQueueEntry()
        assert(newLocation[0] != -2) # Make sure queue isn't full
        agent.queueID = self.ID
        agent.setLocation(newLocation[0], newLocation[1])
        self.agents.append(agent)

    
    def getQueueEntry(self):
        if len(self.agents) == self.length:
            return (-2, -2)
        if self.freeFront == True and (len(self.agents) < (self.length-1)):
            return self.queueSpots[len(self.agents) + 1]
        else:
            return self.queueSpots[len(self.agents)]

    def getQueueEntryNO(self):
        nQueued = len(self.agents)
#        if nQueued == 0:
#            return self.queueSpots[1]
        if nQueued == self.length:
            return self.queueSpots[nQueued - 1]
        if self.freeFront == True and (nQueued < (self.length-1)):
            return self.queueSpots[nQueued + 1]
        else:
            return self.queueSpots[nQueued]

class Agent():
    def __init__(self, ID, row=0, col=0):
        self.ID = ID
        self.row = row
        self.col = col
        self.numResources = 0
        self.maxResources = 1
        self.queueID = None

        self.pheroIntensity = 0.0


    def setLocation(self, row, col):
        self.row = row
        self.col = col


    def getLocation(self):
        return [self.row, self.col]


    def move(self, action):
        if action == 0:
            return 0
        elif action == 1:
            self.row -= 1
        elif action == 2:
            self.col += 1
        elif action == 3:
            self.row += 1
        elif action == 4:
            self.col -= 1
        else:
            print("agent can only move NESW/1234")
            sys.exit()


    def reverseMove(self, action):
        if action == 0:
            return 0
        elif action == 1:
            self.row += 1
        elif action == 2:
            self.col -= 1
        elif action == 3:
            self.row -= 1
        elif action == 4:
            self.col += 1
        else:
            print("agent can only move NESW/1234")
            sys.exit()

    # Generates a unit vector from this agent to the caches
    # ASSUMES CACHES ARE IN THE CENTER OF THE ENVIRONMENT
    def getCacheVector(self, env_shape):
        cachePos = [ (env_shape[0]-1)/2. , (env_shape[1]-1)/2. ]
        agentPos = self.getLocation()
        posVec   = [cachePos[0] -  agentPos[0], cachePos[1] - agentPos[1]]
        posNorm  = np.sqrt(posVec[0]**2 + posVec[1]**2)
        posVec   = [ posVec[0]/posNorm, posVec[1]/posNorm ]

        return posVec, posNorm
        
            
    # This function is only used for test policies, not for actual learning
    # There is undefined behavior for when you try to get the entry point of a full queue
    def findClosestRCEntry(self, RCs, entityType, maxDistance=1000):
        assert((entityType == entity.RESOURCE.value) or
               (entityType == entity.CACHE.value))

        # This distance should always be greater than any potential distance on the map
        closestDist = maxDistance
        closestLoc = (-1, -1)

        currLoc = self.getLocation()

        for rc in RCs:
            if rc.entityType == entityType:
                entryRow, entryCol = rc.queue.getQueueEntry()
                curDist = distance(currLoc, (entryRow, entryCol))
                if closestDist > curDist:
                    closestDist = curDist
                    closestLoc = (entryRow, entryCol)

        rowDif = closestLoc[0] - currLoc[0]
        colDif = closestLoc[1] - currLoc[1]

        if closestDist == 0.0:
            return [0, 0]

        return [rowDif / closestDist, colDif / closestDist]


    def hasFood(self):
        return self.numResources > 0
        
                        
def distance(x1, x2):
    assert(len(x1) == len(x2))
    total = 0
    for i in range(len(x1)):
        total += (x1[i] - x2[i])**2

    return math.sqrt(total)

