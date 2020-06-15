import gym
import time
import math
import random
import numpy as np
import pickle
from threading import Lock
large_width = 400
np.set_printoptions(linewidth=large_width)


TIMESTEP = .1
PHEROMONE_INIT_VALUE = 1
OBS_CENTER = (5, 5)
TROUBLESHOOTING = False



MIN_BEACONS = 5


from enum import Enum

from MAF_gym.envs.entity import *
from MAF_gym.envs.MAF_env import MAF_gym


class Actions(Enum):
    AT_HOME_1 = 1


class policyType(Enum):
    WALKER = 0
    BEACON = 1

class AgentWrapper():
    def __init__(self, agent, numResources):
        self.ID = agent.ID
        self.agent = agent
        self.policyType = policyType.WALKER.value
        self.carryingFood = False
        self.persuedResourceID = None
        
        self.stepCounter = 0
        self.exploreDir = np.random.randint(1,5)

        
        self.prevAction = None
        self.nestBeacon = None
        self.resourceBeacons = [0] * numResources        

        



class BeaconController():
    def __init__(self, env):
        self.agentWrappers = []
        for agent in env.agents:
            self.agentWrappers.append(AgentWrapper(agent, self.getNumResources(env)))
        
        self.resourceBeacons = np.zeros((env.shape[0], env.shape[1], self.getNumResources(env)))
        self.nestBeacon = np.zeros(env.shape)
        self.conversionProbability = .3
        self.stepCounterMax = 4
        self.discoveredResources = []
        self.firstStep = True
                                      
        self.infiniteResources = not env.reappearingResources

        

    def getNumResources(self, env):
        numResources = 0
        for rc in env.RCs:
            if rc.entityType == entity.RESOURCE.value:
                numResources += 1
        return numResources


    def step_all_agents(self, env):
        IDs = list(range(1, len(env.agents) + 1))

        #If we are at the first step, make sure agents in center go first 
        if self.firstStep == True:
            IDs = sorted(IDs, key=lambda ID: distance((env.agents[ID-1].row, env.agents[ID-1].col), (env.shape[0] // 2, env.shape[1] //2)))
            
        
        for agentID in IDs:
            if env.isAgentQueued(agentID):
                continue
            self.stepAgent(agentID, env)
            self.resetBeaconPheromones(env)

        exitedQueue, depletedRCs = env.advanceQueues()

        #agents who were just mining can move in whatever direction they want
        for agentID in exitedQueue:
            self.agentWrappers[agentID-1].prevAction = None
        
        if not self.infiniteResources:
            self.checkDepletedResources(env, depletedRCs)
            
        self.resetBeaconPheromones(env)

        emptyAgents = [agentID for agentID in exitedQueue
                       if env.agents[agentID-1].numResources == 0]
        self.reassignPersuedResourceIDs(emptyAgents, env)

        self.firstStep = False
        

    def checkDepletedResources(self, env, depletedRCs):

        IDs = list(range(1, len(env.agents) + 1))
        beaconIDs = []; walkerIDs = []
        for ID in IDs:
            if self.agentWrappers[ID-1].policyType == policyType.BEACON.value:
                beaconIDs.append(ID)
            elif self.agentWrappers[ID-1].policyType == policyType.WALKER.value:
                walkerIDs.append(ID)
            else:
                print("agent was neither a beacon or walker")
                assert(1==0)

        for k in depletedRCs:

            i = k - 4 #k indexes env.RCs, i indexes self.resourceBeacons
            rc = env.RCs[i+4]
            self.resourceBeacons[:,:,i] = 0
            for agentWrapper in self.agentWrappers:
                agentWrapper.resourceBeacons[i] = 0


            # Sort agent beaconIDs based on proximity to the resource
            beaconIDs = sorted(beaconIDs, key=lambda ID: distance((env.agents[ID-1].row, env.agents[ID-1].col), (rc.row, rc.col)))
                

            # If no one can see the resource, then we're done
            noOneSees = True
            for agentID in beaconIDs:
                if self.resourceInView(agentID, env, rc):
                    noOneSees = False
            if noOneSees == True:
                continue


                
            while True:
                changeMade = False
                print("looping")
                for agentID in beaconIDs:
                    agentWrapper = self.agentWrappers[agentID-1]
                    _, resourceBeacons = self.extractBeacons(agentID, env)                    
                    currBeaconMatrix = resourceBeacons[:,:,i]
                    
                    if self.resourceInView(agentID, env, rc):
                        if agentWrapper.resourceBeacons[i] != 1:
                            agentWrapper.resourceBeacons[i] = 1
                            r, c = agentWrapper.agent.row, agentWrapper.agent.col
                            changeMade = True
                    else:
                        if np.max(currBeaconMatrix) != 0:
                            r, c = agentWrapper.agent.row, agentWrapper.agent.col
                            newVal = np.min(currBeaconMatrix[np.nonzero(currBeaconMatrix)]) + 1
                            oldVal = agentWrapper.resourceBeacons[i]

                            if newVal != oldVal:
                                agentWrapper.resourceBeacons[i] = newVal
                                changeMade = True
                            
                    if changeMade == True:
                        self.resetBeaconPheromones(env)

                if changeMade == False:
                    break
            for agentID in walkerIDs:
                agentWrapper = self.agentWrappers[agentID-1]
                if agentWrapper.persuedResourceID == i:
                    _, resourceBeacons = self.extractBeacons(agentID, env)
                    # agents try to exploit an already found resource first
                    # if the resource they were going to depletes
                    if np.sum(resourceBeacons) != 0:
                        self.reassignPersuedResourceIDs([agentID], env)
                    else:
                        # If you want agents to explore again
                        agentWrapper.persuedResourceID = None
    
    def reassignPersuedResourceIDs(self, agentIDs, env):
        for agentID in agentIDs:
            nestBeacon, resourceBeacons = self.extractBeacons(agentID, env)

            # Find which resource Beacons are available
            visibleBeacons = {}
            for i in range(resourceBeacons.shape[2]):
                if np.max(resourceBeacons[:,:,i]) > 0:
                    visibleBeacons[i] = None

            # give a probability score for each one based on how close they are
            for i in visibleBeacons.keys():
                currBeaconMatrix = resourceBeacons[:,:,i]
                visibleBeacons[i] = np.min(currBeaconMatrix[np.nonzero(currBeaconMatrix)])


            x = visibleBeacons.items()
            beacons = list(visibleBeacons.keys())
            beaconDistances = list(visibleBeacons.values())
            beaconProb = (1 / np.array(beaconDistances)) + .5
            beaconProb = beaconProb / np.sum(beaconProb)

            if len(beacons) >= MIN_BEACONS:
                b = np.random.choice(beacons, 1, p=beaconProb)[0]

            else:
                beacons.append(None)
                beaconProb = beaconProb / 2
                half = (1 - np.sum(beaconProb))
                beaconProb = list(beaconProb)
                beaconProb.append(half)    
                b = np.random.choice(beacons, 1, p=beaconProb)[0]

            self.agentWrappers[agentID-1].persuedResourceID = b

                
    def extractBeacons(self, agentID, env, adjust=0):
        agent = env.agents[agentID-1]
        transform_row = env.observationSize // 2 - agent.row
        transform_col = env.observationSize // 2 - agent.col

        min_row = max((agent.row - env.observationSize // 2) - adjust, 0)
        max_row = min((agent.row + env.observationSize // 2 + 1 + adjust), env.shape[0])
        min_col = max((agent.col - env.observationSize // 2) - adjust, 0)
        max_col = min((agent.col + env.observationSize // 2 + 1 + adjust), env.shape[1])


        nestBeacon = np.zeros((env.observationSize, env.observationSize))
        resourceBeacons = np.zeros((env.observationSize, env.observationSize,
                                    self.getNumResources(env)))

        nestBeacon[
            (min_row + transform_row):(max_row + transform_row),
            (min_col + transform_col):(max_col + transform_col)
        ] = self.nestBeacon[min_row:max_row, min_col:max_col]



        for i in range(self.getNumResources(env)):
            resourceBeacons[
                (min_row + transform_row):(max_row + transform_row),
                (min_col + transform_col):(max_col + transform_col),
                i
            ] = self.resourceBeacons[min_row:max_row, min_col:max_col, i]

        #Don't let agent include itself as a beacon
        center = 5 
        nestBeacon[center, center] = 0
        resourceBeacons[center, center, :] = 0
        return [nestBeacon, resourceBeacons]
        

    def agentOnQueue(self, agentID, env):
        
        allQueueSpots = []
        adjSpots = []
        for rc in env.RCs:
            allQueueSpots = allQueueSpots + rc.queue.queueSpots

            iPoint = rc.queue.queueSpots[0]

            diff = (iPoint[0] - rc.row, iPoint[1] - rc.col)

            if diff[0] == -1 and diff[1] == 0:
                adjSpots.append((iPoint[0], iPoint[1] - 1))
                adjSpots.append((iPoint[0], iPoint[1] + 1))
            elif diff[0] == 0 and diff[1] == 1:
                adjSpots.append((iPoint[0] + 1, iPoint[1]))
                adjSpots.append((iPoint[0] - 1, iPoint[1]))
            elif diff[0] == 1 and diff[1] == 0:
                adjSpots.append((iPoint[0], iPoint[1] - 1))
                adjSpots.append((iPoint[0], iPoint[1] + 1))
            elif diff[0] == 0 and diff[1] == -1:
                adjSpots.append((iPoint[0] + 1, iPoint[1]))
                adjSpots.append((iPoint[0] - 1, iPoint[1]))
            
        row, col = env.agents[agentID-1].row, env.agents[agentID-1].col        
        for coord in (allQueueSpots + adjSpots):
            if row == coord[0] and (col == coord[1]):
                return True

        
        return False
    
    def stepAgent(self, agentID, env):
        agentWrapper = self.agentWrappers[agentID-1]
              
        nestBeacon, resourceBeacons = self.extractBeacons(agentID, env)

        smallNestBeacon, _ = self.extractBeacons(agentID, env, adjust=-1)
        numSeenBeacons = np.sum(smallNestBeacon != 0)
        
        if agentWrapper.policyType == policyType.WALKER.value:            
            if numSeenBeacons >= 2 or self.agentOnQueue(agentID, env):
                if agentWrapper.agent.numResources > 0:
                    self.RCSearch(agentID, env, entity.CACHE.value)
                else:
                    self.RCSearch(agentID, env, entity.RESOURCE.value)
            else:
                agentWrapper.policyType = policyType.BEACON.value
                
                

    
        elif agentWrapper.policyType == policyType.BEACON.value:
            if numSeenBeacons > 2:
                if np.random.random() < self.conversionProbability:
                    agentWrapper.policyType = policyType.WALKER.value

        # If agent ends turn as beacon, update its beacon values
        if agentWrapper.policyType == policyType.BEACON.value:
            if self.cacheInView(agentID, env):
                agentWrapper.nestBeacon = 1
            else:
                try:
                    agentWrapper.nestBeacon = np.min(nestBeacon[np.nonzero(nestBeacon)]) + 1
                except:
                    assert(1==0)
            
            for i in range(self.getNumResources(env)):
                rc = env.RCs[i+4]
                currBeaconMatrix = resourceBeacons[:,:,i]
                if self.resourceInView(agentID, env, rc):
                    agentWrapper.resourceBeacons[i] = 1
                else:
                    if np.max(currBeaconMatrix) != 0:
                        agentWrapper.resourceBeacons[i] = np.min(currBeaconMatrix[np.nonzero(currBeaconMatrix)]) + 1


    def RCSearch(self, agentID, env, entityType):
        nestBeaconObs, resourceBeaconObs = self.extractBeacons(agentID, env)
        agentWrapper = self.agentWrappers[agentID-1]
        eps = .01
        
        # 1st plan - If see RC entry point, go to it
        RCEntryPoints = [rc.queue.getQueueEntry() for rc in env.RCs if rc.entityType == entityType]
        closestCoord = self.getClosestCoord(agentID,
                                            self.getVisibleCoords(agentID, RCEntryPoints))
        if closestCoord is not None:
            if distance((agentWrapper.agent.row, agentWrapper.agent.col), closestCoord) < 1 + eps:
                env.executeAction(agentWrapper.agent, 5)
                return
            self.goToCoord(agentID, env, closestCoord)
            return
        
        # 2nd plan - If see RC itself, go to it
        RCCoords = [(rc.row, rc.col) for rc in env.RCs if rc.entityType == entityType]
        closestCoord = self.getClosestCoord(agentID,
                                            self.getVisibleCoords(agentID, RCCoords))
        
        if closestCoord is not None:
            #print("going to RC")
            self.goToCoord(agentID, env, closestCoord)
            return
        

        # 3rd plan - See RC beacon and follow it.
        # Cache beacon guaranteed to be seen. Resource beacon not guaranteed
        # Cache Case
        if entityType == entity.CACHE.value:
            beaconRows, beaconCols = np.where(nestBeaconObs ==
                                              np.min(nestBeaconObs[np.nonzero(nestBeaconObs)]))
            beaconCoords = self.transposeCoords(agentID, zip(beaconRows, beaconCols))
            bestBeacon = self.getClosestCoord(agentID,
                                              self.getVisibleCoords(agentID, beaconCoords))
            assert(bestBeacon is not None)
            self.goToCoord(agentID, env, bestBeacon)
            return
        
        elif entityType == entity.RESOURCE.value and (agentWrapper.persuedResourceID is not None):
            rID = agentWrapper.persuedResourceID
            currBeaconMatrix = resourceBeaconObs[:,:,rID]
            if np.max(currBeaconMatrix) != 0:
                beaconRows, beaconCols  = np.where(currBeaconMatrix == np.min(currBeaconMatrix[np.nonzero(currBeaconMatrix)]))
                beaconCoords = self.transposeCoords(agentID, zip(beaconRows, beaconCols))
                bestBeacon = self.getClosestCoord(agentID,
                                                  self.getVisibleCoords(agentID, beaconCoords))
                assert(bestBeacon is not None)
                self.goToCoord(agentID, env, bestBeacon)
                return
        # 4th plan - Explore (only applies when not carrying food)
        self.explore(agentID, env)
        

    def getInvalidSpaces(self, agentID, env):
        vectorObs = env.extractObservation(env.agents[agentID-1])
        empty_agent_FOV = vectorObs[0]
        full_agent_FOV = vectorObs[1]
        obstacles = vectorObs[2]
        invalidSpaces = np.logical_or(obstacles, np.logical_or(empty_agent_FOV, full_agent_FOV))
        return invalidSpaces

    def transposeCoords(self, agentID, coords):
        agentWrapper = self.agentWrappers[agentID-1]
        aR = agentWrapper.agent.row; aC = agentWrapper.agent.col;

        tR = aR - 5; tC = aC - 5;
        
        newCoords = []
        for c in coords:
            newCoords.append((c[0] + tR, c[1] + tC))

        return newCoords

    def getClosestCoord(self, agentID, coords):
        minDist = 9e99
        closestCoord = None
        agentWrapper = self.agentWrappers[agentID-1]
        for c in coords:
            d = distance((c[0], c[1]),
                         (agentWrapper.agent.row, agentWrapper.agent.col))

            if d < minDist:
                minDist = d
                closestCoord = c

        return closestCoord
        
    def getVisibleCoords(self, agentID, coords, viewDistance=6):
        visibleCoords = []
        agentWrapper = self.agentWrappers[agentID-1]
        for c in coords:
             if (np.abs(c[0] - agentWrapper.agent.row) < viewDistance and
                 np.abs(c[1] - agentWrapper.agent.col) < viewDistance):

                 visibleCoords.append((c[0], c[1]))
        return visibleCoords

    def resourceInView(self, agentID, env, rc):
        visibleCoords = self.getVisibleCoords(agentID, [(rc.row, rc.col)])

        if len(visibleCoords) != 0:
            return True

        return False

    def cacheInView(self, agentID, env):
        cacheCoords = [(rc.row, rc.col) for rc in env.RCs if rc.entityType == entity.CACHE.value]
        visibleCacheCoords = self.getVisibleCoords(agentID, cacheCoords)

        if len(visibleCacheCoords) != 0:
            return True

        return False

        
    def explore(self, agentID, env):
        agentWrapper = self.agentWrappers[agentID-1]
        validActions = env.listNextValidActions(agentID)

        counter = 0
        while agentWrapper.exploreDir not in validActions:
            agentWrapper.exploreDir = (agentWrapper.exploreDir - 1) % 5

            # skip 0 (no-op)
            if agentWrapper.exploreDir == 0:
                agentWrapper.exploreDir += 1

            counter += 1

            if counter >= 10: #excessively high, but will still work
                agentWrapper.exploreDir = np.random.randint(1,5)
                break

        env.executeAction(agentWrapper.agent, agentWrapper.exploreDir)
        
        agentWrapper.stepCounter += 1
        if agentWrapper.stepCounter == self.stepCounterMax:
            agentWrapper.stepCounter = 0
            
        return


    def nextToCoord(self, agentID, env, coord, diag=True):
        agent = self.agentWrappers[agentID-1].agent
        eps = .1
        d = distance((agent.row, agent.col), coord)

        if diag and (d < (np.sqrt(2) + eps)):
            return True

        if distance < (1 + eps):
            return True

        return False


    def circleCoord(self, agentID, env, coord):
        print("circling coordinate")
        agent = self.agentWrappers[agentID-1].agent
        r, c = agent.row, agent.col

        assert(nextToCoord(agentID, env, coord, diag=True))
        dr = coord[0] - r
        dc = coord[1] - c

        actionMap = {
            -1: {
                -1: [3],
                 0: [4, 1],
                 1: 4,
            },
            0: {
                -1: [3, 4],
                 0: [None],
                 1: [1, 2],
            },
            1: {
                -1: [2],
                 0: [2, 3],
                 1: [1]
            }
        }

        actions = actionMap[dr][dc]
        validActions = env.listNextValidActions(agentID)
        for a in actions:
            if a in validActions:
                env.executeAction(a)
                return

        env.executeAction(np.random.randint(1,5))
        return
        
               
        

    def goToCoord(self, agentID, env, coord):
        agentWrapper  = self.agentWrappers[agentID-1]
        invalidSpaces = self.getInvalidSpaces(agentID, env)

        dr, dc = getVector((agentWrapper.agent.row, agentWrapper.agent.col),
                           (coord[0], coord[1]))

        eps = 0.01
        if np.linalg.norm([dr,dc]) < eps:
            return 0


        avoidReversals = {
            None: -1, # for before world starts
            0: 2,
            1: 3,
            2: 0,
            3: 1
            }
        allMoves = [(-1,0), (0,1), (1,0), (0,-1)]
        movePriorities = np.argsort([np.dot([dr,dc],[r,c]) for r,c in allMoves])[::-1]
        for move in movePriorities:
            if not(invalidSpaces[allMoves[move][0] + 5, allMoves[move][1] + 5]):
                if avoidReversals[agentWrapper.prevAction] != move:
                    agentWrapper.prevAction = move
                    env.executeAction(agentWrapper.agent, move + 1)
                    return

        env.executeAction(agentWrapper.agent, np.random.randint(1,4))

        

    # erases all beacons in controller and redraws them based on current agent positions
    def resetBeaconPheromones(self, env):
        maxVal = 0
        self.nestBeacon[:] = 0
        self.resourceBeacons[:] = 0
        env.phero_map_loaded[:] = 0
        
        for agentWrapper in self.agentWrappers:
            if agentWrapper.policyType == policyType.BEACON.value:
                r = agentWrapper.agent.row; c = agentWrapper.agent.col;
                
                self.nestBeacon[r,c] = agentWrapper.nestBeacon
                for i, resourceBeacon in enumerate(agentWrapper.resourceBeacons):
                    self.resourceBeacons[r,c,i] = resourceBeacon

                env.phero_map_loaded[r,c] = 1.0

def getVector(x1, x2):
    return (x2[0] - x1[0], x2[1] - x1[1])



def distance(x1, x2):
    assert(len(x1) == len(x2))
    total = 0
    for i in range(len(x1)):
        total += (x1[i] - x2[i])**2

    return math.sqrt(total)

    

if __name__ == "__main__":
    env_numAgents = 128
    env_shape = (128,128)
    env_freeAgentPlacement = "nearCache"
    
    
    env = MAF_gym(shape=env_shape, numAgents=env_numAgents, no_agents_queued=0, pheroActionDecay=1, pheroTimeDecay=0.925, episodeNumber=20000, freeAgentPlacement=env_freeAgentPlacement, freeAgentFull=0.0, pheroAutoUpdate=False, tempResources=True)

    beacon = BeaconController(env)
    
    env.render(unloadedPheromones=True)

    for i in range(512):
        beacon.step_all_agents(env)
        env.render()
        time.sleep(TIMESTEP)


    env.close()

