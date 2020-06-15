from __future__ import division

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import threading
import time
import scipy.signal as signal
import os
import GroupLock
import multiprocessing
from MAF_gym.envs.MAF_env import MAF_gym
import pickle
import imageio
from ACNet import ACNet
from MAF_gym.envs.entity import reward

from tensorflow.python.client import device_lib
dev_list = device_lib.list_local_devices()
print(dev_list)
assert len(dev_list) > 1


"""### Helper Functions"""

def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def good_discount(x, gamma):
    return discount(x,gamma)


"""## Worker Agent"""
class Worker:
    def __init__(self, game, metaAgentID, workerID, a_size, groupLock, learning_agent=True, metaAgent_network=None):
        self.workerID = workerID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_"+str(workerID)
        self.agentID = ((workerID-1) % num_workers) + 1
        self.groupLock = groupLock
        self.learning_agent = learning_agent

        self.nextGIF = episode_count # For GIFs output
        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        if learning_agent:
            self.local_AC    = ACNet(self.name,a_size,trainer,True,GRID_SIZE,GLOBAL_NET_SCOPE)
            self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)
        else:
            self.local_AC    = metaAgent_network
            self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, metaAgent_network.scope)

        self.queued = False


    def synchronize(self):
        #handy thing for keeping track of which to release and acquire
        if(not hasattr(self,"lock_bool")):
            self.lock_bool=False
        self.groupLock.release(int(self.lock_bool),self.name)
        self.groupLock.acquire(int(not self.lock_bool),self.name)
        self.lock_bool=not self.lock_bool


    def train(self, rollout, sess, gamma, bootstrap_value, rnn_state0):
        global episode_count

        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        values = rollout[:,5]
        valids = rollout[:,6]
        train_policy = rollout[:,7]
        train_value = rollout[:,-1]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = good_discount(advantages,gamma)

        num_samples = min(EPISODE_SAMPLES,len(advantages))
        sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            global_step:episode_count,
            self.local_AC.target_v:np.stack(discounted_rewards),
            self.local_AC.inputs:np.stack(observations),
            self.local_AC.actions:actions,
            self.local_AC.train_valid:np.stack(valids),
            self.local_AC.advantages:advantages,
            self.local_AC.train_value:train_value,
            self.local_AC.train_policy:train_policy,
            self.local_AC.train_valids:np.vstack(train_policy),
            self.local_AC.state_in[0]:rnn_state0[0],
            self.local_AC.state_in[1]:rnn_state0[1]
        }

        v_l,p_l,valid_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.valid_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l/len(rollout), p_l/len(rollout), valid_l/len(rollout), e_l/len(rollout), g_n, v_n


    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)


    def work(self,max_episode_length,gamma,sess,coord,saver):
        global episode_count, swarm_reward, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops, episode_phero_queues
        total_steps, i_buf = 0, 0
        episode_buffers, s1Values = [ [] for _ in range(NUM_BUFFERS) ], [ [] for _ in range(NUM_BUFFERS) ]

        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                sess.run(self.pull_global)

                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = 0
                d = False
                self.queued = False


                # Initial state from the environment
                if self.agentID==1:
                    swarm_reward[self.metaAgentID] = 0
                    self.env.reset(episode_count)
                self.synchronize() # synchronize starting time of the threads
                validActions           = self.env.listNextValidActions(self.agentID)
                if a_size == 5:
                    validActions       = [a-1 for a in validActions]
                    if len(validActions) == 0: # prevent empty validActions
                        validActions   = [a_size//2]
                else:
                    validActions       = [0] + validActions
                s, _, _                = self.env.observe(self.agentID)
                rnn_state              = self.local_AC.state_init
                rnn_state0             = self.local_AC.state_init
                phero_queues           = []
                max_queue_length       = 0

                self.synchronize() # synchronize starting time of the threads

                saveGIF = False
                if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                    saveGIF = True
                    self.nextGIF = episode_count + 64
                    GIF_episode = int(episode_count)
                    episode_frames = [self.env.render(mode='rgb_array') ]

                while True: # Give me something!

                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,
                                                   self.local_AC.value,
                                                   self.local_AC.state_out],
                                         feed_dict={self.local_AC.inputs:[s[0]],
                                                    self.local_AC.state_in[0]:rnn_state[0],
                                                    self.local_AC.state_in[1]:rnn_state[1]})

                    train_valid = np.zeros(a_size)
                    train_valid[validActions] = 1

                    valid_dist = np.array([a_dist[0,validActions]])
                    valid_dist /= np.sum(valid_dist)

                    a = a_size - 1
                    if TRAINING:
                        if not self.queued:
                            if np.argmax(a_dist.flatten()) not in validActions:
                                episode_inv_count += 1
                                a = validActions[ np.random.choice(range(valid_dist.shape[1])) ]
                                train_val = 0.
                            else:
                                a = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
                                train_val = 1.

                            train_policy = 1.
                        else:
                            train_policy = 0.
                            train_val = 0.
                    else:
                        if not self.queued:
                            a = np.argmax(a_dist.flatten())
                            if a not in validActions or not GREEDY:
                                a = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]

                        train_val = 1.

                    prev_action = self.env.step(self.agentID, a + int(a_size==5))

                    self.synchronize() # synchronize threads after free agents move

                    if self.agentID == 1:
                        self.env.advanceQueues()

                    self.synchronize() # synchronize threads after queues advance

                    # Get common observation for all agents after all individual actions have been performed
                    s1, r, self.queued = self.env.observe(self.agentID)
                    if r > 0:
                        phero_queues.append( np.sum(s1[0][7]) )

                    if not self.queued:
                        validActions = self.env.listNextValidActions(self.agentID, prev_action)
                        if a_size == 5:
                            validActions       = [a-1 for a in validActions]
                            if len(validActions) == 0: # prevent empty validActions
                                validActions   = [a_size//2]
                        else:
                            validActions       = [0] + validActions

                    for rc in self.env.RCs:
                        max_queue_length = max(max_queue_length, len(rc.queue.agents))

                    if saveGIF:
                        episode_frames.append(self.env.render(mode='rgb_array'))

                    episode_buffer.append([s[0],a,r,s1,d,v[0,0],train_valid,train_policy,s[1],train_val])
                    episode_values.append(v[0,0])
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if TRAINING and self.learning_agent and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0):
                        # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                        if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                            episode_buffers[i_buf] = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                        else:
                            episode_buffers[i_buf] = episode_buffer[:]

                        s1Values[i_buf] = sess.run(self.local_AC.value, 
                                 feed_dict={self.local_AC.inputs:np.array([s[0]])
                                            ,self.local_AC.state_in[0]:rnn_state[0]
                                            ,self.local_AC.state_in[1]:rnn_state[1]})[0,0]

                        if (episode_count-EPISODE_START) < NUM_BUFFERS:
                            i_rand = np.random.randint(i_buf+1)
                        else:
                            i_rand = np.random.randint(NUM_BUFFERS)
                            tmp = np.array(episode_buffers[i_rand])
                            while tmp.shape[0] == 0:
                                i_rand = np.random.randint(NUM_BUFFERS)
                                tmp = np.array(episode_buffers[i_rand])
                        v_l,p_l,valid_l,e_l,g_n,v_n = self.train(episode_buffers[i_rand],sess,gamma,s1Values[i_rand],rnn_state0) # b_l

                        i_buf = (i_buf + 1) % NUM_BUFFERS
                        episode_buffers[i_buf] = []
                        rnn_state0 = rnn_state

                    self.synchronize() # synchronize threads
                    if episode_step_count >= max_episode_length:
                        break

                episode_lengths[self.metaAgentID].append(episode_step_count)
                episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
                episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
                if len(phero_queues) > 0:
                    episode_phero_queues[self.metaAgentID].append( np.mean(phero_queues) )

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                    print('                                                                                   ', end='\r')
                    print('{} Episode terminated ({})'.format(episode_count, self.agentID), end='\r')

                swarm_reward[self.metaAgentID] += episode_reward

                self.synchronize() # synchronize threads

                episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])

                if not TRAINING:
                    mutex.acquire()
                    if episode_count < NUM_EXPS:
                        plan_durations[episode_count] = episode_step_count
                    if self.workerID == 1:
                        episode_count += 1
                        print('({}) Thread {}: {} steps, {:.2f} reward ({} invalids).'.format(episode_count, self.workerID, episode_step_count, episode_reward, episode_inv_count))
                    GIF_episode = int(episode_count)
                    mutex.release()
                    
                elif self.learning_agent:
                    episode_count+=1./NUM_ACTIVE

                    if episode_count % SUMMARY_WINDOW == 0:
                        if episode_count % 100 == 0:
                            print ('Saving Model', end='\n')
                            saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                            print ('Saved Model', end='\n')
                        SL = SUMMARY_WINDOW * NUM_ACTIVE
                        mean_reward = np.nanmean(episode_rewards[self.metaAgentID][-SL:])
                        mean_length = np.nanmean(episode_lengths[self.metaAgentID][-SL:])
                        mean_value = np.nanmean(episode_mean_values[self.metaAgentID][-SL:])
                        mean_invalid = np.nanmean(episode_invalid_ops[self.metaAgentID][-SL:])
                        mean_phero_queues = np.nanmean(episode_phero_queues[self.metaAgentID][-SL:])

                        current_learning_rate = sess.run(lr,feed_dict={global_step:episode_count})

                        summary = tf.Summary()

                        summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                        summary.value.add(tag='Perf/Length', simple_value=mean_length)
                        summary.value.add(tag='Perf/Valid Rate', simple_value=(mean_length-mean_invalid)/mean_length)
                        summary.value.add(tag='Perf/Pheromones around Queues', simple_value=mean_phero_queues)
                        summary.value.add(tag='Perf/Max Queue Length', simple_value=max_queue_length)

                        summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                        summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                        summary.value.add(tag='Losses/Valid Loss', simple_value=valid_l)
                        summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                        summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                        global_summary.add_summary(summary, int(episode_count))

                        global_summary.flush()

                        if printQ:
                            print('{} Tensorboard updated ({})'.format(episode_count, self.workerID), end='\r')

                if saveGIF:
                    # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                    time_per_step = 0.1
                    images = np.array(episode_frames)
                    if TRAINING:
                        make_gif(images, '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode,episode_step_count,swarm_reward[self.metaAgentID]))
                    else:
                        make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path,GIF_episode,episode_step_count), duration=len(images)*time_per_step,true_image=True,salience=False)
                if SAVE_EPISODE_BUFFER:
                    with open('gifs3D/episode_{}.dat'.format(GIF_episode), 'wb') as file:
                        pickle.dump(episode_buffer, file)


"""## Training"""

# Learning parameters
max_episode_length     = 512
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = max_episode_length
GRID_SIZE              = 11 #the size of the FOV grid to apply to each agent
ENVIRONMENT_SHAPE      = [(20,96), (20,96), (40,96), (64,128)] # the total size of the environment (length of one side)
OBSTACLE_DENSITY       = (0., 0.05) # average density (no randomization)
DIAG_MVMT              = False # Diagonal movements allowed?
phero_time_decay       = 0.99
phero_action_decay     = 0.97
TEMP_RESOURCES         = False
DONUT_SPACING          = False

a_size                 = 5 # put 6 to allow taking the IDLE action
SUMMARY_WINDOW         = 10
NUM_META_AGENTS        = 4
NUM_ACTIVE             = 8    #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
NUM_PASSIVE            = [8, 16, 24, 40]
NUM_BUFFERS            = 1 # NO EXPERIENCE REPLAY int(NUM_ACTIVE / 2)
EPISODE_SAMPLES        = EXPERIENCE_BUFFER_SIZE # 64
LR_Q                   = 5.e-6 # / (EXPERIENCE_BUFFER_SIZE / 128) # default (conservative, for 256 steps): 1e-5
ADAPT_LR               = False
ADAPT_COEFF            = 5.e-5 #the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
load_model             = False
RESET_TRAINER          = False
model_path             = 'model_ForMIC'
gifs_path              = 'gifs_ForMIC'
train_path             = 'train_ForMIC'
GLOBAL_NET_SCOPE       = 'global'

# Simulation options
FULL_HELP              = False
OUTPUT_GIFS            = True
SAVE_EPISODE_BUFFER    = False

# Testing
TRAINING               = True
GREEDY                 = False
NUM_EXPS               = 100
MODEL_NUMBER           = 313000

# Shared arrays for tensorboard
episode_rewards        = [ [] for _ in range(NUM_META_AGENTS) ]
episode_lengths        = [ [] for _ in range(NUM_META_AGENTS) ]
episode_mean_values    = [ [] for _ in range(NUM_META_AGENTS) ]
episode_invalid_ops    = [ [] for _ in range(NUM_META_AGENTS) ]
episode_phero_queues   = [ [] for _ in range(NUM_META_AGENTS) ]
rollouts               = [ None for _ in range(NUM_META_AGENTS)]
demon_probs=[np.random.rand() for _ in range(NUM_META_AGENTS)]
printQ                 = False # (for headless)
swarm_reward           = [0]*NUM_META_AGENTS

tf.reset_default_graph()
print("Hello World")
if not os.path.exists(model_path):
    os.makedirs(model_path)
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True

if not TRAINING:
    plan_durations = np.array([0 for _ in range(NUM_EXPS)])
    mutex = threading.Lock()
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        os.makedirs('gifs3D')

#Create a directory to save episode playback gifs to
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

with tf.device("/gpu:0"):
    master_network = ACNet(GLOBAL_NET_SCOPE,a_size,None,False,GRID_SIZE,GLOBAL_NET_SCOPE) # Generate global network
    
    global_step = tf.placeholder(tf.float32)
    if ADAPT_LR:
        #computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        #we need the +1 so that lr at step 0 is defined
        lr=tf.divide(tf.constant(LR_Q),tf.sqrt(tf.add(1.,tf.multiply(tf.constant(ADAPT_COEFF),global_step))))
    else:
        lr=tf.constant(LR_Q)
    trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

    if not TRAINING:
        NUM_META_AGENTS = 1

    gameEnvs, workers, groupLocks = [], [], []
    ma_networks = []
    n=1#counter of total number of agents (for naming)
    for ma in range(NUM_META_AGENTS):
        num_workers = NUM_ACTIVE + NUM_PASSIVE[ma]
        ma_net = ACNet("ma_"+str(ma),a_size,None,False,GRID_SIZE,GLOBAL_NET_SCOPE) # Meta-Agent network for passive (non-learning) agents
        ma_networks.append(ma_net)
        
        gameEnv = MAF_gym(numAgents=num_workers, shape=ENVIRONMENT_SHAPE[ma], observationSize=GRID_SIZE, pheroActionDecay=phero_action_decay, pheroTimeDecay=phero_time_decay, no_agents_queued=num_workers, freeAgentPlacement="nearCache", freeAgentFull=0.0, obstacleRatio=OBSTACLE_DENSITY, tempResources=TEMP_RESOURCES, donutSpacing=DONUT_SPACING)
        gameEnvs.append(gameEnv)

        # Create groupLock
        workerNames = ["worker_"+str(i) for i in range(n,n+num_workers)]
        groupLock = GroupLock.GroupLock([workerNames,workerNames])
        groupLocks.append(groupLock)

        # Create worker classes
        workersTmp = []
        # Learning agents
        for i in range(ma*num_workers+1,ma*num_workers+1+NUM_ACTIVE):
            workersTmp.append(Worker(gameEnv,ma,n,a_size,groupLock,True))
            n+=1
        # Non-learning agents
        for i in range(ma*num_workers+1+NUM_ACTIVE,(ma+1)*num_workers+1):
            workersTmp.append(Worker(gameEnv,ma,n,a_size,groupLock,False,ma_net))
            n+=1
        workers.append(workersTmp)

    global_summary = tf.summary.FileWriter(train_path)
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            if not TRAINING:
                with open(model_path+'/checkpoint', 'w') as file:
                    file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
                    file.close()
            ckpt = tf.train.get_checkpoint_state(model_path)
            p=ckpt.model_checkpoint_path
            p=p[p.find('-')+1:]
            p=p[:p.find('.')]
            episode_count=int(p)
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("episode_count set to ",episode_count)
            if RESET_TRAINER:
                trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for ma in range(NUM_META_AGENTS):
            for worker in workers[ma]:
                groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
                worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
                print("Starting worker " + str(worker.workerID))
                t = threading.Thread(target=(worker_work))
                t.start()
                worker_threads.append(t)
        coord.join(worker_threads)

if not TRAINING:
    print([np.mean(plan_durations), np.sqrt(np.var(plan_durations)), np.mean(np.asarray(plan_durations < max_episode_length, dtype=float))])

