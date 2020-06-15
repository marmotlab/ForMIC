import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
#parameters for training
GRAD_CLIP              = 64.0
KEEP_PROB1             = 1
KEEP_PROB2             = 1
RNN_SIZE               = 512


#Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class ACNet:
    def __init__(self, scope, a_size, trainer,TRAINING,GRID_SIZE,GLOBAL_NET_SCOPE):
        self.scope = scope
        with tf.variable_scope(str(scope)+'/qvalues'):
            #The input size may require more work to fit the interface.
            self.inputs = tf.placeholder(shape=[None,8,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.myinput = tf.transpose(self.inputs, perm=[0,2,3,1])
            self.policy, self.value, self.state_out, self.state_in, self.state_init, self.valids = self._build_net(self.myinput, RNN_SIZE,TRAINING,a_size)
        if TRAINING:
            self.actions                = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot         = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.train_valid            = tf.placeholder(shape=[None,a_size], dtype=tf.float32)
            self.target_v               = tf.placeholder(tf.float32, [None], 'Vtarget')
            self.advantages             = tf.placeholder(shape=[None], dtype=tf.float32)
            self.responsible_outputs    = tf.reduce_sum(self.policy * self.actions_onehot, [1])
            self.train_value            = tf.placeholder(tf.float32, [None])
            self.train_policy           = tf.placeholder(tf.float32, [None])
            self.train_valids           = tf.placeholder(tf.float32, [None,1])

            # Loss Functions
            self.value_loss     = 0.5 * tf.reduce_sum(self.train_value*tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))
            self.entropy        = - 0.01 * tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy,1e-10,1.0)))
            self.policy_loss    = - tf.reduce_sum(self.train_policy * tf.log(tf.clip_by_value(self.responsible_outputs,1e-15,1.0)) * self.advantages)
            self.valid_loss     = - 0.001 * tf.reduce_sum(self.train_valids * tf.reduce_sum(tf.log(tf.clip_by_value(self.valids,1e-10,1.0)) * self.train_valid + \
                                   tf.log(tf.clip_by_value(1-self.valids,1e-10,1.0)) * (1-self.train_valid), axis=1))
            
            self.loss           = self.value_loss + self.policy_loss - self.entropy + self.valid_loss


            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            local_vars         = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/qvalues')
            self.gradients     = tf.gradients(self.loss, local_vars)
            self.var_norms     = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)

            # Apply local gradients to global network
            global_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/qvalues')
            self.apply_grads   = trainer.apply_gradients(zip(grads, global_vars))
        print("Hello World... From  "+str(scope))     # :)


    def _build_net(self,inputs,RNN_SIZE,TRAINING,a_size):
        w_init   = layers.variance_scaling_initializer()
        
        conv1    =  layers.conv2d(inputs=inputs,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
        conv1a   =  layers.conv2d(inputs=conv1,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
        conv1b   =  layers.conv2d(inputs=conv1a,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu) 
        pool1    =  layers.max_pool2d(inputs=conv1b,kernel_size=[2,2])
        conv2    =  layers.conv2d(inputs=pool1,    padding="SAME", num_outputs=RNN_SIZE//2, kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
        conv2a   =  layers.conv2d(inputs=conv2,    padding="SAME", num_outputs=RNN_SIZE//2, kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
        conv2b   =  layers.conv2d(inputs=conv2a,    padding="SAME", num_outputs=RNN_SIZE//2, kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
        pool2    =  layers.max_pool2d(inputs=conv2b,kernel_size=[2,2])
        conv3    =  layers.conv2d(inputs=pool2,    padding="VALID", num_outputs=RNN_SIZE, kernel_size=[2, 2],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=None)

        flat     = tf.nn.relu(layers.flatten(conv3))
        h1 = layers.fully_connected(inputs=flat,  num_outputs=RNN_SIZE)
        d1 = layers.dropout(h1, keep_prob=KEEP_PROB1, is_training=TRAINING)
        h2 = layers.fully_connected(inputs=d1,  num_outputs=RNN_SIZE, activation_fn=None)
        d2 = layers.dropout(h2, keep_prob=KEEP_PROB2, is_training=TRAINING)
        self.h3 = tf.nn.relu(d2+flat)
        #Recurrent network for temporal dependencies
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE,state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(self.h3, [0])
        step_size = tf.shape(inputs)[:1]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
        time_major=False)
        lstm_c, lstm_h = lstm_state
        state_out = (lstm_c[:1, :], lstm_h[:1, :])
        self.rnn_out = tf.reshape(lstm_outputs, [-1, RNN_SIZE])

        policy_layer = layers.fully_connected(inputs=self.rnn_out, num_outputs=a_size,weights_initializer=normalized_columns_initializer(1./float(a_size)), biases_initializer=None, activation_fn=None)
        policy       = tf.nn.softmax(policy_layer)
        policy_sig   = tf.sigmoid(policy_layer)
        value        = layers.fully_connected(inputs=self.rnn_out, num_outputs=1, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=None)

        return policy, value, state_out ,state_in, state_init, policy_sig
