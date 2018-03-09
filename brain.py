import numpy as np
import tensorflow as tf
import threading, time

import keras
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D
from keras.models import Model, model_from_json

LEARNING_RATE = 5e-3
DECAY         = 0.99

LOSS_V        = 1   # v loss coefficient
LOSS_ENTROPY  = 0  # entropy coefficient
MIN_BATCH     = 32

class Brain:

    def __init__(self, state_shape, n_actions, gamma, load_path=None):
        self.state_shape = state_shape
        self.n_actions   = n_actions
        self.gamma       = gamma
        
        self.session = tf.Session()
        if load_path == None:       
            self.model = self._build_model()
            self._build_graph(self.model)
            self.session.run(tf.global_variables_initializer())
            self.model._make_predict_function() # Need to call before threading
            self.default_graph = tf.get_default_graph()
            self.saver = tf.train.Saver()

        else:
            self.saver = tf.train.import_meta_graph(load_path)
            self.saver.restore(self.session,tf.train.latest_checkpoint('models/'))
            self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()
        
        self.writer = tf.summary.FileWriter("tmp/log/", self.session.graph)
        

        self.train_queue = [ [], [], [] ,[], [], [] ] # s, a ,r ,s_, mask, total_r
        self.lock_queue  = threading.Lock() 


    def _build_model(self):
        keras.backend.set_session(self.session)
        keras.backend.manual_variable_initialization(True)

        l_input = Input(shape=self.state_shape)

        # Convolution
        # l_conv_1 = Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation='relu')(l_input)
        # l_conv_2 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(l_conv_1)
        # l_conv_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(l_conv_2)
        # l_flatten = Flatten()(l_conv_3)
        # l_final = Dense(512,activation='relu')(l_flatten)

        # MLP
        if len(self.state_shape)>=2:
            l_flatten = Flatten()(l_input)
            l_dense = Dense(16, activation='relu')(l_flatten)
        else:
            l_dense = Dense(16, activation='relu')(l_input)
        l_final = Dense(16,activation='relu')(l_dense)

        out_actions = Dense(self.n_actions, activation='softmax', name='output_actions')(l_final)
        out_value   = Dense(1, activation='linear', name='output_value')(l_final)

        model = Model(inputs=[l_input], outputs=[out_actions,out_value])
        return model

    def _build_graph(self,model):
        assert len(self.state_shape)<=3
        if len(self.state_shape)==1:
            s_t = tf.placeholder(tf.float32, shape=(None, self.state_shape[0]), name='s_t')
        elif len(self.state_shape)==2:
            s_t = tf.placeholder(tf.float32, shape=(None, self.state_shape[0], self.state_shape[1]), name='s_t')
        else:
            s_t = tf.placeholder(tf.float32, shape=(None, self.state_shape[0], self.state_shape[1], self.state_shape[2]), name='s_t')
        a_t = tf.placeholder(tf.float32, shape=(None, self.n_actions), name='a_t')
        r_t = tf.placeholder(tf.float32, shape=(None,1), name='r_t')

        # To count and remember number of frames
        batch_len = tf.placeholder(tf.int32, shape=(), name='batch_len')
        global_step = tf.Variable(0, trainable=False, name='global_step')
        assign_step = tf.assign(global_step, global_step + batch_len, name='assign_step')

        # To compute and log the average total reward
        total_r_t = tf.placeholder(tf.float32, shape=(None,1), name='total_r_t')
        total_r = tf.reduce_mean(total_r_t)
        tf.summary.scalar("total_r", total_r)

        n_step_reward = tf.reduce_mean(r_t)
        tf.summary.scalar("n_step_reward", n_step_reward)

        p, v = model(s_t)

        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)

        advantage = r_t - v
        loss_policy = - log_prob * tf.stop_gradient(advantage)  
        loss_policy_summary = tf.summary.scalar("loss_policy", tf.reduce_mean(loss_policy)) # maximize policy

        loss_value  = LOSS_V * tf.square(advantage)                         # minimize value error
        loss_value_summary = tf.summary.scalar("loss_value", tf.reduce_mean(loss_value))

        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)   # maximize entropy (regularization)
        entropy_summary = tf.summary.scalar("entropy", tf.reduce_mean(entropy))

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        loss_summary = tf.summary.scalar("loss", loss_total)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=DECAY, name='optimizer')
        minimize = optimizer.minimize(loss_total, name='minimize')

        return 

    def train_push(self, s, a, r, s_, done, total_r):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)
            self.train_queue[3].append(s_)
            if done:
                self.train_queue[4].append(0)
            else:
                self.train_queue[4].append(1)
            self.train_queue[5].append(total_r)

    def predict_p(self, s):
        with self.default_graph.as_default():
            output_actions = self.default_graph.get_operation_by_name("output_actions/Softmax").outputs[0]
            feed_dict = {'input_1:0':s}
            p = self.session.run(output_actions, feed_dict)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            output_value = self.default_graph.get_operation_by_name("output_value/BiasAdd").outputs[0]
            feed_dict = {'input_1:0':s}
            v = self.session.run(output_value, feed_dict)
            return v

    def optimize(self):
        global STEP

        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0) # Yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:    # more thread could have passed without lock
                return                                  # we can't yield inside lock

            s, a, r, s_, mask, total_r = self.train_queue
            self.train_queue  = [ [], [], [], [], [], [] ]

        s    = np.array(s)
        a    = np.array(a)
        r    = np.array(r)
        s_   = np.array(s_)
        mask = np.array(mask)
        total_r = np.array(total_r)

        # Keep total_r only if state is done (game is over)
        total_r = np.expand_dims(total_r[mask==0], axis=1)

        if len(s) > 2*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        v = np.reshape(v, len(v))
        r+= self.gamma * v * mask
        
        s_t = self.default_graph.get_operation_by_name('s_t').outputs[0]
        a_t = self.default_graph.get_operation_by_name('a_t').outputs[0]
        r_t = self.default_graph.get_operation_by_name('r_t').outputs[0]
        batch_len = self.default_graph.get_operation_by_name('batch_len').outputs[0]
        total_r_t = self.default_graph.get_operation_by_name('total_r_t').outputs[0]

        minimize = self.default_graph.get_operation_by_name('minimize')
        get_reward = self.default_graph.get_operation_by_name('n_step_reward').outputs[0]
        get_loss = self.default_graph.get_operation_by_name('loss').outputs[0]
        get_loss_policy = self.default_graph.get_operation_by_name('loss_policy').outputs[0]
        get_loss_value = self.default_graph.get_operation_by_name('loss_value').outputs[0]
        get_entropy = self.default_graph.get_operation_by_name('entropy').outputs[0]
        get_total_r = self.default_graph.get_operation_by_name('total_r').outputs[0]
        assign_step = self.default_graph.get_operation_by_name('assign_step')
        
        get_step = self.default_graph.get_tensor_by_name('global_step:0')

        r = np.expand_dims(r,axis=1)

        _, _, step, n_step_reward, loss, loss_policy, loss_value, entropy, mean_total_r = self.session.run([minimize, assign_step, get_step, get_reward, get_loss, get_loss_policy, get_loss_value, get_entropy, get_total_r], feed_dict={s_t: s, a_t: a, r_t: r, batch_len: len(s), total_r_t: total_r})

        self.writer.add_summary(n_step_reward, step)
        self.writer.add_summary(loss, step)
        self.writer.add_summary(loss_policy, step)
        self.writer.add_summary(loss_value, step)
        self.writer.add_summary(entropy, step)
        if(len(total_r)>0):
            self.writer.add_summary(mean_total_r, step)

    
        return


    def save(self, path):
        self.writer.close()
        save_path = self.saver.save(self.session, path)
        return
