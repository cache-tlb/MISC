#!/usr/bin/env python
# from __future__ import print_function

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import cPickle
import numpy as np
from collections import deque
import heapq
import pdb
import json

from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras import initializations

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
# OBSERVE = 100
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

class ReplayMemory():
    def __init__(self, max_replay):
        # self.D = deque()
        self.heap = []
        self.max_replay = max_replay
        self.era = 0
    
    def get_minibatch(self, batch_size, gamma, model):
        # sample a minibatch to train on
        sample_size = 3*batch_size
        # samplebatch = random.sample(self.D, sample_size)
        sample_heap_items = random.sample(self.heap, sample_size)
        samplebatch = [s[2] for s in sample_heap_items]

        # get the batch variables
        s_j_batch = [d[0] for d in samplebatch]
        a_batch = [d[1] for d in samplebatch]
        r_batch = [d[2] for d in samplebatch]
        s_j1_batch = [d[3] for d in samplebatch]
        terminal = [d[4] for d in samplebatch]

        sample_inputs = np.zeros((sample_size, 4, 80, 80))
        sample_inputs_sj1 = np.zeros((sample_size, 4, 80, 80))
        for i in range(sample_size):
            sample_inputs[i,:] = s_j_batch[i]
            sample_inputs_sj1[i,:] = s_j1_batch[i]
        sample_targets = model.predict(sample_inputs)
        sample_targets_sj1 = model.predict(sample_inputs_sj1)
        err = [None]*sample_size
        for i in range(sample_size):
            Q_sa = np.max(sample_targets_sj1[i,:])
            r = r_batch[i]
            if not terminal[i]:
                r += gamma*Q_sa
            st = sample_targets[i,:]*(1 - a_batch[i]) + r*a_batch[i]
            e = np.fabs(r - sample_targets[i,:]).dot(a_batch[i])
            sample_targets[i,:] = st
            err[i] = (e, i)
        err.sort(reverse = True, key=lambda x:x[0])

        inputs = np.zeros((batch_size, 4, 80, 80))
        targets = np.zeros((batch_size, ACTIONS))
        for i in range(batch_size):
            idx = err[i][1]
            inputs[i,:] = sample_inputs[idx, :]
            targets[i,:] = sample_targets[idx, :]

        return inputs, targets

    def add_replay(self, replay, age):
        # (s_t, a_t, r_t, s_t1, terminal)
        self.era += 1
        priorty = self.era*0.01 + age
        # Note: add era to make the keys before replay unique
        heapq.heappush(self.heap, (priorty, self.era, replay))
        if len(self.heap) > self.max_replay:
            heapq.heappop(self.heap)
        # self.D.append(replay)
        # if len(self.D) > self.max_replay:
            # self.D.popleft()

def my_init_w(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def my_init_b(shape, name=None):
    return initializations.one(shape, name=name)*0.01

def ConvLayer(width, height, nb_filter, stride, name, input_shape = None):
    # init_w = uniform(shape=(nb_filter, height, width), scale=0.01)
    # init_b = one(shape=(nb_filter))*0.01
    if input_shape is None:
        return Conv2D(nb_filter, height, width, border_mode='same', subsample=(stride,stride), name=name, init=my_init_w)
    else:
        return Conv2D(nb_filter, height, width, border_mode='same', subsample=(stride,stride), name=name, init=my_init_w, input_shape=input_shape)

def Max_pool_2x2():
    return MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same')

def build_model():
    model = Sequential()
    model.add(ConvLayer(8, 8, 32, 4, 'conv1', input_shape=(4, 80, 80)))
    model.add(Activation('relu'))                 # h_conv1
    model.add(Max_pool_2x2())

    model.add(ConvLayer(4, 4, 64, 2, 'conv2'))
    model.add(Activation('relu'))                 # h_conv2
    # model.add(Max_pool_2x2())

    model.add(ConvLayer(3, 3, 64, 1, 'conv3'))
    model.add(Activation('relu'))                 # h_conv3

    model.add(Flatten())                          # h_conv3_flat

    model.add(Dense(512, input_dim=1600, name='fc1'))
    model.add(Activation('relu'))                 # h_fc1
    model.add(Dense(ACTIONS, input_dim=512, name='fc2'))      # readout

    model.compile(Adam(lr=1e-6), "mse")
    return model


def train_model(model):
    # store the previous observations in replay memory
    display_interval = 1000
    replay_memory = ReplayMemory(REPLAY_MEMORY)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    age = 0
    while t < EXPLORE:
        # choose an action epsilon greedily
        if (t == OBSERVE):
            print "pure observe done"
            # cPickle.dump(replay_memory, open("replay.pkl", "wb"))
        predict_batch = s_t[np.newaxis,:]
        readout_t = model.predict(predict_batch)[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon or t < OBSERVE:
                # print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
            else:
                action_index = np.argmax(readout_t)
        else:
            action_index = 0 # do nothing
        if t < OBSERVE and t % 7 !=0:
            action_index = 0
        a_t[action_index] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (1, 80, 80))
        s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)

        # store the transition in D
        replay_memory.add_replay((s_t, a_t, r_t, s_t1, terminal), age)
        # only train if done observing
        if t > OBSERVE:
            inputs, targets = replay_memory.get_minibatch(BATCH, GAMMA, model)
            loss = model.train_on_batch(inputs, targets)
            if t % display_interval == 0:
                print "iter:", t, "loss:", loss

        # update the old values
        s_t = s_t1
        t += 1
        if terminal:
            age = 0
        else:
            age += 1

def load_model(name):
    with open('%s.json' % (name,), 'r') as inputfile:
        json_string = json.load(inputfile)
    model = model_from_json(json_string)
    model.load_weights('%s.h5' % (name,))
    model.compile(Adam(lr=1e-6), 'mse')
    return model

def train():
    if True:
        model = load_model('model')
    else:
        model = build_model()
    # layer_dict = dict([(layer.name, layer) for layer in model.layers])
    train_model(model)
    try:
        model.save_weights("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)
    except Exception, e:
        pdb.set_trace()

def test():
    model = load_model('model')
    game_state = game.GameState()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    t = 0
    while True:
        predict_batch = s_t[np.newaxis,:]
        readout_t = model.predict(predict_batch)[0]
        a_t = np.zeros([ACTIONS])
        action_index = np.argmax(readout_t)
        q_value = np.max(readout_t)
        print q_value
        a_t[action_index] = 1

        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (1, 80, 80))
        s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)

        s_t = s_t1
        t += 1
        if terminal:
            break

if __name__ == "__main__":
    # train()
    test()
