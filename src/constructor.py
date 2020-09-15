#/usr/bin/env python

import numpy as np
from math import log2
import sys
import pickle
import os.path
import pandas as pd
from os import path
from oracle import TFTableCY101 
class State(object):

    def __init__(self, term, s_index, prop_values):
        self._term = term
        self._s_index = s_index
        self._prop_values = prop_values
        self._name = self.prop_values_to_str()

    def prop_values_to_str(self): 
        if self._term is True:
            return 'terminal'
        else:
            return 's'+str(self._s_index)+'p'+''.join(self._prop_values)

class Action(object):

    def __init__(self, term, name, prop_values):
        self._term = term

        if term == False:
            self._prop_values = None
            self._name = name
        else:
            self._prop_values = prop_values
            self._name = 'a'+''.join(prop_values)


class Obs(object):

    def __init__(self, nonavail, prop_values):
        self._nonavail = nonavail
        if nonavail == False:
            self._prop_values = prop_values
            self._name = 'p'+''.join(prop_values)
        else:
            self._prop_values = None
            self._name = 'na'


class Model:

    def __init__(self, discount, prop_names, high_acc, ask_cost, classifier_nameIn): 
        self._discount = discount
        self._num_comp_states = 5
        self._prop_names = prop_names
        self._high = high_acc
        self._ask_cost = ask_cost

        self._states = []
        self._actions = []
        self._observations = []

        self._classifiers = None

        self.generate_state_set()
        self.generate_action_set()
        self.generate_observation_set()

        self._trans = np.zeros((len(self._actions), len(self._states), len(self._states)))
        self._obs_fun = np.zeros((len(self._actions), len(self._states), len(self._observations)))
        self._reward_fun = np.zeros((len(self._actions), len(self._states)))

        self.generate_trans_fun()
        # self.load_confusion_matrix('../data/icra2014/confusion_matrices_train5.csv')
        self.generate_obs_fun(classifier_nameIn)
        self.generate_reward_fun()


    def generate_state_set(self):

        for i in range(self._num_comp_states):
            self.generate_state_set_helper(i, 0, [], len(self._prop_names))

        self._states.append(State(True, None, None))

        # print(str([s._name for s in self._states]))
        # exit(1)

    def generate_state_set_helper(self, s_index, curr_depth, path, depth):
        
        # print('s_index: ' + str(s_index))
        # print('curr_depth: ' + str(curr_depth))
        # print('path: ' + str(path))
        # print('depth: ' + str(depth))
        # print
        if len(path) == depth:
            self._states.append(State(False, s_index, path))
            return

        self.generate_state_set_helper(s_index, curr_depth+1, path+['0'], depth)
        self.generate_state_set_helper(s_index, curr_depth+1, path+['1'], depth)

    def get_state_index(self, term, s_index, prop_values):
        if term == True:
            return len(self._states) - 1

        else:
            return s_index*pow(2, len(prop_values)) + int(''.join(prop_values), 2)

    def get_action_name(self, a_index):
        for a_idx, a_val in enumerate(self._actions):
            if a_index == a_idx:
                return a_val._name
        else:
            return ""


    def generate_action_set(self):

        # the action names must match the action names in confusion matrices (csv file)
        self._actions.append(Action(False, 'look', None))   #0
        self._actions.append(Action(False, 'grasp', None))  #1
        self._actions.append(Action(False, 'lift_slow', None))   #2
        self._actions.append(Action(False, 'hold', None))   #3
        self._actions.append(Action(False, 'shake', None))   #4
        self._actions.append(Action(False, 'low_drop', None))  #5
        self._actions.append(Action(False, 'tap', None))   #6
        self._actions.append(Action(False, 'push', None))   #7
        self._actions.append(Action(False, 'poke', None))  #8
        self._actions.append(Action(False, 'crush', None))  #9
        self._actions.append(Action(False, 'ask', None))    #10
        self._actions.append(Action(False, 'reinit', None)) #11

        self.generate_action_set_helper(0, [], len(self._prop_names))

        # print(str([a._name for a in self._actions]))
        # exit(1)

    def generate_action_set_helper(self, curr_depth, path, depth):
        if len(path) == depth:
            self._actions.append(Action(True, None, path))
            return

        self.generate_action_set_helper(curr_depth+1, path+['0'], depth)
        self.generate_action_set_helper(curr_depth+1, path+['1'], depth)

    def generate_observation_set(self):


        self.generate_observation_set_helper(0, [], len(self._prop_names))
        self._observations.append(Obs(True, None))

        # print(str([o._name for o in self._observations]))
        # exit(1)

    def generate_observation_set_helper(self, curr_depth, path, depth):
        if len(path) == depth:
            self._observations.append(Obs(False, path))
            return

        self.generate_observation_set_helper(curr_depth+1, path+['0'], depth)
        self.generate_observation_set_helper(curr_depth+1, path+['1'], depth)

    def generate_trans_fun(self):

        # going through all actions based on their names
        for a_idx, a_val in enumerate(self._actions):

            term_idx = self.get_state_index(True, None, None)

            # action of look deterministically leads to state s1, from initial state s0
            if a_val._name == 'look':
                success_rate = 0.95
                for s_idx, s_val in enumerate(self._states):

                    if s_val._term == False and s_val._s_index == 0:
                        tmp_s_idx = self.get_state_index(False, 1, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_rate
                        self._trans[a_idx, s_idx, s_idx] = 1 - success_rate
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0

            # action 'ask' can be executed in any state: a robot can always ask a question anytime
            elif a_val._name == 'ask':
                for s_idx, s_val in enumerate(self._states):
                    self._trans[a_idx, s_idx, s_idx] = 1.0

            # after action 'push', one has to reinitialize the system
            elif a_val._name == 'push':
                success_rate = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = self.get_state_index(False, 4, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_rate
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_rate
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0
            
            elif a_val._name == 'tap':
                success_rate = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = self.get_state_index(False, 4, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_rate
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_rate
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0  
            
            elif a_val._name == 'poke':
                success_rate = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = self.get_state_index(False, 4, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_rate
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_rate
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0

            elif a_val._name == 'crush':
                success_rate = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = self.get_state_index(False, 4, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_rate
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_rate
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0              
            
            # most likely one ends up with s2. With small probability, one remains in s1
            elif a_val._name == 'grasp':
                success_rate = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = self.get_state_index(False, 2, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_rate
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_rate
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0

            # most likely one ends up with s3. With small probability, one ends up with s5
            elif a_val._name == 'lift_slow':
                success_rate = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 2:
                        tmp_s_idx = self.get_state_index(False, 3, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_rate
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_rate
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0

            # most likely one can 'hold' many times. rarely, one ends up with s5 and has to reinit the system
            elif a_val._name == 'hold':
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 3:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0
            
            elif a_val._name == 'shake':
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 3:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0

            elif a_val._name == 'low_drop':
                success_rate = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 3:
                        tmp_s_idx = self.get_state_index(False, 4, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_rate
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_rate
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0

            elif a_val._name == 'reinit':
                success_rate = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 4:
                        tmp_s_idx = self.get_state_index(False, 0, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_rate
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_rate
                    else:
                        self._trans[a_idx, s_idx, term_idx] = 1.0

            elif a_val._term == True:
                for s_idx, s_val in enumerate(self._states):
                    self._trans[a_idx, s_idx, len(self._states)-1] = 1.0
    
        
    def load_classifier(self, path):
        # where to load the classifier
        classifier_file_name = path

        # load classifier
        pkl_load_classifier_file = open(classifier_file_name, 'rb')
        classifier = pickle.load(pkl_load_classifier_file)
        
        self._classifiers = classifier
        #print (self._classifiers[test_object_index])
        print("Classifier loading done")

    
    def load_confusion_matrix(self, path):
        f = open(path, 'r')
        lines = f.readlines()[1:]
        self.dic = {}
        for l in lines:
            words = l.split(',')
            if words[1] in self.dic:
                self.dic[words[1]][words[0]] = [int(w)+1 for w in words[2:]]
            else:
                self.dic[words[1]] = {words[0]: [int(w)+1 for w in words[2:]]}

    def generate_obs_fun(self, path):

        self.load_classifier(path)

        # for a_idx, a_val in enumerate(self._actions):
        #     for s_idx, s_val in enumerate(self._states):
        #         self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0

        for a_idx, a_val in enumerate(self._actions):
            for s_idx, s_val in enumerate(self._states):

                if a_val._term == True or a_val._name == 'reinit' or s_val._term == True:
                    self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                    continue

                for o_idx, o_val in enumerate(self._observations):

                    prob = 1.0
                    if o_val._nonavail == True:
                        # self._obs_fun[a_idx, s_idx, o_idx] = prob
                        continue 

                    if a_val._name == 'ask':
                        if s_val._prop_values == o_val._prop_values:
                            self._obs_fun[a_idx, s_idx, o_idx] = self._high
                        else:
                            self._obs_fun[a_idx, s_idx, o_idx] = \
                            (1.0 - self._high)/(len(self._observations)-2.0)
                        continue 

                    # actions of 'look' and 'press' only make sense when it is taken in state s0 (init state)
                    # otherwise, it won't produce any information
                    if a_val._name == 'look':
                        if s_val._s_index != 1:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'push':
                        if s_val._s_index != 4:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'grasp':
                        if s_val._s_index != 2:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'lift_slow':
                        if s_val._s_index != 3:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'hold':
                        if s_val._s_index != 3:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'shake':
                        if s_val._s_index != 3:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'low_drop':
                        if s_val._s_index != 4:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'tap':
                        if s_val._s_index != 4:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'poke':
                        if s_val._s_index != 4:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'crush':
                        if s_val._s_index != 4:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    #print(s_val._prop_values)
                    for p_s_idx, p_s_val in enumerate(s_val._prop_values):
                        #print(p_s_idx)
                        #print(p_s_val)
                        p_o_val = o_val._prop_values[p_s_idx]

                        # mat = self.dic[a_val._name][self._prop_names[p_s_idx]]
                        mat = self._classifiers.getPredicateBehaviorObservatoinModel(self._prop_names[p_s_idx], a_val._name)
                        #if (type(mat) == int):
                        #    prob = 0.5
                        #    break
                        
                       
                        positives = mat[1][0] + mat[1][1]
                        negatives = mat[0][0] + mat[0][1]
                    
                        #predicated results have only one label: set uniform probabilities

                        if negatives == 0 or positives == 0:
                            prob = 0.5
                            break


                        if p_s_val == '0' and p_o_val == '0':
                            prob = prob * mat[0][0] / (mat[0][0] + mat[1][0])
                        elif p_s_val == '0' and p_o_val == '1':
                            prob = prob * mat[1][0] / (mat[0][0] + mat[1][0])
                        elif p_s_val == '1' and p_o_val == '0':
                            prob = prob * mat[0][1] / (mat[0][1] + mat[1][1])
                        elif p_s_val == '1' and p_o_val == '1':
                            prob = prob * mat[1][1] / (mat[0][1] + mat[1][1])

                    self._obs_fun[a_idx, s_idx, o_idx] = prob
                    #print('prob: ' + str(prob))
    
    def ent(self, s, a):
        entropy = 0
        for o in range(len(self._observations)):
            
            p = self._obs_fun[a, s, o]
            
            if p > 0:
                entropy += (-1)*p*log2(p)
        return entropy
        
        


    def generate_reward_fun_IT(self, a, b, exp, num_preds):
        
        add_on = dict()
        #experience level is related to action only
        for a_idx, a_val in enumerate(self._actions):
            add_on[a_idx] = dict()
            for s_idx, s_val in enumerate(self._states):
                if a_val._name in exp.keys():
                    add_on[a_idx][s_idx] = a * self.ent(s_idx, a_idx) - b * exp[a_val._name]
                    #print(self.ent(s_idx, a_idx))
                    #print (exp[a_val._name])

        if num_preds == 1:            
            for a_idx, a_val in enumerate(self._actions):
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == True:
                        self._reward_fun[a_idx, s_idx] = 0.0
                    elif a_val._term == False and a_val._name == 'ask':
                        self._reward_fun[a_idx, s_idx] = self._ask_cost

                    elif a_val._term == False and a_val._name == 'look':
                        if s_idx == 0: 
                            self._reward_fun[a_idx, s_idx] = -0.5 + add_on[a_idx][2]
                        if s_idx == 1:
                            self._reward_fun[a_idx, s_idx] = -0.5 + add_on[a_idx][3]
                    elif a_val._term == False and a_val._name == 'grasp':
     #                   self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][2]
                        if s_idx == 2: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][4]
                        if s_idx == 3:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][5]
                    elif a_val._term == False and a_val._name == 'lift_slow':
    #                    self._reward_fun[a_idx, s_idx] = -11.1  + add_on[a_idx][3]
                        if s_idx == 4: 
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][6]
                        if s_idx == 5:
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][7]
                
                    elif a_val._term == False and a_val._name == 'hold':
    #                    self._reward_fun[a_idx, s_idx] = -11.1 + add_on[a_idx][3]
                        if s_idx == 6: 
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][6]
                        if s_idx == 7:
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][7]

                    #making up for shake
                    elif a_val._term == False and a_val._name == 'shake':
    #                    self._reward_fun[a_idx, s_idx] = -22.0+ add_on[a_idx][3]
                        if s_idx == 6: 
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][6]
                        if s_idx == 7:
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][7]

                    # making up a cost for this lower action -- actual cost needs to be acquired from Jivko
                    elif a_val._term == False and a_val._name == 'low_drop':
    #                    self._reward_fun[a_idx, s_idx] = -20.4 + add_on[a_idx][4]
                        if s_idx == 6: 
                            self._reward_fun[a_idx, s_idx] = -20.4 + add_on[a_idx][8]
                        if s_idx == 7:
                            self._reward_fun[a_idx, s_idx] = -20.4 + add_on[a_idx][9]
                       
                    elif a_val._term == False and a_val._name == 'push':
                       # self._reward_fun[a_idx, s_idx] = -22.0+ add_on[a_idx][4]
                        if s_idx == 2: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][8]
                        if s_idx == 3:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][9]

                    #making up for tap
                    elif a_val._term == False and a_val._name == 'tap':
    #                    self._reward_fun[a_idx, s_idx] = -22.0+ add_on[a_idx][4]
                        if s_idx == 2: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][8]
                        if s_idx == 3:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][9]

                    #making up for poke
                    elif a_val._term == False and a_val._name == 'poke':
     #                   self._reward_fun[a_idx, s_idx] = -22.0+ add_on[a_idx][4]
                        if s_idx == 2: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][8]
                        if s_idx == 3:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][9]



                    #making up for crush
                    elif a_val._term == False and a_val._name == 'crush':
    #                    self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][4]
                        if s_idx == 2: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][8]
                        if s_idx == 3:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][9]


                   
                    elif a_val._term == False and a_val._name == 'reinit':
                        self._reward_fun[a_idx, s_idx] = -10.0
                    elif a_val._prop_values == s_val._prop_values:
                        self._reward_fun[a_idx, s_idx] = 300.0
                    else:
                        self._reward_fun[a_idx, s_idx] = -300.0                
        
        if num_preds == 2:
            for a_idx, a_val in enumerate(self._actions):
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == True:
                        self._reward_fun[a_idx, s_idx] = 0.0
                    elif a_val._term == False and a_val._name == 'ask':
                        self._reward_fun[a_idx, s_idx] = self._ask_cost

                    elif a_val._term == False and a_val._name == 'look':
                        if s_idx == 0: 
                            self._reward_fun[a_idx, s_idx] = -0.5 + add_on[a_idx][4]
                        if s_idx == 1:
                            self._reward_fun[a_idx, s_idx] = -0.5 + add_on[a_idx][5]
                        if s_idx == 2:
                            self._reward_fun[a_idx, s_idx] = -0.5 + add_on[a_idx][6]
                        if s_idx == 3:
                            self._reward_fun[a_idx, s_idx] = -0.5 + add_on[a_idx][7]
                   
                    elif a_val._term == False and a_val._name == 'grasp':
     #                   self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][2]
                        if s_idx == 4: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][8]
                        if s_idx == 5:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][9]
                        if s_idx == 6: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][10]
                        if s_idx == 7:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][11]
                   
                    elif a_val._term == False and a_val._name == 'lift_slow':
    #                    self._reward_fun[a_idx, s_idx] = -11.1  + add_on[a_idx][3]
                        if s_idx == 8: 
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][12]
                        if s_idx == 9:
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][13]
                        if s_idx == 10: 
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][14]
                        if s_idx == 11:
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][15]
                

                    elif a_val._term == False and a_val._name == 'hold':
    #                    self._reward_fun[a_idx, s_idx] = -11.1 + add_on[a_idx][3]
                        if s_idx == 12: 
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][12]
                        if s_idx == 13:
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][13]
                        if s_idx == 14: 
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][14]
                        if s_idx == 15:
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][15]


                    #making up for shake
                    elif a_val._term == False and a_val._name == 'shake':
    #                    self._reward_fun[a_idx, s_idx] = -22.0+ add_on[a_idx][3]
                        if s_idx == 12: 
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][12]
                        if s_idx == 13:
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][13]
                        if s_idx == 14: 
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][14]
                        if s_idx == 15:
                            self._reward_fun[a_idx, s_idx] = -11.0 + add_on[a_idx][15]


                    # making up a cost for this lower action -- actual cost needs to be acquired from Jivko
                    elif a_val._term == False and a_val._name == 'low_drop':
    #                    self._reward_fun[a_idx, s_idx] = -20.4 + add_on[a_idx][4]
                        if s_idx == 12: 
                            self._reward_fun[a_idx, s_idx] = -20.4 + add_on[a_idx][16]
                        if s_idx == 13:
                            self._reward_fun[a_idx, s_idx] = -20.4 + add_on[a_idx][17]
                        if s_idx == 14: 
                            self._reward_fun[a_idx, s_idx] = -20.4 + add_on[a_idx][18]
                        if s_idx == 15:
                            self._reward_fun[a_idx, s_idx] = -20.4 + add_on[a_idx][19]
                       
                    elif a_val._term == False and a_val._name == 'push':
                       # self._reward_fun[a_idx, s_idx] = -22.0+ add_on[a_idx][4]
                        if s_idx == 4: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][16]
                        if s_idx == 5:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][17]
                        if s_idx == 6: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][18]
                        if s_idx == 7:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][19]
                    #making up for tap
                    elif a_val._term == False and a_val._name == 'tap':
    #                    self._reward_fun[a_idx, s_idx] = -22.0+ add_on[a_idx][4]
                        if s_idx == 4: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][16]
                        if s_idx == 5:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][17]
                        if s_idx == 6: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][18]
                        if s_idx == 7:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][19]
 
                    #making up for poke
                    elif a_val._term == False and a_val._name == 'poke':
     #                   self._reward_fun[a_idx, s_idx] = -22.0+ add_on[a_idx][4]
                        if s_idx == 4: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][16]
                        if s_idx == 5:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][17]
                        if s_idx == 6: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][18]
                        if s_idx == 7:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][19]
 


                    #making up for crush
                    elif a_val._term == False and a_val._name == 'crush':
    #                    self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][4]
                        if s_idx == 4: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][16]
                        if s_idx == 5:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][17]
                        if s_idx == 6: 
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][18]
                        if s_idx == 7:
                            self._reward_fun[a_idx, s_idx] = -22.0 + add_on[a_idx][19]
 

                   
                    elif a_val._term == False and a_val._name == 'reinit':
                        self._reward_fun[a_idx, s_idx] = -10.0
                    elif a_val._prop_values == s_val._prop_values:
                        self._reward_fun[a_idx, s_idx] = 300.0
                    else:
                        self._reward_fun[a_idx, s_idx] = -300.0                
        

       


    def generate_reward_fun(self):

        for a_idx, a_val in enumerate(self._actions):
            for s_idx, s_val in enumerate(self._states):
                if s_val._term == True:
                    self._reward_fun[a_idx, s_idx] = 0.0
                elif a_val._term == False and a_val._name == 'ask':
                    self._reward_fun[a_idx, s_idx] = self._ask_cost

                elif a_val._term == False and a_val._name == 'look':
                    self._reward_fun[a_idx, s_idx] = -0.5
                elif a_val._term == False and a_val._name == 'grasp':
                    self._reward_fun[a_idx, s_idx] = -22.0
                elif a_val._term == False and a_val._name == 'lift_slow':
                    self._reward_fun[a_idx, s_idx] = -11.0              
                elif a_val._term == False and a_val._name == 'hold':
                    self._reward_fun[a_idx, s_idx] = -11.0
                #making up for shake
                elif a_val._term == False and a_val._name == 'shake':
                    self._reward_fun[a_idx, s_idx] = -11.0
                # making up a cost for this lower action -- actual cost needs to be acquired from Jivko
                elif a_val._term == False and a_val._name == 'low_drop':
                    self._reward_fun[a_idx, s_idx] = -20.4                    
                elif a_val._term == False and a_val._name == 'push':
                    self._reward_fun[a_idx, s_idx] = -22.0
                #making up for tap
                elif a_val._term == False and a_val._name == 'tap':
                    self._reward_fun[a_idx, s_idx] = -22.0
                #making up for poke
                elif a_val._term == False and a_val._name == 'poke':
                    self._reward_fun[a_idx, s_idx] = -22.0
                #making up for crush
                elif a_val._term == False and a_val._name == 'crush':
                    self._reward_fun[a_idx, s_idx] = -22.0                
                elif a_val._term == False and a_val._name == 'reinit':
                    self._reward_fun[a_idx, s_idx] = -10.0
                elif a_val._prop_values == s_val._prop_values:
                    self._reward_fun[a_idx, s_idx] = 300.0
                else:
                    self._reward_fun[a_idx, s_idx] = -300.0

    def write_to_file(self, path):
        
        s = 'discount: ' + str(self._discount) + '\nvalues: reward\n\n'
        s += 'states: '
        for state in self._states:
            s += state._name + ' '
        s += '\n\n'
        s += 'actions: '
        for action in self._actions:
            s += action._name + ' '
        s += '\n\n'
        s += 'observations: '
        for observation in self._observations:
            s += observation._name + ' '
        s += '\n\n'

        for a in range(len(self._actions)):
            s += 'T: ' + self._actions[a]._name + '\n'
            for s1 in range(len(self._states)):
                for s2 in range(len(self._states)):
                    s += str(self._trans[a, s1, s2]) + ' '
                s += '\n'
            s += '\n'

        for a in range(len(self._actions)):
            s += 'O: ' + self._actions[a]._name + '\n'
            for s1 in range(len(self._states)):
                for o in range(len(self._observations)):
                    s += str(self._obs_fun[a, s1, o]) + ' '
                s += '\n'
            s += '\n'

        for a in range(len(self._actions)):
            for s1 in range(len(self._states)):
                s += 'R: ' + self._actions[a]._name + ' : ' + self._states[s1]._name + ' : * : * '
                s += str(self._reward_fun[a, s1]) + '\n'

        f = open(path, 'w')
        f.write(s)

def main(argv):

    model = Model(0.99, ['yellow', 'water'], 0.9, -90.0, 1)
    model.write_to_file('model.pomdp')

if __name__ == "__main__":
    main(sys.argv)
        



