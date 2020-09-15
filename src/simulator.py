#! /usr/bin/env python


import random
import os
import numpy as np
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from constructor import Model, State, Action, Obs
from policy import Policy, Solver



# from classifier code
import csv
import copy

class Simulator(object):

    def __init__(self, model = None, policy = None, object_prop_names = None, request_prop_names = None):
        self._model = model
        self.selected_actions = []
        self._policy = policy
        self._object_prop_names = object_prop_names
        self._request_prop_names = request_prop_names
   
        self._legal_actions = {
            0: [0], 
            1: [1, 6, 7, 8, 9],
            2: [2],
            3: [3, 4, 5],
            4: [11]
        }


    def init_state(self):


        # flag = False

        # while flag is False:
        #     s_idx = random.choice(range(len(self._model._states)))
        #     flag = (self._model._states[s_idx]._s_index == 0)

        # return s_idx, self._model._states[s_idx]

        s_name = 's0p'
        for r in self._request_prop_names:
            if r in self._object_prop_names:
                s_name += '1'
            else:
                s_name += '0'

        for s_idx, s_val in enumerate(self._model._states):
            if s_val._name == s_name:
                return s_idx, s_val
        else:
            sys.exit('Error in initializing state')


    def init_belief(self):

        b = np.zeros(len(self._model._states))

        # initialize the beliefs of the states with index=0 evenly
        num_states_each_index = pow(2, len(self._model._states[0]._prop_values))
        for i in range(num_states_each_index):
            b[i] = 1.0/num_states_each_index
            
        return b

    def observe_real_ijcai(self, s_idx, a_idx, request_prop_names, test_object_index): 

        #print('Making an observation... ')
        query_behavior_in_list = self._model.get_action_name(a_idx)

        if query_behavior_in_list == 'ask':
            return self.observe_sim(s_idx, a_idx)

        query_trial_index = random.randrange(1, 5)

        # print('test_object_index: ' + str(test_object_index))
        # print('query_behavior_in_list: ' + str(query_behavior_in_list))
        #print('request_prop_names: ' + str(request_prop_names))
        #print('query_trial_index: ' + str(query_trial_index))

        list_with_single_behavior = []
        list_with_single_behavior.append(query_behavior_in_list)

        query_pred_probs = self._model._classifiers[test_object_index].classifyMultiplePredicates(\
            test_object_index, list_with_single_behavior, request_prop_names, query_trial_index)

        print("\nObservation predicates and probabilities:")
        print(request_prop_names)
        # print(query_pred_probs)

        obs_name = 'p'
        for prob in query_pred_probs:
            if prob > 0.5: 
                obs_name += '1'
            else:
                obs_name += '0'

        for o_idx, o_val in enumerate(self._model._observations): 
            if o_val._name == obs_name:
                return [o_idx, o_val, 0.0]
        else:
            print ('Error in observe_real_ijcai')
    
    """

    def helper(self, array):
        #this is a bubble_sort helper function
        for i in range(0, len(array)):
            for j in range(i, len(array)):
                if  array[i] > array[j]:
                    temp = array[i]
                    array[i] = array[j]
                    array[j] = temp
        return array
    """

    def observe_real(self, s_idx, a_idx):

        behavior = self._model.get_action_name(a_idx)

        if behavior == 'ask' or behavior == 'reinit':
            return self.observe_sim(s_idx, a_idx)

        target_object = '_'.join(self._object_prop_names)

        obs_distribution = np.ones((2,2,2))

        o_name = 'p'
        prob = 1.0

        for prop_name in self._request_prop_names:

            prob_list = []

            if prop_name in self._color_values:
                for v_color in self._color_values:
                    prob_list += [self._classifier.classify(target_object, behavior, v_color)]

                max_idx = prob_list.index(max(prob_list))
                o_name += str(int(prop_name == self._color_values[max_idx]))

            elif prop_name in self._weight_values:

                for v_weight in self._weight_values:
                    prob_list += [self._classifier.classify(target_object, behavior, v_weight)]

                max_idx = prob_list.index(max(prob_list))
                o_name += str(int(prop_name == self._weight_values[max_idx]))
                    
            elif prop_name in self._content_values:

                for v_content in self._color_values:
                    prob_list += [self._classifier.classify(target_object, behavior, v_content)]

                max_idx = prob_list.index(max(prob_list))
                o_name += str(int(prop_name == self._content_values[max_idx]))

            prob *= max(prob_list)

        for o_idx, o_val in enumerate(self._model._observations):
            if o_val._name == o_name:
                return o_idx, o_val, prob
        else:
            sys.exit('Error in making an observation in real')

    def observe_sim(self, s_idx, a_idx):

        rand = np.random.random_sample()
        acc = 0.0
        for i in range(len(self._model._observations)): 
            o_prob = self._model._obs_fun[a_idx, s_idx, i]
            acc += o_prob
            if acc > rand:
                return i, self._model._observations[i], o_prob
        else:
            sys.exit('Error in making an observation in simulation')

    def get_next_state(self, a_idx, s_idx): 

        rand = np.random.random_sample()
        acc = 0.0
        for i in range(len(self._model._states)): 
            acc += self._model._trans[a_idx, s_idx, i]
            if acc > rand:
                return i, self._model._states[i]
        else:
            sys.exit('Error in changing to the next state')

    def get_reward(self, a_idx, s_idx):
        ret = 0
        if a_idx == 0:
            ret = -0.5
        elif a_idx == 1:
            ret = -22.0
        elif a_idx == 2:
            ret = -11.0

        elif a_idx == 3:
            ret = -11.0

        elif a_idx == 4:
            ret = -11.0

        elif a_idx == 5:
            ret = -20.4
        elif a_idx == 6:
            ret = -22.0
        
        elif a_idx == 7:
            ret = -22.0

        elif a_idx == 8:
            ret = -22.0

        elif a_idx == 9:
            ret = -22.0

        elif a_idx == 10:
            ret = -500

        elif a_idx == 11:
            ret = -10.0

        return ret

    def update(self, a_idx, o_idx, b):

        retb = np.dot(b, self._model._trans[a_idx, :])
        num_states = len(self._model._states)
        retb = [retb[i] * self._model._obs_fun[a_idx, i, o_idx] for i in range(num_states)]

        return retb/sum(retb)

    # according to the current belief distribution, the robot is forced to select a
    # report action to terminate the exploration process
    # it is used in random_plus strategy. 
    def select_report_action(self, b):

    	# it's possible that the most likely entry goes to the term state -- we need to 
    	# assign zero to the term state to make sure we find an report action that makes sense
        b_non_term = b
        b_non_term[-1] = 0.0

        prop_values = self._model._states[b_non_term.argmax()]._prop_values
        fake_action = Action(True, None, prop_values)
        for action_idx, action_val in enumerate(self._model._actions):
            if action_val._name == fake_action._name:
                a_idx = action_idx
                break;
        else:
            sys.exit('Error in selecting predefined actions')

        return a_idx
    
    def getActions(self):
        return self.selected_actions

    def run(self, planner, request_prop_names, test_object_index, max_cost):

        [s_idx, s] = self.init_state()
        print('initial state: ' + s._name)
        b = self.init_belief()
        print('initial belief: ' + str(["%0.2f" % i for i in b]))
        trial_reward = 0
        action_cost = 0
        self._action_cnt = 0

        while True:

            # select an action using the POMDP policy
            if planner == 'pomdp':
                a_idx = self._policy.select_action(b)
                #print(type(a_idx))

            elif planner == 'random_plus':
                print ('abs(trial_reward)')
                print (abs(trial_reward))
                if abs(trial_reward)>max_cost:
                    a_idx = self.select_report_action(b)
                else:
                    a_idx = random.choice(self._legal_actions[self._model._states[s_idx]._s_index])

            else:
                sys.exit('planner type unrecognized: ' + planner)


            a = self._model._actions[a_idx]
            print('action selected (' + planner + '): ' + a._name)
            self.selected_actions.append(a._name)
            # computing reward: current state and selected action
            reward = self.get_reward(a_idx, s_idx)
            trial_reward += reward
            

            # state transition
            [s_idx, s] = self.get_next_state(a_idx, s_idx)
            print('resulting state: ' + s._name)

            # compute accumulated reward
            if s._term is True: 
                action_cost = trial_reward - reward
                break

            # make observation
            # if an action (look, grasp, etc) is unsuccessful, one will end up with no state change and a 'na' observation
            if s._name == 'terminal' or 's4': 
                [o_idx, o, o_prob] = self.observe_sim(s_idx, a_idx)
                # [o_idx, o, o_prob] = self.observe_real(s_idx, a_idx)
            else: 
                [o_idx, o, o_prob] = self.observe_real_ijcai(s_idx, a_idx, request_prop_names, test_object_index)

            print('observation made: ' + o._name + '  probability: ' + str(o_prob))

            # update belief
            b = self.update(a_idx, o_idx, b)
            print("Belief: " + str(["%0.2f" % i for i in b]))

        return action_cost

def main(argv):
    return   

if __name__ == "__main__":
    main(sys.argv)



