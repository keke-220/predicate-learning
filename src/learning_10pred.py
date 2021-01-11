#! /usr/bin/env python

"""
This file drives the whole environment on different kinds of strategy of online predicate learning
"""

import pickle
import random
import os
import numpy as np
import sys
import pandas as pd
import csv
import time
from constructor import Model, State, Action, Obs
from policy import Policy, Solver
from simulator import Simulator
from oracle import TFTableCY101
from data_play import DataPlay
from train import Classifier


def print_objects_details(T_oracle, objects, predicates):
    print("Details:\n")
    for p in predicates:
        num_pos = 0
        num_neg = 0
        for o in objects:
            if T_oracle.hasLabel(o, p):
                num_pos += 1
            else:
                num_neg += 1
        print (p + ": " + str(num_pos) + " positives and " + str(num_neg) + " negatives")
    print('\n')

def main(argv):
    
    #test
    test_success = []

    applPath1 = argv[1]
    print ("pomdp solver path: " + str(applPath1))
    print("\n****** Simulation parameters: ******\n")

    min_num_examples_per_class = 3
    T_oracle = TFTableCY101(min_num_examples_per_class)
    
    obj_labels = T_oracle.getObjLabels()
    behaviors = T_oracle.getBehaviors()
    contexts = T_oracle.getContexts()
    objects = T_oracle.getObjects()
    all_words = T_oracle.getWords()
    
   
    datapath = '../data/cy101/normalized_data_without_noobject/'
  
    #set up pretraining set, training set and test set
    random.shuffle(objects)
    
    o_pre = []
    o_train = []
    o_test = {}

    pre_size = 15
    train_size = 40
    test_size = 45

    for i in range(pre_size):
        o_pre.append(objects[i])
    
    #for testing purpose, 10 preds
    #1-pred
    o_pre =  ['pasta_rotini', 'eggcoloringcup_orange', 'smallstuffedanimal_otter', 'ball_blue', 'can_red_bull_small', 'cup_metal', 'medicine_flaxseed_oil', 'eggcoloringcup_yellow', 'ball_base', 'eggcoloringcup_pink', 'ball_yellow_purple', 'tupperware_marbles', 'timber_rectangle', 'cone_3', 'bigstuffedanimal_bunny']
    #2-preds

    #o_pre =  ['cone_2', 'eggcoloringcup_pink', 'cannedfood_tomatoes', 'can_red_bull_large', 'bottle_fuse', 'ball_base', 'pasta_penne', 'smallstuffedanimal_otter', 'egg_wood', 'medicine_flaxseed_oil', 'timber_square', 'smallstuffedanimal_headband_bear', 'cup_metal', 'noodle_3', 'cup_blue', 'pvc_5', 'bigstuffedanimal_pink_dog', 'cannedfood_tomato_paste', 'weight_3', 'tin_snowman', 'timber_squiggle', 'pvc_1', 'smallstuffedanimal_bunny', 'eggcoloringcup_blue', 'medicine_aspirin', 'pasta_cremette', 'noodle_1', 'weight_5', 'cannedfood_cowboy_cookout', 'basket_green']
    
    predicates = []
    min_pos = 3 #at least to be 2 in order to do object based cross validatio

    pre_labels = dict() #store label information of pretraining dataset
    for o in o_pre:
        for p in obj_labels[o]:
            if p not in pre_labels.keys(): #have never see this predicate before
                pre_labels[p] = 1
            else:
                pre_labels[p] += 1 #increase positive number for this predicate

    #filter known predicates according to min_pos
    for p in pre_labels.keys():
        if pre_labels[p] >= min_pos:
            predicates.append(p)
 
    predicates = ['soft', 'green', 'full', 'empty', 'container', 'plastic', 'hard', 'blue', 'metal', 'toy']
    #o_pre_out is objects without pretraining objects
    o_pre_out = []
    for o in objects:
        if o not in o_pre:
            o_pre_out.append(o)

    #select train objects
    random.shuffle(o_pre_out)
    for o in o_pre_out:
        o_train.append(o)
        if len(o_train) >= train_size:
            break


    #for testing 10 preds
    #1-pred query
    o_train =  ['tupperware_rice', 'weight_3', 'bigstuffedanimal_pink_dog', 'pasta_macaroni', 'timber_pentagon', 'smallstuffedanimal_moose', 'cone_4', 'bottle_google', 'tin_snack_depot', 'tupperware_pasta', 'pvc_1', 'weight_2', 'pasta_penne', 'can_starbucks', 'weight_4', 'smallstuffedanimal_headband_bear', 'egg_rough_styrofoam', 'basket_cylinder', 'egg_cardboard', 'can_coke', 'noodle_3', 'basket_green', 'egg_plastic_wrap', 'pasta_pipette', 'metal_thermos', 'cannedfood_tomatoes', 'tin_poker', 'medicine_calcium', 'cannedfood_cowboy_cookout', 'pvc_2', 'pvc_4', 'metal_tea_jar', 'metal_mix_covered_cup', 'medicine_ampicillin', 'bottle_green', 'noodle_2', 'can_red_bull_large', 'tupperware_coffee_beans', 'pvc_3', 'noodle_4']
    #2-pred query

    #o_train =  ['pvc_3', 'eggcoloringcup_yellow', 'bottle_google', 'noodle_2', 'pasta_pipette', 'medicine_bilberry_extract', 'eggcoloringcup_green', 'ball_transparent', 'metal_food_can', 'egg_cardboard', 'bigstuffedanimal_tan_dog', 'bigstuffedanimal_bear', 'eggcoloringcup_orange', 'pasta_rotini', 'bottle_red', 'cannedfood_chili', 'cone_1', 'ball_yellow_purple', 'cup_yellow', 'metal_thermos', 'noodle_5', 'medicine_ampicillin', 'can_arizona', 'can_red_bull_small', 'basket_handle', 'metal_flower_cylinder', 'tin_snack_depot', 'cup_paper_green', 'bigstuffedanimal_frog', 'bottle_sobe', 'smallstuffedanimal_moose', 'tupperware_rice', 'tupperware_marbles', 'weight_1', 'cannedfood_soup']
    random.shuffle(objects)
    
    general_test = []
    for o in objects:
        if o not in o_train and o not in o_pre:
            general_test.append(o)
    
    classifier_all = Classifier(datapath, T_oracle, objects, predicates)
   
    print("predicates =  " + str(predicates) + '\n')

    print("o_pre =  " + str(o_pre) + '\n')
    print_objects_details(T_oracle, o_pre, predicates)

    print("o_train =  " + str(o_train) + '\n')
    print_objects_details(T_oracle, o_train, predicates)
    print("o_test =  " + str(general_test) + '\n')

    print_objects_details(T_oracle, general_test, predicates)
    strategy = ['passive_learning', 'IT_learning']    
    #strategy = ['random_plus', 'passive_learning', 'IT_learning']
    for s in strategy:
        dir_path = "../runtime/classifiers/" + s
        if os.path.exists(dir_path) == False:
            print("Folder for runtime classifiers has been created. ")
            os.mkdir(dir_path)
    
    #batch number: retrain after a batch
    num_batch = 10
    
    #trial number(batch size)
    num_trials = 40
    
    train_times = 1
    
    num_test_trial = 1000
   
    timeout = 5
    
    alpha = 6
    beta = 30
   
    exp_times = 1

    #initialize pretraining data
    pre_path = "../runtime/o_pre/"
    pre_adder = DataPlay(pre_path)
    pre_adder.generate_data(o_pre)
    
    total_success = {}
    total_result_cost = {}
    

    #starting experiment
    for exp_time in range(exp_times):

        result_success = {}
        result_cost = {}
        

        #number of prediacates in query
        for num_preds in [1]:
            

            #create a set of predicate(s) that the agent can ask about
            predicates_set = []
            if num_preds == 1:
                for p in predicates:
                    predicates_set.append([p])
            elif num_preds == 2:
                num_in_set = int(len(predicates) * (len(predicates)-1) / 2)
                
                while True:
                    random.shuffle(predicates)
                    add = [predicates[0], predicates[1]]
                    add.sort()
                    if add not in predicates_set:
                        predicates_set.append(add) 
                    if len(predicates_set) == num_in_set:
                        break
                                     
            for s in strategy:
                if s == 'random_plus':
                    planner = 'random_plus'
                else: planner  ='pomdp' 
                #initialize train set (predicate based)
                train_x = dict()
                train_Y = dict()
                exp = dict()
                #don't forget to add pretrain data into it
                for p in predicates:
                    train_x[p] = dict()
                    train_Y[p] = dict()
                    for c in contexts:
                        train_x[p][c] = []
                        train_Y[p][c] = []
                        for o in o_pre:
                            for t in range(1, 2):
                                features = classifier_all.getFeatures(c, o, t) 
                                train_x[p][c].append(features)
                                if T_oracle.hasLabel(o, p):
                                    train_Y[p][c].append(1)
                                else:
                                    train_Y[p][c].append(0)
                for p_set in predicates_set:
                    exp[tuple(p_set)] = dict()
                    for b in behaviors:
                        exp[tuple(p_set)][b] = 0
         
                #keep track of how many times we tried to each object
                num_obj = dict()
                for o in o_train:
                    num_obj[o] = dict()
                    for b in behaviors:
                        num_obj[o][b] = 0

                result_success[s] = []
                result_cost[s] = []
                result_cost[s].append(0)

                isLearningList = []
                new_data_list = [] 
                print("\nPerforming " + s + "...\n")

                #time.sleep(1)
                
                cls_path = "../runtime/classifiers/" + s + "/"
                if not os.path.exists(cls_path):
                    os.mkdir(cls_path)
                
                #create pre-training classifier
                T_oracle = TFTableCY101(3)
                
                pre_classifier = Classifier(pre_path, T_oracle, o_pre, predicates)
                cm_p_b_dict = dict()
                

                #do training several times, and get mean
                for i in range(0, train_times):

                    pre_classifier.retrain_classifier(train_x, train_Y)
                    cm_p_b_dict[i] = pre_classifier.get_learnable_p_b_dict()
                
                cm_total = dict()
                for p in predicates:
                    cm_total[p] = dict()
                    for b in behaviors:
                        cm_total[p][b] = np.zeros((2,2))
                for i in range(0, train_times):        
                    for p in predicates:
                        for b in behaviors:
                            cm_total[p][b] += cm_p_b_dict[i][p][b]
                pre_classifier.set_cm_p_b_dict(cm_total)
                #save classifier
                pre_classifier.print_results()
           
                #time.sleep(3)
                pkl_file = open(cls_path + 'classifier_batch0.pkl', 'wb')
                pickle.dump(pre_classifier, pkl_file)
                pkl_file.close()
                
          
                    #for every batch  NOTICE: batch 0 for passive learning is a warm up classifier
                for batch_idx in range(1, num_batch + 1):

                    # Testing Phase
                    
                    max_cost = 50
                    
                    success_trials = {}
                    total_cost = {}
                    
                    #for testing purpose, we test the result based on each predicates
                    for request_preds in predicates_set:
                    
                        #for test phase
                        success_trials[tuple(request_preds)] = 0
                        total_cost[tuple(request_preds)] = 0

                        test_model = dict()
                        policies = dict()
                        
                        classifier_name = "../runtime/classifiers/" + s + "/classifier_batch" + str(batch_idx-1) + '.pkl'
                        
                        #for every trial 

                        for trial_idx in range(1, num_test_trial + 1):
                        
                            print('\n================== Test Trial ' + str(trial_idx) + '/' + str(num_test_trial) + ' ==================' + 'BATCH ' + str(batch_idx) + ' EXP ' + str(exp_time) + ' on ' + str(s))
                        
                            query_length = random.randrange(num_preds, num_preds + 1)
                            #random.seed(num_preds + trial_idx)
                            test_object_idx = random.randrange(1, len(general_test)+1)
                            test_object_idx = test_object_idx - 1
                            t_object = general_test[test_object_idx]
                                
                            print('Q: Is #' + str(test_object_idx + 1) + ': ' +t_object+ ' ' + str(request_preds) + '?')
                            
                            object_prop_names = obj_labels[t_object]
                           
                            print ('object ground truth: ', object_prop_names)
                            request_preds.sort()
                            #compute the model if request preds are not seen before
                            if tuple(request_preds) not in test_model.keys():
                                


                                solver = Solver()
                                model = Model(0.99, request_preds, 0.7, -5000.0, classifier_name)
                                
                                #need to modify when it comes to multiple predicates     
                                
                                if (s == 'IT_learning'):
                                    model.generate_reward_fun_IT(alpha, beta, exp[tuple(request_preds)], num_preds)
                                
                                model_name = '../runtime/model/' + s + '_' + str(request_preds) + '.pomdp'
                                
                                print('generating model file "' + model_name + '"')
                                model.write_to_file(model_name)
                                

                                if planner == 'pomdp':
                                    policy_name = '../runtime/policy/' + s + '_' + str(request_preds) + '.policy'
                                    pathlist=[applPath1]
                                    appl=None
                                    for ip in pathlist:
                                        if os.path.exists(ip):
                                            appl=ip   
                                    if appl==None:
                                        print ("ERROR: No path detected for pomdpsol")


                                    dir_path = os.path.dirname(os.path.realpath(__file__))
                                    print('computing policy "' + dir_path + '/' + policy_name + '" for model "' + model_name + '"')
                                    print('this will take at most ' + str(timeout) + ' seconds...')
                                    solver.compute_policy(model_name, policy_name, appl, timeout)

                                    print('parsing policy: ' + policy_name)
                                    policy = Policy(len(model._states), len(model._actions), policy_name)

                                    policies[tuple(request_preds)] = policy
                            
                                test_model[tuple(request_preds)] = model

                            if planner == 'pomdp':
                                print('starting simulation')
                                simulator = Simulator(test_model[tuple(request_preds)], policies[tuple(request_preds)], object_prop_names, request_preds)
                            
                            elif planner == 'random_plus':
                                simulator = Simulator(test_model[tuple(request_preds)], None, object_prop_names, request_preds)
                            action_cost = simulator.run(planner, request_preds,test_object_idx, max_cost)
                            selected_actions = simulator.getActions()
                          
                            print ('overall action cost: ' + str(action_cost))
    #                            print ('overall reward: ' + str(trial_reward) + '\n')
                            
                            p = request_preds
                            if num_preds == 1:

                                if selected_actions[-1] == 'a0' and request_preds[0] not in object_prop_names:
                                    success_trials[tuple(p)] += 1

                             
                                if selected_actions[-1] == 'a1' and request_preds[0] in object_prop_names:
                                    success_trials[tuple(p)] += 1
                            
                            elif num_preds == 2:

                                if selected_actions[-1] == 'a00' and (request_preds[0] not in object_prop_names) and (request_preds[1] not in object_prop_names):
                                    success_trials[tuple(p)] += 1


                                if selected_actions[-1] == 'a01' and (request_preds[0] not in object_prop_names) and (request_preds[1] in object_prop_names):
                                    success_trials[tuple(p)] += 1


                                if selected_actions[-1] == 'a10' and (request_preds[0] in object_prop_names) and (request_preds[1] not in object_prop_names):
                                    success_trials[tuple(p)] += 1


                                if selected_actions[-1] == 'a11' and (request_preds[0] in object_prop_names) and (request_preds[1] in object_prop_names):
                                    success_trials[tuple(p)] += 1


                            #total_cost[tuple(p)] += action_cost
                    
                    #total_c = 0
                    total_s = 0
                    for preds in predicates_set:
                        total_s += success_trials[tuple(preds)]/len(predicates_set)
                        #total_c += total_cost[tuple(preds)]/len(predicates_set)
                    
                    if (num_test_trial > 0):
                        #avg_cost = total_c/num_test_trial
                        avg_success = total_s/num_test_trial
                    else:
                        #avg_cost = 0
                        avg_success = 0
                    result_success[s].append(avg_success)
                    #result_cost[s].append((-1)*avg_cost)
                    #time.sleep(1)
                        
                    print(success_trials)

                    #test
                    test_success.append(success_trials)

             

                    new_data = 0
                    # LEARNING PHASE
                    #time.sleep(1) 
                    print("\n################### BATCH " + str(batch_idx) + ' #####################\n')
                    #time.sleep(1)
                    
                    classifier_name = "../runtime/classifiers/" + s + "/classifier_batch" + str(batch_idx-1) + '.pkl'
                    
                    #for every trial 
                    isLearning = False

                    learn_model = dict()
                    batch_cost = 0
                    for trial_idx in range(1, num_trials + 1):
                        print('\n================== Trial ' + str(trial_idx) + '/' + str(num_trials) + ' ==================' + 'BATCH ' + str(batch_idx) + ' EXP ' + str(exp_time) + ' on ' + str(s))
                    
                        query_length = random.randrange(num_preds, num_preds + 1)
                        #random.seed(num_preds + trial_idx)
                        shuffledwords = predicates[:]
                        random.shuffle(shuffledwords)
                        request_preds = shuffledwords[0:query_length]
                        test_object_idx = random.randrange(1, len(o_train) + 1)
                        ## from 0 to 100
                        test_object_idx = test_object_idx - 1
                        print('Q: Is #' + str(test_object_idx + 1) + ': ' + o_train[test_object_idx] + ' ' + str(request_preds) + '?')
                        
                        object_prop_names = obj_labels[o_train[test_object_idx]]
                       
                        print ('object ground truth: ', object_prop_names)

                        request_preds.sort()
                        #compute the model if request preds are not seen before
                        
                        if tuple(request_preds) not in learn_model.keys():
     
                            solver = Solver()
                            model = Model(0.99, request_preds, 0.7, -5000.0, classifier_name)
                            
                            #need to modify when it comes to multiple predicates     
                            if (s == 'IT_learning'):
                                model.generate_reward_fun_IT(alpha, beta, exp[tuple(request_preds)], num_preds)
                            model_name = '../runtime/model/' + s + '_' + str(request_preds) + '.pomdp'
                            
                            print('generating model file "' + model_name + '"')
                            model.write_to_file(model_name)
                            
                            if planner == 'pomdp':
                                policy_name = '../runtime/policy/' + s + '_' + str(request_preds) + '.policy'
                                pathlist=[applPath1]
                                appl=None
                                for p in pathlist:
                                    if os.path.exists(p):
                                        appl=p   
                                if appl==None:
                                    print ("ERROR: No path detected for pomdpsol")


                                dir_path = os.path.dirname(os.path.realpath(__file__))
                                print('computing policy "' + dir_path + '/' + policy_name + '" for model "' + model_name + '"')
                                print('this will take at most ' + str(timeout) + ' seconds...')
                                solver.compute_policy(model_name, policy_name, appl, timeout)

                                print('parsing policy: ' + policy_name)
                                policy = Policy(len(model._states), len(model._actions), policy_name)

                                policies[tuple(request_preds)] = policy
                            
                            learn_model[tuple(request_preds)] = model


                        if planner == 'pomdp':
                            print('starting simulation')
                            simulator = Simulator(learn_model[tuple(request_preds)], policies[tuple(request_preds)], object_prop_names, request_preds)
                        elif planner == 'random_plus':
                            simulator = Simulator(learn_model[tuple(request_preds)], None, object_prop_names, request_preds)
 
                        
                        action_cost = simulator.run(planner, request_preds,test_object_idx, max_cost)
                        selected_actions = simulator.getActions()

                        is_success = 0

                        if num_preds == 2:

                            if selected_actions[-1] == 'a00' and (request_preds[0] not in object_prop_names) and (request_preds[1] not in object_prop_names):
                                is_success = 1

                            if selected_actions[-1] == 'a01' and (request_preds[0] not in object_prop_names) and (request_preds[1] in object_prop_names):
                                is_success = 1


                            if selected_actions[-1] == 'a10' and (request_preds[0] in object_prop_names) and (request_preds[1] not in object_prop_names):
                                is_success = 1


                            if selected_actions[-1] == 'a11' and (request_preds[0] in object_prop_names) and (request_preds[1] in object_prop_names):
                                is_success = 1

            
                        #check if there is new data we can add to our train set
                        #print (selected_actions)
                        if is_success == 1 or num_preds == 1:
                            for a in selected_actions:
                                if a in T_oracle.getBehaviors():#there is a chance that we have reinit action
                                    if  num_obj[o_train[test_object_idx]][a] < 5:
                                    
                                        num_obj[o_train[test_object_idx]][a] += 1
                                        
                                        for p in request_preds:
                                            for c in contexts:
                                                if '_'.join(c.split('_')[:-1]) == a:
                                                    train_x[p][c].append(classifier_all.getFeatures(c, o_train[test_object_idx], num_obj[o_train[test_object_idx]][a])) 
                                                    isLearning = True
                                                    new_data += 1
                                                    if all(ele in object_prop_names for ele in request_preds):
                                                        train_Y[p][c].append(1)
                                                    else:
                                                        train_Y[p][c].append(0)
                          

                        #for IT_learning, we have to update our experience level
                        for b in behaviors:#there is a chance that we have reinit action
                            for p_set in predicates_set:
                                exp[tuple(p_set)][b] = 0
                                for p in p_set:
                                    for c in contexts:
                                        if b in c:
                                        # update one time
                                            exp[tuple(p_set)][b] += (len(train_x[p][c]) - 1*len(o_pre))/(5*len(o_train))
                                            break
                                exp[tuple(p_set)][b] = exp[tuple(p_set)][b]/num_preds
                        print ('overall action cost: ' + str(action_cost))
                        batch_cost += action_cost
                    #starting to retrian the classifier
                    result_cost[s].append((-1)*batch_cost)
                    #print ("********** Batch " + str(batch_idx) + " results: ***********\n")
                    isLearningList.append(isLearning)
                    new_data_list.append(new_data)

                    cm_p_b_dict = dict()
                    

                    #do training several times, and get mean
                    for i in range(0, train_times):

                        classifier_all.retrain_classifier(train_x, train_Y)
                        cm_p_b_dict[i] = classifier_all.get_learnable_p_b_dict()
                    
                    cm_total = dict()
                    for p in predicates:
                        cm_total[p] = dict()
                        for b in behaviors:
                            cm_total[p][b] = np.zeros((2,2))
                    
                    for i in range(0, train_times):        
                        for p in predicates:
                            for b in behaviors:
                                cm_total[p][b] += cm_p_b_dict[i][p][b]
                    classifier_all.set_cm_p_b_dict(cm_total)

                    classifier_all.print_results()
                    time.sleep(3)
                    #save classifier
                    pkl_file = open(cls_path + 'classifier_batch' + str(batch_idx) + '.pkl', 'wb')
                    pickle.dump(classifier_all, pkl_file)
                    pkl_file.close()

                        
                          
            for p in predicates:
                for c in contexts:

                    print (p + "_" + c + ": " + str(len(train_x[p][c])))
             
            total_exp = 0
            for p_set in predicates_set:
                for b in behaviors:
                    total_exp += exp[tuple(p_set)][b]    
                    print (str(p_set) + "_" + b + ": " + str(exp[tuple(p_set)][b]))
            
            print ("total experience is: " + str(total_exp))


            print('\n')
            print("Result: ")
            print("Num success trials: ")
            print(result_success)
            print(result_cost)

            total_success[exp_time] = result_success
            total_result_cost[exp_time] = result_cost
            print (new_data_list)

    print ("total results: ")
    print (total_success)
    print (total_result_cost)

if __name__ == '__main__':
    main(sys.argv)
    
