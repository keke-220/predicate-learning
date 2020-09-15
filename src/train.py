#! /usr/bin/env python

"""
This file is used for training the sensorimotor data for online predicate learning
"""

import time
import xlsxwriter
import warnings
import random
import os
import numpy as np
import sys
import pandas as pd
import csv
import copy

import pprint
import pickle

from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import normalize

from oracle import TFTableCY101

warnings.filterwarnings("ignore", category=DeprecationWarning)

class Classifier(object):
    
    def __init__(self, data_path, T_oracle, objects, predicates):
        # load data

        #print("\nRetraining classifier...")
        #time.sleep(1)
        
        self._objects = objects

        self._path = data_path
        
        self._behaviors = T_oracle.getBehaviors()

        self._modalities = T_oracle.getModalities()
        
        self._T_oracle = T_oracle
        self._predicates = predicates
        
        # predicates for which we have classifiers
        self._learned_predicates = []
        
        # some constants
        self._num_trials_per_object = 5
        #self._train_test_split_fraction = 2/3 # what percentage of data is used for training when doing internal cross validation on training data
        
        # compute lists of contexts
        self._contexts = []
        self._contexts = T_oracle.getContexts()     
        # print to verify
        #print("Valid contexts:")
        #print(self._contexts)
        
        # dictionary that holds context specific weights for each predicate
        self._pred_context_weights_dict = dict()

        # dictionary holding all data for a given context (the data is a dictionary itself)
        self._context_db_dict = dict()
        
        for context in self._contexts:
            context_filename = self._path+"/"+context+".txt"
            
            data_dict = dict()
            
            with open(context_filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    
                    features = row[1:len(row)]
                    key = row[0]
                    #print(key)       
                    data_dict[key] = features
                    
            self._context_db_dict[context] = data_dict


        #initialize classifier dict for each predicate, each test object
        #at training stage, key will be behavior-modality pair
        self.classifier = dict()
        for p in self._predicates:
            self.classifier[p] = dict()
            for test_object in self._objects:
                self.classifier[p][test_object] = dict()
 
        self._CM_p_dict = dict()
        self._CM_p_b_dict = dict()
        self._CM_p_c_dict = dict()

        #print(self._context_db_dict[context].keys())
        #print (self._context_db_dict["look_surf"]["cup_blue_t1"])

    def getFeatures(self,context,object_name,trial_number):
        key = str(object_name)+"_t"+str(trial_number)
        #print(context)
        return self._context_db_dict[context][key]
    
    
    def isPredicateTrue(self,predicate,object_name):
        return self._T_oracle.getTorF(predicate,str(object_name))
    
    
    def computeKappa(self,confusion_matrix):
        # compute statistics for kappa
        #running error debug
        if (isinstance(confusion_matrix, float)):
            return 0
        CM = np.zeros((2,2))
        if (confusion_matrix == CM).all():
            return 0

        TN = confusion_matrix[0][0]
        TP = confusion_matrix[1][1]
        FP = confusion_matrix[1][0]
        FN = confusion_matrix[0][1]
        
        total_accuracy = (TN+TP)/np.sum(confusion_matrix)
        random_accuracy = ((TN+FP)*(TN+FN) + (FN+TP)*(FP+TP)) / ( np.sum(confusion_matrix) * np.sum(confusion_matrix))
        
        kappa = (total_accuracy - random_accuracy) / (1.0 - random_accuracy)
        return kappa
    
    # inputs: learn_prob_model (either True or False)
    def createScikitClassifier(self, learn_prob_model):
        # SVM
        #return svm.SVC(gamma=0.001, C=100, probability = learn_prob_model)
    
        return svm.SVC(kernel="poly",C=10,degree=2, gamma='scale', probability = learn_prob_model)
    
        # decision tree
        #return tree.DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=None, min_samples_split=2, min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
    
    
    def get_learnable_p_dict(self):
        return self._CM_p_dict
    def get_learnable_p_b_dict(self):
        return self._CM_p_b_dict
    def get_learnable_p_c_dict(self):
        return self._CM_p_c_dict
        
    def set_cm_p_b_dict(self, cm):
        self._CM_p_b_dict = cm


    def getPredicateBehaviorObservatoinModel(self,predicate,behavior):
        """
        # confusion matrix
        CM = np.zeros( (2,2) )
        
        b_contexts = []
        for context in self._contexts:
            if behavior in context:
                b_contexts.append(context)
        #print(b_contexts) 
        for context in b_contexts:
            if (predicate not in self._learned_predicates):
                return 0
            
            CM_c = self._pred_context_cm_dict[predicate][context]
            CM = CM + (self._pred_context_weights_dict[predicate][context] *CM_c)
        
        cm_sum = np.sum(CM)
        #print(cm_sum)

        #running error debug

        CM = CM / cm_sum
        """
        #in case for overfitting
        CM =  self._CM_p_b_dict[predicate][behavior] 
   
        return CM
    
    def get_pred_context_cm_dict(self):

        for key,value in self._pred_context_cm_dict.items():
            if key in self._learned_predicates:
                print("\n######## predicate: " + str(key) + " ########\n")
                for k,v in value.items() :
                    print(str(k) + ": " + str(self.computeKappa(v)))
    
    def get_kappa(self):
        kappa = dict()
        for p in self._predicates:
            kappa[p] = dict()
            for c in self._contexts:
                kappa[p][c] = self.computeKappa(self._pred_context_cm_dict[p][c])
        return kappa

    def set_weights(self):
        weights = {}
        for p in self._predicates:
            weights[p] = dict()
            for c in self._contexts:
                weights[p][c] = self.computeKappa(self._CM_p_c_dict[p][c])
        self._weights = weights


    def learnedPredicates(self):
        return self._learned_predicates
    
    
    # input: the target object, the behavior, and a predicate
    # output: the probability that the object matches the predicate     
    def classify(self, object_id, behaviors, predicate, selected_trial):
        
        # before doing anything, check whether we even have classifiers for the predicate
        if predicate not in self._predicate_classifier_dict.keys():
            # return negative result
            print("[WARNING] no classifier available for predicate "+predicate)
            return 0.0
            
            
        # first, randomly pick which trial we're doing
        num_available = self._num_trials_per_object
        if selected_trial == -1:
            selected_trial = random.randint(1,num_available)
        
        # next, find which contexts are available in that behavior
        b_contexts = []
        for context in self._contexts:
            for behavior in behaviors:
                if behavior in context:
                    b_contexts.append(context)
        
        # call each classifier
        
        # output distribution over class labels (-1 and +1)
        classlabel_distr = [0.0,0.0]
        
        for context in b_contexts:
            
            # get the classifier for context and predicate
            classifier_c = self._predicate_classifier_dict[predicate][context]
            
            # get the data point for the object and the context
            x = self.getFeatures(context,object_id,selected_trial)
            
            # pass the datapoint to the classifier and get output distribuiton
            output = classifier_c.predict_proba([x])
            
            # weigh distribution by context reliability
            context_weight = 1.0
            
            # do this only if weights have been estimated
            if len(self._pred_context_weights_dict) != 0:
                context_weight = self._pred_context_weights_dict[predicate][context]
                
            #print(context_weight)
            
            classlabel_distr += context_weight*output[0]
            #print("Prediction from context "+context+":\t"+str(output))
        
        # normalize so that output distribution sums up to 1.0
        prob_sum = sum(classlabel_distr)
        classlabel_distr /= prob_sum

        #print("Final distribution over labels:\t"+str(classlabel_distr))
        return classlabel_distr[1]
    
    def classifyMultiplePredicates(self, object_id, behavior, predicates, trial):
        output_probs = []
        
        for p in predicates:
            output_probs.append(self.classify(object_id,behavior,p,trial))
        return output_probs

    def retrain_classifier(self, train_x, train_Y):
        
        #split group cross validation
        #num_group = 50
        num_group = dict()
        for p in self._predicates:
            num_group[p] = dict()
            for c in self._contexts:
                num_group[p][c] = len(train_Y[p][c])
        
        for p in self._predicates:
            for b in self._behaviors:
                for c in self._contexts:
                    if b in c:
                        num_group[p][b] = num_group[p][c]
                        break
        

        CM_p_c_dict = dict()
        prob_p_c_i = dict()
        gt_p_c_i = dict()

        for p in self._predicates:
            CM_p_c_dict[p] = dict()
            prob_p_c_i[p] = dict()
            gt_p_c_i[p] = dict()
            print ("Retrain classifier for predicate: " + p)
            
            for c in self._contexts:
                
                #create and train classifier, save it in the dictionary
                #deal with dataset: split to negatives and positives (using index)
                CM_p_c_dict[p][c] = np.zeros((2,2))
                prob_p_c_i[p][c] = dict()
                gt_p_c_i[p][c] = dict()
                
                """
                indexs = []
                    
                for i in range(len(train_Y[p][c])):
                    indexs.append(i)  
                random.shuffle(indexs)
                
                group_size = int(len(indexs)/num_group)
                
                group = dict()
                
                #split index to groups
                for i in range(0, num_group):
                    group[i] = []
                    #last group:
                    if i == num_group-1:
                        idx = i*group_size
                        while idx <= len(indexs)-1:
                            group[i].append(indexs[idx])
                            idx += 1
                    
                    else:
                        for j in range(0, group_size):
                            group[i].append(indexs[i*group_size+j])
                """

                #for every group, train classifier and do behaivor evaluation
                for i in range(0, num_group[p][c]):
                    test_idx = [i] #index list
                    train_idx = [] 
                    for j in range(len(train_Y[p][c])):
                        if j not in test_idx:
                            train_idx.append(j)
                    
                    x_train = []
                    Y_train = []
                    x_test = []
                    Y_test = []
                    for t in train_idx:
                        x_train.append(train_x[p][c][t])
                        Y_train.append(train_Y[p][c][t])
                    for t in test_idx:
                        x_test.append(train_x[p][c][t])
                        Y_test.append(train_Y[p][c][t])

                    classifier_t = self.createScikitClassifier(True)
                    classifier_t.fit(x_train, Y_train)
        
                    #evaluating the classifier using test data
                    #set ground truth
                    gt_p_c_i[p][c][i] = Y_test

                        #get prediction labels
                    #prediction = classifier_t.predict(x_test)
                    prediction = [] 
                        #get prob distribution
                    prob_p_c_i[p][c][i] = classifier_t.predict_proba(x_test)
                    prob = prob_p_c_i[p][c][i]
                    for pb in prob:
                        if float(pb[0])>=float(pb[1]):
                            prediction.append(0)
                        else:
                            prediction.append(1)

                    #if (c == 'look_color'):
                        #print (prob)
                        #print (prediction)
                        #print (Y_test)


                    for j in range (len(gt_p_c_i[p][c][i])):
                        CM_p_c_dict[p][c][prediction[j]][gt_p_c_i[p][c][i][j]] += 1
                    
        #print (prob_p_c_i['soft']['look_color']


        self._CM_p_c_dict = CM_p_c_dict
        self.set_weights()
        
        CM_p_b_dict = dict()
        
        for p in self._predicates:
            CM_p_b_dict[p] = dict()
            
            for b in self._behaviors:
                
                CM_p_b_dict[p][b] = np.zeros((2,2))
                
                #overfitting issue
                CM_p_b_dict[p][b][0][0] += 5
                CM_p_b_dict[p][b][0][1] += 5
                CM_p_b_dict[p][b][1][0] += 5
                CM_p_b_dict[p][b][1][1] += 5
 

                contexts_b = []
                for c in self._contexts:
                    if b in c:
                        contexts_b.append(c)
                

                for i in range(0, num_group[p][b]):

                    
                    
                        
                    prob = np.zeros(2)
                    
                    #give gt an inital value
                    gt = []
                    for c in contexts_b:
                        gt = gt_p_c_i[p][c][i]
                        break

                    for c in contexts_b:
                        
                        gt_i = gt_p_c_i[p][c][i] #set ground truth

                        if gt != gt_i : #check if all gt are the same
                            print ("something wrong with the ground truth label. ")
                            sys.exit()

                        prob_i = prob_p_c_i[p][c][i]
                        if (self._weights[p][c] >= 0):
                            prob[0] += self._weights[p][c]*prob_i[0][0] #add weight
                            prob[1] += self._weights[p][c]*prob_i[0][1] #add weight
                            #prob[0] += prob_i[0][0] #add weight
                            #prob[1] += prob_i[0][1] #add weight
                    
                    #normalize
                    
                    #prob[0] = prob[0]/(prob[0] + prob[1])
                    #prob[1] = 1 - prob[0]
                    if (prob[0]>=prob[1]):
                        prediction = 0
                    else:
                        prediction = 1
                    CM_p_b_dict[p][b][prediction][gt[0]] += 1
                              

        self._CM_p_b_dict = CM_p_b_dict
        #self.print_results()

               

    def train_bm_classifier(self):
        
        for p in self._predicates:
            
            print("Train classifier for prediacate: "+p)
            #do leave one object out train test split

            for test_object in self._objects:

                #train object set
                o_train = []
                for o in self._objects:
                    if o != test_object:
                        o_train.append(o)

                #test object set: test_object

                # ===TRAINING STAGE===
                #train the classifiers for each context
                for c in self._contexts:

                    X_train = []
                    Y_train = []
                    
                    # create X and Y for training
                    for train_object in o_train:
                        
                        # get class label
                        y_o = 0
                        if self.isPredicateTrue(p,train_object):
                            y_o = 1
                        
                        # for each trial, make datapoint
                        for t in range(1,self._num_trials_per_object+1):
                            x_ot = self.getFeatures(c,train_object,t)
                            X_train.append(x_ot)
                            Y_train.append(y_o)
                    
                    #create and train classifier, save it in the dictionary
                    classifier_t = self.createScikitClassifier(False)
                    classifier_t.fit(X_train, Y_train)

                    self.classifier[p][test_object][c] = classifier_t

    def bm_evaluation(self):
        
        CM_p_c_dict = dict()
        prob_p_t_e_c = dict()

        for p in self._predicates:
            print("bm_evaluation for predicate: " + p)
            
            prob_p_t_e_c[p] = dict()
            #initialize 2x2 confusion matrices
            
            #Using all behaviors
            
            CM_p_c_dict[p] = dict()
           
            #for each context
            for c in self._contexts:
                CM_p_c_dict[p][c] = np.zeros( (2,2) )

                
            #do leave one object out train test split

            for test_object in self._objects:

                #classifier dictionary load          
                classifier = self.classifier
                prob_p_t_e_c[p][test_object] = dict()
                # ===TESTING STAGE===
                #evaluate classifiers                    

                for e in range(1, self._num_trials_per_object+1):
              
                    prob_p_t_e_c[p][test_object][e] = dict()
                    # === EVALUATE METHOD 1 === 
                    gt = 0 #set ground truth
                    if self.isPredicateTrue(p, test_object):
                        gt = 1
                        
                    #evaluate the classifiers for each context
                    for c in self._contexts:
                        x_ec = self.getFeatures(c, test_object, e)
                       
                        #get prediction labels
                        prediction = classifier[p][test_object][c].predict([x_ec])
                        #get prob distribution
                        prob_p_t_e_c[p][test_object][e][c] = classifier[p][test_object][c].predict_proba([x_ec])
                        
                        CM_p_c_dict[p][c][prediction[0]][gt] += 1
        
        self._CM_p_c_dict = CM_p_c_dict
        self._prob_p_t_e_c = prob_p_t_e_c


    def behavior_evaluation(self):
        CM_p_b_dict = dict()
      
        prob_p_t_e_c = self._prob_p_t_e_c

        for p in self._predicates:
            print("behavior evaluation for predicate: " + p)
            
            CM_p_b_dict[p] = dict()
           
            #for each context
            for b in self._behaviors:
                CM_p_b_dict[p][b] = np.zeros( (2,2) )

          
            for test_object in self._objects:
               
                classifier = self.classifier
    
                # ===TESTING STAGE===
                #evaluate classifiers                    

                for e in range(1, self._num_trials_per_object+1):
 
                    gt = 0 #set ground truth
                    if self.isPredicateTrue(p, test_object):
                        gt = 1
                
                    # === EVALUATE METHOD 2 ===

                    #evaluate each behavior as classifier
                    for b in self._behaviors:
                        #get all modalities present for behavior b
                        contexts_b = []
                        for c in self._contexts:
                            if b in c:
                                contexts_b.append(c)
                        prob = np.zeros(2)  
                       
                        for c in contexts_b:
                            #update only if kappa value larger than 0
                            if (self._weights[p][c] >= 0):
                                prob[0] += self._weights[p][c]*prob_p_t_e_c[p][test_object][e][c][0][0] #add weight
                                prob[1] += self._weights[p][c]*prob_p_t_e_c[p][test_object][e][c][0][1] #add weight
                        
                        #normalize
                        prob[0] = prob[0]/(prob[0] + prob[1])
                        prob[1] = 1 - prob[0]
                        if (prob[0]>=prob[1]):
                            prediction = 0
                        else:
                            prediction = 1
                        CM_p_b_dict[p][b][prediction][gt] += 1
                                  
        self._CM_p_b_dict = CM_p_b_dict



    def pred_evaluation(self):
        CM_p_dict = dict()
      
        prob_p_t_e_c = self._prob_p_t_e_c

        for p in self._predicates:
            print("Predicate evaluation for predicate: " + p)
            
            CM_p_dict[p] = np.zeros( (2,2) )
           
            for test_object in self._objects:
               
                classifier = self.classifier
    
                # ===TESTING STAGE===
                #evaluate classifiers                    

                for e in range(1, self._num_trials_per_object+1):
 
                    gt = 0 #set ground truth
                    if self.isPredicateTrue(p, test_object):
                        gt = 1
                                  
                    # === EVALUATE METHOD 3 ===
                    
                    #evaluate using all source of information
                    prob = np.zeros(2)
                    for c in self._contexts:
                        if self._weights[p][c] >= 0:
                            prob[0] += self._weights[p][c]*prob_p_t_e_c[p][test_object][e][c][0][0] #add weight
                            prob[1] += self._weights[p][c]*prob_p_t_e_c[p][test_object][e][c][0][1] #add weight

                    #normalize
                    prob[0] = prob[0]/(prob[0] + prob[1])
                    prob[1] = 1 - prob[0]
                    if (prob[0]>=prob[1]):
                        prediction = 0
                    else:
                        prediction = 1
                    
                    CM_p_dict[p][prediction][gt] += 1

        self._CM_p_dict = CM_p_dict


    def output_full_evaluation(self):
        outfile = "../../output/full_evaluation.xlsx"
        
        #create first row label for the spreadsheet
        first_row = ['predicate', 'all']
        for b in self._behaviors:
            first_row.append(b)
        for c in self._contexts:
            first_row.append(c)
        
        first_row.append('num+')
        first_row.append('num-')


        #create workbook
        workbook = xlsxwriter.Workbook(outfile)
        worksheet =  workbook.add_worksheet()

        row = 0
        col = 0
        
        #wirte first row
        for label in first_row:
            worksheet.write(row, col, label)
            col += 1
        
        row = 1
        col = 0
        #write predicate
        for p in self._predicates:
            worksheet.write(row, col, p)
            row += 1
         
        #wirte evaluation result
        for row in range(1, len(self._predicates)+1):
            for col in range(1, len(first_row)):
                if first_row[col] in self._behaviors:
                    worksheet.write(row, col, self.computeKappa(self._CM_p_b_dict[self._predicates[row-1]][first_row[col]]))
                elif first_row[col] in self._contexts:
                    worksheet.write(row, col, self.computeKappa(self._CM_p_c_dict[self._predicates[row-1]][first_row[col]]))
                elif first_row[col] == 'all':
                    worksheet.write(row, col, self.computeKappa(self._CM_p_dict[self._predicates[row-1]]))
                elif first_row[col] == 'num+':
                    num = 0
                    for o in self._objects:
                        if self.isPredicateTrue(self._predicates[row-1], o):
                            num += 1 
                    worksheet.write(row, col, num)
                elif first_row[col] == 'num-':
                    num = 0
                    for o in self._objects:
                        if not self.isPredicateTrue(self._predicates[row-1], o):
                            num += 1    
                    worksheet.write(row, col, num)

        workbook.close()

    def print_results(self):
        #print cm_p_b_dict
         
        for key,value in self._CM_p_b_dict.items():
            if key in self._predicates:
                print("\n######## predicate: " + str(key) + " ########\n")
                for k,v in value.items() :
                    
                    print(str(k) + ": " + str(self.computeKappa(v)))
                    print (v)
        """
        #print cm_p_c_dict
        for key,value in self._CM_p_c_dict.items():
            if key in self._predicates:
                print("\n######## predicate: " + str(key) + " ########\n")
                for k,v in value.items() :
                    print(str(k) + ": " + str(self.computeKappa(v)))
        
        #print cm_p_dict
        for key,value in self._CM_p_dict.items():
            if key in self._predicates:
                print("\n######## predicate: " + str(key) + " ########\n")
                print(str(key) + ": " + str(self.computeKappa(value)))
        """

#create driver code for full evaluation -- to identify learnable predicates
class FullEvaluationDriver(object):
    
    def __init__(self, min_num_examples_per_classIn):
        
        #root_datapath = "../data/cy101/"
        
        datapath = "../../data/cy101/normalized_data_without_noobject/"
        
        #create oracle
        min_num_examples_per_class = min_num_examples_per_classIn #filter prediactes
        T_oracle = TFTableCY101(min_num_examples_per_class)
        
        num_trials_per_object = 5
        
        #get object list from the first file : crush_audio.txt
        objects = []
        with open(datapath + '/crush_audio.txt') as f:
            all_lines = f.read()
            lines = all_lines.split('\n')
            for line in lines:
                obj_name_list = line.split(',')[0].split('_')[:-1];
                s = '_'
                obj_name = s.join(obj_name_list)
                if obj_name not in objects and obj_name != '':
                    objects.append(obj_name)
       
        all_predicates = T_oracle.getAllPredicates()
        
        print(str(len(objects)) + " objects")

        print(str(len(all_predicates)) + " predicates") 
       
        # create classifier
        classifier = Classifier(datapath, T_oracle, objects, all_predicates)
        
        # leave one out training classifiers
        classifier.train_bm_classifier()

        # run b-m evaluation to get weights
        classifier.bm_evaluation()

        # convert previous result to weights
        classifier.set_weights()
        
        # run behavior evaluation
        classifier.behavior_evaluation()

        # run predicate evaluation
        classifier.pred_evaluation()
        
        #save the result into a spreadsheet
        classifier.output_full_evaluation()



class RetrainDriver(object):
    
    def __init__(self, datapathIn, min_num_examples_per_classIn, predicatesIn, strategy, batch):
        root_datapath = "../data/cy101/"
        datapath = datapathIn
        behaviors = ["look","grasp","lift_slow","hold","shake", "low_drop","tap","push","poke", "crush"]
        modalities = ['surf', 'colot', 'flow', 'audio', 'vibro', 'fingers', 'haptics']
        # minumum number of positive and negative examples needed for a predicate to be included in this experiment
        min_num_examples_per_class = min_num_examples_per_classIn # in the actual experiment, this should be 4 or 5 to include more predicates; when doing a quick test, you can set it to 14 or 15 in which case only 3 predicates are valid

        load_classifiers = False
        
        # some train parameters -- only valid if num_object_split_tests is not 32
        num_trials_per_object = 5

        # needs to be true if observation models for behaviors-predicate pairs are needed
        perform_internal_cv = True
        
        # precompute and store train and test set ids for each test
        train_set_dict = dict()
        test_set_dict = dict()
        
        # create oracle
        T_oracle = TFTableCY101(min_num_examples_per_class)
        
        print("\n****** Training parameters: ******")

        objects = []
        with open(datapath + '/crush_audio.txt') as f:
            all_lines = f.read()
            lines = all_lines.split('\n')
            for line in lines:
                obj_name_list = line.split(',')[0].split('_')[:-1];
                s = '_'
                obj_name = s.join(obj_name_list)
                if obj_name not in objects and obj_name != '':

                    objects.append(obj_name)
        print(str(len(objects)) + " objects: " + str(objects))

        known_words = {}
        with open(root_datapath + '/cy101_labels.csv') as csv_file:
            df = pd.read_csv(csv_file, names = ['objects', 'words'])
            data_objects = df.objects.tolist()[1:]
            data_words = df.words.tolist()[1:]

            #for row in data_objects:
                #objects.append(row)

            for i in range(0, len(data_words)):
                if type(data_words[i]) == str:
                    splitted_row = data_words[i].split(', ')
                for item in splitted_row:
                    #print (data_objects[i])
                    if data_objects[i] in objects:
                        if item not in known_words:
                            known_words[item] = 1
                        else:
                            known_words[item] += 1
        
        all_predicates = []
        #all_predicates = T_oracle.getAllPredicates()
        for k, v in known_words.items():
            if v >= min_num_examples_per_class:
                all_predicates.append(k)
        
        #for testing purpose
        all_predicates = predicatesIn

        print(str(len(all_predicates)) + " predicates: " + str(all_predicates)) 
       
        # where to store the confusion matrices for the classification results
        pred_cm_dict = dict()
        for pred in all_predicates:
            cm_p = np.zeros( (2,2) )
            pred_cm_dict[pred]=cm_p
        
        # create classifier
        classifier = ClassifierRetrain(datapath, T_oracle, objects, all_predicates)
        
        #Using all the data for training 
        train_objects = objects;
        test_objects = [];

        #train all the predicates we indicate
        req_train_predicates = all_predicates

        # where to save or load the classifier
        path1 = "retrain_classifiers"
        #path2 = ""
        s = strategy

        path2 = "retrain_classifiers/" + s
    
        
        if os.path.exists(path2) == False:
            os.mkdir(path2)
        if os.path.exists(path1) == False:
            os.mkdir(path1)
        
        if (s == 'passive_learning'):
            classifier_file_name = path2 + "/classifier_batch" + str(batch) + '.pkl'
        else:
            classifier_file_name = path2 + "/classifier.pkl"
        # train classifier
        if load_classifiers == False:
            
            classifier.trainClassifiers(train_objects,num_trials_per_object,req_train_predicates)
            
            #temp change
            test_predicates = all_predicates

        
            if perform_internal_cv:
                classifier.performCrossValidation(test_predicates)  

            # save classifier
            pkl_classifier_file = open(classifier_file_name, 'wb')
            pickle.dump(classifier, pkl_classifier_file)
            pkl_classifier_file.close()
            print(classifier_file_name + " created. ")
            time.sleep(1)
        
        print("\nPrinting predicate learning kappa values...")
        time.sleep(1)
        classifier.get_pred_context_cm_dict()
            

def main(argv):

#Testing for identifying learnable predicates
    x = []
    y = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    T_oracle = TFTableCY101(3)
    datapath = '../../data/cy101/normalized_data_without_noobject/'
    objects = ['basket_cylinder', 'bottle_google', 'cup_isu', 'pvc_3', 'smallstuffedanimal_headband_bear', 'noodle_3', 'bigstuffedanimal_tan_dog', 'cone_2', 'pasta_pipette', 'ball_base']
    random.shuffle(objects)
    predicates = ['soft', 'empty', 'green']
    cls = Classifier(datapath, T_oracle, objects, predicates)
    for o in objects:
        for t in range(1, 6):
                
            x.append(cls.getFeatures("poke_audio", o, t))
            if T_oracle.hasLabel(o, 'green'):
                y.append(1)
            else:
                y.append(0)
    for i in range(0, 40):
        x_train.append(x[i])
        y_train.append(y[i])
    for i in range(40, 50):
        x_test.append(x[i])
        y_test.append(y[i])
    cls_t = cls.createScikitClassifier(True)
    cls_t.fit(x_train, y_train)
    prediction = cls_t.predict(x_test)
    print (prediction)
    print (y_test)

if __name__ == "__main__":
    main(sys.argv)
