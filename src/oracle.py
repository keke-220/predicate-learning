import pandas as pd
import numpy as np

import csv  

import pprint
import pickle

'''The TFTable (true false table) contains two functions,
 the getTorF function returns true/false/none based on given predicate and objectID
 the getAllPredicates function return all predicates as a list (105 int total)'''

class TFTableCY101:
    def __init__(self,min_num_examples_per_class):
        table_path = "../../data/cy101/cy101_labels.csv"
        self.behaviors = ["look","grasp","lift_slow","hold","shake","low_drop","tap","push","poke","crush"]
        self.modalities = ['surf', 'color', 'flow', 'audio', 'vibro', 'fingers', 'haptics']
        #self.df = pd.read_csv(table_path,index_col=0)
        self.missing_as_negative = False    
        self.annotations = None
        self._min_num_examples_per_class = min_num_examples_per_class
        #self.predicates_annotated = pd.read_csv("../data/ijcai2016/test_full.csv",index_col=0)
         
        #print(self.predicates_annotated)
        self.obj_labels = {}
        self._contexts = []
        
        for b in self.behaviors:
            for m in self.modalities:
                if self.isValidContext(b,m):
                    context_bm = b+"_"+m
                    self._contexts.append(context_bm)
       
        self.words = []
        self._objects = []
        with open(table_path) as csv_file:
            df = pd.read_csv(csv_file, names = ['objects', 'words'])
            data_objects = df.objects.tolist()[1:]
            data_words = df.words.tolist()[1:]
        
            for row in data_objects:
                if row != 'no_object':
                    self._objects.append(row)
            i = 0
            for row in data_words:
                if data_objects[i] != 'no_object':
                    self.obj_labels[data_objects[i]] = []
                    if type(row) == str:
                        splitted_row = row.split(', ')
                        for item in splitted_row:
                            self.obj_labels[data_objects[i]].append(item)
                            if item not in self.words:
                                self.words.append(item)
                i += 1

        #load full annotations
        
        self.class_label_dict = dict()
        for file_obj_id in self._objects:
            # for each predicate, we store output here
            obj_label_dict = dict()
            
            # for each predicate, store in dict
            for p in range(0,len(self.words)):
                obj_label_dict[self.words[p]] = 0
            for label in self.obj_labels[file_obj_id]:
                   
                obj_label_dict[label] = 1
            self.class_label_dict[file_obj_id]=obj_label_dict
        


    def getWords(self):
        return self.words
    
    def getObjects(self):
        return self._objects
    
    def getBehaviors(self):
        return self.behaviors
    
    def getModalities(self):
        return self.modalities

    def getContexts(self):
        return self._contexts  

    def getObjLabels(self):
        return self.obj_labels      
        
    def setObjectIDs(self,obj_ids):
        self.obj_ids = obj_ids  
       
    def hasLabel(self, object, predicate):
        
        ret = self.class_label_dict[object][predicate]
        
        if (ret == 1):
            return True
        else:
            return False




    def getTorF(self,predicate, object):
        #self.predicate = predicate
        #self.objectID = objectID
       
        #ret = self.df.ix[predicate,objectID]
        ret = self.class_label_dict[object][predicate]
        
        if (ret==1):
            return True
        else:
            return False

    def isValidContext(self,behavior,modality):
        if behavior == "look":
            if modality == "color" or modality == "surf":
                return True
            else: 
                return False
        elif behavior == 'grasp' and modality == 'fingers':
            return True
        elif modality in ['flow', 'surf', 'audio', 'vibro', 'haptics']:
            return True
        else:
            return False
    
    def getAllPredicates(self):
        
        # all
        pred_candidates = self.words
        
        # for each predicate see if min number of examples is met
        min_num_positive = self._min_num_examples_per_class
        min_num_negative = self._min_num_examples_per_class
        
        pred_counts_pos = np.zeros(len(self.words))
        pred_counts_neg = np.zeros(len(self.words))
        for p in range(0,len(self.words)):
             for o in self._objects:
             
                  if self.getTorF(self.words[p],str(o)):
                      pred_counts_pos[p] = pred_counts_pos[p] + 1       
                  else:
                      pred_counts_neg[p] = pred_counts_neg[p] + 1  

        pred_filtered = []
        
        for p in range(0,len(self.words)):
            if pred_counts_pos[p] > min_num_positive and pred_counts_neg[p] > min_num_negative:
                pred_filtered.append(self.words[p])
                
        return pred_filtered
