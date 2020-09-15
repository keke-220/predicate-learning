import sys
import os
from os.path import isfile, join
from oracle import TFTableCY101
import random
import linecache

class DataPlay:
    
    def __init__(self, path):

        if not os.path.exists(path):
            os.mkdir(path)

        self.path = path
        T_oracle = TFTableCY101(3)
        self.contexts = T_oracle.getContexts()
        self.actions = T_oracle.getBehaviors()
        self.feature_count_dict = {}
        #the original dataset path
        self.ori_path = "../../data/cy101/normalized_data_without_noobject/"
        for c in self.contexts:
            ori_f = open(self.ori_path + c + ".txt", "r")
            for line in ori_f:
                #print (c)
                self.feature_count_dict[c] = (len(line.split(","))-1)
                break
        #print(feature_count_dict)
        
        #objects = T_oracle.getObjects()


    # create 10 objects dataset with all features initialzed to 0       
    def init_data_object_based(self, objects):

        #10 objects waiting for trainning
        #objects = ["ball_blue", "cup_blue", "tin_snowman", "bigstuffedanimal_bear", "noodle_1", "smallstuffedanimal_bunny", "egg_wood", "timber_pentagon", "timber_rectangle", "weight_1"]

        for c in self.contexts:
            if os.path.exists(self.path + c + ".txt") == False:
                f_10 = open(self.path + c + ".txt", 'x')
                f_10.close()
            f_10 = open(self.path + c + ".txt", 'w')
            for obj in objects:
                for t in range(1, 6):
                    f_10.write(obj + '_t' + str(t))
                    for i in range(0, self.feature_count_dict[c]):
                        f_10.write(',0')
                    f_10.write('\n')
    
    def init_data_predicate_based(self, predicates):
        
        for p in predicates:
            path = self.path + p +'/'
            if not os.path.exists(path):
                os.mkdir(path)
            for c in self.contexts:
                
                if os.path.exists(path + c + ".txt") == False:
                    f = open(path + c + ".txt", 'x')
                    f.close()
           


    def isEmpty(self, line):
        #print(line.split(',')[1:])
        for s in line.split(',')[1:]:
            #print(s)
            if s != '0' and s!= '0\n':
                return False
        return True

    def add_data(self, obj_name, act_name):
        inpath = self.ori_path
        outpath = self.path

        #look into the target data to see which trial of data should be added or all trial data exist
        #since all the action have "surf", look into action_surf.txt

        fcheck = open(outpath +  '/' + act_name + "_surf.txt")
        setStart = False
        start_idx = 0
        count = 0
        for line_idx, line in enumerate(fcheck):
            obj = '_'.join(line.split(',')[0].split('_')[:-1])
            if obj == obj_name:
                if setStart == False:# the first time meet the object
                    start_idx = line_idx
                    setStart = True
                if self.isEmpty(line) == False:
                    count += 1
        target_line = start_idx + count
        #all_lines = fcheck.readlines()
        fcheck.close()

        #store old data for all action related file
        data = {}
        for c in self.contexts:
            if c.split('_')[0] == act_name:
                fold = open(outpath + '/' + c + ".txt")
                data[c] = fold.readlines()
                fold.close()


        if (count == 5): 
            print("All data exist for " + obj_name + " " + act_name)
        else:
            #print(count)
            #print(target_line)
            #read new data in original dataset
            new_line = {}
            for c in self.contexts:
                if c.split('_')[0] == act_name:
                    de_count = count
                    fin = open(inpath  + c + ".txt")
                    for line in fin:
                        obj = '_'.join(line.split(',')[0].split('_')[:-1])
                        if obj == obj_name:
                            if de_count == 0:
                                new_line[c] = line
                            de_count -= 1
                    fin.close()

            #replace
            for key in new_line:
                #print(key)
                data[key][target_line] = new_line[key]

            #put new data back to out file
            for c in self.contexts:
                if c.split('_')[0] == act_name:
                    fout = open(outpath + '/' + c + ".txt", 'w')
                    for l in data[c]:
                        fout.write(l)
                    fout.close()
                    print("Data is added for " + obj_name + ' ' + c)



    def generate_data(self, objects):#generate 10 random objects pretrain data
        path = self.path
        if os.path.exists(path) == False:
            os.mkdir(path)
        
        lines = []
        i = 0

        for fin in self.contexts:
            fout = path + "/" + fin + '.txt'
            fread = open(self.ori_path + fin + '.txt')
            fwrite = open(fout, 'w+')
            #all_lines = fread.readlines()
            for i in range(0, 10):       
                j = 1
                for l in fread:
                    obj = '_'.join(l.split(',')[0].split('_')[:-1])
                    if obj in objects:
                        fwrite.write(l) 
                        #print(5*line+i)

def main(argv):
    #the new dataset path
    path = "/Users/xiaohan/perception_pomdp/src_py/10_objects_test/"
    test = DataPlay(path)
    #test.init_data_10()
    test.add_data('ball_blue', 'crush')

if __name__ == "__main__":
    main(sys.argv)
