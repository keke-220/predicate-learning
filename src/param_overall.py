import sys

import numpy as np
import itertools
def main(argv):
    total_success = {}
    total_result_cost = {}
    raw_success = {}
    raw_result_cost = {}

    for alpha in [2, 4, 6, 8, 10]:
        raw_success[alpha] = {}
        raw_result_cost[alpha] = {}
        for beta in [10, 20, 30 ,40, 50]:
            with open('param_result/success_'+str(alpha)+'_'+str(beta)+'.txt', 'r') as file:
                txt = file.read().replace('\n', '')
            raw_success[alpha][beta] = eval(txt)
            with open('param_result/cost_'+str(alpha)+'_'+str(beta)+'.txt', 'r') as file:
                txt = file.read().replace('\n', '')
            raw_result_cost[alpha][beta] = eval(txt)



    strategy = []
    exp_times = len(raw_success[2][10].keys())
    for item in raw_success[2][10][0].keys():
        strategy.append(item)
    num_batch = len(raw_success[2][10][0][strategy[0]])
    
    strategy = []
    for alpha in [2,4,6,8,10]:
        for beta in [10,20,30,40,50]:
            string = str(alpha) + '/' + str(beta)
            strategy.append(string)

    for e in range(0, exp_times):
        total_success[e] = {}
        total_result_cost[e] = {}
        for p in strategy:
            alpha = int(p.split('/')[0])
            beta = int(p.split('/')[1])
            total_success[e][p] = raw_success[alpha][beta][e]['IT_learning']
            total_result_cost[e][p] = raw_result_cost[alpha][beta][e]['IT_learning']
            


    #generating plots:
    markers=itertools.cycle(('v', '*', 'o', 's', 'x', '.',',','^', '<','>','1', '2', '3', '4', '8' ,'p')) 
    slabel = itertools.cycle(strategy)
    import matplotlib.pyplot as plt
    #plt.style.use('seaborn-whitegrid')
    #Combine success rate and cost to a one-dimention graph
    x= {}
    y = {}
    y_err = {}
    x_err = {}
    ymin = {}
    ymax = {}
    xmin = {}
    xmax = {}
    for s in strategy:
        y[s] = []
        x[s] = []
        y_err[s] = []
        x_err[s] = []
        for i in range(0, num_batch):
            total = 0
            y_data = []
            for j in range(exp_times):
                    y_data.append(total_success[j][s][i])
                    total += total_success[j][s][i]
            y_err[s].append(100*np.std(y_data))
            avg = total/exp_times
            y[s].append(100*avg)

        ymin[s] = np.array(y[s]) - np.array(y_err[s])
        ymax[s] = np.array(y[s]) + np.array(y_err[s])

        for i in range(0, num_batch):
            total = 0
            x_data = []
            for j in range(exp_times):
                appendeditem = 0
                for t in range(0, i+1):
                    appendeditem += total_result_cost[j][s][t]
                    x_data.append(appendeditem)
                total += appendeditem
                    #y_data.append(total_result_cost[j][s][i])
                    #total += total_result_cost[j][s][i]
            x_err[s].append((np.std(x_data))/3600)
            avg = total/exp_times
            x[s].append(avg/3600)

        xmin[s] = np.array(x[s]) - np.array(x_err[s])
        xmax[s] = np.array(x[s]) + np.array(x_err[s])
    
    """
    fig, ax = plt.subplots()
    plt.grid(True)
    #plt.ylim(0.675, 0.875)
   # plt.xlim(40, 550)
    plt.xlabel("Training time (h)", fontsize = 13)
    plt.ylabel("Accuracy (%)", fontsize = 13)
    #plt.title("Accuracy", fontsize = 18)
    for s in strategy:
        ax.errorbar(x[s], y[s], label = next(slabel), yerr = None, marker = next(markers), mec = 'black', markersize = 7, fmt = '-', elinewidth = 2)
        #ax.fill_between(x[s], ymax[s], ymin[s],alpha=0.5)
        
        #ax.fill_between(y, xmax[s], xmin[s],alpha=0.5)
    #ax.set_xscale("log")
    ax.legend(loc='lower right', ncol = 5)
    fig.savefig('auc_acc_test.pdf') 
    """
    data = []

    budget = 2
    for alpha in [2,4,6,8,10]:
        row = []
        for beta in [10, 20, 30, 40, 50]:
            s = str(alpha) + '/' + str(beta)
            for i in range(len(x[s])):
                if x[s][i] >= budget:
                    row.append(y[s][i])
                    break
        data.append(row)
    print (data)
    plt.ylabel('Alpha')
    plt.xlabel('Beta')
    plt.imshow(data, origin = 'lower',  extent = [5, 55, 1, 11], aspect = 5, cmap = 'binary') 
    plt.colorbar()
    plt.show()

# Just some example data (random)
    #data = np.random.rand(10,5)

    #rows,cols = data.shape

    #print (data)
   
if __name__ == '__main__':
    main(sys.argv)
                #appendeditem = 0
                #for t in range(0, i+1):
                #    appendeditem += total_result_cost[j][s][t]

                #y_data.append(appendeditem)
                #total += appendeditem
 
