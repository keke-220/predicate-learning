import sys

import numpy as np
import itertools
def main(argv):
    
    #Success related separate graph
    with open('success_2pred.txt', 'r') as file:
        txt = file.read().replace('\n', '')
    total_success = eval(txt)

    with open('cost_2pred.txt', 'r') as file:
        txt = file.read().replace('\n', '')
    total_result_cost = eval(txt)

    strategy = []
    exp_times = len(total_success.keys())
    for item in total_success[0].keys():
        strategy.append(item)
    num_batch = len(total_success[0][strategy[0]])
    #generating plots:
    markers=itertools.cycle(('v', '*', 'o')) 
    slabel = itertools.cycle(('Random', 'Passive', 'ITRS'))
    color = itertools.cycle(('#ff8243', '#c043ff', '#82ff43'))
    import matplotlib.pyplot as plt
    #plt.style.use('seaborn-whitegrid')
    '''
    x= []
    for i in range(num_batch):
        x.append(i)

    y = {}
    y_err = {}
    ymin = {}
    ymax = {}
    for s in strategy:
        y[s] = []
        y_err[s] = []
        for i in range(num_batch):
            total = 0
            y_data = []
            for j in range(exp_times):
                    y_data.append(total_success[j][s][i])
                    total += total_success[j][s][i]
            y_err[s].append(np.std(y_data))
            avg = total/exp_times
            y[s].append(avg)

        ymin[s] = np.array(y[s]) - np.array(y_err[s])
        ymax[s] = np.array(y[s]) + np.array(y_err[s])

    fig, ax = plt.subplots()
    plt.xlim(0, num_batch-1)
    plt.grid(True)
    plt.ylim(0.55, 0.9)
    plt.xlabel("Batch Index", fontsize = 13)
    plt.ylabel("Accuracy (%)", fontsize = 13)
    #plt.title("Task Completion Accuracy", fontsize = 18)
    for s in strategy:
        ax.errorbar(x, y[s], label = next(slabel), yerr = None, marker = next(markers), mec = 'black', markersize = 7, fmt = '-', elinewidth = 2)
        ax.fill_between(x, ymax[s], ymin[s],alpha=0.5)
        
    ax.legend(loc='lower right', numpoints=1)
    fig.savefig('success.pdf')
    
    

    #COst related separate graph
    x= []
    for i in range(num_batch):
        x.append(i)

    y = {}
    y_err = {}
    ymin = {}
    ymax = {}
    for s in strategy:
        y[s] = []
        y_err[s] = []
        for i in range(num_batch):
            total = 0

                    #appendeditem = 0
                    #for t in range(0, i+1):
                    #appendeditem += total_result_cost[j][s][t]

                    #y_data.append(appendeditem)
                    #total += appendeditem
                y_data.append(total_result_cost[j][s][i])
                total += total_result_cost[j][s][i]
            y_err[s].append(np.std(y_data))
            avg = total/exp_times
            y[s].append(avg)

        ymin[s] = np.array(y[s]) - np.array(y_err[s])
        ymax[s] = np.array(y[s]) + np.array(y_err[s])

    fig, ax = plt.subplots()
    plt.xlim(-0.5, num_batch-0.5)
    plt.grid(True)
    #plt.ylim(15, 65)
    plt.xlabel("Batch Index", fontsize = 13)
    plt.ylabel("Action Cost", fontsize = 13)
    #plt.title("Average Action Cost", fontsize = 18)
    for s in strategy:
        ax.errorbar(x, y[s], label = next(slabel), yerr = None, marker = next(markers), mec = 'black', markersize = 7, fmt = '-', elinewidth = 2)
        ax.fill_between(x, ymax[s], ymin[s],alpha=0.5)
    
    ax.legend(loc='upper right', numpoints=1)
    fig.savefig('cost.pdf') 
    '''

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

    fig, ax = plt.subplots()
    plt.grid(True)
    #plt.ylim(65, 87.5)
    plt.ylim(27.5, 72.5)
    #plt.xlim(-0.1, 5)
    plt.xlim(-0.25, 17)
    plt.xlabel("Exploration time (h)", fontsize = 14)
    plt.ylabel("Accuracy (%)", fontsize = 14)
    #plt.title("Accuracy", fontsize = 18)
    for s in strategy:
        color1 = next(color)
        ax.errorbar(x[s], y[s], label = next(slabel),color = color1,yerr = None, marker = next(markers), mec = 'black', markersize = 7, fmt = '-', elinewidth = 2)
        ax.fill_between(x[s], ymax[s], ymin[s],alpha=0.5, color = color1)
        
        #ax.fill_between(y, xmax[s], xmin[s],alpha=0.5)
    #ax.set_xscale("log")
    ax.legend(loc='lower right', numpoints=1)
    fig.savefig('auc_acc.pdf') 

   
if __name__ == '__main__':
    main(sys.argv)
                #appendeditem = 0
                #for t in range(0, i+1):
                #    appendeditem += total_result_cost[j][s][t]

                #y_data.append(appendeditem)
                #total += appendeditem
 
