#!/usr/bin/env python

import sys
import numpy as np
import os.path
import subprocess

class Policy(object):
  actions = None
  policy = None

  def __init__(self, num_states, num_actions, filename='output.policy'):
    try:
      f = open(filename, 'r')
    except:
      print('\nError: unable to open file: ' + filename)

    lines = f.readlines()

    # the first three and the last lines are not related to the actual policy
    lines = lines[3:]

    self.actions = -1 * np.ones((len(lines), 1, ))
    self.policy = np.zeros((len(lines), num_states, ))

    for i in range(len(lines)):
      # print("this line:\n\n" + lines[i])
      if lines[i].find('/AlphaVector') >= 0:
        break
      l = lines[i].find('"')
      r = lines[i].find('"', l + 1)
      self.actions[i] = int(lines[i][l + 1 : r])

      ll = lines[i].find('>')
      rr = lines[i].find(' <')
      # print(str(i))
      self.policy[i] = np.matrix(lines[i][ll + 1 : rr])

    f.close()
    
  def select_action(self, b):
    
    # sanity check if probabilities sum up to 1
    if sum(b) - 1.0 > 0.00001:
      print('Error: belief does not sum to 1, diff: ', sum(b)[0] - 1.0)
      sys.exit()

    return int(self.actions[np.argmax(np.dot(self.policy, b.T)), 0])
    # return np.argmax(b) + 12
    # return np.random.randint(24, size=1)[0]


class Solver(object):

    def __init__(self):
        pass

    def compute_policy(self, model_name, policy_name, appl_path, timeout):
        if 'pomdp' not in model_name or 'policy' not in policy_name:
            sys.exit('Error: model name or policy name incorrect')

        if os.path.isfile(appl_path) is False:
            sys.exit('Error: appl binary does not exist')

        subprocess.check_output([appl_path, model_name, \
            '--timeout', str(timeout), '--output', policy_name])



