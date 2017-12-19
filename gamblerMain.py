#######################################################################
# Copyright (C)                                                       #
# 2017 James LIAO(453143942@qq.com)                                   #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib.pyplot as plt

# In this program, we evaluate and improve the policy simultaneously
stNum = 100  # The number of the possible states
# Epsilon threshold has twofold functions:
# 1. minimum improvement of policy improvement for each state to change state action;
# 2. maximum improvement of entire function to end policy improvement
epsThr = 1.0e-5
probH = 0.4

states = list(range(0,stNum+1)) # The states
# Value function.
# Caution: the value of state 100 is not 1, the "state 100" is the terminal state
valVec = np.zeros(stNum+1)
policy = [0 for ii in range(stNum+1)]  # Action for each state
rewVec = np.zeros(stNum+1)  # Reward function
rewVec[stNum] = 1.0  # Caution: the reward is 1 only when the capital hits 100

# Function for value evaluation
runTime = 1
while True:
    deltaVal = 0.0
    gf=[]
    for qq in (range(1,stNum)):
        oldVal = valVec[qq]
        optValTmp = valVec[qq]
        for ee in range(min(qq+1,stNum+1-qq)):
            futVal = probH * (rewVec[qq+ee] + valVec[qq+ee]) \
                     + (1 - probH) * (rewVec[qq-ee] + valVec[qq-ee])
            # Caution: the action changes if and only if the improvement is larger than epsThr
            if futVal > (optValTmp + epsThr):
                optValTmp = futVal
                valVec[qq] = futVal
                policy[qq] = ee
        gg = abs(valVec[qq]-oldVal)
        gf.append(gg)  # Record the policy improvement for each state
    # Caution: the policy improvement ends if and only if the maximum improvement is smaller than epsThr
    if (max(gf)<epsThr):
        break
    runTime += 1
plt.plot(states, policy)
plt.show()
