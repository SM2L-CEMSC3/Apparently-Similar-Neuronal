# -*- coding: utf-8 -*-

# ///--- KC quiescent ----///

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime

a1 = datetime.datetime.now()

#Parameters
N = 5000  #Neurons
Avg_degree = 30    #Connectivity
pi = 0.6  #Rewiring probability 
r1 = 0.001 # Spontaneous neuron activation
n_s = 1000 #time steps

# ///---- States ----///
#Q: quiescent, E: excited, R:refractory
Q = 0
E = 1
R = 2 or 3 or 4 
S = np.random.randint(0, 5, size=N)

# ///---- Network + weightsMatrix ----///
conectivityMatrix = nx.watts_strogatz_graph(N, Avg_degree, pi, seed=None)
NeighborMat = np.zeros((N, 5*Avg_degree), dtype=int)-(int(1))
count = np.zeros(N,dtype=int)
c=0
for i in list(conectivityMatrix.nodes): #For the list of nodes i
        pos=0
        
        for j in list(conectivityMatrix.neighbors(i)): #For each neighbors j of i
            NeighborMat[i,pos]=int(j)
            pos=pos+1
            count[c]=count[c]+1
        c=c+1

weightsMatrix = np.zeros((N, 5*Avg_degree))+-1 #Memory
for i in range(N):
    for j in range(count[i]):
        if weightsMatrix[i,j]==-1:
            weightsMatrix[i,j]=np.random.uniform(0,1)
            for h in range(count[NeighborMat[i,j]]):
                if i==NeighborMat[NeighborMat[i,j],h]:
                    weightsMatrix[NeighborMat[i,j],h]=weightsMatrix[i,j]

del conectivityMatrix
activity = np.zeros(n_s)
Time=np.zeros(n_s)
f_s = []

# ///---- Dynamics ----///
sigma_0 = 0.5
sigma_f = 2
Delta_sigma = 16

sigma_values = np.linspace(sigma_0 , sigma_f, Delta_sigma)
for sigma in sigma_values:
    S = np.random.randint(0, 4, size=N)
    print(sigma)
    pmax = (2*sigma)/(Avg_degree-1)
    for t in range(n_s):
        activity[t] = np.sum(S == E)
        Time[t]=t
        S_prev = S.copy()
        
        for i in range(N):
            if S_prev[i] == Q:
                if np.random.rand() <= r1:
                    S[i] = E
            elif S_prev[i] == E:
                pos=0
                for j in range(count[i]):
                    if S_prev[NeighborMat[i,j]] == Q and np.random.rand() <= weightsMatrix[i,j]*pmax:
                        S[NeighborMat[i,j]]=E
                    pos=pos+1
            
            if S_prev[i] >= 1:
                S[i] = (S[i] + 1) % 5

    f_s.append(np.mean(activity)/N)      
# ///---- Data file ----///
    filename = 'act%sK%sPI%sSigma%s.txt' % (N,Avg_degree,int(pi*100),int(sigma*1000))
    file=open(filename,"w")
    np.savetxt(filename , np.column_stack([Time, activity]), fmt=['%d','%d'])
    file.close() 
filename2 = 'f_svsTN%sK%sPI%s.txt' % (N,Avg_degree,int(pi*100))
file1=open(filename2,"w")
np.savetxt(filename2 , np.column_stack([sigma_values, f_s]), fmt=['%f','%f'])
file1.close()
plt.xlabel("$\sigma$")
plt.ylabel("$f_s$")
plt.plot(sigma_values,f_s,linestyle='solid',marker='o',color='blue')
plt.show()
a2 = datetime.datetime.now()
b=a2-a1
print("time",a2-a1)
