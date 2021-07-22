# -*- coding: utf-8 -*-

# ///--- Greenberg Hastings Model ----///
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import random

a1 = datetime.datetime.now()

#Parameters
N = 5000  #Neurons
Avg_degree = 10    #Connectivity
pi = 0.6  #Rewiring probability 
scale = 1/12.5 #Lambda=12.5
r1 = 1e-3  # Spontaneous neuron activation
r2 = 0.3 #Probability of quiescent state
n_s = 1000 #time steps

# ///---- States and Control Parameter ----///
# Q = quiescent, E = excited, R = refractory, T = threshold
Q = 0
E = 1
R = 2
T_0 = 0
T_F= 0.4
n_t= 40
T_list = np.linspace(T_0, T_F, n_t)
S = np.random.randint(0, 3, size=N)
activity = np.zeros(n_s)
Time=np.zeros(n_s)
f_s = []

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
            weightsMatrix[i,j]=np.random.exponential(scale)
            for h in range(count[NeighborMat[i,j]]):
                if i==NeighborMat[NeighborMat[i,j],h]:
                    weightsMatrix[NeighborMat[i,j],h]=weightsMatrix[i,j]

del conectivityMatrix
# ///---- Dynamics ----///
for T in T_list:
    print(float("{0:.2f}".format(T)))
    S = np.random.randint(0, 3, size=N)
    for t in range(n_s):
        activity[t] = np.sum(S == E)
        Time[t]=t
        S_prev = S.copy()
        for i in range(N):
            if S_prev[i] == Q:
                if np.random.uniform(0,1) <= r1:
                    S[i] = E
                else:
                    activity_neighbors=0
                    for j in range(count[i]): #for all j active neighbors
                        if (S_prev[NeighborMat[i,j]]==E):
                            activity_neighbors=activity_neighbors+weightsMatrix[i,j]
                    if activity_neighbors >= T:
                        S[i] = E
            elif S_prev[i] == E:
                S[i] = R
            else:
                if np.random.uniform(0,1) <= r2:
                    S[i] = Q
    f_s.append(np.mean(activity)/N)   
# ///---- Data file ----///
    filename = 'actN%sK%sPI%sT%s.txt' % (N,Avg_degree,int(pi*100),int(T*1000))
    file=open(filename,"w")
    np.savetxt(filename , np.column_stack([Time, activity]), fmt=['%d','%d'])
    file.close()
filename2 = 'f_svsTN%sK%sPI%sT%s.txt' % (N,Avg_degree,int(pi*100))
file1=open(filename2,"w")
np.savetxt(filename2 , np.column_stack([T_list, f_s]), fmt=['%f','%f'])
file1.close()
plt.xlabel("T")
plt.ylabel("$f_s$")
plt.plot(T_list,f_s,linestyle='solid',marker='o',color='blue')
plt.show()
a2 = datetime.datetime.now()
b=a2-a1
print("time",a2-a1)
