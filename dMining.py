import numpy as np
import os
from sklearn.cluster import KMeans
import scipy

def Objective(A, k, colors, N ):
  objectTotal = 0
  for i in range(k):
    B = np.ones(N, int)                                           ##### to initialze the array for each V_i, if m is in V_i then B[m] = 1 else 0.
    C = [] 
    edges = 0 
    for j in range(N):
      if colors[j] == i:
        B[j] = 0
        C.append(j)     
    for r in C:
      edges = edges+np.dot(A[r], B)                       #### get the total edges V_i with (V \ V_i)
    objectTotal = objectTotal + edges/(len(C))
  print(objectTotal)
  file_write = open("output-v.txt", 'w')
  file_write.write(str(objectTotal)+'\n')
  file_write.close()
  return objectTotal


f= open("./graphs_processed/Oregon-1.txt")     ##### Open the inputfile
#f= open("./graphs_processed/ca-GrQc.txt")
#f= open("./graphs_processed/soc-Epinions1.txt")
#f= open("./graphs_processed/web-NotreDame.txt")
k = 5                                        ###### define the cluster number k

line = f.readline()
fileTitle = line
b = line.split()
N = int(b[2])                              ###### extract the vertice number of the graph

#A = np.zeros((N,N), dtype='uint8')              ###### Construct the graph A
A = np.zeros((N,N),int)
print(type(A[0,0]))
temp = 0
while line:
  if temp > 0 :
    a= line.split()  
    i = int(a[0])
    j = int(a[1])
    A[i,j] = 1
    A[j,i] = 1
  temp = temp +1 
  line = f.readline()
#print A[0,4016]      ######## upper is used to extract the graph and build the graph A
#print A[0,0]
#######  then we will go to build the D and calculate L 
f.close()
D = np.zeros((N,N), dtype ='uint16')
#D = np.diag(A.sum(axis=1))                                ###### sum the degree of each node.

for var in range(N):
    temp = 0
    for var1 in range(N):
        temp = temp + A[var,var1]
    D[var,var] = temp
#print(type(D[0,0]))
#for m in range(N):
  
 # if D[m,m] > 65535:
   # print (D[m,m])
#print('done') 
#print D
#L = np.zeros((N,N),dtype = 'float32')
M = D - A
L = M.astype(np.float32)
print(type(L[0,0]))
#print L

#np.savetxt('001',L)
print('will calculate the eigenvector')
#vals, vecs = np.linalg.eigh(L)
vals, vecs = scipy.sparse.linalg.eigsh(L,k=k, which = 'SM')
print('eingen is done')
#index = np.argsort(vals)[:k]
#T = np.real(vecs[:,index])
kmeans = KMeans(algorithm ='elkan' , n_clusters = k, max_iter = 500, n_init = 5000).fit(vecs)

print('go to the objective value')
###### color is an array that  
colors = kmeans.labels_       

if os.path.exists("output.txt"):
    os.remove("output.txt")
file_write = open("output.txt", 'w')

file_write.write(fileTitle)
for var in range(N):
  file_write.write(str(var)+' '+str(colors[var])+'\n')
file_write.close()

#np.savetxt('001',colors)

#### in the next step we will calculate the the objective function. in this function we will use the matrix dot function and 
######  the method is for Vi set and we could set the color elements from Vi to 1. and the new array set B
######  And then let the Sum(A[k]*B) that the total edges Vi with other vertices. 
####### below code is for objective function.

obeject = Objective(A,k,colors, N)

## print(object)


