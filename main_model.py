# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:43:22 2020

@author: Dnyanesh
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
import time
import winsound

tic=time.time()

grid_size=250
#Probability exponent for flow accumulation area #beta
beta=1
alpha=1
#flow direction and flow accumulation matrices
FD=np.zeros((grid_size,grid_size))
Facc=np.zeros((grid_size,grid_size))    

Down_length=np.zeros((grid_size,grid_size)) 
Potential=[]

Label=dict()

#Initial Setup

#All pixel labelled as unassigned "n"

for i in range(grid_size):
    for j in range(grid_size):
        Label.update({(i,j):"n"})
        

#Outward pixels setup
FD[0]=64
FD[grid_size-1]=4
FD[:,0]=16
FD[:,grid_size-1]=1
FD[0][0]=32
FD[0][grid_size-1]=128
FD[grid_size-1][0]=8
FD[grid_size-1][grid_size-1]=2

#All surrounding pixels as drainage outlets and their down_lengths 1
Facc[:,0]=Facc[0]=Facc[grid_size-1]=Facc[:,grid_size-1]=1  
Down_length[1,1:grid_size-1]=Down_length[1:grid_size-1,1]=Down_length[grid_size-2,1:grid_size-1]=Down_length[1:grid_size-1,grid_size-2]=1       
    
#Now updating lists first only border elements without corner elements 
#And later corner elements
   
#For list of potential pixels in first step
for i in range(grid_size-4):    
    Potential.append((i+2,1))
    Potential.append((1,i+2))
    Potential.append((grid_size-2,i+2))
    Potential.append((i+2,grid_size-2))


#inner
Potential.append((1,1))
Potential.append((grid_size-2,1))
Potential.append((1,grid_size-2))
Potential.append((grid_size-2,grid_size-2))

#Updating initial labels of all surrounding pixels to assigned or "y"
for i in range(grid_size):
    Label.update({(i,0):"y"})
    Label.update({(0,i):"y"})
    Label.update({(grid_size-1,i):"y"})
    Label.update({(i,grid_size-1):"y"})

#Updating labels for potential first inside layer pixels
for i in range(grid_size-2):
    Label.update({(i+1,1):"p"})
    Label.update({(1,i+1):"p"})
    Label.update({(grid_size-2,i+1):"p"})
    Label.update({(i+1,grid_size-2):"p"})


#Initial Setup Done
#Now running FD assigning algorithm for all pixels

with open('FD_search.py') as afile:
        exec(afile.read()) 
        
toc=time.time()
time_elapsed=toc - tic

duration=1000 #milliseconds
freq=440 #Hz
winsound.Beep(freq, duration)

'''
#Saving flow direction matrix in csv
np.savetxt('FD_matrix.csv', FD, delimiter=",")

#plotting flowacc
plt.imshow(Facc, cmap='bone')
plt.title('Drainage Network with beta='+str(beta), fontsize=8)
#plt.savefig("drainage_network 1k beta_"+str(beta)+".png", bbox_inches='tight')
#Plotting the sreams 

W=np.where(Facc>100,1,0)
plt.imshow(W, cmap=('gray',N=2),origin='lower',extent=[0,grid_size,0,grid_size])
#plt.title('stream Network with beta='+str(beta), fontsize=8)
plt.savefig("stream_network_blue.png",bbox_inches='tight',dpi=300)
#plt.savefig("stream_network 1k beta_"+str(beta)+".png", bbox_inches='tight')#Plotting flow accumulation
'''

