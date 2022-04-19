# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:43:22 2020

@author: Dnyanesh

This code does initial setup for probabilistic drainage network evolution model
And then with FD_search algorithm code it runs simulation and gives resulting drainage network

This code is written for generl boundary raster input which is obtained from vector shapefiles of required area
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
from osgeo import gdal
import time
import winsound

tic=time.time()


img_boundary = gdal.Open('tasmania1.tif')
img_shp = gdal.Open('tasmania_polygon1.tif') # read the polygon which is within/overlapping with boundary

arr_boundary = np.array(img_boundary.GetRasterBand(1).ReadAsArray())
arr_shp =  np.array(img_shp.GetRasterBand(1).ReadAsArray())

grid_size=arr_shp.shape

#Probability exponent for flow accumulation area #beta
beta=1
alpha=1

#flow direction and flow accumulation matrices
FD=np.zeros((grid_size[0],grid_size[1]))
Facc=np.zeros((grid_size[0],grid_size[1]))    

Down_length=np.zeros((grid_size[0],grid_size[1])) 
Potential=[]

Label=dict()

## Initial Setup

true_area=np.zeros((grid_size[0],grid_size[1])) # matrix true for area and boundary

#All pixel labelled as unassigned "n" & true area matrix defination

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        if arr_boundary[i,j]>0 or arr_shp[i,j]>0 :
            Label.update({(i,j):"n"})
            true_area[i,j]=True
        
# Let's define boundary pixels first
for pi in Label:
    if arr_boundary[pi[0],pi[1]]>0:
        Label.update({(pi[0],pi[1]):"b"})
        Facc[pi[0],pi[1]]=1
            
#Now creating list of potential pixels and updating their labels

ij=((0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128))
border={'left':((-1,0,64),(-1,1,128),(0,1,1),(1,1,2),(1,0,4)),'top':((0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16)),'right':((1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64)),'bottom':((0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128),(0,1,1))}
corner={'bottom_left':((-1,0,64),(-1,1,128),(0,1,1)),'left_top':((0,1,1),(1,1,2),(1,0,4)),'top_right':((1,0,4),(1,-1,8),(0,-1,16)),'right_bottom':((0,-1,16),(-1,-1,32),(-1,0,64))}

for pi in Label:
    if Label[pi[0],pi[1]]=="b":
        if (pi[0]==0 or pi[0]==grid_size[0]-1 or pi[1]==0 or pi[1]==grid_size[1]-1):
            if (pi[0]==grid_size[0]-1 and pi[1]==0):
                ij_list=corner['bottom_left']
            elif (pi[0]==0 and pi[1]==0):
                ij_list=corner['left_top']
            elif (pi[0]==0 and pi[1]==grid_size[1]-1):
                ij_list=corner['top_right']
            elif (pi[0]==grid_size[0]-1 and pi[1]==grid_size[1]-1):
                ij_list=corner['right_bottom']
                
            elif (pi[1]==0):
                ij_list=border['left']
            elif (pi[0]==0):
                ij_list=border['top']
            elif (pi[1]==grid_size[1]-1):
                ij_list=border['right']
            elif (pi[0]==grid_size[0]-1):
                ij_list=border['bottom']
        else:
            ij_list=ij
        for y in ij_list:
            if arr_shp[(pi[0]+y[0],pi[1]+y[1])]>0:
                if Label[(pi[0]+y[0],pi[1]+y[1])]=='n':
                    Potential.append((pi[0]+y[0],pi[1]+y[1]))
                    Label.update({(pi[0]+y[0],pi[1]+y[1]):"p"})
                    Down_length[pi[0]+y[0],pi[1]+y[1]]=1

# No need to assign flow directions to the border pixels i.e. outlets
        
'''
#Cross checking initial setup
fig=np.zeros((grid_size[0],grid_size[1]))       
'''          
 

#Initial Setup Done
#Now running FD assigning algorithm for all pixels

with open('FD_search.py') as afile:
        exec(afile.read()) 
        
toc=time.time()
time_elapsed=toc - tic
'''
duration=1000 #milliseconds
freq=440 #Hz
winsound.Beep(freq, duration)
'''
'''
#Saving flow direction matrix in csv
np.savetxt('FD_matrix.csv', FD, delimiter=",")

#plotting flowacc
plt.imshow(Facc, cmap='gray')
plt.title('Drainage Network with beta='+str(beta), fontsize=8)
#plt.savefig("drainage_network 1k beta_"+str(beta)+".png", bbox_inches='tight')
#Plotting the sreams 

W=np.where(Facc>40,1,0)
plt.imshow(W, cmap='gray')
plt.title('stream Network with beta='+str(beta), fontsize=8)
#plt.savefig("stream_network 1k beta_"+str(beta)+".png", bbox_inches='tight')#Plotting flow accumulation
'''

