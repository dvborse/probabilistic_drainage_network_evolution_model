# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:46:21 2023

@author: dnyan
"""

# Get initial drainage network from this code It'scombined rand_walk_lem and FD_search code from single basin case

import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
#from osgeo import gdal
import random 
import time
import winsound

def FD_cordinates(xy):
    '''
    Function for updating flow accumulation values along the stream
    Parameters
    ----------
    xy : tupule
        function which returns cordinates of pixel where flow is flowing
        suppose FD for Pi is 1 then this will return a tuple with cordinates of pixel right of pi

    Returns
    tupule of cordinates of nerby pixel as per flow direction
    '''
    if FD[xy[0]][xy[1]]==1:
        return((xy[0],xy[1]+1))
    if FD[xy[0]][xy[1]]==2:
        return((xy[0]+1,xy[1]+1))
    if FD[xy[0]][xy[1]]==4:
        return((xy[0]+1,xy[1]))
    if FD[xy[0]][xy[1]]==8:
        return((xy[0]+1,xy[1]-1))
    if FD[xy[0]][xy[1]]==16:
        return((xy[0],xy[1]-1))
    if FD[xy[0]][xy[1]]==32:
        return((xy[0]-1,xy[1]-1))
    if FD[xy[0]][xy[1]]==64:
        return((xy[0]-1,xy[1]))
    if FD[xy[0]][xy[1]]==128:
        return((xy[0]-1,xy[1]+1))

def Get_cord(ix):
    i=int(ix/grid_size[1])
    j=ix%grid_size[1]
    return((i,j))

def Get_id(i,j):
    return grid_size[1]*i+j

#colourmap for figures
norm=plt.Normalize(0,1)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","blue"])

#%% Initial setup

tic=time.time()

# #img_boundary = gdal.Open('UG_basin_0.01.tif')
# arr_shp =imageio.imread('CBG_resampled_0005.tif')>0 # read the polygon which is within/overlapping with boundary
# #arr_shp=np.flip(arr_shp,axis=0)
# grid_size=arr_shp.shape
# arr_shp[0,:]=arr_shp[grid_size[0]-1,:]=arr_shp[:,0]=arr_shp[:,grid_size[1]-1]=0
# We need only one polygon no need of polyline this code
#arr_boundary = np.array(img_boundary.GetRasterBand(1).ReadAsArray())
#arr_shp =  np.array(img_shp)

watershed_sb=np.load("watershed_sb.npy")
zeros_row = np.zeros((1, watershed_sb.shape[1]))
# Add the row of zeros to the existing array
arr_shp = np.vstack((watershed_sb, zeros_row))
grid_size=arr_shp.shape
#Probability exponent for flow accumulation area #beta
outlet=(173,116) #Enter manually 

beta=1
alpha=1

r=0  #optional parameter - rate of channel growth r=0.1~ 10% potential pixels grow every time

#flow direction and flow accumulation matrices
FD=np.zeros((grid_size[0],grid_size[1]))
Facc=np.zeros((grid_size[0],grid_size[1]))    
Area=np.zeros(grid_size[0]*grid_size[1])      # single array
Length=np.zeros(grid_size[0]*grid_size[1])      # single array
Down_length=np.zeros((grid_size[0],grid_size[1])) 

Pot_ids=[]
Label=dict()

## Initial Setup
true_area=np.zeros((grid_size[0],grid_size[1])) # matrix true for area and boundary

#All pixel labelled as unassigned "n" & true area matrix defination
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        if arr_shp[i,j]>0 :
            Label.update({(i,j):"n"})
            true_area[i,j]=True
        
# Let's define boundary pixels first
# for pi in Label:
#     if arr_boundary[pi[0],pi[1]]>0:
#         Label.update({(pi[0],pi[1]):"b"})
        #Facc[pi[0],pi[1]]=1
            
#Now creating list of potential pixels and updating their labels

ij=((0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128))
#border={'left':((-1,0,64),(-1,1,128),(0,1,1),(1,1,2),(1,0,4)),'top':((0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16)),'right':((1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64)),'bottom':((0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128),(0,1,1))}
#corner={'bottom_left':((-1,0,64),(-1,1,128),(0,1,1)),'left_top':((0,1,1),(1,1,2),(1,0,4)),'top_right':((1,0,4),(1,-1,8),(0,-1,16)),'right_bottom':((0,-1,16),(-1,-1,32),(-1,0,64))}

## Now only for outlet
pi=outlet

# if (pi[0]==0 or pi[0]==grid_size[0]-1 or pi[1]==0 or pi[1]==grid_size[1]-1):
#     if (pi[0]==grid_size[0]-1 and pi[1]==0):
#         ij_list=corner['bottom_left']
#     elif (pi[0]==0 and pi[1]==0):
#         ij_list=corner['left_top']
#     elif (pi[0]==0 and pi[1]==grid_size[1]-1):
#         ij_list=corner['top_right']
#     elif (pi[0]==grid_size[0]-1 and pi[1]==grid_size[1]-1):
#         ij_list=corner['right_bottom']
        
#     elif (pi[1]==0):
#         ij_list=border['left']
#     elif (pi[0]==0):
#         ij_list=border['top']
#     elif (pi[1]==grid_size[1]-1):
#         ij_list=border['right']
#     elif (pi[0]==grid_size[0]-1):
#         ij_list=border['bottom']
# else:
#     ij_list=ij
for y in ij:
    if arr_shp[(pi[0]+y[0],pi[1]+y[1])]>0:
        if Label[(pi[0]+y[0],pi[1]+y[1])]=='n':
            iid=Get_id(pi[0]+y[0],pi[1]+y[1])
            #Pot_ids.append(grid_size[1]*(pi[0]+y[0])+(pi[1]+y[1]))
            Pot_ids.append(iid)
            Label.update({(pi[0]+y[0],pi[1]+y[1]):"p"})
            Length[iid]=1
            Down_length[pi[0]+y[0],pi[1]+y[1]]=1

Label.update({(outlet[0],outlet[1]):"o"})

#%%  FD_seach
while(len(Pot_ids)>0):
    pot_len=Length[Pot_ids]
    len_alpha=pot_len**alpha
    cum_len=np.cumsum(len_alpha)
    
    rand_n=random.randint(1,int(cum_len[-1]))
    
    pi_id=-1
    for j in range(len(Pot_ids)):
        if rand_n<=cum_len[j]:
            pi_id=Pot_ids[j]
            break
        
    pi=Get_cord(pi_id)
    
    # Choose flow direction
    Values=[] #list of tupules containing FD, Facc and Facc cumulative for surrounding drainage pixels
    Facc_cum=0
    for y in ij:
        if true_area[pi[0]+y[0],pi[1]+y[1]]: 
            if Label[(pi[0]+y[0],pi[1]+y[1])]=='y' or Label[(pi[0]+y[0],pi[1]+y[1])]=='o':
                  Facc_cum+=int((1+Facc[pi[0]+y[0]][pi[1]+y[1]])**beta)
                  Values.append((y[2],Facc[pi[0]+y[0]][pi[1]+y[1]],Facc_cum))

    sum_facc=Values[-1][2]
    value_rand=random.randint(1,sum_facc)
    
    for i in range(len(Values)):
        if value_rand<=Values[i][2]:
            FD[pi[0]][pi[1]]=Values[i][0]
            break
    #Now updating labels and list for pi 
    #Potential.remove((pi))
    Pot_ids.remove(pi_id)
    Label.update({pi:"y"})
    
    #Updating Down_length for pi
    cordinates=FD_cordinates(pi)
    if (FD[pi[0]][pi[1]]==2 or FD[pi[0]][pi[1]]==8 or FD[pi[0]][pi[1]]==32 or FD[pi[0]][pi[1]]==128):
        Down_length[pi[0]][pi[1]]=Down_length[cordinates[0]][cordinates[1]]+np.sqrt(2)
    else:
        Down_length[pi[0]][pi[1]]=Down_length[cordinates[0]][cordinates[1]]+1
        
    #Updating labels and lists for and surrounding pixels of pi
    for x in ij:
        if true_area[pi[0]+x[0],pi[1]+x[1]]:
            if Label[(pi[0]+x[0],pi[1]+x[1])]=='n':
                Label[(pi[0]+x[0],pi[1]+x[1])]='p'
                Pot_ids.append(grid_size[1]*(pi[0]+x[0])+pi[1]+x[1])
                Down_length[pi[0]+x[0]][pi[1]+x[1]]=1+Down_length[pi[0]][pi[1]]
                Length[grid_size[1]*(pi[0]+x[0])+pi[1]+x[1]]=1+Length[grid_size[1]*pi[0]+pi[1]]
    #Now updating flow accumulation value for pi and then along the stream
    current_pi=pi
    Facc[pi[0]][pi[1]]=1
    Area[grid_size[1]*pi[0]+pi[1]]=1

    while(not(Label[(current_pi[0],current_pi[1])]=="o")):
        current_pi=FD_cordinates(current_pi)
        Facc[current_pi[0],current_pi[1]]+=1  
        Area[grid_size[1]*current_pi[0]+current_pi[1]]+=1
#%%
W=np.where(Facc>50,1,0)
#plt.imshow(W, cmap=cmap, origin='lower',extent=[0,grid_size[0],0,grid_size])
plt.imshow(W, cmap=cmap)
plt.title(f"Stream Network_\u03B1_{alpha}_\u03B2_{beta}_r_{r}", fontsize=8)
plt.savefig(f"Stream Network_\u03B1_{alpha}_\u03B2_{beta}_r_{r}.png",dpi=300)

#%%
# W=np.where(Facc>50,1,0)
# #plt.imshow(W, cmap=cmap, origin='lower',extent=[0,grid_size[0],0,grid_size])
# plt.imshow(W, cmap=cmap)
# plt.title("Initial stream Network", fontsize=8)
# plt.savefig("Initial Network.png",dpi=300)
