# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:17:17 2020

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt


def FD_cordinates(xy):
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

#Watershed 
max_l=np.where(Down_length == np.amax(Down_length))
max_l=(max_l[0][0],max_l[1][0])
x_pi=max_l 
while(not(x_pi[0]==0 or x_pi[0]==grid_size-1 or x_pi[1]==0 or x_pi[1]==grid_size-1)):
    c_pi=x_pi
    x_pi=FD_cordinates(x_pi)
outlet=x_pi
#c_pi is pixel just before outlet
watershed=np.zeros((grid_size,grid_size))
watershed[outlet[0]][outlet[1]]=1
xij=[(0,1,16),(1,1,32),(1,0,64),(1,-1,128),(0,-1,1),(-1,-1,2),(-1,0,4),(-1,1,8)]

#Now making list of neighbouring pixels flowing into outlet
unassigned={c_pi}  # set so that repeat pixel is not added

for x in xij:
    if FD_cordinates((c_pi[0]+x[0],c_pi[1]+x[1]))==outlet:
        unassigned.add((c_pi[0]+x[0],c_pi[1]+x[1]))
        c_pi=(c_pi[0]+x[0],c_pi[1]+x[1])
#if one more pixel is added along with c_pi then we should check surrounding pixels of that too
if len(unassigned)>1:
    for x in xij:
        if FD_cordinates((c_pi[0]+x[0],c_pi[1]+x[1]))==outlet:
            unassigned.add((c_pi[0]+x[0],c_pi[1]+x[1]))

unassigned1=[] #converting set object unassigned to list
for a in unassigned:
    unassigned1.append(a)
unassigned=unassigned1
for pixel in unassigned:
    watershed[pixel[0]][pixel[1]]=1
 

while(len(unassigned)>0):
    for pixel in unassigned:
        for ix in xij:
            if FD[(pixel[0]+ix[0],pixel[1]+ix[1])]==ix[2]:
                watershed[pixel[0]+ix[0]][pixel[1]+ix[1]]=1
                unassigned.append((pixel[0]+ix[0],pixel[1]+ix[1]))
        unassigned.remove(pixel)

#UP_length
Up_length=np.zeros((grid_size,grid_size)) 

#Calculation for Up length for all pixels of matrix 
for i in range(grid_size):
    for j in range(grid_size):
        Up_list=[0] #list for storing uplengths in a stream
        if Up_length[i][j]==0:
            xpi=(i,j)
            while(not(xpi[0]==0 or xpi[0]==grid_size-1 or xpi[1]==0 or xpi[1]==grid_size-1)):
                next_pi=FD_cordinates((xpi[0],xpi[1]))
                if (FD[xpi[0]][xpi[1]]==2 or FD[xpi[0]][xpi[1]]==8 or FD[xpi[0]][xpi[1]]==32 or FD[xpi[0]][xpi[1]]==128):
                    Up_list.append(Up_list[-1]+np.sqrt(2))
                else:
                    Up_list.append(Up_list[-1]+1)
                if Up_length[next_pi[0]][next_pi[1]] < Up_list[-1]:
                    Up_length[next_pi[0]][next_pi[1]]=Up_list[-1]
                    xpi=next_pi
                    continue
                else:
                    break
        else:
            continue

#Hack's law
# Making lists for plotting
H_length=[]
H_area=[]
for i in range(grid_size):
    for j in range(grid_size):
        if Facc[i][j]>50:
        #if Up_length[i][j]>150:   
            
            H_area.append(Facc[i][j])
            H_length.append(Up_length[i][j])
            

h, k = np.polyfit(np.log(H_area),np.log(H_length), 1)

k=np.exp(k)

H_length_cal=k*H_area**h

#Exceedance probabilitiues are for single largest watresheds

#Exceedance probability of drainage length
Up_dist=np.zeros((grid_size,grid_size))  #contributing area matrix clipped for watershed
for i in range(grid_size):
    for j in range(grid_size):
        if watershed[i][j]>0:
            Up_dist[i][j]=Up_length[i][j]
            
max_Uplength=np.amax(Up_dist)
Total_pi_uplength=(Up_dist>0).sum()
P_uplength=np.zeros((int(max_Uplength),2))

for i in range(int(max_Uplength)):
    P_uplength[i][0]=i+1
    P_uplength[i][1]=((Up_dist>=i+1).sum())/Total_pi_uplength

#####    
#Exceedance probability of drainage area
con_area=np.zeros((grid_size,grid_size))  #contributing area matrix clipped for watershed
for i in range(grid_size):
    for j in range(grid_size):
        if watershed[i][j]>0:
            con_area[i][j]=Facc[i][j]
         
max_conarea=np.amax(con_area)
Total_pi_conarea=(con_area>0).sum()
P_conarea=np.zeros((int(max_conarea),2)) 
    
for i in range(int(max_conarea)):
    P_conarea[i][0]=i+1
    P_conarea[i][1]=((con_area>=i+1).sum())/Total_pi_conarea

###****  Slope Break  ****####

#choose number of points till slopebreak

sb_a=0
sb_l=0

for ix in P_conarea[:,1]:
    if ix>0.03:
        sb_a+=1
    else:
        break

for iy in P_uplength[:,1]:
    if iy>0.05:
        sb_l+=1
    else:
        break
#W/o slope break
#slope_a, intercept_a = np.polyfit(np.log(P_conarea[:,0]),np.log(P_conarea[:,1]), 1)
#slope_l, intercept_l = np.polyfit(np.log(P_uplength[:,0]),np.log(P_uplength[:,1]), 1)

#with slope break
slope_a, intercept_a = np.polyfit(np.log(P_conarea[3:sb_a,0]),np.log(P_conarea[3:sb_a,1]), 1)
slope_l, intercept_l = np.polyfit(np.log(P_uplength[:sb_l,0]),np.log(P_uplength[:sb_l,1]), 1)

intercept_a=np.exp(intercept_a)
P_conarea_cal=intercept_a*P_conarea[:sb_a,0]**slope_a

intercept_l=np.exp(intercept_l)
P_uplength_cal=intercept_l*P_uplength[:sb_l,0]**slope_l

'''
####
#Plotting     
#####

#Hack's law
plt.scatter((H_area),(H_length),s=1, color='royalblue')
plt.loglog(H_area,H_length_cal,color='black',linewidth=0.5)
plt.xlabel('Area (Ad)',fontsize='12',fontdict=None)  
plt.ylabel('length (l)',fontsize='12')
#plt.title("Hack's law relationship")
plt.savefig("Hack's law Relationship",bbox_inches='tight',dpi=300)
plt.show()

#plots with slope breaks

plt.loglog((P_conarea[:,0]),(P_conarea[:,1]),linewidth=0, color='royalblue',marker='o',markersize=2)
plt.loglog((P_conarea[:sb_a,0]),(P_conarea_cal) ,color='black',linewidth=0.5)
plt.xlabel('Area(\u03B4)',fontsize='12')
plt.ylabel('P[Ad>=\u03B4]',fontsize='12')
#plt.title("Exceeding probability for contributing Area")
plt.savefig("Exceeding probability for contributing Area",bbox_inches='tight',dpi=300)
plt.show()

#P(length)
plt.loglog((P_uplength[:,0]),(P_uplength[:,1]),linewidth=0, color='royalblue',marker='o',markersize=3)
plt.loglog((P_uplength[:sb_l,0]),(P_uplength_cal),color='black',linewidth=0.5)
plt.xlabel('length(l)',fontsize='12')
plt.ylabel('P[x>=l]',fontsize='12') 
#plt.title("Exceeding probability for Upstream length")
plt.savefig("Exceeding probability for Upstream length1",bbox_inches='tight',dpi=300)
plt.show()

'''

'''
#Subplots
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(3, 3)

#Hack's law
ax1 = fig.add_subplot(gs[-1, 0])
ax1.scatter((H_area),(Up_length),s=1, color='black')
ax1.loglog(H_area,Up_length_cal,color='b')
ax1.set_xlabel('Ad')
ax1.set_ylabel('length')
#ax1.set_title("Hack's law relationship")

#exceeding probability for drainage area
ax2 = fig.add_subplot(gs[-1, 1])
ax2.loglog((P_conarea[:,0]),(P_conarea[:,1]),color='b')
ax2.loglog((P_conarea[:sb_a,0]),(P_conarea_cal),ls='--' ,color='k')
ax2.set_xlabel('Area(\u03B4)')
ax2.set_ylabel('P[Ad>=\u03B4]')
#ax2.set_title("Exceeding probability for contributing Area")

#Exceeding probability for upstream length
ax3 = fig.add_subplot(gs[-1, 2])
ax3.loglog((P_uplength[:,0]),(P_uplength[:,1]),color='b')
ax3.loglog((P_uplength[:sb_l,0]),(P_uplength_cal),ls='--' ,color='k')
ax3.set_xlabel('length(l)')
ax3.set_ylabel('P[x>=l]')
#ax3.set_title("Exceeding probability for Upstream length")
fig.savefig('scaling Relationships',dpi=300)
'''
