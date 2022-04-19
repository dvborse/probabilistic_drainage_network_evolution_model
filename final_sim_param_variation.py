# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 07:31:05 2020

@author: admin
"""

#Ths final simulation is for calculating hack's coefficient and energy expenditure for different values of aplha 
#Each of 5 beta values iterated 5 times i.e 25 iteartions 
import numpy as np
import matplotlib.pyplot as plt
import time
#import winsound
import random
import multiprocessing as mp


#Energy expenditure calculation setup -0.5,2.1,0.1
tic=time.time()


#hack's value is calculated for beta values with number of iterations for 4 different thresholds 5,10,20,50
#lengh of thresholds should be equal to this number

# with open('All_functions.py') as afile:
#     exec(afile.read())
 
#Function for multiprocssing
#Thos fun takes alpha beta set and runs it for desired iterations
#hence we shall run multiple processes of iterations with different set of parameters at a time
#that is e.g.5*5 alpha beta combinations will run simulataniously each for 20 ietartions 

#FD=np.zeros((25,25))

def All_calculations(id_alpha,id_beta,alpha,beta,iterations, Energy_with_beta, Hacks_hk,Sinuosity,GC):
    
    for it in range(iterations):
        grid_size=250
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
                
        ij=[(0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128)]
       
        while(len(Potential)>0):
            
            #Randomly choosing one pixel from potential list
            #choosing one pixel from potential list based on probability given by Down_length
            potential_array=np.zeros((len(Potential),4))
            pot_cum=0
            pi=[-1,-1]
            ix=0 #iterator
            for i in Potential:
                pot_cum+=int(10*(Down_length[i[0]][i[1]])**alpha) 
                potential_array[ix][0]=i[0]
                potential_array[ix][1]=i[1]
                potential_array[ix][2]=Down_length[i[0]][i[1]]
                potential_array[ix][3]=pot_cum
                ix+=1
                
            sum_pot=potential_array[-1][3]
            rand_pot=random.randint(1,sum_pot)
                
            for j in range(len(potential_array)):
                if rand_pot<=potential_array[j][3]:
                    pi[0]=int(potential_array[j][0])
                    pi[1]=int(potential_array[j][1])
                    break
        
            pi=tuple(pi)
            
            #Considering flow acc in surrounding pixels to assign FD from probability 
            Values=[] #list of tupules containing FD, Facc and Facc cumulative for surrounding drainage pixels
            Facc_cum=0 
        
            for y in ij:
                if Label[(pi[0]+y[0],pi[1]+y[1])]=='y': 
                          Facc_cum+=int(10*((1+Facc[pi[0]+y[0]][pi[1]+y[1]])**beta)) 
                          Values.append((y[2],Facc[pi[0]+y[0]][pi[1]+y[1]],Facc_cum))
        
            sum_facc=Values[-1][2]
            value_rand=random.randint(1,sum_facc)
            
            for i in range(len(Values)):
                if value_rand<=Values[i][2]:
                    FD[pi[0]][pi[1]]=Values[i][0]
                    break
            
            #Now updating labels and list for pi 
            Potential.remove((pi))
            Label.update({pi:"y"})
            #Updating Down_length for pi
            cordinates=FD_cordinates(pi)
            if (FD[pi[0]][pi[1]]==2 or FD[pi[0]][pi[1]]==8 or FD[pi[0]][pi[1]]==32 or FD[pi[0]][pi[1]]==128):
                Down_length[pi[0]][pi[1]]=Down_length[cordinates[0]][cordinates[1]]+np.sqrt(2)
            else:
                Down_length[pi[0]][pi[1]]=Down_length[cordinates[0]][cordinates[1]]+1
                
            #Updating labels and lists for and surrounding pixels of pi
            for x in ij:
                if Label[(pi[0]+x[0],pi[1]+x[1])]=='n':
                    Label[(pi[0]+x[0],pi[1]+x[1])]='p'
                    Potential.append((pi[0]+x[0],pi[1]+x[1]))
                    Down_length[pi[0]+x[0]][pi[1]+x[1]]=1+Down_length[pi[0]][pi[1]]
                
        #Now updating flow accumulation value for pi and then along the stream
            current_pi=(pi[0],pi[1])
            Facc[pi[0]][pi[1]]=1
        
            while(not(current_pi[0]==0 or current_pi[0]==grid_size-1 or current_pi[1]==0 or current_pi[1]==grid_size-1)):
            
                current_pi=FD_cordinates(current_pi)
                Facc[current_pi[0]][current_pi[1]]+=1    
        
        
        li=np.zeros((grid_size,grid_size))
            
        for i in range(grid_size):
            for j in range(grid_size):
                if (FD[i][j]==1 or FD[i][j]==4 or FD[i][j]==16 or FD[i][j]==64):   
                    li[i][j]=1
                else :
                    li[i][j]=np.sqrt(2)
            
            #Get flow acc matrix
            #formula for total energy expenditure
        energy=np.multiply(np.power(Facc,0.5),li)
        total_energy=np.sum(energy)
        
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
                
        #threshold=[5,10,25,50,75,100,150,200]
        threshold=[50]
        hk_list=[[],[]]
        
        for th in threshold:
            H_length=[]
            H_area=[]
            for i in range(grid_size):
                for j in range(grid_size):
                    if Facc[i][j]>th:
                    #if Up_length[i][j]>150:   
                        H_area.append(Facc[i][j])
                        H_length.append(Up_length[i][j])
                        
            
            h, k = np.polyfit(np.log(H_area),np.log(H_length), 1)
            
            k=np.exp(k)
            
            hk_list[0].append(h)
            hk_list[1].append(k)
                
        
        
        #Sinuosity and GC Gravelius coefficient calculation 
        #First watershed delination and then calculation of dist bet outlet and max upstream point
        #finding outlet from down length matrix using max down length
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
        
        unassigned=list(unassigned)
        for pixel in unassigned:
            watershed[pixel[0]][pixel[1]]=1
         
        
        while(len(unassigned)>0):
            for pixel in unassigned:
                for ix in xij:
                    if FD[(pixel[0]+ix[0],pixel[1]+ix[1])]==ix[2]:
                        watershed[pixel[0]+ix[0]][pixel[1]+ix[1]]=1
                        unassigned.append((pixel[0]+ix[0],pixel[1]+ix[1]))
                unassigned.remove(pixel)
        #watershed is 1/0 matrix now  outlet, max_l is cordinates for two ends of stream
        stream_displacement=np.sqrt((max_l[0]-outlet[0])**2+(max_l[1]-outlet[1])**2)
        stream_distance=np.amax(Down_length)
    
        #GC calculaion
        #Logic: number of surfaces of watershed pixels sharing lateral side with non-watershed pixels contribute to the perimeter
        #Logic: number of surfaces of watershed pixels sharing lateral side with non-watershed pixels contribute to the perimeter
        perimeter=3 #three sides of single outlet pixel
        yij=[(0,1),(1,0),(0,-1),(-1,0)]
        
        Area=(watershed>0).sum()
        #iterate only no border elements
        
        for i in np.arange(1,grid_size-1,1):
            for j in np.arange(1,grid_size-1,1):
                count=0
                if watershed[i][j]==1:
                    for y in yij:
                        if watershed[i+y[0]][j+y[1]]==0:
                            count+=1
                perimeter+=count
        
        
        Energy_with_beta[id_alpha*130+id_beta*10+it]=total_energy
        #this is to avoid array and use it as single list in multiprocessing 
        #we are trying to decide position in that single list using dimentional cordinates of array   
        
        Hacks_hk[id_alpha*260+id_beta*20+it*2+0]=hk_list[0][0]
        Hacks_hk[id_alpha*260+id_beta*20+it*2+1]=hk_list[1][0] #to incorporate coefficient
        # for iz,_ in enumerate(threshold):
        #     Hacks_hk[id_alpha*400+id_beta*80+it*8+iz]=hk_list[0][iz]
        Sinuosity[id_alpha*130+id_beta*10+it]=stream_distance/stream_displacement
        GC[id_alpha*130+id_beta*10+it]=perimeter/(2*np.sqrt(np.pi*Area))

if __name__ == "__main__":
        
    alpha_values=np.arange(0,2.1,0.5) 
    #beta_values=[0,6.1,1]
    beta_values=np.arange(0,6.1,0.5)
    
    iterations=10
    
    #we shall make a single large ctype array instead of row-column array for energy and hack's coefficient
    #Because it is not processing well with other data types in multiprocessing
    #later on we shall deconstruct large list to rows and columns
    
    Energy_with_beta=mp.Array('d',len(alpha_values)*len(beta_values)*iterations) 
    Hacks_hk=mp.Array('d',len(alpha_values)*len(beta_values)*iterations*2) #8 is length of thresholds  
    Sinuosity=mp.Array('d',len(alpha_values)*len(beta_values)*iterations)
    GC=mp.Array('d',len(alpha_values)*len(beta_values)*iterations)
    
    processes=[]
    
    for ih,alpha in enumerate(alpha_values):
        for jh,beta in enumerate(beta_values):
            p=mp.Process(target=All_calculations, args=(ih,jh,alpha,beta,iterations,Energy_with_beta,Hacks_hk,Sinuosity,GC))
            processes.append(p)
            
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()
        
    import pickle
    
    Energy_list=list(Energy_with_beta)
    Hacks_hk_list=list(Hacks_hk)
    Sinuosity_list=list(Sinuosity)
    GC_list=list(GC)
    
    
    with open("energy_exp.txt", "wb") as fp:   #Pickling
        pickle.dump(Energy_list, fp)
    with open("Hacks_hk.txt", "wb") as fp:   #Pickling
        pickle.dump(Hacks_hk_list, fp)
    with open("Sinuosity.txt", "wb") as fp:   #Pickling
        pickle.dump(Sinuosity_list, fp)
    with open("GC.txt", "wb") as fp:   #Pickling
        pickle.dump(GC_list, fp)
  
  
# #Box plot 

# data=[x for x in Hacks_hk[:,:,3]]
# plt.boxplot(data,labels=["0", "0.5", "1", "1.5", "2"])
# plt.xlabel('beta values')
# plt.ylabel("Hack's exponent")
# plt.title("Variation of Hack's exponent with beta")
# plt.savefig("Variation of Hack's exponent with beta_facc_50",dpi=300)
# plt.show()

# #Box plot for energy evolution
# data1=[x for x in Energy_with_beta[:,:]]
# plt.boxplot(data1,labels=["0", "0.5", "1", "1.5", "2"])
# plt.xlabel('beta values')
# plt.ylabel("Energy Evolution")
# plt.title("Variation of Energy Expenditure with beta")
# plt.savefig("Variation of Energy Expenditure with beta",dpi=300)
# plt.show()

'''
plt.imshow(Facc, cmap='gray')
plt.title('Drainage Network with beta='+str(beta), fontsize=8)
plt.savefig("drainage_network 1k beta_"+str(beta)+".png", bbox_inches='tight')

W=np.where(Facc>500,1,0)
plt.imshow(W, cmap='gray')
plt.title('stream Network iteration_='+str(b), fontsize=8)
plt.savefig("stream_network iteration_"+str(b)+".png", bbox_inches='tight')        
'''
toc=time.time()
time_elapsed=toc - tic



#duration=1000 #milliseconds
#freq=440 #Hz
#winsound.Beep(freq, duration)

#import csv

# Energy_list=list(Energy_with_beta)
# Hacks_exponent=list(Hacks_hk)

# with open('energy_expenditure','w') as f:
#     wr=csv.writer(f) 
#     wr.writerow(Energy_list)
    
# with open('hacks_exponente','w') as f1:
#     wr=csv.writer(f1) 
#     wr.writerow(Hacks_exponent)
    




