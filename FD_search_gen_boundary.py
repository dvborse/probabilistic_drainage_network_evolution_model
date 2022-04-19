# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:32:18 2020

@author: Dnyanesh
"""
'''
At any step
>A pixel chosen among potential pixels randomly say Pi 
>Then surrounding pixels around Pi are considered and  
>among those pixels a list is prepared based on probability distribution based on areas
>Then randomly one element is chosen from that list and flow direction assiged to that pixel
>Once FD assigned then label of that pixel is changed and surrounding pixels to that pixels are added 
>in the potential list
>Then flow acc of pi is also updated to 1 and the subsequent flow acc
> values in that streamline are also updated or increses by one 
>Ok that's it 
>Figure is ready for next iteration 
> This should go on till no. of pixels in unassigned set becomes zero 
'''
import random

#colourmap for figures
norm=plt.Normalize(0,1)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","blue"])

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

#Surrounding pixel and their correcsponding flow direction values
ij=((0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128))
border={'left':((-1,0,64),(-1,1,128),(0,1,1),(1,1,2),(1,0,4)),'top':((0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16)),'right':((1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64)),'bottom':((0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128),(0,1,1))}
corner={'bottom_left':((-1,0,64),(-1,1,128),(0,1,1)),'left_top':((0,1,1),(1,1,2),(1,0,4)),'top_right':((1,0,4),(1,-1,8),(0,-1,16)),'right_bottom':((0,-1,16),(-1,-1,32),(-1,0,64))}

ixx=0 #iterator to plot intermediate images
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
    '''
    #If pixel is either border/corner  pixel or inner
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
    '''
    for y in ij:
        if true_area[pi[0]+y[0],pi[1]+y[1]]: 
            if Label[(pi[0]+y[0],pi[1]+y[1])]=='y' or Label[(pi[0]+y[0],pi[1]+y[1])]=='b':
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
        if true_area[pi[0]+x[0],pi[1]+x[1]]:
            if Label[(pi[0]+x[0],pi[1]+x[1])]=='n':
                Label[(pi[0]+x[0],pi[1]+x[1])]='p'
                Potential.append((pi[0]+x[0],pi[1]+x[1]))
                Down_length[pi[0]+x[0]][pi[1]+x[1]]=1+Down_length[pi[0]][pi[1]]
        
#Now updating flow accumulation value for pi and then along the stream
    current_pi=pi
    Facc[pi[0]][pi[1]]=1

    while(not(Label[(current_pi[0],current_pi[1])]=="b")):
        current_pi=FD_cordinates(current_pi)
        Facc[current_pi[0],current_pi[1]]+=1    
    '''
    if ixx%10000==0:
        W=np.where(Facc>50,1,0)
        plt.imshow(W, cmap=cmap, origin='lower',extent=[0,grid_size[1],0,grid_size[0]])
        #plt.title('stream Network for iteration '+str(ixx), fontsize=8)♥
        plt.savefig("stream_network for "+str(ixx)+".png", bbox_inches='tight',dpi=300) 
    ixx+=1
    '''


