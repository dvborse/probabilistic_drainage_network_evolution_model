# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:01:37 2021

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt

alpha_values=np.arange(0,2.1,0.5) 
beta_values=np.arange(0,6.1,0.5)
iterations=10

#Energy_list=list(Energy_with_beta)
#Hacks_exponent=list(Hacks_hk)
#threshold=[5,10,25,50,75,100,150,200] #threshold values for reference
#we have Energy list 250 values and Hacks list with 2000 values
#First lets bring it to array format for easier understanding

Energy_array=np.zeros((len(alpha_values),len(beta_values),iterations)) 
Hack_array=np.zeros((len(alpha_values),len(beta_values),iterations,2))
Sinuosity_array=np.zeros((len(alpha_values),len(beta_values),iterations))
GC_array=np.zeros((len(alpha_values),len(beta_values),iterations))

it=0 #iterator for list
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        for k in range(iterations):
            Energy_array[i][j][k]=Energy_list[it]
            it+=1
            
it=0 #iterator for list
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        for k in range(iterations):
            Sinuosity_array[i][j][k]=Sinuosity_list[it]
            it+=1

it=0 #iterator for list
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        for k in range(iterations):
            GC_array[i][j][k]=GC_list[it]
            it+=1            

itt=0 #iterator for list
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        for k in range(iterations):
            for l in range(2):
                Hack_array[i][j][k][l]=Hacks_hk_list[itt]
                itt+=1

#averaged list
GC_avg=np.zeros((len(alpha_values),len(beta_values)))
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        GC_avg[i][j]=sum(GC_array[i,j,:])/iterations

Sinuosity_avg=np.zeros((len(alpha_values),len(beta_values)))
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        Sinuosity_avg[i][j]=sum(Sinuosity_array[i,j,:])/iterations

Energy_avg=np.zeros((len(alpha_values),len(beta_values)))
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        Energy_avg[i][j]=sum(Energy_array[i,j,:])/iterations
        
Hack_avg_it=np.zeros((len(alpha_values),len(beta_values),2)) #average with respect to iterations
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        for k in range(2):    
            Hack_avg_it[i][j][k]=sum(Hack_array[i,j,:,k])/iterations

#We shall calculate for diff th e.g threshold 50 i.e 4th value in threshold array        
Hack_avg_th=np.zeros((len(alpha_values),len(beta_values))) #average with respect to iterations
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        Hack_avg_th[i][j]=sum(Hack_array[i,j,:,0])/iterations
        
Hack_avg_alpha=np.zeros((len(alpha_values),iterations)) #average with respect to iterations
for i in range(len(alpha_values)):
    for j in range(iterations):
        Hack_avg_alpha[i][j]=sum(Hack_array[i,:,j,0])/len(beta_values)

Hack_avg_beta=np.zeros((len(beta_values),iterations)) #average with respect to iterations
for i in range(len(beta_values)):
    for j in range(iterations):
        Hack_avg_beta[i][j]=sum(Hack_array[:,i,j,0])/len(alpha_values)
        


font = {'size': 14}
plt.rc('font', **font)



data=[x for x in GC_array[2,:,:]]
plt.boxplot(data,labels=["0","0.5","1","1.5", "2","2.5", "3","3.5", "4","4.5", "5","5.5", "6"])
plt.xlabel("\u03B2 ",fontsize=14)
plt.ylabel("Gravelius coefficient",fontsize=14)
#plt.title("Variation GC with beta for alpha=0.5")
plt.savefig("Variation GC with beta for alpha 1",bbox_inches='tight',dpi=300)
plt.show()

data=[x for x in GC_array[:,2,:]]
plt.boxplot(data,labels=["0","0.5","1","1.5", "2"])
plt.xlabel("\u03B1",fontsize=14)
plt.ylabel("Gravelius coefficient",fontsize=14)
#plt.title("Variation GC with beta for alpha=0.5")
plt.savefig("Variation GC with alpha for beta 1",bbox_inches='tight',dpi=300)
plt.show()


plt.plot(beta_values,GC_avg[0],marker='o', label='alpha_0.1')
plt.plot(beta_values,GC_avg[1],marker='*',label="alpha_0.5")
#plt.title("Gravellius Coefficient variation with beta values for different alpha")
plt.xlabel("\u03B2 values",fontsize=14) #beta_values
plt.ylabel("Gravelius Coefficient",fontsize=14)           
#plt.savefig("Gravellius Coefficient variation with beta values for different alpha.png", bbox_inches='tight',dpi=300)
plt.show()




data=[x for x in GC_array[:,0,:]]
plt.boxplot(data,labels=["0.1","0.5","0.9","1.3", "1.7"])
plt.xlabel("\u03B1",fontsize=14)
plt.ylabel("Gravelius coefficient",fontsize=14)
#plt.title("Variation GC with beta for alpha=0.5")
plt.savefig("Variation GC with alpha for beta 0",bbox_inches='tight',dpi=300)
plt.show()

plt.imshow(Energy_avg, cmap='gray')
plt.title('Energy Expenditure with parameters', fontsize=8)
plt.xticks(ticks=np.arange(0,len(beta_values),1),labels=list(beta_values))
plt.xlabel('Beta_values')
plt.yticks(ticks=np.arange(len(alpha_values)),labels=['%.1f' % elem for elem in alpha_values])
plt.ylabel('Alpha_values')
plt.colorbar()
plt.savefig("Energy Expenditure Avg_10it_alpha_1_beta_4 variation.png", bbox_inches='tight',dpi=300)


plt.imshow(Hack_avg_th, cmap='Oranges')
plt.title('Hacks exponent variation Threshold=100', fontsize=8)
plt.xticks(ticks=[0,1,2,3,4],labels=[0,0.5,.0,1.5,2.0])
plt.xlabel('Beta_values')
plt.yticks(ticks=[0,1,2],labels=[0,0.5,1])
plt.ylabel('Alpha_values')
plt.colorbar()
plt.savefig("Hacks exponent variation_alpha1_beta_4 variation Threshold=100.png", bbox_inches='tight',dpi=300)



# #Box plot 
#with alpha
data=[x for x in Hack_avg_alpha[:]]
plt.boxplot(data,labels=["0","1", "2", "3", "4", "5", "6"])
plt.xlabel('alpha values')
plt.ylabel("Hack's exponent")
plt.title("Variation of avg Hack's exponent with alpha")
plt.savefig("Variation of avg Hack's exponent with alpha_facc_50",dpi=300)
plt.show()

#with beta
data=[x for x in Hack_avg_beta[:]]
plt.boxplot(data,labels=["0","1", "2", "3", "4", "5", "6"])
plt.xlabel('beta values')
plt.ylabel("Hack's exponent")
plt.title("Variation of avg Hack's exponent with beta")
plt.savefig("Variation of avg Hack's exponent with beta_facc_50",dpi=300)
plt.show()

data=[x for x in Hack_array[2,:,:,0]]
plt.boxplot(data,labels=["0","0.5","1","1.5", "2","2.5", "3","3.5", "4","4.5", "5","5.5", "6"])
plt.xlabel('\u03B2')
plt.ylabel("Hack's exponent")
#plt.title("Variation of Hack's exponent with beta")
plt.savefig("Variation of Hack's exponent with beta for alpha 0.9.png",bbox_inches='tight',dpi=300)
plt.show()

# #Box plot for energy evolution with alpha/beta
data1=[x for x in Energy_array[:,1,:]]
plt.boxplot(data1,labels=[ "0","1", "2", "3", "4", "5", "6"])
plt.xlabel('alpha values')
plt.ylabel("Energy Expenditure")
plt.title("Variation of Energy Expenditure with alpha for beta 1")
plt.savefig("Variation of Energy Expenditure with alpha for beta 1",dpi=300)
plt.show()