# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:29:05 2021

@author: Dnyanesh
"""
import pickle

# Energy_list=list(Energy_with_beta)
# Hacks_hk_list=list(Hacks_hk)
# Sinuosity_list=list(Sinuosity)


# with open("energy_exp.txt", "wb") as fp:   #Pickling
#     pickle.dump(Energy_list, fp)
# with open("Hacks_hk.txt", "wb") as fp:   #Pickling
#     pickle.dump(Hacks_hk_list, fp)
# with open("Sinuosity.txt", "wb") as fp:   #Pickling
#     pickle.dump(Sinuosity_list, fp)
# with open("Sinuosity.txt", "wb") as fp:   #Pickling
#     pickle.dump(Sinuosity_list, fp)

with open("energy_exp.txt", "rb") as fp:   # Unpickling
    Energy_list = pickle.load(fp)
with open("Hacks_hk.txt", "rb") as fp:   # Unpickling
    Hacks_hk_list = pickle.load(fp)
with open("Sinuosity.txt", "rb") as fp:   # Unpickling
    Sinuosity_list = pickle.load(fp)
with open("GC.txt", "rb") as fp:   # Unpickling
    GC_list = pickle.load(fp)
