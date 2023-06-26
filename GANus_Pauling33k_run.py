#!/usr/bin/env python
# coding: utf-8

# In[28]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torchvision.models as models 
from data_treatment import DataSet, DataAtts
from discriminator import *
from generator import *
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os
from itertools import cycle
from numpy import genfromtxt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from torch.autograd import Variable
import time
import os


# # Load Model
# 

# In[30]:


discriminator = DiscriminatorNet(192)
checkpoint = torch.load('16mer_models/discriminator.pt')
discriminator.load_state_dict(checkpoint['model_state_dict'])
discriminator.eval() 

generator = GeneratorNet(192)
checkpoint = torch.load('16mer_models/generatorpeptide.pt')
generator.load_state_dict(checkpoint['model_state_dict'])
generator.eval()


# In[31]:


# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): 
        return n.cuda() 
    return n

# Start with random seeds
newdata = generator(noise(200)).detach().cpu().numpy()


# Set some parameter values

AAlist=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

HydrophobicityValues=[0.61, 1.07, 0.46, 0.47, 2.02, 0.07, 0.61, 2.22, 1.15, 1.53, 1.18, 0.06, 1.95, 0.0, 0.6, 0.05, 0.05, 1.32, 2.65, 1.88]
Alpha_CH_Chem_ShiftValues=[4.35, 4.65, 4.76, 4.29, 4.66, 3.97, 4.63, 3.95, 4.36, 4.17, 4.52, 4.75, 4.44, 4.37, 4.38, 4.5, 4.35, 3.95, 4.7, 4.6]
Conf_Par_of_Beta_StructureValues = [0.77, 0.65, 0.65, 0.55, 0.98, 0.65, 0.83, 0.98, 0.55, 0.83, 0.98, 0.55, 0.55, 0.72, 0.72, 0.55, 0.83, 0.98, 0.77, 0.83]
Conf_Par_of_Inner_HelixValues=[1.0, 0.06, 0.44, 0.73, 0.6, 0.35, 0.6, 0.73, 0.6, 1.0, 1.0, 0.35, 0.06, 0.44, 0.52, 0.35, 0.44, 0.82, 0.73, 0.44]
Conf_Par_of_Beta_TurnValues=[0.37, 0.84, 0.97, 0.53, 0.53, 0.97, 0.75, 0.37, 0.75, 0.53, 0.64, 0.97, 0.97, 0.64, 0.84, 0.84, 0.75, 0.37, 0.97, 0.84]
Residue_VolumeValues=[52.6, 68.3, 68.4, 84.7, 113.9, 36.3, 91.9, 102.0, 105.1, 102.0, 97.7, 75.7, 73.6, 89.7, 109.1, 54.9, 71.2, 85.1, 135.4, 116.2]
Steri_ParValues=[0.52, 0.62, 0.76, 0.68, 0.7, 0.0, 0.7, 1.02, 0.68, 0.98, 0.78, 0.76, 0.36, 0.68, 0.68, 0.53, 0.5, 0.76, 0.7, 0.7]
Norm_Freq_of_Beta_Turn_aValues=[0.66, 1.19, 1.46, 0.74, 0.6, 1.56, 0.95, 0.47, 1.01, 0.59, 0.6, 1.56, 1.52, 0.98, 0.95, 1.43, 0.96, 0.5, 0.96, 1.14]
Norm_Freq_of_Alpha_Helix_aValues=[1.42, 0.7, 1.01, 1.51, 1.13, 0.57, 1.0, 1.08, 1.16, 1.21, 1.45, 0.67, 0.57, 1.11, 0.98, 0.77, 0.83, 1.06, 1.08, 0.69]
Norm_Freq_of_Beta_SheetValues=[0.83, 1.19, 0.54, 0.37, 1.38, 0.75, 0.87, 1.6, 0.74, 1.3, 1.05, 0.89, 0.55, 1.1, 0.93, 0.75, 1.19, 1.7, 1.37, 1.47]
Beta_Strand_IndicesValues=[0.84, 1.27, 0.59, 0.57, 1.15, 0.94, 0.81, 1.29, 0.86, 1.1, 0.88, 0.66, 0.8, 1.02, 1.04, 1.05, 1.2, 1.56, 1.15, 1.39]
Alpha_Helix_IndicesValues=[1.29, 0.79, 1.1, 1.49, 1.13, 0.63, 1.33, 1.05, 1.33, 1.31, 1.54, 0.81, 0.63, 1.07, 1.0, 0.78, 0.77, 0.81, 1.18, 0.71]


# find the closest amino acid in parameter space using scaled L1 norm
def closestAA(i,k): 

    d1min = 10000.0;
    
    for j in range(0,20): # loop over 20 amino acids

        # distance function is a scaled L1 norm
        
        d1 = 0.0
        d1 = abs(HydrophobicityValues[j]-newdata[i,12*k])/(max(HydrophobicityValues)-min(HydrophobicityValues))
        d1 = d1+abs(Alpha_CH_Chem_ShiftValues[j]-newdata[i,12*k+1])/(max(Alpha_CH_Chem_ShiftValues)-min(Alpha_CH_Chem_ShiftValues))
        d1 = d1+abs(Conf_Par_of_Beta_StructureValues[j]-newdata[i,12*k+2])/(max(Conf_Par_of_Beta_StructureValues)-min(Conf_Par_of_Beta_StructureValues))
        d1 = d1+abs(Conf_Par_of_Inner_HelixValues[j]-newdata[i,12*k+3])/(max(Conf_Par_of_Inner_HelixValues)-min(Conf_Par_of_Inner_HelixValues))
        d1 = d1+abs(Conf_Par_of_Beta_TurnValues[j]-newdata[i,12*k+4])/(max(Conf_Par_of_Beta_TurnValues)-min(Conf_Par_of_Beta_TurnValues))
        d1 = d1+abs(Residue_VolumeValues[j]-newdata[i,12*k+5])/(max(Residue_VolumeValues)-min(Residue_VolumeValues))
        d1 = d1+abs(Steri_ParValues[j]-newdata[i,12*k+6])/(max(Steri_ParValues)-min(Steri_ParValues))
        d1 = d1+abs(Norm_Freq_of_Beta_Turn_aValues[j]-newdata[i,12*k+7])/(max(Norm_Freq_of_Beta_Turn_aValues)-min(Norm_Freq_of_Beta_Turn_aValues))
        d1 = d1+abs(Norm_Freq_of_Alpha_Helix_aValues[j]-newdata[i,12*k+8])/(max(Norm_Freq_of_Alpha_Helix_aValues)-min(Norm_Freq_of_Alpha_Helix_aValues))
        d1 = d1+abs(Norm_Freq_of_Beta_SheetValues[j]-newdata[i,12*k+9])/(max(Norm_Freq_of_Beta_SheetValues)-min(Norm_Freq_of_Beta_SheetValues))
        d1 = d1+abs(Beta_Strand_IndicesValues[j]-newdata[i,12*k+10])/(max(Beta_Strand_IndicesValues)-min(Beta_Strand_IndicesValues))
        d1 = d1+abs(Alpha_Helix_IndicesValues[j]-newdata[i,12*k+11])/(max(Alpha_Helix_IndicesValues)-min(Alpha_Helix_IndicesValues))
            
        if d1 < d1min:
            d1min = d1
            indmin = j
    
    return indmin

# convert a feature vector to its "closest" sequence
def feature2sequence(ind,featurevector):
    sequence = "                 "
    list1 = list(sequence)
    for k in range(0,16):
        indmin = closestAA(ind,k)
        list1[k] = AAlist[indmin]
    sequence = ''.join(list1)
    #print(sequence)
    return sequence




# In[32]:


# normalize the feature data to correspond to a real AA sequence

for i in range(0,len(newdata[:,1])):  
        
    for k in range(0,16): # number of residues in peptide


        indmin = closestAA(i,k)
        
        # reassign descriptor values to correspond to a real sequence
        
        newdata[i,12*k] =   HydrophobicityValues[indmin]     
        newdata[i,12*k+1] = Alpha_CH_Chem_ShiftValues[indmin] 
        newdata[i,12*k+2] = Conf_Par_of_Beta_StructureValues[indmin]
        newdata[i,12*k+3] = Conf_Par_of_Inner_HelixValues[indmin]
        newdata[i,12*k+4] = Conf_Par_of_Beta_TurnValues[indmin]
        newdata[i,12*k+5] = Residue_VolumeValues[indmin]
        newdata[i,12*k+6] = Steri_ParValues[indmin]
        newdata[i,12*k+7] = Norm_Freq_of_Beta_Turn_aValues[indmin]
        newdata[i,12*k+8] = Norm_Freq_of_Alpha_Helix_aValues[indmin]
        newdata[i,12*k+9] = Norm_Freq_of_Beta_SheetValues[indmin]
        newdata[i,12*k+10] = Beta_Strand_IndicesValues[indmin]
        newdata[i,12*k+11] = Alpha_Helix_IndicesValues[indmin]
        
        
                                                                        


# # Classification with DCNet

# In[33]:


Xpep=np.array(newdata, dtype=np.float64)


# In[34]:


# Pass test data
from torch.autograd import Variable
XpepT = torch.FloatTensor(Xpep)
Xpepy_hat_test = discriminator(XpepT)  
Xpepy_hat_test_class = np.where(Xpepy_hat_test.detach().numpy()<0.5, 0, 1)


# In[35]:


Xpepy_hat_test_classContinuous=Xpepy_hat_test.detach().numpy()


# In[36]:


len(Xpepy_hat_test_classContinuous)


# # Collect new Beta-hairpins 

# In[37]:

# screen for beta hairpins
filename = "16mer_models/16mer_beta_hairpin_sequences.fasta"
file = open(filename, "w")

for i in range(len(Xpep[:,1])):
    peptideID='peptide'+str(i)
    peptideSeq=feature2sequence(i,Xpep[i,:])
    Xpepi=np.array(Xpep[i,:], dtype=np.float64)
    XpepiT = torch.FloatTensor(Xpepi)
    Xpepiy_hat_test = discriminator(XpepiT)
    Xpepiy_hat_test_classContinuous=Xpepiy_hat_test.detach().numpy()
    if Xpepiy_hat_test_classContinuous>=.95:
        print(peptideID, peptideSeq, Xpepiy_hat_test_classContinuous[0])
        file = open(filename, "a")    
        file.write(">" + peptideID+ "|" + str(Xpepiy_hat_test_classContinuous[0]) + "\n")
        file.write(peptideSeq + "\n")



# In[ ]:




