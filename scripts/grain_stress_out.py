#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import hexrd.rotations as rot

import scipy.ndimage as img
from skimage.transform import iradon, radon, rescale

import hexrd.matrixutil as mutil

from scipy.stats import gaussian_kde
from PIL import Image


from hexrd import matrixutil as mutil
from hexrd import rotations  as rot
from hexrd import symmetry   as sym

import pandas as pd

import copy
from cycler import cycler


#%%
#------------Import data into new array-------------#
#please note, forloops and labelling for importing data will be based on however you saved your analysis folders for grains.out scans
#please change accordingly

#input grains.out below: here it is hardcoded to find it on the file system with a presumed saving path strategy. 

grain_out_root = '/nfs/chess/aux/user/ken38/galaxy_workspace/ti_data/'
test_name = 'ti7-05'
scan = 11
grains_out_file= grain_out_root + test_name + '-scan-%d/grains.out' % (initial_scan)

ptemp = np.loadtxt('strain_ecorr.txt')


#%%
#----------Good grain parameters ------------#
#this block defines what should be considered a "good" grain

completeness_thresh = 0.8
chi_thresh = 0.009
vol_strain_thresh = 0.001

good_grains=np.where(np.logical_and(np.logical_and(comp_0>completeness_thresh,chi_0<chi_thresh),np.abs(v_strain_0)<vol_strain_thresh))[0]

#==============================================================================
# %% ELASTIC MODULI (Ti-7Al) #user should edit for their particular material
#==============================================================================
#Values in GPa

C11=176.1 #Venkataraman DOI: 10.1007/s11661-017-4024-y (2017)
C33=190.5
C44=50.8
C66=44.6

C12=86.9
C13=68.3

B=110

c_mat_C=np.array([[C11,C12,C13,0.,0.,0.],
                  [C12,C11,C13,0.,0.,0.],
                  [C13,C13,C33,0.,0.,0.],
                  [0.,0.,0.,C44,0.,0.],
                  [0.,0.,0.,0.,C44,0.],
                  [0.,0.,0.,0.,0.,C66]])*1e9

#==============================================================================
# %% Calculating Stresses
#==============================================================================
#this block is used to calculate stress tensors and particular stresses
#you will want to edit this for your own scripts

great_grains = good_grains
num_grains=len(great_grains)

great_grains_stresses = np.zeros([num_grains,6])
great_grains_chi = np.zeros([num_grains])
great_grains_comp = np.zeros([num_grains])

#plt.close("all")

tmp_data_x=np.loadtxt(grains_out_file)
strain_x=tmp_data_x[:,15:]
pos_x=tmp_data_x[:,6:9]
strain_x[:,0]=strain_x[:,0]-(p[0]*pos_x[:,1]+p[1])
strain_x[:,1]=strain_x[:,1]-(p[0]*pos_x[:,1]+p[1])
strain_x[:,2]=strain_x[:,2]-(p[0]*pos_x[:,1]+p[1])
exmap_x=tmp_data_x[:,3:6]
strainTmp=np.atleast_2d(strain_x[great_grains]).T
expMap=np.atleast_2d(exmap_x[great_grains]).T

stress_S=np.zeros([num_grains,6])
stress_C=np.zeros([num_grains,6])
stress_prin=np.zeros([num_grains,3])
hydrostatic=np.zeros([num_grains,1])
pressure=np.zeros([num_grains,1])
max_shear=np.zeros([num_grains,1])
von_mises=np.zeros([num_grains,1])
           #Turn exponential map into an orientation matrix
chi_x_t= tmp_data_x[:,2]
chi_x=chi_x_t[great_grains]

comp_x_t=tmp_data_x[:,1]
comp_x=comp_x_t[great_grains]

id_x_t=tmp_data_x[:,0]
id_x=id_x_t[great_grains]
#%%
       # plt.figure(ii)
for ii in range (0,num_grains-1,1):
      
           
    great_grains_chi[ii]=chi_x[ii]
    great_grains_comp[ii]=comp_x[ii]
      
    Rsc=rot.rotMatOfExpMap(expMap[:,ii])

    strainTenS = np.zeros((3, 3), dtype='float64')
    strainTenS[0, 0] = strainTmp[0,ii]
    strainTenS[1, 1] = strainTmp[1,ii]
    strainTenS[2, 2] = strainTmp[2,ii]
    strainTenS[1, 2] = strainTmp[3,ii]
    strainTenS[0, 2] = strainTmp[4,ii]
    strainTenS[0, 1] = strainTmp[5,ii]
    strainTenS[2, 1] = strainTmp[3,ii]
    strainTenS[2, 0] = strainTmp[4,ii]
    strainTenS[1, 0] = strainTmp[5,ii]

    strainTenC=np.dot(np.dot(Rsc.T,strainTenS),Rsc)
    strainVecC = mutil.strainTenToVec(strainTenC)

    v_strain_gg=np.trace(strainTenS)/3

#Calculate stress
    stressVecC=np.dot(c_mat_C,strainVecC)
    stressTenC = mutil.stressVecToTen(stressVecC)
    stressTenS = np.dot(np.dot(Rsc,stressTenC),Rsc.T)
    stressVecS = mutil.stressTenToVec(stressTenS)

#Calculate hydrostatic stress
    hydrostaticStress=(stressVecS[:3].sum()/3)
    w,v = np.linalg.eig(stressTenC)
    maxShearStress=(np.max(w)-np.min(w))/2.

#Calculate Von Mises Stress
    devStressS=stressTenS-hydrostaticStress*np.identity(3)
    vonMisesStress=np.sqrt((3/2)*(devStressS**2).sum())

    stress_data=dict()

    stress_data['stress_S']=stress_S
    stress_data['stress_C']=stress_C
    stress_data['hydrostatic']=hydrostaticStress
    stress_data['max_shear']=max_shear
    stress_data['pressure']=pressure
    stress_data['von_mises']=von_mises
    stress_data['principal']=stress_prin

#
    great_grains_stresses[ii,:] = stressVecS.T
#
# Save the image in memory in PNG format

#%%LOAD DIC DATA --- this is dic stress-strain data saved in npy format

np.savetxt('stress_tensors.txt', great_grains_stresses)
