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

import copy
from cycler import cycler


#%%
#------------Import data into new array-------------#
#please note, forloops and labelling for importing data will be based on however you saved your analysis folders for grains.out scans
#please change accordingly

#input grains.out below: here it is hardcoded to find it on the file system with a presumed saving path strategy. 

grain_out_root = '/nfs/chess/aux/user/ken38/galaxy_workspace/ti_data/'
test_name = 'ti7-05'
initial_scan = 11
init_grain_out_file= grain_out_root + test_name + '-scan-%d/grains.out' % (initial_scan)

#load grains.out 

tmp_data_0=np.loadtxt(init_grain_out_file)

id_0=tmp_data_0[:,0]
comp_0=tmp_data_0[:,1]
chi_0=tmp_data_0[:,2]
strain_0=tmp_data_0[:,15:]
ori_0=tmp_data_0[:,3:6]
pos_0=tmp_data_0[:,6:9]
v_strain_0=np.sum(strain_0[:,0:3],axis=1)/3.

#%%
#----------Good grain parameters ------------#
#this block defines what should be considered a "good" grain

completeness_thresh = 0.8
chi_thresh = 0.009
vol_strain_thresh = 0.001

good_grains=np.where(np.logical_and(np.logical_and(comp_0>completeness_thresh,chi_0<chi_thresh),np.abs(v_strain_0)<vol_strain_thresh))[0]

#%%
#---------- Correcting for vertical variation in beam energy ----------
#this block corrects the vertical variation in beam energy based on the initial scan

x = pos_0[good_grains,1]
y = v_strain_0[good_grains]
p =  np.polyfit(x, y, 1) #polynomial fit of gradient p[0] will be the slope p[1] the intercept

#updating v_strain_0 to corrected values
v_strain_0 = v_strain_0-p[0]*pos_0[:,1]-p[1]

#add corrected values to plot
plt.figure(1)
plt.plot(pos_0[good_grains,1],v_strain_0[good_grains],'bo', label='corrected')
legend = plt.legend(loc='lower right')
#%%
n_grains=len(good_grains)

astrain=np.zeros(n_grains)
astrain2=np.zeros(n_grains)
cstrain=np.zeros(n_grains)

strain_0[:,0]=strain_0[:,0]-(p[0]*pos_0[:,1]+p[1])
strain_0[:,1]=strain_0[:,1]-(p[0]*pos_0[:,1]+p[1])
strain_0[:,2]=strain_0[:,2]-(p[0]*pos_0[:,1]+p[1])

for ii in np.arange(n_grains):
    ti=good_grains[ii]
    strain_ten=np.array([[strain_0[ti,0],strain_0[ti,5],strain_0[ti,4]],[strain_0[ti,5],strain_0[ti,1],strain_0[ti,3]],[strain_0[ti,4],strain_0[ti,3],strain_0[ti,2]]])
    R=rot.rotMatOfExpMap(ori_0[ti,:])

    strain_ten_c=np.dot(R.T,np.dot(strain_ten,R))
    #print(strain_ten_c)
    astrain[ii]=strain_ten_c[0,0]
    astrain2[ii]=strain_ten_c[1,1]
    cstrain[ii]=strain_ten_c[2,2]

#print figure

plt.figure(2)
plt.plot(astrain,'x')
plt.plot(astrain2,'gx')
plt.plot(cstrain,'rx')

#%%
np.savetxt('strain_ecorr.txt', p )
