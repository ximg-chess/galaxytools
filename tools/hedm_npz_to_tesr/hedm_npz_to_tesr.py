#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:32:06 2018

@author: dcp99
"""


#%%

import argparse
import numpy as np
import re

def __main__():

    parser = argparse.ArgumentParser(
        description='Convert an HEDM_map.npz file to a neper tesr file')
    parser.add_argument(
        'input',
        type=argparse.FileType('rb'),
        help='HEDM_map.npz')
    parser.add_argument(
        'output',
        type=argparse.FileType('w'),
        help='neper tesr file' )
    parser.add_argument('-s', '--voxel_spacing', type=float, default=0.005, help='voxel spacing')
    parser.add_argument('-x', '--x_name', default='^.*X.*$', help='X array name')
    parser.add_argument('-y', '--y_name', default='^.*Y.*$', help='Y array name')
    parser.add_argument('-z', '--z_name', default='^.*Z.*$', help='Z array name')
    parser.add_argument('-g', '--grain_map', default='^.*grain.*$', help='grain map name')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug')
    args = parser.parse_args()
    data=np.load(args.input)
    grain_map = Xs = Ys = Zs = None
    for f in data.files:
        if re.match(args.grain_map, f):
            grain_map = data[f]
        elif re.match(args.x_name, f):
            Xs = data[f]
        elif re.match(args.y_name, f):
            Ys = data[f]
        elif re.match(args.z_name, f):
            Zs = data[f]
    voxel_spacing = args.voxel_spacing

    #CREATE ASSEMBLED DATA -- LIST OF [VOXEL COORDINATES (X,Y,Z),GRAIN ID]

    coordinate_list=np.vstack((Xs.ravel(),Ys.ravel(),Zs.ravel()))
    assembled_data=np.hstack((coordinate_list.T,np.atleast_2d(grain_map.ravel()).T))

    #%% SORT BY ROWS Z AND THEN Y

    assembled_data=assembled_data[assembled_data[:,2].argsort()]
    total_size=int(grain_map.shape[0]*grain_map.shape[1]*grain_map.shape[2])

    stack_size=int(grain_map.shape[0]*grain_map.shape[1])
    for ii in np.arange(int(total_size/stack_size)):
         tmp_args=assembled_data[ii*stack_size:(ii+1)*stack_size,1].argsort()
         assembled_data[ii*stack_size:(ii+1)*stack_size,:]=assembled_data[ii*stack_size+tmp_args,:]
     
    stack_size=grain_map.shape[1]
    for ii in np.arange(int(total_size/stack_size)):
         tmp_args=assembled_data[ii*stack_size:(ii+1)*stack_size,0].argsort()
         assembled_data[ii*stack_size:(ii+1)*stack_size,:]=assembled_data[ii*stack_size+tmp_args,:]

    #%%

    np.set_printoptions(threshold=np.inf)
    l1  = '***tesr'
    l2  = ' **format'
    l3  = '   2.0 ascii'
    l4  = ' **general'
    l5  = '   3'
    l6  = '   ' + str(grain_map.shape[1]) + ' ' + str(grain_map.shape[0])  + ' ' + str(grain_map.shape[2]) 
    l7  = '   ' + str(voxel_spacing) + ' ' + str(voxel_spacing) + ' ' + str(voxel_spacing)
    l8  = ' **cell';
    l9  = '   ' + str(np.max(grain_map).astype('int'))
    l10 = '  *id';
    l11 = '   ' + str(np.arange(1,np.max(grain_map)+1).astype('int'))[1:-1]
    l12 = ' **data'
    #l13 = '   ' + str(assembled_data[:,3].astype('int'))[1:-1]
    l14 = '***end'
    
    #%%
    output = args.output
    output.write('%s\n' % l1)
    output.write('%s\n' % l2)
    output.write('%s\n' % l3)
    output.write('%s\n' % l4)
    output.write('%s\n' % l5)
    output.write('%s\n' % l6)
    output.write('%s\n' % l7)
    output.write('%s\n' % l8)
    output.write('%s\n' % l9)
    output.write('%s\n' % l10)
    output.write('%s\n' % l11)
    output.write('%s\n' % l12)
    output.write('   ')
    np.savetxt(output,np.atleast_2d(assembled_data[:,3]).T,fmt='%d')
    #output.write('%s\n' % l13)
    output.write('%s\n' % l14)
    output.close()


if __name__ == "__main__":
    __main__()
