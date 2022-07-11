#!/usr/bin/env python3

import argparse

import numpy as np

from hexrd import matrixutil as mutil
from hexrd import rotations as rot


def calc(grains_file, c_mat,
         completeness_thresh=0.8, chi_thresh=0.009, vol_strain_thresh=0.001):
    tmp_data_0 = np.loadtxt(grains_file)

    # id_0 = tmp_data_0[:, 0]
    comp_0 = tmp_data_0[:, 1]
    chi_0 = tmp_data_0[:, 2]
    strain_0 = tmp_data_0[:, 15:]
    # ori_0 = tmp_data_0[:, 3:6]
    pos_0 = tmp_data_0[:, 6:9]
    v_strain_0 = np.sum(strain_0[:, 0:3], axis=1) / 3.

    good_grains = np.where(np.logical_and(
                           np.logical_and(comp_0 > completeness_thresh,
                                          chi_0 < chi_thresh),
                           np.abs(v_strain_0) < vol_strain_thresh))[0]

    # this block corrects the vertical variation
    # in beam energy based on the initial scan
    x = pos_0[good_grains, 1]
    y = v_strain_0[good_grains]
    # polynomial fit of gradient p[0] will be the slope p[1] the intercept
    p = np.polyfit(x, y, 1)

    # this block is used to calculate stress tensors and particular stresses
    great_grains = good_grains
    num_grains = len(great_grains)

    great_grains_stresses = np.zeros([num_grains, 7])
    great_grains_chi = np.zeros([num_grains])
    great_grains_comp = np.zeros([num_grains])

    tmp_data_x = np.loadtxt(grains_file)
    strain_x = tmp_data_x[:, 15:]
    pos_x = tmp_data_x[:, 6:9]
    strain_x[:, 0] = strain_x[:, 0]-(p[0]*pos_x[:, 1]+p[1])
    strain_x[:, 1] = strain_x[:, 1]-(p[0]*pos_x[:, 1]+p[1])
    strain_x[:, 2] = strain_x[:, 2]-(p[0]*pos_x[:, 1]+p[1])
    exmap_x = tmp_data_x[:, 3:6]
    strainTmp = np.atleast_2d(strain_x[great_grains]).T
    expMap = np.atleast_2d(exmap_x[great_grains]).T

    stress_S = np.zeros([num_grains, 6])
    stress_C = np.zeros([num_grains, 6])
    stress_prin = np.zeros([num_grains, 3])
    # hydrostatic = np.zeros([num_grains, 1]) #not used
    pressure = np.zeros([num_grains, 1])
    max_shear = np.zeros([num_grains, 1])
    von_mises = np.zeros([num_grains, 1])
    # Turn exponential map into an orientation matrix
    chi_x_t = tmp_data_x[:, 2]
    chi_x = chi_x_t[great_grains]

    comp_x_t = tmp_data_x[:, 1]
    comp_x = comp_x_t[great_grains]

    # id_x_t = tmp_data_x[:, 0] #not used
    # id_x = id_x_t[great_grains] #not used
    for ii in range(0, num_grains-1, 1):
        great_grains_chi[ii] = chi_x[ii]
        great_grains_comp[ii] = comp_x[ii]
        Rsc = rot.rotMatOfExpMap(expMap[:, ii])
        strainTenS = np.zeros((3, 3), dtype='float64')
        strainTenS[0, 0] = strainTmp[0, ii]
        strainTenS[1, 1] = strainTmp[1, ii]
        strainTenS[2, 2] = strainTmp[2, ii]
        strainTenS[1, 2] = strainTmp[3, ii]
        strainTenS[0, 2] = strainTmp[4, ii]
        strainTenS[0, 1] = strainTmp[5, ii]
        strainTenS[2, 1] = strainTmp[3, ii]
        strainTenS[2, 0] = strainTmp[4, ii]
        strainTenS[1, 0] = strainTmp[5, ii]
        strainTenC = np.dot(np.dot(Rsc.T, strainTenS), Rsc)
        strainVecC = mutil.strainTenToVec(strainTenC)
        # v_strain_gg = np.trace(strainTenS)/3 # not used

        # Calculate stress
        stressVecC = np.dot(c_mat, strainVecC)
        stressTenC = mutil.stressVecToTen(stressVecC)
        stressTenS = np.dot(np.dot(Rsc, stressTenC), Rsc.T)
        stressVecS = mutil.stressTenToVec(stressTenS)

        # Calculate hydrostatic stress
        hydrostaticStress = (stressVecS[:3].sum()/3)
        w, v = np.linalg.eig(stressTenC)
        # maxShearStress = (np.max(w)-np.min(w))/2. # not used

        # Calculate Von Mises Stress
        # devStressS = stressTenS-hydrostaticStress*np.identity(3) # not used
        # vonMisesStress = np.sqrt((3/2)*(devStressS**2).sum()) # not used

        stress_data = dict()
        stress_data['stress_S'] = stress_S
        stress_data['stress_C'] = stress_C
        stress_data['hydrostatic'] = hydrostaticStress
        stress_data['max_shear'] = max_shear
        stress_data['pressure'] = pressure
        stress_data['von_mises'] = von_mises
        stress_data['principal'] = stress_prin
        great_grains_stresses[ii, 0] = great_grains[ii]
        great_grains_stresses[ii, 1:] = stressVecS.T
    return great_grains_stresses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate stress tensors from HEXRD grains.out"
    )
    parser.add_argument(
        'grains_file',
        help="HEXRD grains.out file"
    )
    parser.add_argument(
        'matrix',
        help="Tabular file (6x6 matrix of floating point values)"
    )
    parser.add_argument(
        '-m', '--completeness_thresh',
        type=float,
        default=0.8,
        help="completeness threshold"
    )
    parser.add_argument(
        '-c', '--chi_thresh',
        type=float,
        default=0.009,
        help="chi threshold"
    )
    parser.add_argument(
        '-v', '--vol_strain_thresh',
        type=float,
        default=0.001,
        help="volume strain threshold"
    )
    parser.add_argument(
        '-o', '--output',
        default='stress_tensors.txt',
        help="Tabular file (6x6 matrix of floating point values")
    args = parser.parse_args()

    cmat = np.loadtxt(args.matrix)
    grains_stresses = calc(args.grains_file, cmat,
                           completeness_thresh=args.completeness_thresh,
                           chi_thresh=args.chi_thresh,
                           vol_strain_thresh=args.vol_strain_thresh)
    np.savetxt(args.output, grains_stresses)
