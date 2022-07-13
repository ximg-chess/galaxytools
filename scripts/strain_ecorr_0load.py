#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from hexrd import rotations as rot


def calc(grains_file,
         completeness_thresh=0.8, chi_thresh=0.009, vol_strain_thresh=0.001,
         fmt='png'):

    tmp_data_0 = np.loadtxt(grains_file)

    # id_0 = tmp_data_0[:, 0]
    comp_0 = tmp_data_0[:, 1]
    chi_0 = tmp_data_0[:, 2]
    strain_0 = tmp_data_0[:, 15:]
    ori_0 = tmp_data_0[:, 3:6]
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

    # updating v_strain_0 to corrected values
    v_strain_0 = v_strain_0 - p[0] * pos_0[:, 1] - p[1]

    n_grains = len(good_grains)

    astrain = np.zeros(n_grains)
    astrain2 = np.zeros(n_grains)
    cstrain = np.zeros(n_grains)

    strain_0[:, 0] = strain_0[:, 0] - (p[0] * pos_0[:, 1] + p[1])
    strain_0[:, 1] = strain_0[:, 1] - (p[0] * pos_0[:, 1] + p[1])
    strain_0[:, 2] = strain_0[:, 2] - (p[0] * pos_0[:, 1] + p[1])

    for ii in np.arange(n_grains):
        ti = good_grains[ii]
        strain_ten = np.array([
            [strain_0[ti, 0], strain_0[ti, 5], strain_0[ti, 4]],
            [strain_0[ti, 5], strain_0[ti, 1], strain_0[ti, 3]],
            [strain_0[ti, 4], strain_0[ti, 3], strain_0[ti, 2]]
            ])
        R = rot.rotMatOfExpMap(ori_0[ti, :])

        strain_ten_c = np.dot(R.T, np.dot(strain_ten, R))
        # print(strain_ten_c)
        astrain[ii] = strain_ten_c[0, 0]
        astrain2[ii] = strain_ten_c[1, 1]
        cstrain[ii] = strain_ten_c[2, 2]

    np.savetxt('strain_ecorr.txt', p)

    if fmt == 'pdf':
        with PdfPages('figures.pdf') as pdf:
            plt.figure(1)
            plt.plot(pos_0[good_grains, 1],
                     v_strain_0[good_grains],
                     'bo',
                     label='corrected')
            plt.legend(loc='lower right')
            pdf.savefig()
            plt.figure(2)
            plt.plot(astrain, 'x')
            plt.plot(astrain2, 'gx')
            plt.plot(cstrain, 'rx')
            pdf.savefig()
    else:
        plt.figure(1)
        plt.plot(pos_0[good_grains, 1],
                 v_strain_0[good_grains],
                 'bo',
                 label='corrected')
        plt.legend(loc='lower right')
        fname = 'figure1' + '.' + fmt
        plt.savefig(fname, dpi='figure', format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None)
        plt.figure(2)
        plt.plot(astrain, 'x')
        plt.plot(astrain2, 'gx')
        plt.plot(cstrain, 'rx')
        fname = 'figure2' + '.' + fmt
        plt.savefig(fname, dpi='figure', format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None)


if __name__ == '__main__':
    """
    """
    parser = argparse.ArgumentParser(
        description="Calculate strain_ecorr from grains.out"
    )
    parser.add_argument(
        'grains_file',
        help="grains.out file"
    )
    parser.add_argument(
        '-m', '--completeness_thresh',
        type=float,
        default=0.75,
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
        '-f', '--format', default='png',
        choices=['png', 'pdf', 'svg'],
        help="plot format"
    )
    args = parser.parse_args()
    calc(args.grains_file,
         completeness_thresh=args.completeness_thresh,
         chi_thresh=args.chi_thresh,
         vol_strain_thresh=args.vol_strain_thresh,
         fmt=args.format)
