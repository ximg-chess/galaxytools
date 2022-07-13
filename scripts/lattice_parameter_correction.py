#!/usr/bin/env python3

import argparse

import numpy as np

from hexrd import rotations as rot
from hexrd import matrixutil as mutil
from hexrd import config


def _lp_calc(v_mat, f_mat, r_mat):
    """
    do strained lappi

    Parameters
    ----------
    vmat : TYPE
        DESCRIPTION.
    f_mat : TYPE
        DESCRIPTION.
    r_mat : TYPE
        DESCRIPTION.

    Returns
    -------
    lparams : TYPE
        DESCRIPTION.

    """
    f_prime = np.dot(v_mat, np.dot(r_mat, f_mat))
    f_prime_hat = mutil.unitVector(f_prime)

    lp_mags = mutil.columnNorm(f_prime)

    lp_angs = np.hstack(
        [np.arccos(np.dot(f_prime_hat[:, 1], f_prime_hat[:, 2])),
         np.arccos(np.dot(f_prime_hat[:, 2], f_prime_hat[:, 0])),
         np.arccos(np.dot(f_prime_hat[:, 0], f_prime_hat[:, 1]))]
    )
    return np.hstack([lp_mags, np.degrees(lp_angs)])


def extract_lattice_parameters(
        cfg_file, grains_file,
        min_completeness=0.90, force_symmetry=False
        ):
    """
    """
    cfg = config.open(cfg_file)[0]

    pd = cfg.material.plane_data
    qsym_mats = rot.quatProductMatrix(pd.getQSym())

    f_mat = pd.latVecOps['F']

    gt = np.loadtxt(grains_file)

    idx = gt[:, 1] >= min_completeness

    expMaps = gt[idx, 3:6].T
    vinv = gt[idx, 9:15]

    lparms = []
    for i in range(sum(idx)):
        quat = rot.quatOfExpMap(expMaps[:, i].reshape(3, 1))
        v_mat = np.linalg.inv(mutil.vecMVToSymm(vinv[i, :].reshape(6, 1)))
        if force_symmetry:
            for qpm in qsym_mats:
                lparms.append(
                    _lp_calc(v_mat, f_mat, rot.rotMatOfQuat(np.dot(qpm, quat)))
                )
        else:
            lparms.append(
                    _lp_calc(v_mat, f_mat, rot.rotMatOfQuat(quat))
                )
    lp_avg = np.average(np.vstack(lparms), axis=0)
    return lp_avg, sum(idx), len(idx)


if __name__ == '__main__':
    """
    maybe make this smart enough to average properly for symmetry?
    """
    parser = argparse.ArgumentParser(
        description="Extract average lattice parameters from grains.out"
    )
    parser.add_argument(
        'cfg', help="ff-HEDM config YAML file", type=str
        )
    parser.add_argument(
        'grains_file', help="grains.out file", type=str
    )
    parser.add_argument(
        '-m', '--min-completeness', help="completeness threshold", type=float,
        default=0.75
    )

    parser.add_argument(
        '-f', '--force-symmetry',
        help="symmetrize results",
        action="store_true"
    )

    args = parser.parse_args()

    cfg_file = args.cfg
    grains_file = args.grains_file
    min_completeness = args.min_completeness
    force_symmetry = args.force_symmetry

    lp_avg, n_used, n_total = extract_lattice_parameters(
        cfg_file, grains_file,
        min_completeness=min_completeness, force_symmetry=force_symmetry
    )
    print("Using %d grains out of %d above %.2f%% completeness threshold:"
          % (n_used, n_total, 100*min_completeness))
    print("---------------------------------------------------------------")
    print("a       \tb       \tc       "
          + "\talpha   \tbeta    \tgamma \n"
          + "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" % tuple(lp_avg))
