#!/usr/bin/env python

import argparse
import pathlib

from hexrd.material import load_materials_hdf5, save_materials_hdf5, Material
import numpy as np


def make_matl(mat_name, sgnum, lparms, hkl_ssq_max=50):
    matl = Material(mat_name)
    matl.name = mat_name
    matl.sgnum = sgnum
    matl.latticeParameters = lparms
    matl.hklMax = hkl_ssq_max
    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls, dtype=bool))
    return matl


def __main__():
    parser = argparse.ArgumentParser(description='', epilog='')
    parser.add_argument('-i', '--input',
                        type=pathlib.Path,
                        nargs='?', default=None,
                        help='A materials file to edit')
    parser.add_argument('-c', '--cif',
                        type=pathlib.Path,
                        ## nargs='*',
                        default=[], action='append',
                        help='material.cif file')
    parser.add_argument('-o', '--output',
                        type=pathlib.Path,
                        default='materials.h5',
                        help='Materials output file')
    parser.add_argument('-m', '--material',
                        ## nargs='*', 
                        default=[], action='append',
                        help='Material ')
    args = parser.parse_args()

    materials = {}
    if args.input:
        materials = load_materials_hdf5(args.input)
    for i, cifpath in enumerate(args.cif):
        cif = str(cifpath.resolve()) 
        print(cif)
        try:
            name = 'cif_' + str(i)
            mat = Material(name, material_file=cif)
            ## print(mat) 
            materials[name] = mat
        except Exception as e:
            print(cif, e)
    for mat in args.material:
        print(mat)
        try:
            fields = [x.strip() for x in mat.split(',')] 
            # ftypes = (str, int, float, float, float, float, float, float)
            # (name, spacegroup, a, b, c, alpha, beta, gamma) = [t(s) for t, s in zip(ftypes, fields[0:8])]
            name = str(fields[0])
            spacegroup = int(fields[1])
            lattice_params = [float(x) for x in fields[2:]]
            materials[name] = make_matl(name, spacegroup, lattice_params)
        except Exception as e:
            print(mat, e)
    print(materials.keys())
    save_materials_hdf5(args.output, materials)


if __name__ == "__main__":
    __main__()
