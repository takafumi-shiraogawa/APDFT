#!/usr/bin/env python

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import functools

# calculate Hessian for de/dz1dz2, get eigenvectors
# test benzene compare to 10.1021/acs.jpclett.8b02805, then do naphthalene

#%%
basepath = "/mnt/c/Users/guido/workcopies/apdft/prototyping/symmetry-rules"


@functools.lru_cache(maxsize=100)
def get_electronic_energy(basefolder, sitei, sitej, direction):
    if sitei == None and sitej == None:
        fn = f"order-0/site-all-cc"
    else:
        if sitej is None:
            sitej, sitei = sitei, sitej
        if sitei == None:
            fn = f"order-1/site-{abs(sitej)}-{direction}"
    if sitei != None and sitej != None:
        A = min(abs(sitei), abs(sitej))
        B = max(abs(sitei), abs(sitej))
        fn = f"order-2/site-{A}-{B}-{direction}"
    with open(f"{basefolder}/{fn}/run.log") as fh:
        lines = fh.readlines()
        energy = float([_ for _ in lines if _.startswith("TOTAL_ENERGY")][0].split()[1])
        # e_nn = float([_ for _ in lines if _.startswith("NN_ENERGY")][0].split()[1])
    return energy  # - e_nn


def hessian(molname, natoms):
    basefolder = f"{basepath}/{molname}-apdft/QM/"
    E = lambda i, j, d: get_electronic_energy(basefolder, i, j, d)
    hessian = np.zeros((natoms, natoms))
    for i in range(natoms):
        for j in range(i, natoms):
            if i != j:
                dE = (
                    E(i, j, "up")
                    - E(i, None, "up")
                    - E(None, i, "up")
                    + 2 * E(None, None, "none")
                    - E(-i, None, "dn")
                    - E(None, -j, "dn")
                    + E(-i, -j, "dn")
                )
                dE /= 0.05 ** 2 * 2
            else:
                dE = E(i, None, "up") - 2 * E(None, None, "none") + E(-i, None, "dn")
                dE /= 0.05 ** 2
            hessian[i, j] = dE
            hessian[j, i] = dE

    return hessian


# %%
np.savetxt(f"{basepath}/naphthalene-apdft/totalhessian.txt", hessian("naphthalene", 10))
np.savetxt(f"{basepath}/benzene-apdft/totalhessian.txt", hessian("benzene", 6))
vals, vectors = np.linalg.eig(hessian("naphthalene", 10))
for i in range(10):
    plt.bar(range(10), vectors.T[i], bottom=i)

# %%
vals
# %%
dz = np.array((0, 0, 1.0, 0, -1.0, 0))
plt.bar(range(6), np.dot(vectors.T, dz))

# %%

# %%
np.dot((1, 0, -1, -1, 0, 1), (1, 0, -1, 1, 0, -1))
# %%
