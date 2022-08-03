import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["VECLIB_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

import numpy as np
import numpy
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
from pyscf.data import nist

angstrom = 1 / 0.52917721067

mol = pyscf.gto.Mole()
mol.atom = "{{ atoms }}"
mol.basis = {{basisset}}
mol.verbose = 0
mol.build()

method = "{{ method }}"
# "CCSD", "HF", "PBE", "PBE0", and "B3LYP" are allowed.
if method not in ["CCSD", "HF", "PBE", "PBE0", "B3LYP"]:
    raise NotImplementedError("Method %s not supported." % method)

deltaZ = np.array(({{deltaZ}}))
includeonly = np.array(({{includeonly}}))


def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords()[includeonly]/ angstrom, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(float)
        q[includeonly] += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


if method == "HF":
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    hfe = calc.kernel(verbose=0)
    dm1_ao = calc.make_rdm1()
    total_energy = calc.e_tot
    Enn = calc.energy_nuc()
if method == "CCSD":
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    hfe = calc.kernel(verbose=0)
    mycc = pyscf.cc.CCSD(calc).run()
    dm1 = mycc.make_rdm1()
    dm1_ao = np.einsum("pi,ij,qj->pq", calc.mo_coeff, dm1, calc.mo_coeff.conj())
    total_energy = mycc.e_tot
    Enn = calc.energy_nuc()
if method in ["PBE", "PBE0", "B3LYP"]:
    # Set SCF calculation condition
    calc = add_qmmm(pyscf.scf.RKS(mol), mol, deltaZ)
    # kernel() function is the simple way to call HF driver.
    # verbose is the output level.
    calc.max_cycle = 1000
    if method == "PBE":
        calc.xc = 'pbe,pbe'
    elif method == "PBE0":
        calc.xc = 'pbe0'
    elif method == "B3LYP":
        calc.xc = 'b3lyp'
    calc.kernel(verbose=0)
    # One-particle density matrix in AO representation:
    # MO occupation number
    # * MO coefficients
    # * conjugated MO coefficients
    dm1_ao = calc.make_rdm1()
    total_energy = calc.e_tot
    # Calculate nuclear-nuclear repulsion energy of
    # of the reference molecule and
    Enn = calc.energy_nuc()
    # of the target molecular geometry
    # target_Enn = target_mol.energy_nuc()

# GRIDLESS, as things should be ############################
# Total energy of SCF run

print("TOTAL_ENERGY", total_energy)
print("NN_ENERGY", Enn)

# Electronic EPN from electron density
for site in includeonly:
    mol.set_rinv_orig_(mol.atom_coords()[site])
    print("ELECTRONIC_EPN", site, np.matmul(dm1_ao, mol.intor("int1e_rinv")).trace())

# Electronic Dipole w.r.t to center of geometry
with mol.with_common_orig(mol.atom_coords().mean(axis=0)):
    ao_dip = mol.intor_symmetric("int1e_r", comp=3)
dipole = -numpy.einsum("xij,ji->x", ao_dip, dm1_ao).real
print("ELECTRONIC_DIPOLE", *dipole)

# GRID, as things were #####################################
grid = pyscf.dft.gen_grid.Grids(mol)
grid.level = 3
grid.build()
ao_value = pyscf.dft.numint.eval_ao(mol, grid.coords, deriv=0)
rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_ao, xctype="LDA")

# Ionic Forces
for site in includeonly:
    rvec = grid.coords - mol.atom_coords()[site]
    force = [
        (rho * grid.weights * rvec[:, _] / np.linalg.norm(rvec, axis=1) ** 3).sum()
        for _ in range(3)
    ]
    print("IONIC_FORCE", site, *force)

# Quadrupole moments
rs = grid.coords - mol.atom_coords().mean(axis=0)
ds = np.linalg.norm(rs, axis=1) ** 2
# Q = np.zeros((3,3))
for i in range(3):
    for j in range(i, 3):
        q = 3 * rs[:, i] * rs[:, j]
        if i == j:
            q -= ds
        print("ELECTRONIC_QUADRUPOLE", i, j, (rho * q * grid.weights).sum())
