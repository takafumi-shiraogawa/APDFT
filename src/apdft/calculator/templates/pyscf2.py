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

# Set information on a reference molecule
mol = pyscf.gto.Mole()
mol.atom = "{{ atoms }}"
mol.basis = {{basisset}}
mol.verbose = 0
mol.build()

# Set information on a original reference molecule
original_mol = pyscf.gto.Mole()
original_mol.atom = "{{ original_atoms }}"
original_mol.build()

# Set information on a target molecule
target_mol = pyscf.gto.Mole()
target_mol.atom = "{{ target_atoms }}"
target_mol.build()

# Set information on all atom geometries for generating numerical grids
all_mol = pyscf.gto.Mole()
all_mol.atom = "{{ all_atoms }}"
all_mol.spin = {{all_spin}}
all_mol.build()

# Set a quantum chemical computation method
# Only CCSD or HF is allowed.
method = "{{ method }}"
if method not in ["CCSD", "HF"]:
    raise NotImplementedError("Method %s not supported." % method)

deltaZ = np.array(({{deltaZ}}))
includeonly = np.array(({{includeonly}}))


def add_qmmm(calc, mol, deltaZ):
    # Background charges are patched to the underlying SCF calculation
    # The qmmm module implements the elctronic embedding model.
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords()[includeonly]/ angstrom, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(float)
        q[includeonly] += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


if method == "HF":
    # Set SCF calculation condition
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    # kernel() function is the simple way to call HF driver.
    # verbose is the output level.
    hfe = calc.kernel(verbose=0)
    # One-particle density matrix in AO representation:
    # MO occupation number
    # * MO coefficients
    # * conjugated MO coefficients
    dm1_ao = calc.make_rdm1()
    total_energy = calc.e_tot
    # Calculate nuclear-nuclear repulsion energy of
    # of the reference molecule and
    # Enn = calc.energy_nuc()
    # of the target molecular geometry
    # target_Enn = target_mol.energy_nuc()

if method == "CCSD":
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    hfe = calc.kernel(verbose=0)
    mycc = pyscf.cc.CCSD(calc).run()
    # Unrelaxed density matrix is evaluated in the MO basis
    dm1 = mycc.make_rdm1()
    # Convert the density matrix into the AO basis
    dm1_ao = np.einsum("pi,ij,qj->pq", calc.mo_coeff, dm1, calc.mo_coeff.conj())
    total_energy = mycc.e_tot
    # Calculate nuclear-nuclear repulsion energy of
    # of the reference molecule and
    # Enn = calc.energy_nuc()
    # of the target molecular geometry
    # target_Enn = target_mol.energy_nuc()

# GRIDLESS, as things should be ############################
# Total energy of SCF run

print("TOTAL_ENERGY", total_energy)
# print("NN_ENERGY", Enn)
# print("NN_ENERGY2", target_Enn)

# Electronic EPN from electron density
# For the reference molecule
for site in includeonly:
    # Update origin for operator `\frac{1}{|r-R_O|}`.
    mol.set_rinv_orig_(original_mol.atom_coords()[site])
    print("ELECTRONIC_EPN", site, np.matmul(dm1_ao, mol.intor("int1e_rinv")).trace())

# For the target molecule
for site in includeonly:
    # Update origin for operator `\frac{1}{|r-R_O|}`
    mol.set_rinv_orig_(target_mol.atom_coords()[site])
    print("ELECTRONIC_EPN2", site, np.matmul(dm1_ao, mol.intor("int1e_rinv")).trace())

# Electronic Dipole w.r.t to center of geometry (geometrical center)
with mol.with_common_orig(mol.atom_coords().mean(axis=0)):
    ao_dip = mol.intor_symmetric("int1e_r", comp=3)
dipole = -numpy.einsum("xij,ji->x", ao_dip, dm1_ao).real
print("ELECTRONIC_DIPOLE", *dipole)

# Target electronic Dipole w.r.t to center of geometry (geometrical center)
# of the target molecule
with mol.with_common_orig(target_mol.atom_coords().mean(axis=0)):
    target_ao_dip = mol.intor_symmetric("int1e_r", comp=3)
target_dipole = -numpy.einsum("xij,ji->x", target_ao_dip, dm1_ao).real
print("TARGET_ELECTRONIC_DIPOLE", *target_dipole)

# GRID, as things were #####################################
# For evaluation of atomic forces, numerical grids are small near both
# reference and target atoms.
grid = pyscf.dft.gen_grid.Grids(all_mol)
# level = 3 is a standard condition of PySCF
grid.level = 3
grid.build()
# Calculate AO values on the grids
ao_value = pyscf.dft.numint.eval_ao(mol, grid.coords, deriv=0)
# Calculate electron density on the grids
# Using xctype="LDA", rho only contains the electron density
# at the grids
rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_ao, xctype="LDA")

# Electronic Dipole evaluated with grids
# Set distance vectors to the grids from a mass center
# of the molecule
num_r = grid.coords - mol.atom_coords().mean(axis=0)
num_dipole = -np.einsum('g,g,gx->x', rho, grid.weights, num_r)
print("NUMERICAL_DIPOLE", *num_dipole)

# Ionic Forces
for site in includeonly:
    rvec = grid.coords - mol.atom_coords()[site]
    force = [
        (rho * grid.weights * rvec[:, _] / np.linalg.norm(rvec, axis=1) ** 3).sum()
        for _ in range(3)
    ]
    print("IONIC_FORCE", site, *force)

# Target ionic Forces
# Numerical grids are small near both reference and
# target atoms.
for site in includeonly:
    target_rvec = grid.coords - target_mol.atom_coords()[site]
    target_force = [
        (rho * grid.weights * target_rvec[:, _] /
         np.linalg.norm(target_rvec, axis=1) ** 3).sum()
        for _ in range(3)
    ]
    print("TARGET_IONIC_FORCE", site, *target_force)

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
