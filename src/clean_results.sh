#!/bin/sh
#
# Common outputs of "energies" and "energies_geometries" modes
# in modified APDFT
rm apdft.conf
rm commands.sh
rm -rf QM/
rm energies.csv
rm ele_energies.csv
rm nuc_energies.csv
rm dipoles.csv
rm ele_dipoles.csv
rm nuc_dipoles.csv
rm atomic_forces.csv
rm ele_atomic_forces.csv
rm nuc_atomic_forces.csv

# "energies_geometries" mode
rm energies_total_contributions.csv
rm energies_reference_contributions.csv
rm energies_target_contributions.csv
rm deriv_rho_contributions.csv
rm hf_ionic_force_contributions.csv
rm hf_ionic_forces.csv
rm ele_hf_ionic_forces.csv
rm nuc_hf_ionic_forces.csv
rm ver_atomic_forces.csv
rm ver_ele_atomic_forces.csv
