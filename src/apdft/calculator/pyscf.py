#!/usr/bin/env python

import basis_set_exchange as bse
import apdft.calculator as apc
import os
import jinja2 as j
import numpy as np
from apdft import log
import functools


class PyscfCalculator(apc.Calculator):
    _methods = {"CCSD": "CCSD", "HF": "HF"}

    @staticmethod
    def _format_coordinates(nuclear_numbers, coordinates):
        """ Converts the vector representation into the atomspec format for PySCF."""
        ret = []
        for z, pos in zip(nuclear_numbers, coordinates):
            ret.append("%d %f %f %f" % (z, *pos))
        return ";".join(ret)

    @staticmethod
    def _format_basis(nuclear_numbers, basisset):
        basis = {}
        for nuclear_number in set(nuclear_numbers):
            basis[nuclear_number] = bse.get_basis(
                basisset, int(nuclear_number), fmt="nwchem"
            )
        return str(basis)

    @staticmethod
    def _format_list(values):
        return ",".join([str(_) for _ in values])

    def get_input(
        self,
        coordinates,
        nuclear_numbers,
        nuclear_charges,
        grid,
        iscomparison=False,
        includeonly=None,
    ):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open("%s/templates/pyscf.py" % basedir) as fh:
            template = j.Template(fh.read())

        env = {}
        env["atoms"] = PyscfCalculator._format_coordinates(nuclear_numbers, coordinates)
        env["basisset"] = PyscfCalculator._format_basis(nuclear_numbers, self._basisset)
        env["method"] = self._methods[self._method]

        if includeonly is None:
            includeonly = range(len(nuclear_numbers))
        env["includeonly"] = PyscfCalculator._format_list(includeonly)

        deltaZ = np.array(nuclear_charges) - np.array(nuclear_numbers)
        deltaZ = deltaZ[includeonly]
        env["deltaZ"] = PyscfCalculator._format_list(deltaZ)

        return template.render(**env)

    # For a different target geometry from the reference
    # Used in mode "energies_geometries"
    def get_input_general(
        self,
        coordinates,
        original_coordinates,
        target_coordinates,
        nuclear_numbers,
        nuclear_charges,
        grid,
        iscomparison=False,
        includeonly=None,
    ):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open("%s/templates/pyscf2.py" % basedir) as fh:
            template = j.Template(fh.read())

        env = {}
        env["atoms"] = PyscfCalculator._format_coordinates(
            nuclear_numbers, coordinates)
        # Original atoms with a original coordinate
        env["original_atoms"] = PyscfCalculator._format_coordinates(
            nuclear_numbers, original_coordinates)
        # Target atoms
        # The atom types are identical with the reference ("atoms").
        env["target_atoms"] = PyscfCalculator._format_coordinates(
            nuclear_numbers, target_coordinates)
        env["basisset"] = PyscfCalculator._format_basis(
            nuclear_numbers, self._basisset)
        env["method"] = self._methods[self._method]

        # Collect all different coordinates of atoms
        # TODO: this algorithm is not smart and should be changed.
        all_coordinates = original_coordinates
        all_coordinates = np.vstack([original_coordinates, coordinates])
        all_coordinates = np.vstack([all_coordinates, target_coordinates])

        collect_all_nuclear_numbers = nuclear_numbers
        collect_all_nuclear_numbers = np.vstack(
            [collect_all_nuclear_numbers, nuclear_numbers])
        collect_all_nuclear_numbers = np.vstack(
            [collect_all_nuclear_numbers, nuclear_numbers])
        onedim_collect_all_nuclear_numbers = collect_all_nuclear_numbers.flatten()

        # print(all_coordinates)

        # Obtain unique coordinates
        all_coordinates, all_nuclear_numbers_id = np.unique(all_coordinates, return_index=True, axis=0)
        all_nuclear_numbers = np.zeros(
            len(all_nuclear_numbers_id), dtype=np.int64)
        for id_1 in range(len(all_nuclear_numbers_id)):
            for id_2 in range(len(onedim_collect_all_nuclear_numbers)):
                if all_nuclear_numbers_id[id_1] == id_2:
                    all_nuclear_numbers[id_1] = onedim_collect_all_nuclear_numbers[id_2]

        # print(all_nuclear_numbers)
        # print(all_coordinates)

        # "all_atom" collects all atom geometries used in each QM calculation
        env["all_atoms"] = PyscfCalculator._format_coordinates(
            all_nuclear_numbers, all_coordinates
        )

        # To avoid error in PySCF, a spin state is setted.
        if np.sum(all_nuclear_numbers) % 2 == 0:
            # Singlet spin state
            env["all_spin"] = 0
        else:
            # Doublet spin state
            env["all_spin"] = 1

        if includeonly is None:
            includeonly = range(len(nuclear_numbers))
        env["includeonly"] = PyscfCalculator._format_list(includeonly)

        deltaZ = np.array(nuclear_charges) - np.array(nuclear_numbers)
        deltaZ = deltaZ[includeonly]
        env["deltaZ"] = PyscfCalculator._format_list(deltaZ)

        return template.render(**env)

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def _cached_log_read(folder):
        return open("%s/run.log" % folder).readlines()

    @staticmethod
    def _read_value(folder, label, multiple, lines=None):
        if lines is None:
            lines = PyscfCalculator._cached_log_read(folder)
        res = []
        for line in lines:
            parts = line.strip().split()
            if parts[0] == label:
                res.append([float(_) for _ in parts[1:]])
                # check for nan / inf values
                if not np.isfinite(res[-1]).all():
                    raise ValueError("Invalid value in log file.")
                if not multiple:
                    return np.array(res[0])

        return np.array(res)

    @staticmethod
    def get_total_energy(folder):
        return PyscfCalculator._read_value(folder, "TOTAL_ENERGY", False)

    def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open("%s/templates/pyscf-run.sh" % basedir) as fh:
            template = j.Template(fh.read())
        return template.render()

    @staticmethod
    def get_epn(folder, coordinates, includeatoms, nuclear_charges):
        epns = PyscfCalculator._read_value(folder, "ELECTRONIC_EPN", True)
        if len(epns.flatten()) == 0:
            raise ValueError("Incomplete calculation.")

        # check that all included sites are in fact present
        included_results = epns[:, 0].astype(np.int)
        if not set(included_results) == set(includeatoms):
            log.log(
                "Atom selections do not match. Likely the configuration has changed in the meantime.",
                level="error",
            )

        included_results = list(included_results)
        return epns[[included_results.index(_) for _ in includeatoms], 1]

    # For an "energies_geometries" mode
    # Obtain EPNs for the reference and target geometries
    @staticmethod
    def get_epn2(folder, coordinates, includeatoms, nuclear_charges):
        epns = PyscfCalculator._read_value(folder, "ELECTRONIC_EPN2", True)
        if len(epns.flatten()) == 0:
            raise ValueError("Incomplete calculation.")

        # check that all included sites are in fact present
        included_results = epns[:, 0].astype(np.int)
        if not set(included_results) == set(includeatoms):
            log.log(
                "Atom selections do not match. Likely the configuration has changed in the meantime.",
                level="error",
            )

        included_results = list(included_results)
        # Return only EPNs of each atom
        return epns[[included_results.index(_) for _ in includeatoms], 1]

    @staticmethod
    # Get the electronic part of Hellmann-Feynman atomic forces
    # from log files of PySCF calculations.
    def get_target_ionic_force(folder, coordinates, includeatoms, nuclear_charges):
        ionic_forces = PyscfCalculator._read_value(
            folder, "TARGET_IONIC_FORCE", True)
        # If no data are read, raise error.
        if len(ionic_forces.flatten()) == 0:
            raise ValueError("Incomplete calculation.")

        # check that all included sites are in fact present
        included_results = ionic_forces[:, 0].astype(np.int)
        if not set(included_results) == set(includeatoms):
            log.log(
                "Atom selections do not match. Likely the configuration has changed in the meantime.",
                level="error",
            )

        included_results = list(included_results)
        # Return only ionic forces of each atom
        return ionic_forces[[included_results.index(_) for _ in includeatoms], 1:4]

    @staticmethod
    # Get the electronic part of Hellmann-Feynman atomic forces
    # from log files of PySCF calculations.
    # For only "energies" mode.
    def get_ionic_force(folder, includeatoms):
        ionic_forces = PyscfCalculator._read_value(
            folder, "IONIC_FORCE", True)
        # If no data are read, raise error.
        if len(ionic_forces.flatten()) == 0:
            raise ValueError("Incomplete calculation.")

        # check that all included sites are in fact present
        included_results = ionic_forces[:, 0].astype(np.int)
        if not set(included_results) == set(includeatoms):
            log.log(
                "Atom selections do not match. Likely the configuration has changed in the meantime.",
                level="error",
            )

        included_results = list(included_results)
        # Return only ionic forces of each atom
        # ionic_forces[, 0] is the site number and ionic_forces[, 1:3] is the forces.
        return ionic_forces[[included_results.index(_) for _ in includeatoms], 1:4]

    @staticmethod
    def get_electronic_dipole(folder):
        dipoles = PyscfCalculator._read_value(folder, "ELECTRONIC_DIPOLE", True)
        # If no data are read, raise error.
        if len(dipoles.flatten()) == 0:
            raise ValueError("Incomplete calculation.")
        # Since dipoles has (1, 3) np.array, the first element [0, :]
        # is returned
        return dipoles[0]

    @staticmethod
    def get_target_electronic_dipole(folder):
        dipoles = PyscfCalculator._read_value(
            folder, "TARGET_ELECTRONIC_DIPOLE", True)
        # If no data are read, raise error.
        if len(dipoles.flatten()) == 0:
            raise ValueError("Incomplete calculation.")
        # Since dipoles has (1, 3) np.array, the first element [0, :]
        # is returned
        return dipoles[0]

    @staticmethod
    # Get the electronic part of Hellmann-Feynman atomic forces
    # from log files of PySCF calculations.
    # For only "energies" mode.
    def get_target_hf_ionic_force(folder, includeatoms):
        ionic_forces = PyscfCalculator._read_value(
            folder, "TARGET_IONIC_FORCE", True)
        # If no data are read, raise error.
        if len(ionic_forces.flatten()) == 0:
            raise ValueError("Incomplete calculation.")

        # check that all included sites are in fact present
        included_results = ionic_forces[:, 0].astype(np.int)
        if not set(included_results) == set(includeatoms):
            log.log(
                "Atom selections do not match. Likely the configuration has changed in the meantime.",
                level="error",
            )

        included_results = list(included_results)
        # Return only ionic forces of each atom
        # ionic_forces[, 0] is the site number and ionic_forces[, 1:3] is the forces.
        return ionic_forces[[included_results.index(_) for _ in includeatoms], 1:4]
