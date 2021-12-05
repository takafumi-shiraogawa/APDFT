#!/usr/bin/env python
import numpy as np
import basis_set_exchange as bse
import apdft
import os
import itertools as it
import pandas as pd

#: Conversion factor from Angstrom to Bohr
angstrom = 1 / 0.52917721067
#: Conversion factor from electron charges and Angstrom to Debye
debye = 1 / 0.20819433


class Coulomb(object):
    """ Collects functions for Coulomb interaction."""

    @staticmethod
    def nuclei_nuclei(coordinates, charges):
        """ Calculates the nuclear-nuclear interaction energy from Coulomb interaction.

		Sign convention assumes positive charges for nuclei.

		Args:
			coordinates:		A (N, 3) array of nuclear coordinates :math:`\\mathbf{r_i}`. [Angstrom]
			charges:			A N array of point charges :math:`q_i`. [e]
		Returns:
			Coulombic interaction energy. [Hartree]
		"""
        natoms = len(coordinates)
        ret = 0.0
        for i in range(natoms):
            for j in range(i + 1, natoms):
                d = np.linalg.norm((coordinates[i] - coordinates[j]) * angstrom)
                ret += charges[i] * charges[j] / d
        return ret

    @staticmethod
    def nuclear_potential(coordinates, charges, at):
        natoms = len(coordinates)
        ret = 0.0
        for i in range(natoms):
            if i == at:
                continue
            d = np.linalg.norm((coordinates[i] - coordinates[at]) * angstrom)
            ret += charges[i] / d
        return ret

    @staticmethod
    def nuclei_atom_force(coordinates, charges):
        """ Calculates the nuclear part of atomic forces.

		Sign convention assumes positive charges for nuclei.

		Args:
			coordinates:		A (N, 3) array of nuclear coordinates :math:`\\mathbf{r_i}`. [Angstrom]
			charges:			A N array of point charges :math:`q_i`. [e]
		Returns:
			Nuclear part of atomic forces. [Hartree]
		"""
        natoms = len(coordinates)

        # Set atomic force vectors for all atoms of the molecule
        ret = np.zeros((natoms, 3))

        # Roops for all combinations between atoms
        # Roop for specifying a target atom for the atomic force
        for i in range(natoms):
            # Roop for specifying the other atoms
            for j in range(natoms):
                if j == i:
                    continue
                # Roop for specifying three Cartesian coordinates
                for k in range(3):

                    # Distance between two atoms in the absolute value
                    abs_d = np.linalg.norm(
                        (coordinates[i] - coordinates[j]) * angstrom)
                    vec_d = (coordinates[i, k] - coordinates[j, k]) * angstrom

                    # ret += charges[i] * charges[j] / (abs_d ** 3.0)
                    ret[i, k] += charges[i] * charges[j] * vec_d / (abs_d ** 3.0)
        return ret


class Dipoles(object):
    """ Collects functions regarding the calculation of dipole moments. This code follows the physics convention of the sign: the dipole moment vector points from the negative charge center to the positive charge center."""

    @staticmethod
    def point_charges(reference_point, coordinates, charges):
        """ Calculates the dipole moment of point charges.

		Note that for sets of point charges of a net charge, the resulting dipole moment depends on the chosen reference point. A common choice in the molecular context is the center of mass.
		Sign convention is such that nuclei should be given as positive charges.

		.. math::

			\\mathbf{p}(\\mathbf{r}) = \\sum_I q_i(\\mathbf{r_i}-\\mathbf{r})

		Args:
			reference_point:	A 3 array of the reference point :math:`\\mathbf{r}`. [Angstrom]
			coordinates: 		A (3,N) array of point charge coordinates :math:`\\mathbf{r_i}`. [Angstrom]
			charges:			A N array of point charges :math:`q_i`. [e]
		Returns:
			Dipole moment :math:`\\mathbf{p}`. [Debye]
		"""
        shift = coordinates - reference_point
        return np.sum(shift.T * charges, axis=1) * debye

    # Compute point charges in the atomic unit
    @staticmethod
    def point_charges_au(reference_point, coordinates, charges):
        """ Calculates the dipole moment of point charges.

		Note that for sets of point charges of a net charge, the resulting dipole moment depends on the chosen reference point. A common choice in the molecular context is the center of mass.
		Sign convention is such that nuclei should be given as positive charges.

		.. math::

			\\mathbf{p}(\\mathbf{r}) = \\sum_I q_i(\\mathbf{r_i}-\\mathbf{r})

		Args:
			reference_point:	A 3 array of the reference point :math:`\\mathbf{r}`. [Angstrom]
			coordinates: 		A (3,N) array of point charge coordinates :math:`\\mathbf{r_i}`. [Angstrom]
			charges:			A N array of point charges :math:`q_i`. [e]
		Returns:
			Dipole moment :math:`\\mathbf{p}`. [Debye]
		"""
        shift = coordinates - reference_point
        shift *= angstrom
        return np.sum(shift.T * charges, axis=1)

    @staticmethod
    def electron_density(reference_point, coordinates, electron_density):
        """ Calculates the dipole moment of a charge distribution.

		Note that for a charge density, the resulting dipole momennt depends on the chosen reference point. A common choice in the molecular context is the center of mass.
		Electron density is a positive quantity.

		.. math::

			\\mathbf{p}(\\mathbf{r}) = \\int_\\Omega \\rho(\\mathbf{r_i})(\\mathbf{r_i}-\\mathbf{r})
		
		Args:
			reference_point:	A 3 array of the reference point :math:`\\mathbf{r}`. [Angstrom]
			coordinates: 		A (3,N) array of grid coordinates :math:`\\mathbf{r_i}`. [Angstrom]
			electron_density:	A N array of electron density values :math:`\\rho` at `coordinates`. [e/Angstrom^3]
		Returns:
			Dipole moment :math:`\\mathbf{p}`. [Debye]
		"""
        shift = coordinates - reference_point
        return -np.sum(shift.T * electron_density, axis=1) * debye


def charge_to_label(Z):
    """ Converts a nuclear charge to an element label.

	Uncharged (ghost) sites are assigned a dash.

	Args:
		Z 					Nuclear charge. [e]
	Returns:
		Element label. [String]
	"""
    if Z == 0:
        return "-"
    return bse.lut.element_sym_from_Z(Z, normalize=True)


class APDFT(object):
    """ Implementation of alchemical perturbation density functional theory.
    
    Requires a working directory `basepath` which allows for storing the intermediate calculation results."""

    def __init__(
        self,
        highest_order,
        nuclear_numbers,
        coordinates,
        basepath,
        calculator,
        max_charge=0,
        max_deltaz=3,
        include_atoms=None,
        targetlist=None,
        target_cartesian="z",
        small_deltaZ = 0.05,
        small_deltaR = 0.005,
        mix_lambda = 1.0,
        calc_der = False
    ):
        # Exception handling for the apdft.conf input
        # For APDFT order
        if highest_order > 2:
            raise NotImplementedError("apdft_maxorder in apdft.conf must be smaller than 3.")
        # For molecular geometry change in the "energies_geometries" mode
        if target_cartesian not in ["z", "full"]:
            raise NotImplementedError("apdft_cartesian in apdft.conf must be z or full.")
        self._orders = list(range(0, highest_order + 1))
        self._nuclear_numbers = np.array(nuclear_numbers)
        self._coordinates = coordinates
        self._cartesian = target_cartesian
        self._delta = small_deltaZ
        self._R_delta = small_deltaR
        self._basepath = basepath
        self._calculator = calculator
        self._max_charge = max_charge
        self._max_deltaz = max_deltaz
        self._targetlist = targetlist
        self._mix_lambda = mix_lambda
        self._calc_der = calc_der

        # For a combination of APDFT order and vertical energy derivative calculations
        if max(self._orders) > 1 and self._calc_der:
            raise NotImplementedError(
                "Combination of vertical energy derivatives and APDFTn (n > 2) is not implemented yet.")

        if include_atoms is None:
            self._include_atoms = list(range(len(self._nuclear_numbers)))
        else:
            included = []
            for part in include_atoms:
                if isinstance(part, int):
                    included.append(part)
                else:
                    included += list(
                        np.where(
                            self._nuclear_numbers == bse.lut.element_Z_from_sym(part)
                        )[0]
                    )
            self._include_atoms = sorted(list(set(included)))

    def _calculate_delta_Z_vector(self, numatoms, order, sites, direction):
        baseline = np.zeros(numatoms)

        if order > 0:
            sign = {"up": 1, "dn": -1}[direction] * self._delta
            baseline[list(sites)] += sign

        return baseline

    # For molecular geometry changes
    def _calculate_delta_R_vector(self, numatoms, order, sites, direction, axis):
        baseline = np.zeros((numatoms, 3))

        if order > 0:
            sign = {"up": 1, "dn": -1}[direction] * (self._R_delta)
            baseline[list(sites), axis] += sign

        return baseline

    def prepare(self, explicit_reference=False):
        """ Builds a complete folder list of all relevant calculations."""
        if os.path.isdir("QM"):
            apdft.log.log(
                "Input folder exists. Reusing existing data.", level="warning"
            )

        commands = []

        for order in self._orders:
            # only upper triangle with diagonal
            for combination in it.combinations_with_replacement(
                self._include_atoms, order
            ):
                if len(combination) == 2 and combination[0] == combination[1]:
                    continue
                if order > 0:
                    label = "-" + "-".join(map(str, combination))
                    directions = ["up", "dn"]
                else:
                    directions = ["cc"]
                    label = "-all"

                for direction in directions:
                    path = "QM/order-%d/site%s-%s" % (order, label, direction)
                    commands.append("( cd %s && bash run.sh )" % path)
                    if os.path.isdir(path):
                        continue
                    os.makedirs(path)

                    charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                        len(self._nuclear_numbers), order, combination, direction
                    )
                    inputfile = self._calculator.get_input(
                        self._coordinates,
                        self._nuclear_numbers,
                        charges,
                        None,
                        includeonly=self._include_atoms,
                    )
                    with open("%s/run.inp" % path, "w") as fh:
                        fh.write(inputfile)
                    with open("%s/run.sh" % path, "w") as fh:
                        fh.write(
                            self._calculator.get_runfile(
                                self._coordinates, self._nuclear_numbers, charges, None
                            )
                        )

        if explicit_reference:
            targets = self.enumerate_all_targets()
            apdft.log.log(
                "All targets listed for comparison run.",
                level="info",
                count=len(targets),
            )
            for target in targets:
                path = "QM/comparison-%s" % ("-".join(map(str, target)))
                os.makedirs(path, exist_ok=True)

                inputfile = self._calculator.get_input(
                    self._coordinates, self._nuclear_numbers, target, None
                )
                with open("%s/run.inp" % path, "w") as fh:
                    fh.write(inputfile)
                with open("%s/run.sh" % path, "w") as fh:
                    fh.write(
                        self._calculator.get_runfile(
                            self._coordinates, self._nuclear_numbers, target, None
                        )
                    )
                commands.append("( cd %s && bash run.sh )" % path)

        # write commands
        with open("commands.sh", "w") as fh:
            fh.write("\n".join(commands))

    # For mode "energies_geometries"
    def prepare_general(self, target_coordinate=None, explicit_reference=False):
        """ Builds a complete folder list of all relevant calculations."""
        if os.path.isdir("QM"):
            apdft.log.log(
                "Input folder exists. Reusing existing data.", level="warning"
            )

        if target_coordinate is None:
            apdft.log.log(
                "Target molecular coordinate is not given.", level="error"
            )

        commands = []

        for order in self._orders:
            # only upper triangle with diagonal

            # Loop for nuclear charge changes
            for combination_z in it.combinations_with_replacement(
                self._include_atoms, order
            ):
                # If the order is 2 and selected two atoms are
                # equivalent, QM calculations are not needed for
                # constructing perturbed electron densities, and hence
                # the directories are not necessary, at least within
                # APDFT3 or lower levels.
                # TODO: need to consider the derivatives of the density
                #       for the higher level APDFT (n > 3)
                if len(combination_z) == 2 and \
                    combination_z[0] == combination_z[1]:
                    continue
                if order > 0:
                    # For nuclear charge changes
                    # e.g., -1- and -0-1-
                    label = "-" + "-".join(map(str, combination_z))
                    directions = ["up", "dn"]
                else:
                    directions = ["cc"]
                    label = "-all"

                for direction in directions:
                    # In QM/order-0/, site-all-cc is unique.
                    if direction == "cc":
                        site_name = 'site'
                    else:
                        site_name = 'z-site'
                    path = "QM/order-%d/%s%s-%s" % (order, site_name, label, direction)
                    commands.append("( cd %s && bash run.sh )" % path)
                    if os.path.isdir(path):
                        continue
                    os.makedirs(path)

                    charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                        len(self._nuclear_numbers), order, combination_z, direction
                    )
                    inputfile = self._calculator.get_input_general(
                        self._coordinates,
                        self._coordinates,
                        target_coordinate,
                        self._nuclear_numbers,
                        charges,
                        None,
                        includeonly=self._include_atoms,
                    )
                    with open("%s/run.inp" % path, "w") as fh:
                        fh.write(inputfile)
                    with open("%s/run.sh" % path, "w") as fh:
                        fh.write(
                            self._calculator.get_runfile(
                                self._coordinates, self._nuclear_numbers, charges, None
                            )
                        )

            # Loop for nuclear coordinate changes
            for combination_r in it.combinations_with_replacement(
                self._include_atoms, order
            ):
                # If this is a calculation of the analytical derivative of potential
                # energy with respect to nuclear coordinates by a vertical manner,
                # this roop is skipped
                if self._calc_der:
                    continue

                # For z-Cartesian coordinate changes
                if self._cartesian == "z":
                    # If the order is 2 and selected two atoms are
                    # equivalent, QM calculations are not needed for
                    # constructing perturbed electron densities, and hence
                    # the directories are not necessary, at least within
                    # APDFT3 or lower levels.
                    # TODO: need to consider the derivatives of the density
                    #       for the higher level APDFT (n > 3)
                    if len(combination_r) == 2 and combination_r[0] == combination_r[1]:
                        continue
                    if order > 0:
                        # For nuclear coordinate changes
                        # e.g., -1- and -0-1-
                        label = "-" + "-".join(map(str, combination_r))
                        directions = ["up", "dn"]
                    else:
                        # In QM/order-0/, site-all-cc is unique.
                        continue

                    for direction in directions:
                        path = "QM/order-%d/r-site%s-%s" % (order, label, direction)
                        commands.append("( cd %s && bash run.sh )" % path)
                        if os.path.isdir(path):
                            continue
                        os.makedirs(path)

                        # It is assumed that only Z coordinate changes
                        # TODO: generalize to three Cartesian components
                        nuclear_positions = self._coordinates + self._calculate_delta_R_vector(
                            len(self._nuclear_numbers), order, combination_r, direction, 2
                        )
                        inputfile = self._calculator.get_input_general(
                            nuclear_positions,
                            self._coordinates,
                            target_coordinate,
                            self._nuclear_numbers,
                            self._nuclear_numbers,
                            None,
                            includeonly=self._include_atoms,
                        )
                        with open("%s/run.inp" % path, "w") as fh:
                            fh.write(inputfile)
                        with open("%s/run.sh" % path, "w") as fh:
                            fh.write(
                                self._calculator.get_runfile(
                                    self._coordinates, self._nuclear_numbers, charges, None
                                )
                            )
                # For full-Cartesian coordinate changes
                else:
                    # If the order is 2 and selected two atoms are
                    # equivalent, QM calculations are not needed for
                    # constructing perturbed electron densities, and hence
                    # the directories are not necessary, at least within
                    # APDFT3 or lower levels.
                    # TODO: need to consider the derivatives of the density
                    #       for the higher level APDFT (n > 3)
                    if len(combination_r) == 2 and combination_r[0] == combination_r[1]:
                        continue
                    if order > 0:
                        # For nuclear coordinate changes
                        # e.g., -1- and -0-1-
                        label = "-" + "-".join(map(str, combination_r))
                        directions = ["up", "dn"]
                    else:
                        # In QM/order-0/, site-all-cc is unique.
                        continue

                    # For 3 Cartesian axes
                    for didx, dim in enumerate("XYZ"):
                        for direction in directions:
                            path = "QM/order-%d/r%s-site%s-%s" % (
                                order, dim, label, direction)
                            commands.append("( cd %s && bash run.sh )" % path)
                            if os.path.isdir(path):
                                continue
                            os.makedirs(path)

                            # It is assumed that only Z coordinate changes
                            # TODO: generalize to three Cartesian components
                            nuclear_positions = self._coordinates + self._calculate_delta_R_vector(
                                len(self._nuclear_numbers), order, combination_r, direction, didx
                            )
                            inputfile = self._calculator.get_input_general(
                                nuclear_positions,
                                self._coordinates,
                                target_coordinate,
                                self._nuclear_numbers,
                                self._nuclear_numbers,
                                None,
                                includeonly=self._include_atoms,
                            )
                            with open("%s/run.inp" % path, "w") as fh:
                                fh.write(inputfile)
                            with open("%s/run.sh" % path, "w") as fh:
                                fh.write(
                                    self._calculator.get_runfile(
                                        self._coordinates, self._nuclear_numbers, charges, None
                                    )
                                )

            # Loop for mixed changes of nuclear charge and coordinate
            for combination_zr in it.product(
                self._include_atoms, repeat = order
            ):
                # If this is a calculation of the analytical derivative of potential
                # energy with respect to nuclear coordinates by a vertical manner,
                # this roop is skipped
                if self._calc_der:
                    continue

                # For z-Cartesian coordinate changes
                if self._cartesian == "z":
                    # If the order is 2 and selected two atoms are
                    # equivalent, QM calculations are not needed for
                    # constructing perturbed electron densities, and hence
                    # the directories are not necessary, at least within
                    # APDFT3 or lower levels.
                    # TODO: need to consider the derivatives of the density
                    #       for the higher level APDFT (n > 3)
                    # if len(combination_zr) == 2 and combination_zr[0] == combination_zr[1]:
                    #     continue
                    if order > 1:
                        # For nuclear mixed changes of nuclear charge and coordinate
                        # e.g., -0-1-
                        label = "-" + "-".join(map(str, combination_zr))
                        directions = ["up", "dn"]
                    else:
                        # In QM/order-0/, site-all-cc is unique, and
                        # zr-site-?-?-dn or -up do not appear in QM/order-1/.
                        continue

                    for direction in directions:
                        path = "QM/order-%d/zr-site%s-%s" % (order, label, direction)
                        commands.append("( cd %s && bash run.sh )" % path)
                        if os.path.isdir(path):
                            continue
                        os.makedirs(path)

                        # For nuclear charge changes
                        # TODO:generalize to higher-order APDFT (n > 3)
                        charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                            len(self._nuclear_numbers), order, tuple([combination_zr[0]]), direction
                        )
                        # For molecular geometry changes
                        # TODO:generalize to higher-order APDFT (n > 3)
                        nuclear_positions = self._coordinates + self._calculate_delta_R_vector(
                            len(self._nuclear_numbers), order, tuple([combination_zr[1]]), direction, 2
                        )
                        inputfile = self._calculator.get_input_general(
                            nuclear_positions,
                            self._coordinates,
                            target_coordinate,
                            self._nuclear_numbers,
                            charges,
                            None,
                            includeonly=self._include_atoms,
                        )
                        with open("%s/run.inp" % path, "w") as fh:
                            fh.write(inputfile)
                        with open("%s/run.sh" % path, "w") as fh:
                            fh.write(
                                self._calculator.get_runfile(
                                    self._coordinates, self._nuclear_numbers, charges, None
                                )
                            )
                # For full-Cartesian coordinate changes
                else:
                    # If the order is 2 and selected two atoms are
                    # equivalent, QM calculations are not needed for
                    # constructing perturbed electron densities, and hence
                    # the directories are not necessary, at least within
                    # APDFT3 or lower levels.
                    # TODO: need to consider the derivatives of the density
                    #       for the higher level APDFT (n > 3)
                    # if len(combination_zr) == 2 and combination_zr[0] == combination_zr[1]:
                    #     continue
                    if order > 1:
                        # For nuclear mixed changes of nuclear charge and coordinate
                        # e.g., -0-1-
                        label = "-" + "-".join(map(str, combination_zr))
                        directions = ["up", "dn"]
                    else:
                        # In QM/order-0/, site-all-cc is unique, and
                        # zr-site-?-?-dn or -up do not appear in QM/order-1/.
                        continue

                    # For 3 Cartesian axes
                    for didx, dim in enumerate("XYZ"):
                        for direction in directions:
                            path = "QM/order-%d/zr%s-site%s-%s" % (order, dim, label, direction)
                            commands.append("( cd %s && bash run.sh )" % path)
                            if os.path.isdir(path):
                                continue
                            os.makedirs(path)

                            # For nuclear charge changes
                            # TODO:generalize to higher-order APDFT (n > 3)
                            charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                                len(self._nuclear_numbers), order, tuple([combination_zr[0]]), direction
                            )
                            # For molecular geometry changes
                            # TODO:generalize to higher-order APDFT (n > 3)
                            nuclear_positions = self._coordinates + self._calculate_delta_R_vector(
                                len(self._nuclear_numbers), order, tuple([combination_zr[1]]), direction, didx
                            )
                            inputfile = self._calculator.get_input_general(
                                nuclear_positions,
                                self._coordinates,
                                target_coordinate,
                                self._nuclear_numbers,
                                charges,
                                None,
                                includeonly=self._include_atoms,
                            )
                            with open("%s/run.inp" % path, "w") as fh:
                                fh.write(inputfile)
                            with open("%s/run.sh" % path, "w") as fh:
                                fh.write(
                                    self._calculator.get_runfile(
                                        self._coordinates, self._nuclear_numbers, charges, None
                                    )
                                )

            # If this is a calculation of the analytical derivative of potential
            # energy with respect to nuclear coordinates by a vertical manner
            if self._calc_der:
                # Loop for mixed changes of nuclear charge and coordinates
                # for nuclear coordinate changes for analytical derivative
                # of potential energy with respect to nuclear coordinates
                #   APDFT1 <- del_rho / del_R
                #       e.g., QM/order-1/rz-site-*-up
                #   APDFT2 <- del^2_rho / (del_R * del_Z)
                #       e.g., QM/order-2/rz-site-*-*-dn
                # TODO: implementation of APDFT3 derivatives
                #   APDFT3 <- del^3_rho / (del_R * del_Z * del_Z)
                #       e.g., QM/order-3/rz-site-*-*-*-up
                for combination_rz in it.product(
                    self._include_atoms, repeat=order + 1
                ):
                    # For z-Cartesian coordinate changes
                    if self._cartesian == "z":
                        # For nuclear mixed changes of nuclear charge and coordinate
                        # e.g., -0-1-
                        label = "-" + "-".join(map(str, combination_rz))
                        directions = ["up", "dn"]

                        for direction in directions:
                            # Because the analytical derivative requires one-order higher
                            # derivative of the electron density in comparison with
                            # the energy, order + 1 is used here.
                            path = "QM/order-%d/rz-site%s-%s" % (
                                order + 1, label, direction)
                            commands.append("( cd %s && bash run.sh )" % path)
                            if os.path.isdir(path):
                                continue
                            os.makedirs(path)

                            # For molecular geometry changes
                            # TODO: generalize to higher-order APDFT (n > 3)
                            nuclear_positions = self._coordinates + self._calculate_delta_R_vector(
                                len(self._nuclear_numbers), order + 1, tuple(
                                    [combination_rz[0]]), direction, 2
                            )
                            # For nuclear charge changes
                            # TODO: generalize to higher-order APDFT (n > 3)
                            if len(combination_rz) > 1:
                                charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                                    len(self._nuclear_numbers), order + 1, tuple(
                                        [combination_rz[1]]), direction
                                )
                            else:
                                charges = self._nuclear_numbers
                            inputfile = self._calculator.get_input_general(
                                nuclear_positions,
                                self._coordinates,
                                target_coordinate,
                                self._nuclear_numbers,
                                charges,
                                None,
                                includeonly=self._include_atoms,
                            )
                            with open("%s/run.inp" % path, "w") as fh:
                                fh.write(inputfile)
                            with open("%s/run.sh" % path, "w") as fh:
                                fh.write(
                                    self._calculator.get_runfile(
                                        self._coordinates, self._nuclear_numbers, charges, None
                                    )
                                )

                    # For full-Cartesian coordinate changes
                    # elif self._cartesian == "full":
                    else:
                        # For nuclear mixed changes of nuclear charge and coordinate
                        # e.g., -0-1-
                        label = "-" + "-".join(map(str, combination_rz))
                        directions = ["up", "dn"]

                        # For 3 Cartesian axes
                        for didx, dim in enumerate("XYZ"):
                            for direction in directions:
                                path = "QM/order-%d/r%sz-site%s-%s" % (
                                    order + 1, dim, label, direction)
                                commands.append(
                                    "( cd %s && bash run.sh )" % path)
                                if os.path.isdir(path):
                                    continue
                                os.makedirs(path)

                                # For molecular geometry changes
                                # TODO:generalize to higher-order APDFT (n > 3)
                                nuclear_positions = self._coordinates + self._calculate_delta_R_vector(
                                    len(self._nuclear_numbers), order + 1, tuple(
                                        [combination_rz[0]]), direction, didx
                                )
                                # For nuclear charge changes
                                # TODO: generalize to higher-order APDFT (n > 3)
                                if len(combination_rz) > 1:
                                    charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                                        len(self._nuclear_numbers), order + 1, tuple(
                                            [combination_rz[1]]), direction
                                    )
                                else:
                                    charges = self._nuclear_numbers
                                inputfile = self._calculator.get_input_general(
                                    nuclear_positions,
                                    self._coordinates,
                                    target_coordinate,
                                    self._nuclear_numbers,
                                    charges,
                                    None,
                                    includeonly=self._include_atoms,
                                )
                                with open("%s/run.inp" % path, "w") as fh:
                                    fh.write(inputfile)
                                with open("%s/run.sh" % path, "w") as fh:
                                    fh.write(
                                        self._calculator.get_runfile(
                                            self._coordinates, self._nuclear_numbers, charges, None
                                        )
                                    )

        if explicit_reference:
            targets = self.enumerate_all_targets()
            apdft.log.log(
                "All targets listed for comparison run.",
                level="info",
                count=len(targets),
            )
            for target in targets:
                path = "QM/comparison-%s" % ("-".join(map(str, target)))
                os.makedirs(path, exist_ok=True)

                inputfile = self._calculator.get_input(
                    self._coordinates, self._nuclear_numbers, target, None
                )
                with open("%s/run.inp" % path, "w") as fh:
                    fh.write(inputfile)
                with open("%s/run.sh" % path, "w") as fh:
                    fh.write(
                        self._calculator.get_runfile(
                            self._coordinates, self._nuclear_numbers, target, None
                        )
                    )
                commands.append("( cd %s && bash run.sh )" % path)

        # write commands
        with open("commands.sh", "w") as fh:
            fh.write("\n".join(commands))

    def _get_stencil_coefficients(self, deltaZ, shift):
        """ Calculates the prefactors of the density terms outlined in the documentation of the implementation, e.g. alpha and beta.

        In general, this collects all terms of the taylor expansion for one particular target and returns their coefficients.
        For energies, the n-th order derivative of the density is divided by :math:`(n+1)!`, while the target density is obtained
        from a regular Taylor expansion, i.e. the density derivative is divided by :math:`n!`. Therefore, a `shift` of 1 returns
        energy coefficients and a `shift` of 0 returns density coefficients.

        Args:
            self:   APDFT instance to obtain the stencil from.
            deltaZ: Array of length N. Target system as described by the change in nuclear charges. [e]
            shift:  Integer. Shift of the factorial term in the energy expansion or density expansion.
        """

        # build alphas
        N = len(self._include_atoms)
        nvals = {0: 1, 1: N * 2, 2: N * (N - 1)}
        alphas = np.zeros((sum([nvals[_] for _ in self._orders]), len(self._orders)))

        # test input
        if N != len(deltaZ):
            raise ValueError(
                "Mismatch of array lengths: %d dZ values for %d nuclei."
                % (len(deltaZ), N)
            )

        # order 0
        if 0 in self._orders:
            alphas[0, 0] = 1

            # Mixing reference and target molecules by using non-integer lambda
            # If target molecule is not targeted
            if self._mix_lambda != 1.0:
                alphas[0, 0] *= self._mix_lambda ** shift

        # order 1
        if 1 in self._orders:
            prefactor = 1 / (2 * self._delta) / np.math.factorial(1 + shift)

            # Mixing reference and target molecules by using non-integer lambda
            # If target molecule is not targeted
            if self._mix_lambda != 1.0:
                prefactor *= self._mix_lambda ** 2.0

            for siteidx in range(N):
                alphas[1 + siteidx * 2, 1] += prefactor * deltaZ[siteidx]
                alphas[1 + siteidx * 2 + 1, 1] -= prefactor * deltaZ[siteidx]

        # order 2
        if 2 in self._orders:
            pos = 1 + N * 2 - 2
            for siteidx_i in range(N):
                for siteidx_j in range(siteidx_i, N):
                    if siteidx_i != siteidx_j:
                        pos += 2
                    if deltaZ[siteidx_j] == 0 or deltaZ[siteidx_i] == 0:
                        continue
                    if self._include_atoms[siteidx_j] > self._include_atoms[siteidx_i]:
                        prefactor = (1 / (2 * self._delta ** 2)) / np.math.factorial(
                            2 + shift
                        )

                        # Mixing reference and target molecules by using non-integer lambda
                        # If target molecule is not targeted
                        if self._mix_lambda != 1.0:
                            prefactor *= self._mix_lambda ** (2 + shift)

                        prefactor *= deltaZ[siteidx_i] * deltaZ[siteidx_j]
                        alphas[pos, 2] += prefactor
                        alphas[pos + 1, 2] += prefactor
                        alphas[0, 2] += 2 * prefactor
                        alphas[1 + siteidx_i * 2, 2] -= prefactor
                        alphas[1 + siteidx_i * 2 + 1, 2] -= prefactor
                        alphas[1 + siteidx_j * 2, 2] -= prefactor
                        alphas[1 + siteidx_j * 2 + 1, 2] -= prefactor
                    if self._include_atoms[siteidx_j] == self._include_atoms[siteidx_i]:
                        prefactor = (1 / (self._delta ** 2)) / np.math.factorial(
                            2 + shift
                        )

                        # Mixing reference and target molecules by using non-integer lambda
                        # If target molecule is not targeted
                        if self._mix_lambda != 1.0:
                            prefactor *= self._mix_lambda ** (2 + shift)

                        prefactor *= deltaZ[siteidx_i] * deltaZ[siteidx_j]
                        alphas[0, 2] -= 2 * prefactor
                        alphas[1 + siteidx_i * 2, 2] += prefactor
                        alphas[1 + siteidx_j * 2 + 1, 2] += prefactor

        return alphas

    # For an "energies_geometries" mode
    def _get_stencil_coefficients_general(self, deltaZ, deltaR, shift):
        """ Calculates the prefactors of the density terms outlined in the documentation of the implementation, e.g. alpha and beta.

        In general, this collects all terms of the taylor expansion for one particular target and returns their coefficients.
        For energies, the n-th order derivative of the density is divided by :math:`(n+1)!`, while the target density is obtained
        from a regular Taylor expansion, i.e. the density derivative is divided by :math:`n!`. Therefore, a `shift` of 1 returns
        energy coefficients and a `shift` of 0 returns density coefficients.

        Args:
            self:   APDFT instance to obtain the stencil from.
            deltaZ: Array of length N. Target system as described by the change in nuclear charges. [e]
            shift:  Integer. Shift of the factorial term in the energy expansion or density expansion.
        """

        # build alphas and betas
        # alphas: coefficients for EPNs (epn_matrix in predict_all_targets)
        # betas: coefficients for atomic forces which originates from
        #        the derivative of the perturbed density (deriv_rho_force
        #        in predict_all_targets)
        N = len(self._include_atoms)
        nvals = {0: 1, 1: 2 * (N * 2) + 0, 2: (2 * (N * (N - 1))) + (2 * N * N)}
        # Dimension of alphas and beta is
        # (the number of QM calculations, the order of APDFT).
        alphas = np.zeros(
            (sum([nvals[_] for _ in self._orders]), len(self._orders)))
        betas = np.zeros(
            (sum([nvals[_] for _ in self._orders]), len(self._orders), N))

        # Convert unit of a small number for nuclear differentiation
        # from Angstrom to a.u.
        R_delta_ang = self._R_delta * angstrom

        # test input
        if N != len(deltaZ):
            raise ValueError(
                "Mismatch of array lengths: %d dZ values for %d nuclei."
                % (len(deltaZ), N)
            )

        # This function can not specify included atoms.
        # TODO: generalization to specify atoms.
        if N != len(self._nuclear_numbers):
            raise ValueError(
                "Cannot specify atoms in the energies_geometries mode: %d included atoms for %d atoms."
                % (N, len(self._nuclear_numbers))
            )

        # order 0
        # APDFT(0 + 1) = APDFT1
        # For energy, APDFT1 uses the raw density of the reference molecule.
        if 0 in self._orders:
            alphas[0, 0] = 1
        # betas for the force is zero for [0, 0].

        # order 1
        # APDFT(1 + 1) = APDFT2
        # For energy, the 1st-order perturbed density of APDFT1 consider
        # the effects of individual changes of atomic charges and coordinates.
        if 1 in self._orders:
            # self._delta is 0.05, a small fraction for the finite difference
            # with respect to atomic charge changes
            prefactor = 1 / (2 * self._delta) / np.math.factorial(1 + shift)
            # Set the position for the loop for an atomic geometry change
            pos = 0
            # Loop for an atomic charge change
            for siteidx in range(N):
                # For "up" change of the charge,
                # alphas[n, 1] (n = 1, 3, ..., 2N - 1).
                # For "dn" change of the charge,
                # alphas[n, 1] (n = 2, 4, ..., 2N).
                # deltaZ arises from the chain rule of the derivative.
                alphas[1 + siteidx * 2, 1] += prefactor * deltaZ[siteidx]
                alphas[1 + siteidx * 2 + 1, 1] -= prefactor * deltaZ[siteidx]

                # betas for the force are zero for the atomic charge changes.

            # self._delta is 0.005, a small fraction for the finite difference
            # with respect to atomic coordinate changes
            prefactor = 1 / (2 * R_delta_ang) / \
                np.math.factorial(1 + shift)
            # Loop for an atomic geometry change
            for siteidx in range(N):
                # Current implementation only can deal with one Cartesian
                # coordinate change (Z vector here).
                # TODO: generalization to three Cartesian coordinates
                alphas[1 + 2 * N + siteidx * 2, 1] += prefactor * deltaR[siteidx, 2]
                alphas[1 + 2 * N + siteidx * 2 + 1, 1] -= prefactor * deltaR[siteidx, 2]
                betas[1 + 2 * N + siteidx * 2, 1, siteidx] = prefactor
                betas[1 + 2 * N + siteidx * 2 + 1, 1, siteidx] = -prefactor

        # order 2
        # APDFT(2 + 1) = APDFT3
        # For energy, the 2nd-order perturbed density of APDFT1 consider
        # the effects of combinatorial changes of atomic charges.
        if 2 in self._orders:
            # alphas has the following structure:
            #   0: the row reference density
            #   from 1 to 2N: the double changes of charges at one atom
            #                     odd number is for "up".
            #                     even number is for "dn".
            #   from 2N + 1 to the end: the double changes of charges
            #                           at different atoms
            #                           odd number is for "up".
            #                           even number is for "dn".
            # pos is used to specify the position in alphas
            # Position of double changes of nuclear charges is set.
            pos = 1 + 2 * (N * 2) - 2

            # For atomic charge changes
            # Loops for the combination of two atoms
            # (duplication is allowed)
            for siteidx_i in range(N):
                for siteidx_j in range(siteidx_i, N):
                    if siteidx_i != siteidx_j:
                        pos += 2

                    # If there is no charge change in selected atoms of
                    # target and reference molecules, the contribution to
                    # the target energy (or property) becomes zero.
                    # Note that the derivative of the density with respect to
                    # the charge is not zero.
                    if deltaZ[siteidx_j] == 0 or deltaZ[siteidx_i] == 0:
                        continue

                    # If selected atoms with the charge change are different
                    # (It is same as siteidx_j > siteidx_i if all atoms are targeted.)
                    if self._include_atoms[siteidx_j] > self._include_atoms[siteidx_i]:
                        #                 2 * (delta ** 2) = 2 * (0.05 ** 2)
                        #                                  = 2 * 0.025
                        #                                  = 0.005
                        prefactor = (1 / (2 * (self._delta ** 2))) / np.math.factorial(
                            2 + shift
                        )
                        prefactor *= deltaZ[siteidx_i] * deltaZ[siteidx_j]
                        # Following alphas are for the seven terms in the mixed derivatives
                        # with respect to the two different charges
                        alphas[pos, 2] += prefactor
                        alphas[pos + 1, 2] += prefactor
                        # For the raw reference density
                        alphas[0, 2] += 2 * prefactor
                        # For the single change of the atomic charge
                        alphas[1 + siteidx_i * 2, 2] -= prefactor
                        alphas[1 + siteidx_i * 2 + 1, 2] -= prefactor
                        alphas[1 + siteidx_j * 2, 2] -= prefactor
                        alphas[1 + siteidx_j * 2 + 1, 2] -= prefactor

                        # betas for the force are zero for the atomic charge changes.

                    # If selected atoms with the charge change are same
                    # (It is same as siteidx_j == siteidx_i if all atoms are targeted.)
                    if self._include_atoms[siteidx_j] == self._include_atoms[siteidx_i]:
                        # To use same perturbed density with the first-order one, 2h -> h
                        # is used, and therefore in prefactor 1 / ((2 * self._delta) ** 2)
                        # becomes 1 / (self._delta ** 2).
                        prefactor = (1 / (self._delta ** 2)) / np.math.factorial(
                            2 + shift
                        )
                        prefactor *= deltaZ[siteidx_i] * deltaZ[siteidx_j]
                        # For the raw electron density
                        alphas[0, 2] -= 2 * prefactor
                        # For the double changes at the same atom
                        # Note that here siteidx_i == siteidx_j.
                        alphas[1 + siteidx_i * 2, 2] += prefactor
                        alphas[1 + siteidx_j * 2 + 1, 2] += prefactor

                        # betas for the force are zero for the atomic charge changes.

            # For atomic coordinate changes
            # Loops for the combination of two atoms
            # (duplication is allowed)
            # TODO: generalization to three Cartesian coordinates
            for siteidx_i in range(N):
                for siteidx_j in range(siteidx_i, N):
                    if siteidx_i != siteidx_j:
                        pos += 2

                    # Commented out for betas
                    # # If there is no coordinate change in selected atoms of
                    # # target and reference molecules, the contribution to
                    # # the target energy (or property) becomes zero.
                    # # Note that the derivative of the density with respect to
                    # # the coordinate is not zero.
                    # if deltaR[siteidx_j, 2] == 0 or deltaR[siteidx_i, 2] == 0:
                    #     continue

                    # If selected atoms with the coordinate change are different
                    # (It is same as siteidx_j > siteidx_i if all atoms are targeted.)
                    if self._include_atoms[siteidx_j] > self._include_atoms[siteidx_i]:
                        #                 2 * (delta ** 2) = 2 * (0.05 ** 2)
                        #                                  = 2 * 0.025
                        #                                  = 0.005
                        prefactor = (1 / (2 * (R_delta_ang ** 2))) / np.math.factorial(
                            2 + shift
                        )
                        # Set a prefactor for betas.
                        # betas's prefactor 2 is included in prefactor.
                        prefactor_betas = prefactor * deltaR[siteidx_j, 2]
                        prefactor_betas_rev = prefactor * deltaR[siteidx_i, 2]
                        # Set a prefactor for alphas.
                        prefactor *= deltaR[siteidx_i, 2] * deltaR[siteidx_j, 2]

                        # Following alphas are for the seven terms in the mixed derivatives
                        # with respect to the two different coordinates
                        alphas[pos, 2] += prefactor
                        alphas[pos + 1, 2] += prefactor
                        # For the raw reference density
                        alphas[0, 2] += 2 * prefactor
                        # For the single change of the atomic coordinate
                        alphas[1 + 2 * N + siteidx_i * 2, 2] -= prefactor
                        alphas[1 + 2 * N + siteidx_i * 2 + 1, 2] -= prefactor
                        alphas[1 + 2 * N + siteidx_j * 2, 2] -= prefactor
                        alphas[1 + 2 * N + siteidx_j * 2 + 1, 2] -= prefactor

                        # Following betas are for the seven terms in the mixed derivatives
                        # with respect to the two different coordinates
                        betas[pos, 2, siteidx_i] += prefactor_betas
                        betas[pos + 1, 2, siteidx_i] += prefactor_betas
                        # For the raw reference density
                        betas[0, 2, siteidx_i] += 2 * prefactor_betas
                        # For the single change of the atomic coordinate
                        betas[1 + 2 * N + siteidx_i * 2, 2, siteidx_i] -= prefactor_betas
                        betas[1 + 2 * N + siteidx_i * 2 + 1, 2, siteidx_i] -= prefactor_betas
                        betas[1 + 2 * N + siteidx_j * 2, 2, siteidx_i] -= prefactor_betas
                        betas[1 + 2 * N + siteidx_j * 2 + 1, 2, siteidx_i] -= prefactor_betas

                        # Reverse indexes for betas
                        betas[pos, 2, siteidx_j] += prefactor_betas_rev
                        betas[pos + 1, 2, siteidx_j] += prefactor_betas_rev
                        # For the raw reference density
                        betas[0, 2, siteidx_j] += 2 * prefactor_betas_rev
                        # For the single change of the atomic coordinate
                        betas[1 + 2 * N + siteidx_j * 2, 2, siteidx_j] -= prefactor_betas_rev
                        betas[1 + 2 * N + siteidx_j * 2 + 1, 2, siteidx_j] -= prefactor_betas_rev
                        betas[1 + 2 * N + siteidx_i * 2, 2, siteidx_j] -= prefactor_betas_rev
                        betas[1 + 2 * N + siteidx_i * 2 + 1, 2, siteidx_j] -= prefactor_betas_rev

                    # If selected atoms with the coordinate change are same
                    # (It is same as siteidx_j == siteidx_i if all atoms are targeted.)
                    if self._include_atoms[siteidx_j] == self._include_atoms[siteidx_i]:
                        # To use same perturbed density with the first-order one, 2h -> h
                        # is used, and therefore in prefactor 1 / ((2 * self._delta) ** 2)
                        # becomes 1 / (self._delta ** 2).
                        prefactor = (1 / (R_delta_ang ** 2)) / np.math.factorial(
                            2 + shift
                        )
                        # Set a prefactor for betas
                        prefactor_betas = 2 * prefactor * deltaR[siteidx_i, 2]
                        # Set a prefactor for alphas
                        prefactor *= deltaR[siteidx_i, 2] * deltaR[siteidx_j, 2]
                        # For the raw electron density
                        alphas[0, 2] -= 2 * prefactor
                        betas[0, 2, siteidx_i] -= 2 * prefactor_betas
                        # For the double changes at the same atom
                        # Note that here siteidx_i == siteidx_j.
                        alphas[1 + 2 * N + siteidx_i * 2, 2] += prefactor
                        alphas[1 + 2 * N + siteidx_j * 2 + 1, 2] += prefactor
                        betas[1 + 2 * N + siteidx_i * 2, 2, siteidx_i] += prefactor_betas
                        betas[1 + 2 * N + siteidx_j * 2 + 1, 2, siteidx_j] += prefactor_betas

            # For both atomic charge and coordinate changes
            # TODO: generalization to three Cartesian coordinates
            # Loop for the atomic charge change
            for siteidx_i in range(N):
                # Loop for the atomic coordinate change
                for siteidx_j in range(N):
                    pos += 2

                    # Commented out for betas
                    # # If there is no charge or coordinate change in selected atoms
                    # # of target and reference molecules, the contribution to
                    # # the target energy (or property) becomes zero.
                    # # Note that the derivative of the density with respect to
                    # # the coordinate is not zero.
                    # if deltaR[siteidx_j, 2] == 0 or deltaZ[siteidx_i] == 0:
                    #     continue

                    prefactor = (1 / (4 * self._delta * R_delta_ang)
                                 ) / np.math.factorial(2 + shift)
                    # prefactor = (1 / (2 * (self._delta ** 2))) / np.math.factorial(
                    #     2 + shift
                    # )
                    # Set a prefactor for betas
                    # betas's prefactor 2 is included in prefactor.
                    prefactor_betas = 2.0 * prefactor * deltaZ[siteidx_i]
                    # Set a prefactor for alphas
                    # Here 2 comes from the duplication of Z and R.
                    prefactor *= 2 * deltaZ[siteidx_i] * deltaR[siteidx_j, 2]
                    # prefactor *= deltaZ[siteidx_i] * deltaR[siteidx_j, 2]

                    # Following alphas are for the seven terms in the mixed derivatives
                    # with respect to atomic charge and coordinate.
                    alphas[pos, 2] += prefactor
                    alphas[pos + 1, 2] += prefactor
                    # For the raw reference density
                    alphas[0, 2] += 2 * prefactor
                    # For the single change of the atomic charge
                    alphas[1 + siteidx_i * 2, 2] -= prefactor
                    alphas[1 + siteidx_i * 2 + 1, 2] -= prefactor
                    # For the single change of the atomic coordinate
                    alphas[1 + 2 * N + siteidx_j * 2, 2] -= prefactor
                    alphas[1 + 2 * N + siteidx_j * 2 + 1, 2] -= prefactor

                    # Following betas are for the seven terms in the mixed derivatives
                    # with respect to atomic charge and coordinate.
                    betas[pos, 2, siteidx_j] += prefactor_betas
                    betas[pos + 1, 2, siteidx_j] += prefactor_betas
                    # For the raw reference density
                    betas[0, 2, siteidx_j] += 2 * prefactor_betas
                    # For the single change of the atomic charge
                    betas[1 + siteidx_i * 2, 2, siteidx_j] -= prefactor_betas
                    betas[1 + siteidx_i * 2 + 1, 2, siteidx_j] -= prefactor_betas
                    # For the single change of the atomic coordinate
                    betas[1 + 2 * N + siteidx_j * 2, 2, siteidx_j] -= prefactor_betas
                    betas[1 + 2 * N + siteidx_j * 2 + 1, 2, siteidx_j] -= prefactor_betas

        if shift == 1:
            return alphas, betas
        elif shift == 0:
            return alphas

    def get_epn_coefficients(self, deltaZ):
        """ EPN coefficients are the weighting of the electronic EPN from each of the finite difference calculations.

        The weights depend on the change in nuclear charge, i.e. implicitly on reference and target molecule as well as the finite difference stencil employed."""
        return self._get_stencil_coefficients(deltaZ, 1)

    # For an "energies_geometries" mode
    def get_epn_coefficients_general(self, deltaZ, deltaR):
        """ EPN coefficients are the weighting of the electronic EPN from each of the finite difference calculations.

        The weights depend on the change in nuclear charge, i.e. implicitly on reference and target molecule as well as the finite difference stencil employed.
        In the energies_geometries mode, geometries also change."""
        return self._get_stencil_coefficients_general(deltaZ, deltaR, 1)

    def _print_energies(self, targets, energies, comparison_energies):
        for position in range(len(targets)):
            targetname = APDFT._get_target_name(targets[position])
            kwargs = dict()
            for order in self._orders:
                kwargs["order%d" % order] = energies[position][order]
            if comparison_energies is not None:
                kwargs["reference"] = comparison_energies[position]
                kwargs["error"] = energies[position][-1] - kwargs["reference"]

            apdft.log.log(
                "Energy calculated",
                level="RESULT",
                value=energies[position][-1],
                kind="total_energy",
                target=targets[position],
                targetname=targetname,
                **kwargs
            )

    def _print_dipoles(self, targets, dipoles, comparison_dipoles):
        if comparison_dipoles is None:
            for target, dipole in zip(targets, dipoles):
                targetname = APDFT._get_target_name(target)
                kwargs = dict()
                for order in self._orders:
                    kwargs["order%d" % order] = list(dipole[:, order])
                apdft.log.log(
                    "Dipole calculated",
                    level="RESULT",
                    kind="total_dipole",
                    value=list(dipole[:, -1]),
                    target=target,
                    targetname=targetname,
                    **kwargs
                )
        else:
            for target, dipole, comparison in zip(targets, dipoles, comparison_dipoles):
                targetname = APDFT._get_target_name(target)
                apdft.log.log(
                    "Dipole calculated",
                    level="RESULT",
                    kind="total_dipole",
                    reference=list(comparison),
                    value=list(dipole),
                    target=target,
                    targetname=targetname,
                )

    def _print_forces(self, targets, forces, comparison_forces):
        if comparison_forces is None:
            for target, force in zip(targets, forces):
                targetname = APDFT._get_target_name(target)
                kwargs = dict()
                for order in self._orders:
                    for atomidx in range(len(self._nuclear_numbers)):
                        kwargs["order%d-atom%d" % (order, atomidx)] = list(
                            force[:, atomidx, order])
                apdft.log.log(
                    "Force calculated",
                    level="RESULT",
                    kind="total_force",
                    value=list(force[:, :, -1]),
                    target=target,
                    targetname=targetname,
                    **kwargs
                )
        else:
            for target, force, comparison in zip(targets, forces, comparison_forces):
                targetname = APDFT._get_target_name(target)
                apdft.log.log(
                    "Force calculated",
                    level="RESULT",
                    kind="total_force",
                    reference=list(comparison),
                    value=list(force),
                    target=target,
                    targetname=targetname,
                )

    @staticmethod
    def _get_target_name(target):
        return ",".join([apdft.physics.charge_to_label(_) for _ in target])

    def get_folder_order(self):
        """ Returns a static order of calculation folders to build the individual derivative entries.

        To allow for a more efficient evaluation of APDFT, terms are collected and most of the evaluation
        is done with the combined cofficients of those terms. This requires the terms to be handled in a certain
        fixed order that is stable in the various parts of the code. Depending on the selected expansion order, this 
        function builds the list of folders to be included.

        Returns: List of strings, the folder names."""

        folders = []

        # order 0
        folders.append("%s/QM/order-0/site-all-cc/" % self._basepath)

        # order 1
        if 1 in self._orders:
            for site in self._include_atoms:
                folders.append("%s/QM/order-1/site-%d-up/" % (self._basepath, site))
                folders.append("%s/QM/order-1/site-%d-dn/" % (self._basepath, site))

        # order 2
        if 2 in self._orders:
            for site_i in self._include_atoms:
                for site_j in self._include_atoms:
                    if site_j <= site_i:
                        continue

                    folders.append(
                        "%s/QM/order-2/site-%d-%d-up/"
                        % (self._basepath, site_i, site_j)
                    )
                    folders.append(
                        "%s/QM/order-2/site-%d-%d-dn/"
                        % (self._basepath, site_i, site_j)
                    )

        return folders

    # For an "energies_geometries" mode
    def get_folder_order_general(self):
        """ Returns a static order of calculation folders to build the individual derivative entries.

        To allow for a more efficient evaluation of APDFT, terms are collected and most of the evaluation
        is done with the combined cofficients of those terms. This requires the terms to be handled in a certain
        fixed order that is stable in the various parts of the code. Depending on the selected expansion order, this 
        function builds the list of folders to be included.

        Returns: List of strings, the folder names."""

        folders = []

        # order 0
        folders.append("%s/QM/order-0/site-all-cc/" % self._basepath)

        # order 1
        if 1 in self._orders:
            # For the atomic charge change
            for site in self._include_atoms:
                folders.append("%s/QM/order-1/z-site-%d-up/" %
                               (self._basepath, site))
                folders.append("%s/QM/order-1/z-site-%d-dn/" %
                               (self._basepath, site))

            # For the atomic position change
            # For z-Cartesian coordinate changes
            if self._cartesian == "z":
                for site in self._include_atoms:
                    folders.append("%s/QM/order-1/r-site-%d-up/" %
                                (self._basepath, site))
                    folders.append("%s/QM/order-1/r-site-%d-dn/" %
                                (self._basepath, site))
            # For full-Cartesian coordinate changes
            else:
                for site in self._include_atoms:
                    for didx, dim in enumerate("XYZ"):
                        folders.append("%s/QM/order-1/r%s-site-%d-up/" %
                                    (self._basepath, dim, site))
                        folders.append("%s/QM/order-1/r%s-site-%d-dn/" %
                                    (self._basepath, dim, site))

        # order 2
        if 2 in self._orders:
            # For the atomic charge changes
            for site_i in self._include_atoms:
                for site_j in self._include_atoms:
                    if site_j <= site_i:
                        continue

                    folders.append(
                        "%s/QM/order-2/z-site-%d-%d-up/"
                        % (self._basepath, site_i, site_j)
                    )
                    folders.append(
                        "%s/QM/order-2/z-site-%d-%d-dn/"
                        % (self._basepath, site_i, site_j)
                    )

            # For the atomic position changes
            # For z-Cartesian coordinate changes
            if self._cartesian == "z":
                for site_i in self._include_atoms:
                    for site_j in self._include_atoms:
                        if site_j <= site_i:
                            continue

                        folders.append(
                            "%s/QM/order-2/r-site-%d-%d-up/"
                            % (self._basepath, site_i, site_j)
                        )
                        folders.append(
                            "%s/QM/order-2/r-site-%d-%d-dn/"
                            % (self._basepath, site_i, site_j)
                        )
            # For full-Cartesian coordinate changes
            else:
                for site_i in self._include_atoms:
                    for site_j in self._include_atoms:
                        if site_j <= site_i:
                            continue

                        for didx, dim in enumerate("XYZ"):
                            folders.append(
                                "%s/QM/order-2/r%s-site-%d-%d-up/"
                                % (self._basepath, dim, site_i, site_j)
                            )
                            folders.append(
                                "%s/QM/order-2/r%s-site-%d-%d-dn/"
                                % (self._basepath, dim, site_i, site_j)
                            )

            # For both changes of atomic charge and position
            # Loop for the atomic charge change
            # For z-Cartesian coordinate changes
            if self._cartesian == "z":
                for site_i in self._include_atoms:
                    # Loop for the atomic position change
                    for site_j in self._include_atoms:
                        folders.append(
                            "%s/QM/order-2/zr-site-%d-%d-up/"
                            % (self._basepath, site_i, site_j)
                        )
                        folders.append(
                            "%s/QM/order-2/zr-site-%d-%d-dn/"
                            % (self._basepath, site_i, site_j)
                        )
            # For full-Cartesian coordinate changes
            else:
                for site_i in self._include_atoms:
                    # Loop for the atomic position change
                    for site_j in self._include_atoms:
                        for didx, dim in enumerate("XYZ"):
                            folders.append(
                                "%s/QM/order-2/zr%s-site-%d-%d-up/"
                                % (self._basepath, dim, site_i, site_j)
                            )
                            folders.append(
                                "%s/QM/order-2/zr%s-site-%d-%d-dn/"
                                % (self._basepath, dim, site_i, site_j)
                            )

        return folders

    # Used in physics.predict_all_targets(_general) to obtain epn_matrix
    def get_epn_matrix(self):
        """ Collects :math:`\int_Omega rho_i(\mathbf{r}) /|\mathbf{r}-\mathbf{R}_I|`. """
        N = len(self._include_atoms)
        # folders have the dimension of the number of QM calculations
        folders = self.get_folder_order()

        # Dimension is (the number of QM calculations, the number of atoms).
        coeff = np.zeros((len(folders), N))

        # This function is iteratively used in get_epn_matrix.
        # folder: a specific folder of a QM calculation
        # order: the order of APDFT - 1
        # direction: "up" or "down"
        # combination: atoms whose charges change
        def get_epn(folder, order, direction, combination):
            res = 0.0
            charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                len(self._nuclear_numbers), order, combination, direction
            )
            try:
                # For PySCF, self._coordinates and charges are not used.
                res = self._calculator.get_epn(
                    folder, self._coordinates, self._include_atoms, charges
                )
            except ValueError:
                apdft.log.log(
                    "Calculation with incomplete results.",
                    level="error",
                    calulation=folder,
                )
            except FileNotFoundError:
                apdft.log.log(
                    "Calculation is missing a result file.",
                    level="error",
                    calculation=folder,
                )
            return res

        # order 0
        pos = 0

        # order 0
        coeff[pos, :] = get_epn(folders[pos], 0, "up", 0)
        pos += 1

        # order 1
        if 1 in self._orders:
            for site in self._include_atoms:
                coeff[pos, :] = get_epn(folders[pos], 1, "up", [site])
                coeff[pos + 1, :] = get_epn(folders[pos + 1], 1, "dn", [site])
                pos += 2

        # order 2
        if 2 in self._orders:
            for site_i in self._include_atoms:
                for site_j in self._include_atoms:
                    if site_j <= site_i:
                        continue

                    coeff[pos, :] = get_epn(folders[pos], 2, "up", [site_i, site_j])
                    coeff[pos + 1, :] = get_epn(
                        folders[pos + 1], 2, "dn", [site_i, site_j]
                    )
                    pos += 2

        return coeff

    # For an "energies_geometries" mode
    # Used in physics.predict_all_targets(_general) to obtain epn_matrix
    def get_epn_matrix_general(self):
        """ Collects :math:`\int_Omega rho_i(\mathbf{r}) /|\mathbf{r}-\mathbf{R}_I|`. """
        N = len(self._include_atoms)
        # folders have the dimension of the number of the computed densities
        # (QM calculations)
        folders = self.get_folder_order_general()

        # Dimension is (the number of QM calculations, the number of atoms).
        #              (the types of densities)
        # EPNs at the reference geometry
        coeff = np.zeros((len(folders), N))
        # EPNs for the target geometry
        coeff2 = np.zeros((len(folders), N))

        # This function is iteratively used in get_epn_matrix.
        # folder: a specific folder of a QM calculation
        # order: the order of APDFT - 1
        # direction: "up" or "down"
        # combination: atoms whose charges change
        # In PySCF, only folder is used.
        def get_epn(folder, order, direction, combination):
            # For EPNs of the reference
            res = 0.0
            # For EPNs of the target
            res2 = 0.0
            # This is not used in get_epn of PySCF!
            charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                len(self._nuclear_numbers), order, combination, direction
            )
            try:
                # For PySCF, self._coordinates and charges are not used.
                # Therefore, direction and combination are also not used.
                # For EPNs of the reference
                res = self._calculator.get_epn(
                    folder, self._coordinates, self._include_atoms, charges
                )
                # For EPNs of the reference
                res2 = self._calculator.get_epn2(
                    folder, self._coordinates, self._include_atoms, charges
                )
            except ValueError:
                apdft.log.log(
                    "Calculation with incomplete results.",
                    level="error",
                    calulation=folder,
                )
            except FileNotFoundError:
                apdft.log.log(
                    "Calculation is missing a result file.",
                    level="error",
                    calculation=folder,
                )
            return res, res2

        # order 0
        pos = 0

        # order 0
        # "up" is meaningless here.
        coeff[pos, :], coeff2[pos, :] = get_epn(folders[pos], 0, "up", 0)
        # For the next order
        pos += 1

        # order 1
        if 1 in self._orders:
            # For the atomic charge change
            for site in self._include_atoms:
                coeff[pos, :], coeff2[pos, :] = get_epn(folders[pos], 1, "up", [site])
                coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(folders[pos + 1], 1, "dn", [site])
                # For the next site
                pos += 2
            # For the atomic position change
            # TODO: generalization to three Cartesian coordinates
            for site in self._include_atoms:
                coeff[pos, :], coeff2[pos, :] = get_epn(folders[pos], 1, "up", [site])
                coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(folders[pos + 1], 1, "dn", [site])
                # For the next site
                pos += 2

        # order 2
        if 2 in self._orders:
            # For the atomic charge changes
            for site_i in self._include_atoms:
                for site_j in self._include_atoms:
                    if site_j <= site_i:
                        continue

                    coeff[pos, :], coeff2[pos, :] = get_epn(
                        folders[pos], 2, "up", [site_i, site_j])
                    coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(
                        folders[pos + 1], 2, "dn", [site_i, site_j]
                    )
                    # For the next site
                    pos += 2

            # For the atomic position changes
            for site_i in self._include_atoms:
                for site_j in self._include_atoms:
                    if site_j <= site_i:
                        continue

                    coeff[pos, :], coeff2[pos, :] = get_epn(
                        folders[pos], 2, "up", [site_i, site_j])
                    coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(
                        folders[pos + 1], 2, "dn", [site_i, site_j]
                    )
                    # For the next site
                    pos += 2

            # For both changes of atomic charge and position
            # Loop for the atomic charge change
            for site_i in self._include_atoms:
                # Loop for the atomic position change
                for site_j in self._include_atoms:

                    coeff[pos, :], coeff2[pos, :] = get_epn(
                        folders[pos], 2, "up", [site_i, site_j])
                    coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(
                        folders[pos + 1], 2, "dn", [site_i, site_j]
                    )
                    # For the next site
                    pos += 2

        # # For check
        # print("epn")
        # print(coeff)
        # print("folders")
        # [print(i) for i in folders]
        # print('')

        # For check
        # print("epn2")
        # print(coeff2)
        # print("folders")
        # [print(i) for i in folders]
        # print('')

        return coeff, coeff2

    # For an "energies_geometries" mode
    # Used in physics.predict_all_targets(_general) to obtain epn_matrix
    # and hf_ionic_force_matrix
    def get_property_matrix_general(self):
        """ Collects :math:`\int_Omega rho_i(\mathbf{r}) /|\mathbf{r}-\mathbf{R}_I|`. """
        N = len(self._include_atoms)
        # folders have the dimension of the number of the computed densities
        # (QM calculations)
        folders = self.get_folder_order_general()

        # Dimension is (the number of QM calculations, the number of atoms).
        #              (the types of densities)
        # EPNs at the reference geometry
        coeff = np.zeros((len(folders), N))
        # EPNs for the target geometry
        coeff2 = np.zeros((len(folders), N))

        # Hellmann-Feynman atomic forces
        # The dimension is (the types of densities, the number of atoms,
        #                   three Cartesian coordinates)
        force_coeff = np.zeros((len(folders), N, 3))

        # This function is iteratively used in get_epn_matrix.
        # folder: a specific folder of a QM calculation
        # order: the order of APDFT - 1
        # direction: "up" or "down"
        # combination: atoms whose charges change
        # In PySCF, only folder is used.
        def get_epn(folder, order, direction, combination):
            # For EPNs of the reference
            res = 0.0
            # For EPNs of the target
            res2 = 0.0
            # This is not used in get_epn of PySCF!
            charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                len(self._nuclear_numbers), order, combination, direction
            )

            # If this is a calculation of vertical energy derivatives
            # TODO: generalization to three Cartesian coordinates
            if self._calc_der and "/r-site" in folder or "/zr-site" in folder:
                res = 0.0
                res2 = 0.0
            else:
                try:
                    # For PySCF, self._coordinates and charges are not used.
                    # Therefore, direction and combination are also not used.
                    # For EPNs of the reference
                    res = self._calculator.get_epn(
                        folder, self._coordinates, self._include_atoms, charges
                    )
                    # For EPNs for the target
                    # Here self._coordinates is not used
                    res2 = self._calculator.get_epn2(
                        folder, self._coordinates, self._include_atoms, charges
                    )
                except ValueError:
                    apdft.log.log(
                        "Calculation with incomplete results.",
                        level="error",
                        calulation=folder,
                    )
                except FileNotFoundError:
                    apdft.log.log(
                        "Calculation is missing a result file.",
                        level="error",
                        calculation=folder,
                    )
            return res, res2

        # This function is iteratively used in get_hf_ionic_force_matrix.
        # folder: a specific folder of a QM calculation
        # order: the order of APDFT - 1
        # direction: "up" or "down"
        # combination: atoms whose charges change
        # In PySCF, only folder is used.
        def get_ionic_force(folder, order, direction, combination):
            res = 0.0
            # This is not used in get_epn of PySCF!
            charges = self._nuclear_numbers + self._calculate_delta_Z_vector(
                len(self._nuclear_numbers), order, combination, direction
            )

            # If this is a calculation of vertical energy derivatives and
            # TODO: generalization to three Cartesian coordinates
            if self._calc_der and "/r-site" in folder or "/zr-site" in folder:
                res = 0.0
            else:
                try:
                    res = self._calculator.get_target_ionic_force(
                        folder, self._coordinates, self._include_atoms, charges
                    )
                except ValueError:
                    apdft.log.log(
                        "Calculation with incomplete results.",
                        level="error",
                        calulation=folder,
                    )
                except FileNotFoundError:
                    apdft.log.log(
                        "Calculation is missing a result file.",
                        level="error",
                        calculation=folder,
                    )
            return res

        # order 0
        pos = 0

        # order 0
        # "up" is meaningless here.
        # Read EPN
        coeff[pos, :], coeff2[pos, :] = get_epn(folders[pos], 0, "up", 0)
        # Read ionic force
        force_coeff[pos, :, :] = get_ionic_force(folders[pos], 0, "up", 0)

        # For the next order
        pos += 1

        # order 1
        if 1 in self._orders:
            # For the atomic charge change
            for site in self._include_atoms:
                # Read EPNs
                coeff[pos, :], coeff2[pos, :] = get_epn(
                    folders[pos], 1, "up", [site])
                coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(
                    folders[pos + 1], 1, "dn", [site])

                # Read ionic forces
                force_coeff[pos, :, :] = get_ionic_force(
                    folders[pos], 1, "up", [site])
                force_coeff[pos + 1, :, :] = get_ionic_force(
                    folders[pos + 1], 1, "dn", [site])

                # For the next site
                pos += 2

            # For the atomic position change
            # TODO: generalization to three Cartesian coordinates
            for site in self._include_atoms:
                # Read EPNs
                coeff[pos, :], coeff2[pos, :] = get_epn(
                    folders[pos], 1, "up", [site])
                coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(
                    folders[pos + 1], 1, "dn", [site])

                # Read ionic forces
                force_coeff[pos, :, :] = get_ionic_force(
                    folders[pos], 1, "up", [site])
                force_coeff[pos + 1, :, :] = get_ionic_force(
                    folders[pos + 1], 1, "dn", [site])

                # For the next site
                pos += 2

        # order 2
        if 2 in self._orders:
            # For the atomic charge changes
            for site_i in self._include_atoms:
                for site_j in self._include_atoms:
                    if site_j <= site_i:
                        continue

                    # Read EPNs
                    coeff[pos, :], coeff2[pos, :] = get_epn(
                        folders[pos], 2, "up", [site_i, site_j])
                    coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(
                        folders[pos + 1], 2, "dn", [site_i, site_j])

                    # Read ionic forces
                    force_coeff[pos, :, :] = get_ionic_force(
                        folders[pos], 2, "up", [site_i, site_j])
                    force_coeff[pos + 1, :, :] = get_ionic_force(
                        folders[pos + 1], 2, "dn", [site_i, site_j])

                    # For the next site
                    pos += 2

            # For the atomic position changes
            for site_i in self._include_atoms:
                for site_j in self._include_atoms:
                    if site_j <= site_i:
                        continue

                    # Read EPNs
                    coeff[pos, :], coeff2[pos, :] = get_epn(
                        folders[pos], 2, "up", [site_i, site_j])
                    coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(
                        folders[pos + 1], 2, "dn", [site_i, site_j])

                    # Read ionic forces
                    force_coeff[pos, :, :] = get_ionic_force(
                        folders[pos], 2, "up", [site_i, site_j])
                    force_coeff[pos + 1, :, :] = get_ionic_force(
                        folders[pos + 1], 2, "dn", [site_i, site_j])

                    # For the next site
                    pos += 2

            # For both changes of atomic charge and position
            # Loop for the atomic charge change
            for site_i in self._include_atoms:
                # Loop for the atomic position change
                for site_j in self._include_atoms:

                    # Read EPNs
                    coeff[pos, :], coeff2[pos, :] = get_epn(
                        folders[pos], 2, "up", [site_i, site_j])
                    coeff[pos + 1, :], coeff2[pos + 1, :] = get_epn(
                        folders[pos + 1], 2, "dn", [site_i, site_j])

                    # Read ionic forces
                    force_coeff[pos, :, :] = get_ionic_force(
                        folders[pos], 2, "up", [site_i, site_j])
                    force_coeff[pos + 1, :, :] = get_ionic_force(
                        folders[pos + 1], 2, "dn", [site_i, site_j])

                    # For the next site
                    pos += 2

        # # For check
        # print("epn")
        # print(coeff)
        # print("folders")
        # [print(i) for i in folders]
        # print('')

        # For check
        # print("epn2")
        # print(coeff2)
        # print("folders")
        # [print(i) for i in folders]
        # print('')

        return coeff, coeff2, force_coeff

    def get_linear_density_coefficients(self, deltaZ):
        """ Obtains the finite difference coefficients for a property linear in the density. 
        
        Args:
            deltaZ:     Array of integers of length N. Target system expressed in the change in nuclear charges from the reference system. [e]
        Returns:
            Vector of coefficients."""
        return self._get_stencil_coefficients(deltaZ, 0)

    # For an "energies_geometries" mode
    def get_linear_density_coefficients_general(self, deltaZ, deltaR):
        """ Obtains the finite difference coefficients for a property linear in the density.

        Args:
            deltaZ:     Array of integers of length N. Target system expressed in the change in nuclear and coordinate charges from the reference system. [e]
        Returns:
            Vector of coefficients."""
        return self._get_stencil_coefficients_general(deltaZ, deltaR, 0)

    def enumerate_all_targets(self):
        """ Builds a list of all possible targets.

		Note that the order is not guaranteed to be stable.

		Args:
			self:		Class instance from which the total charge and number of sites is determined.
		Returns:
			A list of lists with the integer nuclear charges."""
        # there might be a user-specified explicit list
        if self._targetlist is not None:
            return self._targetlist

        # Generate targets
        if self._max_deltaz is None:
            around = None
            limit = None
        else:
            around = np.array(self._nuclear_numbers)
            limit = self._max_deltaz

        res = []
        nsites = len(self._nuclear_numbers)
        nprotons = sum(self._nuclear_numbers)
        for shift in range(-self._max_charge, self._max_charge + 1):
            if nprotons + shift < 1:
                continue
            # If the number of protons of the system is lower than 0
            res += apdft.math.IntegerPartitions.partition(
                nprotons + shift, nsites, around, limit
            )

        # filter for included atoms
        ignore_atoms = list(
            set(range(len(self._nuclear_numbers))) - set(self._include_atoms)
        )
        if len(self._include_atoms) != len(self._nuclear_numbers):
            res = [
                _
                for _ in res
                if [_[idx] for idx in ignore_atoms]
                == [self._nuclear_numbers[idx] for idx in ignore_atoms]
            ]
        return res

    def estimate_cost_and_coverage(self):
        """ Estimates number of single points (cost) and number of targets (coverage).

		Args:
			self:		Class instance from which the total charge and number of sites is determined.
		Returns:
			Tuple of ints: number of single points, number of targets."""

        N = len(self._include_atoms)
        cost = sum({0: 1, 1: 2 * N, 2: N * (N - 1)}[_] for _ in self._orders)

        coverage = len(self.enumerate_all_targets())
        return cost, coverage

    def estimate_cost_and_coverage_general(self):
        """ Estimates number of single points (cost) and number of targets (coverage).
            This is a modified estimate_cost_and_coverage and can
            treat molecular geometry changes.

		Args:
			self:		Class instance from which the total charge and number of sites is determined.
		Returns:
			Tuple of ints: number of single points, number of targets."""

        N = len(self._include_atoms)
        # Required order of APDFTn is up to n - 1.
        # (In the current implementation, the maximum order of perturbed
        # electron densities calculated by the central finite difference
        # is 2 (APDFT3).)
        # Here, self._orders is 0, 1, ..., n - 1.
        cost = sum({0: 1, 1: 2 * N, 2: N * (N - 1)}[_] for _ in self._orders)

        # If this is not a vertical energy derivative calculation,
        # QM calculations for atomic coordinate changes are required.
        if not self._calc_der:
            # Add a cost with respect to molecular geometry changes.
            # For z-Cartesian coordinate changes
            if self._cartesian == "z":
                cost += sum({0: 0, 1: 2 * N, 2: N * (N - 1)}
                            [_] for _ in self._orders)
            # For full-Cartesian coordinate changes
            else:
                cost += sum({0: 0, 1: 3 * (2 * N), 2: 3 * (N * (N - 1)
                                                           ) + 3 * (2 * N * N)}[_] for _ in self._orders)

            # Add a cost with respect to mixed changes for atomic charge
            # and geometry.
            # In the order 2, the prefactor 2 is for "up" and "dn".
            # For z-Cartesian coordinate changes
            if self._cartesian == "z":
                cost += sum({0: 0, 1: 0, 2: 2 * N * N}[_] for _ in self._orders)
            # For full-Cartesian coordinate changes
            else:
                cost += sum({0: 0, 1: 0, 2: 3 * (2 * N * N)}[_] for _ in self._orders)

        # If this is a vertical energy derivative calculation,
        # QM calculations for atomic coordinate changes are required.
        else:
            # TODO: Consider the cost for APDFT3
            #       0 is added for now.
            # For z-Cartesian coordinate changes
            if self._cartesian == "z":
                cost += sum({0: 2 * N , 1: 2 * N * N, 2: 0}[_] for _ in self._orders)
            # For full-Cartesian coordinate changes
            else:
                cost += sum({0: 3 * (2 * N), 1: 3 * (2 * N * N), 2: 0}[_] for _ in self._orders)


        # The number of candidates does not change with nuclear charge transformations
        # because it is assumed that the molecular geometry is determined by the nuclear
        # charges of atoms.
        # self.enumerate_all_targets() is the list of all the target systems.
        coverage = len(self.enumerate_all_targets())
        return cost, coverage

    def get_energy_from_reference(self, nuclear_charges, is_reference_molecule=False):
        """ Retreives the total energy from a QM reference. 

		Args:
			nuclear_charges: 	Integer list of nuclear charges. [e]
		Returns:
			The total energy. [Hartree]"""
        if is_reference_molecule:
            return self._calculator.get_total_energy(
                "%s/QM/order-0/site-all-cc" % self._basepath
            )
        else:
            return self._calculator.get_total_energy(
                "%s/QM/comparison-%s"
                % (self._basepath, "-".join(map(str, nuclear_charges)))
            )

    def get_reference_energy_derivatives(self):
        """ Retreives analytical energy derivatives of a reference molecule from a QM reference. 

		Args:
			nuclear_charges: 	Integer list of nuclear charges. [e]
		Returns:
			analytical energy derivatives of a reference molecule. [a.u.]"""
        return self._calculator.get_reference_anal_energy_derivatives(
            "%s/QM/order-0/site-all-cc" % self._basepath, self._include_atoms
        )

    def get_linear_density_matrix(self, propertyname):
        """ Retrieves the value matrix for properties linear in density.

        Valid properties are: ELECTRONIC_DIPOLE, IONIC_FORCE, ELECTRONIC_QUADRUPOLE.
        Args:
            self:           APDFT instance.
            propertyname:   String. One of the choices above.
        Returns: 
            (N, m) array for an m-dimensional property over N QM calculations or None if the property is not implemented with this QM code."""

        functionname = "get_%s" % propertyname.lower()
        try:
            function = getattr(self._calculator, functionname)
        except AttributeError:
            return None

        folders = self.get_folder_order()
        results = []
        for folder in folders:
            try:
                if functionname == "get_ionic_force":
                    results.append(function(folder, self._include_atoms))
                else:
                    results.append(function(folder))
            except ValueError:
                apdft.log.log(
                    "Calculation with incomplete results.",
                    level="error",
                    calulation=folder,
                )

        # Only meaningful if all calculations are present.
        if len(results) == len(folders):
            return np.array(results)

    # For an "energies_geometries" mode
    def get_linear_density_matrix_general(self, propertyname):
        """ Retrieves the value matrix for properties linear in density.

        Valid properties are: ELECTRONIC_DIPOLE, IONIC_FORCE, ELECTRONIC_QUADRUPOLE.
        Args:
            self:           APDFT instance.
            propertyname:   String. One of the choices above.
        Returns:
            (N, m) array for an m-dimensional property over N QM calculations or None if the property is not implemented with this QM code."""

        # if the target property is not calculated, raise error.
        if not propertyname in [
            'TARGET_ELECTRONIC_DIPOLE', 'TARGET_HF_IONIC_FORCE', 'ELECTRONIC_QUADRUPOLE']:
            raise ValueError(
                "Calculation routine of the target property %s of molecules has not been implemented yet."
                % propertyname
                )

        if self._calc_der and propertyname is 'ELECTRONIC_QUADRUPOLE':
            raise NotImplemented(
                "Electronic quadrupole is not implemented yet in a calculation of vertical energy derivatives."
            )

        functionname = "get_%s" % propertyname.lower()
        try:
            # For functionname, "ELECTRONIC_DIPOLE" can be used.
            # If self._calculator is pyscf,
            # pyscf.get_target_electronic_dipole is function.
            function = getattr(self._calculator, functionname)
        except AttributeError:
            return None

        # Obtain 0 values corresponding to properties
        def get_empty_value(label):
            res = []
            if label == 'TARGET_ELECTRONIC_DIPOLE':
                # For xyz components
                res.append([0.0, 0.0, 0.0])
                return res[0]
            elif label == 'TARGET_HF_IONIC_FORCE':
                for i in range(len(self._nuclear_numbers)):
                    res.append([0.0, 0.0, 0.0])

            return res

        # Obtain folders corresponding to "energies_geometries"
        folders = self.get_folder_order_general()
        results = []
        for folder in folders:
            # If this is a calculation of vertical energy derivatives
            # TODO: generalization to three Cartesian coordinates
            if self._calc_der and "/r-site" in folder or "/zr-site" in folder:
                results.append(get_empty_value(propertyname))
            else:
                try:
                    # Properties are obtained.
                    if functionname == "get_target_hf_ionic_force":
                        # For "TARGET_IONIC_FORCE"
                        results.append(function(folder, self._include_atoms))
                    else:
                        results.append(function(folder))
                except ValueError:
                    apdft.log.log(
                        "Calculation with incomplete results.",
                        level="error",
                        calulation=folder,
                    )

        # Only meaningful if all calculations are present.
        if len(results) == len(folders):
            return np.array(results)

    def predict_all_targets(self):
        # assert one order of targets
        targets = self.enumerate_all_targets()
        own_nuc_nuc = Coulomb.nuclei_nuclei(self._coordinates, self._nuclear_numbers)

        energies = np.zeros((len(targets), len(self._orders)))
        ele_energies = np.zeros((len(targets), len(self._orders)))
        nuc_energies = np.zeros((len(targets), len(self._orders)))
        dipoles = np.zeros((len(targets), 3, len(self._orders)))
        nuc_dipoles = np.zeros((len(targets), 3, len(self._orders)))
        ele_dipoles = np.zeros((len(targets), 3, len(self._orders)))
        forces = np.zeros((len(targets), len(self._nuclear_numbers), 3, len(self._orders)))
        ele_forces = np.zeros((len(targets), len(self._nuclear_numbers), 3, len(self._orders)))
        nuc_forces = np.zeros((len(targets), len(self._nuclear_numbers), 3, len(self._orders)))

        # get base information
        refenergy = self.get_energy_from_reference(
            self._nuclear_numbers, is_reference_molecule=True
        )
        epn_matrix = self.get_epn_matrix()
        dipole_matrix = self.get_linear_density_matrix("ELECTRONIC_DIPOLE")
        force_matrix = self.get_linear_density_matrix("IONIC_FORCE")

        alchemical_target = np.zeros(len(self._nuclear_numbers))

        # get target predictions
        for targetidx, target in enumerate(targets):
            deltaZ = target - self._nuclear_numbers

            deltaZ_included = deltaZ[self._include_atoms]
            alphas = self.get_epn_coefficients(deltaZ_included)

            # Mixing reference and target molecules by using non-integer lambda
            # If target molecule is not targeted
            if self._mix_lambda != 1.0:
                alchemical_target = self._mix_lambda * np.array(target) + \
                    (1.0 - self._mix_lambda) * self._nuclear_numbers

            # energies
            # Mixing reference and target molecules by using non-integer lambda
            # If target molecule is not targeted
            if self._mix_lambda == 1.0:
                deltaEnn = Coulomb.nuclei_nuclei(self._coordinates, target) - own_nuc_nuc
                Enn = Coulomb.nuclei_nuclei(self._coordinates, target)
            else:
                alchemical_target = self._mix_lambda * np.array(target) + \
                    (1.0 - self._mix_lambda) * self._nuclear_numbers
                deltaEnn = Coulomb.nuclei_nuclei(
                    self._coordinates, alchemical_target) - own_nuc_nuc
                Enn = Coulomb.nuclei_nuclei(
                    self._coordinates, alchemical_target)

            for order in sorted(self._orders):
                contributions = -np.multiply(
                    np.outer(alphas[:, order], deltaZ_included), epn_matrix
                ).sum()
                ele_energies[targetidx, order] = contributions
                if order > 0:
                    ele_energies[targetidx, order] += ele_energies[targetidx, order - 1]
            ele_energies[targetidx, :] += refenergy - own_nuc_nuc
            energies[targetidx, :] += ele_energies[targetidx, :] + Enn
            nuc_energies[targetidx, :] = Enn

            # dipoles
            if dipole_matrix is not None:
                betas = self.get_linear_density_coefficients(deltaZ_included)
                # Compute nuclear dipole moment centered at the geometrical center
                nuc_dipole = Dipoles.point_charges_au(
                    self._coordinates.mean(axis=0), self._coordinates, target
                )
                for order in sorted(self._orders):
                    ed = np.multiply(dipole_matrix, betas[:, order, np.newaxis]).sum(
                        axis=0
                    )
                    dipoles[targetidx, :, order] = ed
                    if order > 0:
                        dipoles[targetidx, :, order] += dipoles[targetidx, :, order - 1]
                    ele_dipoles[targetidx, :, order] = dipoles[targetidx, :, order]
                    nuc_dipoles[targetidx, :, order] = nuc_dipole
                dipoles[targetidx] += nuc_dipole[:, np.newaxis]

            # forces
            # Atomic force which originates from the nuclei repulsion term
            # nuc_Fnn(len(self._nuclear_numbers), 3)
            nuc_Fnn = Coulomb.nuclei_atom_force(
                self._coordinates, target
            )
            if force_matrix is not None:
                for order in sorted(self._orders):
                    for atomidx in range(len(self._include_atoms)):
                        # Electronic part of the atomic force
                        electron_force = np.multiply(
                            target[atomidx] * force_matrix[:, atomidx, :], betas[:, order, np.newaxis]).sum(
                            axis=0
                        )
                        forces[targetidx, atomidx, :, order] = electron_force
                        if order > 0:
                            forces[targetidx, atomidx, :, order] += forces[
                                targetidx, atomidx, :, order - 1
                                ]
                        ele_forces[targetidx, atomidx, :, order] = forces[targetidx, atomidx, :, order]
                        nuc_forces[targetidx, atomidx, :, order] = nuc_Fnn[atomidx, :]
                for order in sorted(self._orders):
                    for atomidx in range(len(self._include_atoms)):
                        forces[targetidx, atomidx, :, order] += nuc_forces[targetidx, atomidx, :, order]

        # return results
        return targets, energies, ele_energies, nuc_energies, dipoles, ele_dipoles, nuc_dipoles, forces, ele_forces, nuc_forces

    # For an "energies_geometries" mode
    # target_coordinate is in angstrom.
    def predict_all_targets_general(self, target_coordinate):
        # assert one order of targets
        targets = self.enumerate_all_targets()
        # calculate the nuclear-nuclear repulsion energy of
        # the reference molecule
        own_nuc_nuc = Coulomb.nuclei_nuclei(
            self._coordinates, self._nuclear_numbers)

        # Energy
        energies = np.zeros((len(targets), len(self._orders)))
        ele_energies = np.zeros((len(targets), len(self._orders)))
        nuc_energies = np.zeros((len(targets), len(self._orders)))
        reference_energy_contributions = np.zeros((len(targets), len(self._orders)))
        target_energy_contributions = np.zeros((len(targets), len(self._orders)))
        total_energy_contributions = np.zeros((len(targets), len(self._orders)))

        # Atomic force
        atomic_forces = np.zeros((
            len(targets), len(self._orders), len(self._nuclear_numbers), 3))
        # ele_atomic_forces is a electronic term of atomic forces and
        # is a sum of hf_ionic_force_contributions and deriv_rho_contributions
        ele_atomic_forces = np.zeros((
            len(targets), len(self._orders), len(self._nuclear_numbers), 3))
        nuc_atomic_forces = np.zeros((
            len(targets), len(self._orders), len(self._nuclear_numbers), 3))
        hf_ionic_force_contributions = np.zeros((
            len(targets), len(self._orders), len(self._nuclear_numbers), 3))
        deriv_rho_contributions = np.zeros((
            len(targets), len(self._orders), len(self._nuclear_numbers), 3))

        # Hellmann-Feynman ionic force
        hf_ionic_forces = np.zeros((len(targets), len(self._nuclear_numbers),
                                    3, len(self._orders)))
        ele_hf_ionic_forces = np.zeros((len(targets), len(self._nuclear_numbers),
                                        3, len(self._orders)))
        nuc_hf_ionic_forces = np.zeros((len(targets), len(self._nuclear_numbers),
                                        3, len(self._orders)))

        # Electric dipole moment
        dipoles = np.zeros((len(targets), 3, len(self._orders)))
        nuc_dipoles = np.zeros((len(targets), 3, len(self._orders)))
        ele_dipoles = np.zeros((len(targets), 3, len(self._orders)))

        # get base information
        # refenergy is the total energy
        refenergy = self.get_energy_from_reference(
            self._nuclear_numbers, is_reference_molecule=True
        )
        # If this is a calculation of vertical energy derivatives,
        # analytical energy derivatives of a reference molecule are extracted.
        if self._calc_der:
            atomic_forces_reference = -self.get_reference_energy_derivatives()
        # Dimension of epn_matrix is
        # (the number of QM calculations, the number of atoms).
        # TODO: need to be generalized to three Cartesian coordinates
        # epn_matrix, epn_matrix_target = self.get_epn_matrix_general()
        epn_matrix, epn_matrix_target, ionic_force_matrix = self.get_property_matrix_general()
        # Dipole matrix
        # TODO: need to be generalized to three Cartesian coordinates
        dipole_matrix = self.get_linear_density_matrix_general("TARGET_ELECTRONIC_DIPOLE")
        # Hellmann-Feynmann force matrix
        hf_ionic_force_matrix = self.get_linear_density_matrix_general(
            "TARGET_HF_IONIC_FORCE")

        # hf_ionic_force_matrix and ionic_force_matrix are identical
        # print("hf_ionic_force_matrix", np.shape(hf_ionic_force_matrix))
        # print("ionic_force_matrix", np.shape(ionic_force_matrix))
        # for i in range(len(self._nuclear_numbers)):
        #     for j in range(3):
        #         print("")
        #         print(ionic_force_matrix[:, i, j] - hf_ionic_force_matrix[:, i, j])

        # get difference between reference and target geometries
        deltaR = target_coordinate - self._coordinates
        # Convert angstrom to Bohr (a.u.)
        deltaR *= angstrom

        # get target predictions
        # target is target nuclear charges of atoms
        for targetidx, target in enumerate(targets):
            deltaZ = target - self._nuclear_numbers

            deltaZ_included = deltaZ[self._include_atoms]
            alphas, force_alphas = self.get_epn_coefficients_general(deltaZ_included, deltaR)

            # energies
            # Calculate nuclear-nuclear repulsion energy of the target
            Enn = Coulomb.nuclei_nuclei(
                target_coordinate, target)

            # Atomic force which originates from the nuclei repulsion term
            # targetFnn = np.zeros((len(self._nuclear_numbers), 3))
            targetFnn = Coulomb.nuclei_atom_force(
                target_coordinate, target
            )

            for order in sorted(self._orders):
                # Energy contributions from the target
                contributions_target = -np.multiply(
                    np.outer(alphas[:, order], target), epn_matrix_target
                ).sum()

                # Energy contributions from the reference
                contributions_reference = np.multiply(
                    np.outer(alphas[:, order], self._nuclear_numbers), epn_matrix
                ).sum()

                # Force contributions from the Hellmann-Feynman ionic force
                contributions_hf_ionic_force = np.zeros((len(self._nuclear_numbers), 3))
                # Calculation of the forces on each atom
                for i in range(len(self._nuclear_numbers)):
                    contributions_hf_ionic_force[i, :] = np.multiply(
                        target[i] * hf_ionic_force_matrix[:,
                                                          i, :], alphas[:, order, np.newaxis]
                    ).sum(axis=0)

                # Force contributions from the force of derivatives of
                # the perturbed density
                contributions_target_deriv_rho = np.zeros(
                    (len(self._nuclear_numbers), 3))
                contributions_reference_deriv_rho = np.zeros(
                    (len(self._nuclear_numbers), 3))

                # Force contributions from the target
                for i in range(len(self._nuclear_numbers)):
                    for j in range(3):
                        # Z axis
                        if j == 2:
                            contributions_target_deriv_rho[i, j] = np.multiply(
                                np.outer(force_alphas[:, order, i], target),
                                         epn_matrix_target
                                ).sum()

                # Force contributions from the reference
                for i in range(len(self._nuclear_numbers)):
                    for j in range(3):
                        # Z axis
                        if j == 2:
                            contributions_reference_deriv_rho[i, j] = -np.multiply(
                                np.outer(force_alphas[:, order, i],
                                         self._nuclear_numbers),
                                         epn_matrix
                                ).sum()

                # # For check
                # for i in range(len(self._nuclear_numbers)):
                #     print("order", order)
                #     print("target", target)
                #     print("atom", i)
                #     print(contributions_target_deriv_rho[i, 2])
                #     print(contributions_reference_deriv_rho[i, 2])
                #     print('')

                # Energy
                ele_energies[targetidx, order] = contributions_target + contributions_reference

                # Atomic forces
                # Sum of the target and reference contributions to
                # the Hellmann-Feynman term of the atomic force
                hf_ionic_force_contributions[targetidx, order, :, :] = \
                    contributions_hf_ionic_force[:, :]

                # Sum of the target and reference contributions to
                # the atromic force originates from  derivatives of
                # the perturbed density
                deriv_rho_contributions[targetidx, order, :, :] = \
                    contributions_target_deriv_rho[:, :] + \
                    contributions_reference_deriv_rho[:, :]

                # Sum of the electronic contributions to atomic forces
                ele_atomic_forces[targetidx, order, :, :] = \
                    hf_ionic_force_contributions[targetidx, order, :, :] + \
                    deriv_rho_contributions[targetidx, order, :, :]
                atomic_forces[targetidx, order, :, :] = ele_atomic_forces[targetidx, order, :, :]

                # Save energy contributions
                reference_energy_contributions[targetidx, order] = contributions_reference
                target_energy_contributions[targetidx, order] = contributions_target
                total_energy_contributions[targetidx, order] = contributions_target + \
                    contributions_reference

                # For check by vertical charge changes
                # ele_energies[targetidx, order] = contributions
                if order > 0:
                    ele_energies[targetidx, order] += ele_energies[targetidx, order - 1]

                if order > 0:
                    atomic_forces[targetidx, order] += atomic_forces[targetidx, order - 1]
                    ele_atomic_forces[targetidx, order] += atomic_forces[targetidx, order - 1]

            ele_energies[targetidx, :] += refenergy - own_nuc_nuc
            energies[targetidx, :] += ele_energies[targetidx, :] + Enn
            nuc_energies[targetidx, :] += Enn

            atomic_forces[targetidx, :] += targetFnn

            # Save the nuclear term of atomic forces
            nuc_atomic_forces[targetidx, :] = targetFnn

            # dipoles
            if dipole_matrix is not None:
                betas = self.get_linear_density_coefficients_general(
                    deltaZ_included, deltaR
                )
                # Compute nuclear dipole moment centered at the geometrical center
                # for the target coordinated
                nuc_dipole = Dipoles.point_charges_au(
                    target_coordinate.mean(axis=0), target_coordinate, target
                )
                for order in sorted(self._orders):
                    ed = np.multiply(dipole_matrix, betas[:, order, np.newaxis]).sum(
                        axis=0
                    )
                    dipoles[targetidx, :, order] = ed
                    if order > 0:
                        dipoles[targetidx, :, order] += dipoles[targetidx, :, order - 1]
                    ele_dipoles[targetidx, :, order] = dipoles[targetidx, :, order]
                    nuc_dipoles[targetidx, :, order] = nuc_dipole
                dipoles[targetidx] += nuc_dipole[:, np.newaxis]

            # Hellmann-Feynmann forces
            if hf_ionic_force_matrix is not None:
                for order in sorted(self._orders):
                    for atomidx in range(len(self._include_atoms)):
                        # Electronic part of the atomic force
                        electron_force = np.multiply(
                            target[atomidx] * hf_ionic_force_matrix[:, atomidx, :], betas[:, order, np.newaxis]).sum(
                            axis=0
                        )
                        hf_ionic_forces[targetidx, atomidx,
                                        :, order] = electron_force
                        if order > 0:
                            hf_ionic_forces[targetidx, atomidx, :, order] += hf_ionic_forces[
                                targetidx, atomidx, :, order - 1
                            ]
                        ele_hf_ionic_forces[targetidx, atomidx, :,
                                            order] = hf_ionic_forces[targetidx, atomidx, :, order]
                        nuc_hf_ionic_forces[targetidx, atomidx,
                                            :, order] = targetFnn[atomidx, :]
                hf_ionic_forces[targetidx, :, :,
                                :] += nuc_hf_ionic_forces[targetidx, :, :, :]

        # return results
        return targets, energies, ele_energies, nuc_energies, dipoles, ele_dipoles, nuc_dipoles, \
               reference_energy_contributions, target_energy_contributions, total_energy_contributions, \
               atomic_forces, ele_atomic_forces, nuc_atomic_forces, hf_ionic_force_contributions, \
               deriv_rho_contributions, hf_ionic_forces, ele_hf_ionic_forces, nuc_hf_ionic_forces

    def analyse(self, explicit_reference=False):
        """ Performs actual analysis and integration. Prints results"""
        try:
            targets, energies, ele_energies, nuc_energies, dipoles, ele_dipoles, nuc_dipoles, \
                atomic_forces, ele_atomic_forces, nuc_atomic_forces = self.predict_all_targets()
        except (FileNotFoundError, AttributeError):
            apdft.log.log(
                "At least one of the QM calculations has not been performed yet. Please run all QM calculations first.",
                level="warning",
            )
            return

        if explicit_reference:
            comparison_energies = np.zeros(len(targets))
            comparison_dipoles = np.zeros((len(targets), 3))
            comparison_atomic_forces = np.zeros((len(targets), 3))
            for targetidx, target in enumerate(targets):
                path = "QM/comparison-%s" % "-".join(map(str, target))
                try:
                    comparison_energies[targetidx] = self._calculator.get_total_energy(
                        path
                    )
                except FileNotFoundError:
                    apdft.log.log(
                        "Comparison calculation is missing. Predictions are unaffected. Will skip this comparison.",
                        level="warning",
                        calculation=path,
                        target=target,
                    )
                    comparison_energies[targetidx] = np.nan
                    comparison_dipoles[targetidx] = np.nan
                    comparison_atomic_forces[targetidx] = np.nan
                    continue
                except ValueError:
                    apdft.log.log(
                        "Comparison calculation is damaged. Predictions are unaffected. Will skip this comparison.",
                        level="warning",
                        calculation=path,
                        target=target,
                    )
                    comparison_energies[targetidx] = np.nan
                    comparison_dipoles[targetidx] = np.nan
                    comparison_atomic_forces[targetidx] = np.nan
                    continue

                nd = apdft.physics.Dipoles.point_charges(
                    [0, 0, 0], self._coordinates, target
                )
                # TODO: load dipole
                # comparison_dipoles[targetidx] = ed + nd
        else:
            comparison_energies = None
            comparison_dipoles = None
            comparison_atomic_forces = None

        self._print_energies(targets, energies, comparison_energies)
        self._print_dipoles(targets, dipoles, comparison_dipoles)
        self._print_forces(targets, atomic_forces, comparison_atomic_forces)

        # persist results to disk
        targetnames = [APDFT._get_target_name(_) for _ in targets]
        # Energy
        result_energies = {"targets": targetnames, "total_energy": energies[:, -1]}
        result_ele_energies = {"targets": targetnames, "ele_energy": ele_energies[:, -1]}
        result_nuc_energies = {"targets": targetnames, "nuc_energy": nuc_energies[:, -1]}
        for order in self._orders:
            result_energies["total_energy_order%d" % order] = energies[:, order]
            result_ele_energies["ele_energy_order%d" % order] = ele_energies[:, order]
            result_nuc_energies["nuc_energy_order%d" % order] = nuc_energies[:, order]
        # Dipole
        result_dipoles = {
            "targets": targetnames,
            "dipole_moment_x": dipoles[:, 0, -1],
            "dipole_moment_y": dipoles[:, 1, -1],
            "dipole_moment_z": dipoles[:, 2, -1],
        }
        for order in self._orders:
            for didx, dim in enumerate("xyz"):
                result_dipoles["dipole_moment_%s_order%d" % (dim, order)] = dipoles[
                    :, didx, order
                ]
        # Electronic dipole
        ele_result_dipoles = {
            "targets": targetnames,
            "ele_dipole_moment_x": ele_dipoles[:, 0, -1],
            "ele_dipole_moment_y": ele_dipoles[:, 1, -1],
            "ele_dipole_moment_z": ele_dipoles[:, 2, -1],
        }
        for order in self._orders:
            for didx, dim in enumerate("xyz"):
                ele_result_dipoles["ele_dipole_moment_%s_order%d" % (dim, order)] = ele_dipoles[
                    :, didx, order
                ]
        # Nuclear dipole
        nuc_result_dipoles = {
            "targets": targetnames,
            "nuc_dipole_moment_x": nuc_dipoles[:, 0, -1],
            "nuc_dipole_moment_y": nuc_dipoles[:, 1, -1],
            "nuc_dipole_moment_z": nuc_dipoles[:, 2, -1],
        }
        for order in self._orders:
            for didx, dim in enumerate("xyz"):
                nuc_result_dipoles["nuc_dipole_moment_%s_order%d" % (dim, order)] = nuc_dipoles[
                    :, didx, order
                ]

        # Force
        result_atomic_forces = {
            "targets": targetnames,
        }
        # The best results with the highest APDFT order
        for atomidx in range(len(self._nuclear_numbers)):
            for didx, dim in enumerate("xyz"):
                result_atomic_forces["force_atom%d_%s" % (atomidx, dim)] = atomic_forces[
                    :, atomidx, didx, -1
                ]
        # All the results
        for order in self._orders:
            for atomidx in range(len(self._nuclear_numbers)):
                for didx, dim in enumerate("xyz"):
                    result_atomic_forces["force_atom%d_%s_order%d" % (atomidx, dim, order)] = atomic_forces[
                        :, atomidx, didx, order
                    ]

        # Electronic force
        ele_result_atomic_forces = {
            "targets": targetnames,
        }
        # The best results with the highest APDFT order
        for atomidx in range(len(self._nuclear_numbers)):
            for didx, dim in enumerate("xyz"):
                ele_result_atomic_forces["ele_force_atom%d_%s" % (atomidx, dim)] = ele_atomic_forces[
                    :, atomidx, didx, -1
                ]
        # All the results
        for order in self._orders:
            for atomidx in range(len(self._nuclear_numbers)):
                for didx, dim in enumerate("xyz"):
                    ele_result_atomic_forces["force_atom%d_%s_order%d" % (atomidx, dim, order)] = ele_atomic_forces[
                        :, atomidx, didx, order
                    ]

        # Nuclear force
        nuc_result_atomic_forces = {
            "targets": targetnames,
        }
        # The best results with the highest APDFT order
        for atomidx in range(len(self._nuclear_numbers)):
            for didx, dim in enumerate("xyz"):
                nuc_result_atomic_forces["force_atom%d_%s" % (atomidx, dim)] = nuc_atomic_forces[
                    :, atomidx, didx, -1
                ]
        # All the results
        for order in self._orders:
            for atomidx in range(len(self._nuclear_numbers)):
                for didx, dim in enumerate("xyz"):
                    nuc_result_atomic_forces["force_atom%d_%s_order%d" % (atomidx, dim, order)] = nuc_atomic_forces[
                        :, atomidx, didx, order
                    ]

        if explicit_reference:
            result_energies["reference_energy"] = comparison_energies
            result_dipoles["reference_dipole_x"] = comparison_dipoles[:, 0]
            result_dipoles["reference_dipole_y"] = comparison_dipoles[:, 1]
            result_dipoles["reference_dipole_z"] = comparison_dipoles[:, 2]
            for atomidx in range(len(self._nuclear_numbers)):
                result_atomic_forces["reference_force_x"] = comparison_atomic_forces[:, atomidx, 0]
                result_atomic_forces["reference_force_y"] = comparison_atomic_forces[:, atomidx, 1]
                result_atomic_forces["reference_force_z"] = comparison_atomic_forces[:, atomidx, 2]

        pd.DataFrame(result_energies).to_csv("energies.csv", index=False)
        pd.DataFrame(result_ele_energies).to_csv("ele_energies.csv", index=False)
        pd.DataFrame(result_nuc_energies).to_csv("nuc_energies.csv", index=False)
        pd.DataFrame(result_dipoles).to_csv("dipoles.csv", index=False)
        pd.DataFrame(ele_result_dipoles).to_csv("ele_dipoles.csv", index=False)
        pd.DataFrame(nuc_result_dipoles).to_csv("nuc_dipoles.csv", index=False)
        pd.DataFrame(result_atomic_forces).to_csv("atomic_forces.csv", index=False)
        pd.DataFrame(ele_result_atomic_forces).to_csv("ele_atomic_forces.csv", index=False)
        pd.DataFrame(nuc_result_atomic_forces).to_csv(
            "nuc_atomic_forces.csv", index=False)

    # For an "energies_geometries" mode
    def analyse_general(self, target_coordinate=None, explicit_reference=False):
        """ Performs actual analysis and integration. Prints results"""
        # If the target coordinate is not given, the error message is displayed.
        if target_coordinate is None:
            apdft.log.log(
                "Target molecular coordinate is not given.", level="error"
            )

        try:
            targets, energies, ele_energies, nuc_energies, dipoles, ele_dipoles, nuc_dipoles, \
            energies_reference_contributions, energies_target_contributions, energies_total_contributions, \
            atomic_forces, ele_atomic_forces, nuc_atomic_forces, hf_ionic_force_contributions, \
            deriv_rho_contributions, hf_ionic_forces, ele_hf_ionic_forces, nuc_hf_ionic_forces \
                = self.predict_all_targets_general(target_coordinate)

        except (FileNotFoundError, AttributeError):
            apdft.log.log(
                "At least one of the QM calculations has not been performed yet. Please run all QM calculations first.",
                level="warning",
            )
            return

        if explicit_reference:
            comparison_energies = np.zeros(len(targets))
            comparison_dipoles = np.zeros((len(targets), 3))
            for targetidx, target in enumerate(targets):
                path = "QM/comparison-%s" % "-".join(map(str, target))
                try:
                    comparison_energies[targetidx] = self._calculator.get_total_energy(
                        path
                    )
                except FileNotFoundError:
                    apdft.log.log(
                        "Comparison calculation is missing. Predictions are unaffected. Will skip this comparison.",
                        level="warning",
                        calculation=path,
                        target=target,
                    )
                    comparison_energies[targetidx] = np.nan
                    comparison_dipoles[targetidx] = np.nan
                    continue
                except ValueError:
                    apdft.log.log(
                        "Comparison calculation is damaged. Predictions are unaffected. Will skip this comparison.",
                        level="warning",
                        calculation=path,
                        target=target,
                    )
                    comparison_energies[targetidx] = np.nan
                    comparison_dipoles[targetidx] = np.nan
                    continue

                nd = apdft.physics.Dipoles.point_charges(
                    [0, 0, 0], self._coordinates, target
                )
                # TODO: load dipole
                # comparison_dipoles[targetidx] = ed + nd
        else:
            comparison_energies = None
            comparison_dipoles = None

        self._print_energies(targets, energies, comparison_energies)
        self._print_dipoles(targets, dipoles, comparison_dipoles)

        # Sum of contributions obtained at each APDFT order
        sum_energies_reference_contributions = np.zeros(len(targets))
        sum_energies_target_contributions = np.zeros(len(targets))
        sum_energies_total_contributions = np.zeros(len(targets))
        for order in self._orders:
            sum_energies_reference_contributions[:] += energies_reference_contributions[:, order]
            sum_energies_target_contributions[:] += energies_target_contributions[:, order]
            sum_energies_total_contributions[:] += energies_total_contributions[:, order]

        # persist results to disk
        targetnames = [APDFT._get_target_name(_) for _ in targets]
        result_energies = {"targets": targetnames,
                           "total_energy": energies[:, -1]}
        result_ele_energies = {"targets": targetnames,
                               "ele_energy": ele_energies[:, -1]}
        result_nuc_energies = {"targets": targetnames,
                               "nuc_energy": nuc_energies[:, -1]}
        result_energies_reference_contributions = {"targets": targetnames,
                                                   "reference_contributions":
                                                   sum_energies_reference_contributions[:]}
        result_energies_target_contributions = {"targets": targetnames,
                                                "target_contributions":
                                                sum_energies_target_contributions[:]}
        result_energies_total_contributions = {"targets": targetnames,
                                               "total_contributions":
                                               sum_energies_total_contributions[:]}
        for order in self._orders:
            result_energies["total_energy_order%d" %
                            order] = energies[:, order]
            result_ele_energies["ele_energy_order%d" %
                                order] = ele_energies[:, order]
            result_nuc_energies["nuc_energy_order%d" %
                                order] = nuc_energies[:, order]
            result_energies_reference_contributions["reference_contributions_order%d" %
                                           order] = energies_reference_contributions[:, order]
            result_energies_target_contributions["target_contributions_order%d" %
                                        order] = energies_target_contributions[:, order]
            result_energies_total_contributions["total_contributions_order%d" %
                                       order] = energies_total_contributions[:, order]
        # Dipole
        result_dipoles = {
            "targets": targetnames,
            "dipole_moment_x": dipoles[:, 0, -1],
            "dipole_moment_y": dipoles[:, 1, -1],
            "dipole_moment_z": dipoles[:, 2, -1],
        }
        for order in self._orders:
            for didx, dim in enumerate("xyz"):
                result_dipoles["dipole_moment_%s_order%d" % (dim, order)] = dipoles[
                    :, didx, order
                ]
        # Electronic dipole
        ele_result_dipoles = {
            "targets": targetnames,
            "ele_dipole_moment_x": ele_dipoles[:, 0, -1],
            "ele_dipole_moment_y": ele_dipoles[:, 1, -1],
            "ele_dipole_moment_z": ele_dipoles[:, 2, -1],
        }
        for order in self._orders:
            for didx, dim in enumerate("xyz"):
                ele_result_dipoles["ele_dipole_moment_%s_order%d" % (dim, order)] = ele_dipoles[
                    :, didx, order
                ]
        # Nuclear dipole
        nuc_result_dipoles = {
            "targets": targetnames,
            "nuc_dipole_moment_x": nuc_dipoles[:, 0, -1],
            "nuc_dipole_moment_y": nuc_dipoles[:, 1, -1],
            "nuc_dipole_moment_z": nuc_dipoles[:, 2, -1],
        }
        for order in self._orders:
            for didx, dim in enumerate("xyz"):
                nuc_result_dipoles["nuc_dipole_moment_%s_order%d" % (dim, order)] = nuc_dipoles[
                    :, didx, order
                ]
        if explicit_reference:
            result_energies["reference_energy"] = comparison_energies
            result_dipoles["reference_dipole_x"] = comparison_dipoles[:, 0]
            result_dipoles["reference_dipole_y"] = comparison_dipoles[:, 1]
            result_dipoles["reference_dipole_z"] = comparison_dipoles[:, 2]

        # Set results of atomic forces
        result_atomic_forces = {}
        result_atomic_forces["targets"] = targetnames
        # Results of electronic contributions of atomic forces
        result_ele_atomic_forces = {}
        result_ele_atomic_forces["targets"] = targetnames
        # Results of the nuclear term of atomic forces
        result_nuc_atomic_forces = {}
        result_nuc_atomic_forces["targets"] = targetnames
        result_hf_ionic_force_contributions = {}
        result_hf_ionic_force_contributions["targets"] = targetnames
        result_deriv_rho_contributions = {}
        result_deriv_rho_contributions["targets"] = targetnames

        # TODO: generalization to specify target atoms
        natoms = len(self._coordinates)
        for order in self._orders:
            for atom_pos in range(natoms):
                # Only Z component is presented.
                # TODO: generalization to three Cartesian coordinates
                result_atomic_forces["atomic_force_%s_order%d" % (atom_pos, order)] = \
                    atomic_forces[:, order, atom_pos, 2]
                result_ele_atomic_forces["ele_atomic_force_%s_order%d" % (atom_pos, order)] = \
                    ele_atomic_forces[:, order, atom_pos, 2]
                result_nuc_atomic_forces["nuc_atomic_force_%s_order%d" % (atom_pos, order)] = \
                    nuc_atomic_forces[:, order, atom_pos, 2]
                result_hf_ionic_force_contributions["atomic_force_%s_order%d" % (atom_pos, order)] = \
                    hf_ionic_force_contributions[:, order, atom_pos, 2]
                result_deriv_rho_contributions["atomic_force_%s_order%d" % (atom_pos, order)] = \
                    deriv_rho_contributions[:, order, atom_pos, 2]

        # Hellmann-Feynman ionic force
        result_hf_ionic_forces = {
            "targets": targetnames,
        }
        # The best results with the highest APDFT order
        for atomidx in range(len(self._nuclear_numbers)):
            for didx, dim in enumerate("xyz"):
                result_hf_ionic_forces["hf_ionic_force_atom%d_%s" % (atomidx, dim)] = hf_ionic_forces[
                    :, atomidx, didx, -1
                ]
        # All the results
        for order in self._orders:
            for atomidx in range(len(self._nuclear_numbers)):
                for didx, dim in enumerate("xyz"):
                    result_hf_ionic_forces["hf_ionic_force_atom%d_%s_order%d" % (atomidx, dim, order)] = hf_ionic_forces[
                        :, atomidx, didx, order
                    ]

        # Electronic Hellmann-Feynman ionic force
        result_ele_hf_ionic_forces = {
            "targets": targetnames,
        }
        # The best results with the highest APDFT order
        for atomidx in range(len(self._nuclear_numbers)):
            for didx, dim in enumerate("xyz"):
                result_ele_hf_ionic_forces["ele_hf_ionic_force_atom%d_%s" % (atomidx, dim)] = ele_hf_ionic_forces[
                    :, atomidx, didx, -1
                ]
        # All the results
        for order in self._orders:
            for atomidx in range(len(self._nuclear_numbers)):
                for didx, dim in enumerate("xyz"):
                    result_ele_hf_ionic_forces["hf_ionic_force_atom%d_%s_order%d" % (atomidx, dim, order)] = ele_hf_ionic_forces[
                        :, atomidx, didx, order
                    ]

        # Nuclear Hellmann-Feynman ionic force
        result_nuc_hf_ionic_forces = {
            "targets": targetnames,
        }
        # The best results with the highest APDFT order
        for atomidx in range(len(self._nuclear_numbers)):
            for didx, dim in enumerate("xyz"):
                result_nuc_hf_ionic_forces["force_atom%d_%s" % (atomidx, dim)] = nuc_hf_ionic_forces[
                    :, atomidx, didx, -1
                ]
        # All the results
        for order in self._orders:
            for atomidx in range(len(self._nuclear_numbers)):
                for didx, dim in enumerate("xyz"):
                    result_nuc_hf_ionic_forces["force_atom%d_%s_order%d" % (atomidx, dim, order)] = nuc_hf_ionic_forces[
                        :, atomidx, didx, order
                    ]

        pd.DataFrame(result_energies).to_csv("energies.csv", index=False)
        pd.DataFrame(result_ele_energies).to_csv("ele_energies.csv", index=False)
        pd.DataFrame(result_nuc_energies).to_csv("nuc_energies.csv", index=False)
        pd.DataFrame(result_energies_reference_contributions).to_csv(
            "energies_reference_contributions.csv", index=False)
        pd.DataFrame(result_energies_target_contributions).to_csv(
            "energies_target_contributions.csv", index=False)
        pd.DataFrame(result_energies_total_contributions).to_csv(
            "energies_total_contributions.csv", index=False)
        pd.DataFrame(result_dipoles).to_csv("dipoles.csv", index=False)
        pd.DataFrame(ele_result_dipoles).to_csv("ele_dipoles.csv", index=False)
        pd.DataFrame(nuc_result_dipoles).to_csv("nuc_dipoles.csv", index=False)

        pd.DataFrame(result_atomic_forces).to_csv(
            "atomic_forces.csv", index=False)
        pd.DataFrame(result_ele_atomic_forces).to_csv(
            "ele_atomic_forces.csv", index=False)
        pd.DataFrame(result_nuc_atomic_forces).to_csv(
            "nuc_atomic_forces.csv", index=False)
        pd.DataFrame(result_hf_ionic_force_contributions).to_csv(
            "hf_ionic_force_contributions.csv", index=False)
        pd.DataFrame(result_deriv_rho_contributions).to_csv(
            "deriv_rho_contributions.csv", index=False)

        pd.DataFrame(result_hf_ionic_forces).to_csv(
            "hf_ionic_forces.csv", index=False)
        pd.DataFrame(result_ele_hf_ionic_forces).to_csv(
            "ele_hf_ionic_forces.csv", index=False)
        pd.DataFrame(result_nuc_hf_ionic_forces).to_csv(
            "nuc_hf_ionic_forces.csv", index=False)

        return targets, energies, comparison_energies
