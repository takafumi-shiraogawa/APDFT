#!/usr/bin/env python
import os
import shutil
import numpy as np
import functools
import itertools as it
import copy
import apdft
import apdft.physics as ap
import apdft.proc_output as apo
import platform
from concurrent.futures import ProcessPoolExecutor

# Conversion factor from Angstrom to Bohr
angstrom = 1 / 0.52917721067

path_data = os.path.dirname(__file__).replace('src/apdft', 'src/apdft/mini_qml')
mini_qml_files = os.listdir(path_data)
flag_mini_qml = False
for idx, file in enumerate(mini_qml_files):
    if ".so" in file:
        flag_mini_qml = True
        import apdft.mini_qml.representations as amr

class IntegerPartitions(object):
    @staticmethod
    @functools.lru_cache(maxsize=64)
    def _do_partition(total, maxelements, around=None, maxdz=None):
        """ Builds all integer partitions of *total* split into *maxelements* parts.

		Note that ordering matters, i.e. (2, 1) and (1, 2) are district partitions. Moreover, elements of zero value are allowed. In all cases, the sum of all elements is equal to *total*.
		There is no guarantee as for the ordering of elements.

		If a center *around* is given, then a radius *maxdz* is required.
		Only those partitions are listed where the L1 norm of the distance between partition and *around* is less or equal to *maxdz*.

		Args:
			total:			The sum of all entries. [Integer]
			maxelements:	The number of elements to split into. [Integer]
			around:			Tuple of N entries. Center around which partitions are listed. [Integer]
			maxdz:			Maximum absolute difference in Z space from center *around*. [Integer]
		Returns:
			A list of all partitions as lists.
		"""
        if (around is None) != (maxdz is None):
            raise ValueError("Cannot define center or radius alone.")

        # If the site number is 1
        if maxelements == 1:
            # If the nuclear number of the atom is given and
            # maxdz is smaller than the absolute difference in
            # the allowed total charge and the number of protons
            # That is, the latter condition means that it is not
            # possible to reach the allowed charge by the change
            # of the atomic charge.
            if around is not None and maxdz < abs(total - around[-maxelements]):
                return []
            else:
                return [[total]]
        res = []

        # get range to cover
        if around is None:
            first = 0
            last = total
            limit = None
        else:
            first = max(0, around[-maxelements] - maxdz)
            last = min(total, around[-maxelements] + maxdz)
        for x in range(first, last + 1):
            if around is not None:
                limit = maxdz - abs(x - around[-maxelements])
            for p in IntegerPartitions._do_partition(
                total - x, maxelements - 1, around, limit
            ):
                res.append([x] + p)
        return res

    @staticmethod
    def partition(total, maxelements, around=None, maxdz=None):
        """ Builds all integer partitions of *total* split into *maxelements* parts.

		Note that ordering matters, i.e. (2, 1) and (1, 2) are district partitions. Moreover, elements of zero value are allowed. In all cases, the sum of all elements is equal to *total*.
		There is no guarantee as for the ordering of elements.

		If a center *around* is given, then a radius *maxdz* is required.
		Only those partitions are listed where the L1 norm of the distance between partition and *around* is less or equal to *maxdz*.

		Args:
			total:			The sum of all entries. [Integer]
			maxelements:	The number of elements to split into. [Integer]
			around:			Iterable of N entries. Center around which partitions are listed. [Integer]
			maxdz:			Maximum absolute difference in Z space from center *around*. [Integer]
		Returns:
			A list of all partitions as lists.
		"""
        if around is not None:
            return IntegerPartitions._do_partition(
                total, maxelements, tuple(around), maxdz
            )
        else:
            return IntegerPartitions._do_partition(total, maxelements)

    @staticmethod
    def arbitrary_partition(nuclear_numbers, target_atom_number, target_atom_positions, limit_mutations):
        """ Get a list of target molecules with mutated atoms with [-1, 0, 1] nuclear number changes
        Args:
            nuclear_numbers   : Iterable of N entries. Nuclear numbers are listed. [Integer]
            target_atom_number : A target atom number. [Integer]
            target_atom_positions : List begins from 0 [Integer]

		Returns:
			A list of all partitions as lists.
        """

        # Exception handling
        if target_atom_number == 1:
            raise ValueError("Error: Specification of the mutated atom is invalid.")

        # Set target mutated atoms
        target_nuclear_numbers = []
        target_nuclear_numbers.append(target_atom_number - 1)
        target_nuclear_numbers.append(target_atom_number)
        target_nuclear_numbers.append(target_atom_number + 1)

        # Get the number of target atoms
        num_target_atoms = len(target_atom_positions)

        # Get a sum of number of protons of the target atoms
        num_target_protons = 0
        for i in range(num_target_atoms):
            num_target_protons += nuclear_numbers[target_atom_positions[i]]

        # Copy nuclear numbers of a reference molecule
        instant_nuclear_numbers = copy.copy(nuclear_numbers)

        # Get a list of target molecules
        res = []
        for mut_nuclear_numbers in it.product(target_nuclear_numbers, repeat=num_target_atoms):
            # It is assumed that a sum of the number of protons of mutated
            # atoms unchanges.
            if sum(mut_nuclear_numbers) != num_target_protons:
                continue

            # Get nuclear numbers of a molecule with mutated atoms
            diff_Z = 0
            for i in range(num_target_atoms):
                instant_nuclear_numbers[target_atom_positions[i]] = mut_nuclear_numbers[i]

                diff_Z += abs(mut_nuclear_numbers[i] - nuclear_numbers[target_atom_positions[i]])

            # If the number of mutation atoms exceeds the limit,
            # it is not included in target molecules.
            if diff_Z > limit_mutations:
                continue

            res.append(list(instant_nuclear_numbers))

        return res

    def judge_unique(this_eigen_value, eigen_value):
        dist = ap.Coulomb.get_distance_mols_with_coulomb_matrix(this_eigen_value, eigen_value)
        # This is possibly used in a PRR article
        if dist < 0.01:
            flag_unique_mol = False
        else:
            flag_unique_mol = True

        return flag_unique_mol

    def wrapper_judge_unique(args):
        return IntegerPartitions.judge_unique(*args)

    @staticmethod
    def systematic_partition(nuclear_numbers, target_atom_positions, \
        limit_mutations, nuclear_coordinates, mol_identity=True, gener_output=True, gener_coord=True):
        """ Get a list of target molecules with mutated atoms with [-1, 0, 1] nuclear number changes
        Args:
            nuclear_numbers       : Iterable of N entries. Nuclear numbers are listed. [Integer]
            target_atom_positions : List begins from 0 [Integer]
            nuclear_coordinates   : (N, 3) entries. Nuclear coordinates are listed. [Integer]
            mol_identity          : Wheteher to remove same molecules [boolean]
            gener_output          : Whether to save a list of obtained target molecules [boolean]

		Returns:
			A list of all partitions as lists.
        """

        name_os = platform.system()
        if name_os == 'Darwin':
            num_smp_core = 8
        else:
            num_smp_core = os.cpu_count()

        # Set changes of nuclear numbers
        change_nuclear_numbers = []
        change_nuclear_numbers.append(-1)
        change_nuclear_numbers.append(1)

        # Get the number of target atoms
        num_target_atoms = len(target_atom_positions)

        # Set a list of target molecules
        res = []

        # Count processed molecules
        count_proc_mol = 0

        # Mutation covers from a reference molecule (num_mut = 0)
        # to molecules with mutated atoms in the limited mutation number.
        for num_mut_atoms in range(num_target_atoms + 1):
            # Here mutation atoms should be pair because increase or decrease
            # of the charge is +1 or -1 and target molecule must be charge neutral.
            if num_mut_atoms % 2 != 0:
                continue

            # If the number of mutation atoms exceeds the limit
            if num_mut_atoms > limit_mutations:
                break

            unique_eigen_value = []

            # Select mutated atom positions
            for mut_atom_positions in it.combinations(target_atom_positions, num_mut_atoms):

                # Select atom types of mutated atoms
                # for mut_nuclear_numbers in it.product(target_nuclear_numbers, repeat=num_mut_atoms):
                # Because of the neutral charge condition, only atoms with the positive charge change
                # are specified.
                positions_mut_atom_positions = []
                for i in range(num_mut_atoms):
                    positions_mut_atom_positions.append(i)

                # Select positive mutation positions
                for pos_positions_mut_atom_positions in it.combinations(positions_mut_atom_positions, int(num_mut_atoms / 2)):
                    count_proc_mol += 1

                    # Copy nuclear numbers of a reference molecule
                    instant_nuclear_numbers = copy.copy(nuclear_numbers)

                    # Initialize with negative mutations
                    mut_nuclear_numbers = [change_nuclear_numbers[0]] * num_mut_atoms

                    # Specify positions of atoms with the positive charge change
                    for i in range(int(num_mut_atoms / 2)):
                        mut_nuclear_numbers[pos_positions_mut_atom_positions[i]] = change_nuclear_numbers[1]

                    # Get a target molecule
                    flag_He_checker = False
                    for i in range(num_mut_atoms):
                        if instant_nuclear_numbers[mut_atom_positions[i]] == 1 and mut_nuclear_numbers[i] == 1:
                            flag_He_checker = True
                        instant_nuclear_numbers[mut_atom_positions[i]] += mut_nuclear_numbers[i]
                    if flag_He_checker:
                        continue

                    # Whether to remove same molecules
                    if mol_identity:
                        # Eigenvalues of Coulomb matrices
                        flag_unique_mol = False
                        if not flag_mini_qml:
                            this_eigen_value = ap.Coulomb.gener_eigenvalues_from_coulomb_matrix(
                                instant_nuclear_numbers, nuclear_coordinates)
                        else:
                            this_eigen_value = amr.generate_eigenvalue_coulomb_matrix(
                                instant_nuclear_numbers, nuclear_coordinates * angstrom, len(nuclear_coordinates))
                        if len(unique_eigen_value) != 0:
                            # Parallelized
                            with ProcessPoolExecutor(max_workers=num_smp_core) as executor:
                                values = [(this_eigen_value, y) for y in unique_eigen_value]
                                flags = executor.map(IntegerPartitions.wrapper_judge_unique, values)
                            if all(list(flags)):
                                flag_unique_mol = True

                            # Non-parallelized
                            # for idx, eigen_value in enumerate(unique_eigen_value):
                            #     dist = ap.Coulomb.get_distance_mols_with_coulomb_matrix(this_eigen_value, eigen_value)
                            #     # This is possibly used in a PRR article
                            #     if dist < 0.01:
                            #         flag_unique_mol = False
                            #         break
                            #     else:
                            #         flag_unique_mol = True
                            #         continue
                        else:
                            flag_unique_mol = True

                    if flag_unique_mol:
                        unique_eigen_value.append(this_eigen_value)
                        res.append(list(instant_nuclear_numbers))

                        with open('num_target_mol.dat', mode='w') as fh:
                            print("Processed molecules:", count_proc_mol, file=fh)
                            print("Unique molecules:", len(res), file=fh)

        # Whether to save a list of obtained target molecules
        if gener_output:
            # Generate a list of target molecules (target_molecules.inp)
            fh = open('target_molecules.inp', 'w')
            for i in range(len(res)):
                print(*res[i], sep=',', file=fh)
            fh.close()

        # Whether to save geometry outputs of target molecules
        if gener_coord:
            if os.path.isdir("./target_geom/"):
                shutil.rmtree("./target_geom/")
            os.mkdir("./target_geom/")

            pre_name = "./target_geom/geom_target"
            for idx, target in enumerate(res):
                name = "%s%s" % (pre_name, str(idx + 1))
                apo.Geom_Output.gjf_output(target, nuclear_coordinates, name)

        return res

    @staticmethod
    def read_target_molecules(input_path = None):
        """ Read a list of target molecules
        Args:
            input_path  : path of input including a file name, e.g., /home/test/target_molecules.inp
        Returns:
			A list of all partitions as lists [numpy.ndarrray]
        """

        # If the input path is given
        if input_path is not None:
            with open(input_path, 'r') as fh:
                targetlist = apdft.commandline.parse_target_list(fh.readlines())
        # If the input path is not given, target_molecules.inp in the current directory
        # is used as the input.
        else:
            with open('target_molecules.inp', 'r') as fh:
                targetlist = apdft.commandline.parse_target_list(fh.readlines())

        # TODO: remove redundant conversion
        return list(map(list, targetlist))
