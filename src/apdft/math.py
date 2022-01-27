#!/usr/bin/env python
import numpy as np
import functools
import itertools as it
import copy


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
    def arbitrary_partition(nuclear_numbers, target_atom_number, target_atom_positions):
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

        # Set range of mutated atoms
        range_nuclear_numbers = []
        range_nuclear_numbers.append(target_atom_number - 1)
        range_nuclear_numbers.append(target_atom_number)
        range_nuclear_numbers.append(target_atom_number + 1)

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
        for mol in it.product(range_nuclear_numbers, repeat=num_target_atoms):
            # It is assumed that a sum of the number of protons of mutated
            # atoms unchanges.
            if sum(mol) != num_target_protons:
                continue

            # Get nuclear numbers of a molecule with mutated atoms
            for i in range(num_target_atoms):
                instant_nuclear_numbers[target_atom_positions[i]] = mol[i]

            res.append(list(instant_nuclear_numbers))

        return res
