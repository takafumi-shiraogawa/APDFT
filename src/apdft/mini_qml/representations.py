# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen, Lars Andersen Bratholm and Bing Huang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .frepresentations import fgenerate_eigenvalue_coulomb_matrix

def generate_eigenvalue_coulomb_matrix(nuclear_charges, coordinates, size = 23):
    """ Creates an eigenvalue Coulomb Matrix representation of a molecule.
        A matrix :math:`M` is constructed with elements

        .. math::

            M_{ij} =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}       & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i` and :math:`j` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.
        The molecular representation of the molecule is then the sorted eigenvalues of M.
        The representation is calculated using an OpenMP parallel Fortran routine.

        :param nuclear_charges: Nuclear charges of the atoms in the molecule
        :type nuclear_charges: numpy array
        :param coordinates: 3D Coordinates of the atoms in the molecule
        :type coordinates: numpy array
        :param size: The size of the largest molecule supported by the representation
        :type size: integer

        :return: 1D representation - shape (size, )
        :rtype: numpy array
    """
    return fgenerate_eigenvalue_coulomb_matrix(nuclear_charges,
        coordinates, size)
