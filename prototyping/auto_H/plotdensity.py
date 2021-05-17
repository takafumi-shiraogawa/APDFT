#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:15:49 2019
@author: misa
get density on grid from cube files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial as scs
import scipy.integrate
from scipy.interpolate import CubicSpline
import scipy.spatial.transform as sst
import functools
import copy
from ase.units import Bohr

class CUBE(object):
    
    def __init__(self, fname):
        f = open(fname, 'r')
        for i in range(2): f.readline() # echo comment
        tkns = f.readline().split() # number of atoms included in the file followed by the position of the origin of the volumetric data
        self.natoms = int(tkns[0])
        self.origin = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
        tkns = f.readline().split() #
        self.NX = int(tkns[0])
        self.X = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
        tkns = f.readline().split() #
        self.NY = int(tkns[0])
        self.Y = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
        tkns = f.readline().split() #
        self.NZ = int(tkns[0])
        self.Z = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])

        self.dv = np.linalg.det(np.array([self.X, self.Y, self.Z])) # volume per gridpoint
        
        self.atoms = []
        for i in range(self.natoms):
          tkns = f.readline().split()
          self.atoms.append([float(tkns[0]), float(tkns[2]), float(tkns[3]), float(tkns[4])])
          
        self.atoms = np.array(self.atoms)
        
        self.data = np.zeros((self.NX,self.NY,self.NZ))
        i=0
        for s in f:
          for v in s.split():
            self.data[i//(self.NY*self.NZ), (i//self.NZ)%self.NY, i%self.NZ] = float(v)
            i+=1
        if i != self.NX*self.NY*self.NZ: raise NameError("FSCK!")
        f.close()
        
    def project(self, axes):
        """
        scales density by gridvolume and projects density on specified axes (1D or 2D)
        """
        projected_density = np.sum(self.data*self.dv, axis=axes)
        return(projected_density)
        
    def get_grid(self):
        """
        returns the coordinates of the grid points where the density values are given as a meshgrid
        works so far only for orthogonal coordinate axes
        """
        # length along the axes
        l_x = self.X[0]*self.NX
        l_y = self.Y[1]*self.NX
        l_z = self.Z[2]*self.NZ
        # gpts along every axis
        x_coords = np.linspace(self.origin[0], l_x-self.X[0], self.NX)
        y_coords = np.linspace(self.origin[1], l_y-self.Y[1], self.NY)
        z_coords = np.linspace(self.origin[2], l_z-self.Z[2], self.NZ)
        # create gridpoints
        meshgrid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        return(meshgrid)
        
        
    def scale(self):
        """
        calculate density scaled by volume of gridpoints
        """
        self.data_scaled = (self.data*self.dv).copy()
        
    def plot_projection(self, axes):
        """
        plot scaled projection of density along specified projection axis (1D, 2D)
        """
        projected_density = self.project(axes)
        
        if type(axes) == tuple:
            coordinate = np.linspace(self.origin[0], self.X[0]*self.NX*Bohr, self.NX)          # origin need * Bohr as well???
            fig, ax = plt.subplots(1,1)
            ax.plot(coordinate, projected_density)
            ax.set_xlabel(r'Cell coordinate $x_0$ (Ang)')
            ax.set_ylabel(r'$\rho (x_0)$')
            with open ("rho.txt","w+") as f:
                for i in range(len(coordinate)):
                    f.write(str(coordinate[i])+"          "+str(projected_density[i])+"\n")
            f.close()

            
        if type(axes) == int:
            coordinate0 = np.linspace(self.origin[0]+0.5, self.X[0]*self.NX*Bohr-0.5, self.NX)  # +/- 0.5 ????
            coordinate1 = np.linspace(self.origin[0], self.X[0]*self.NX*Bohr, self.NX)          # both of them are self.X[0] if it is not cubic???
            fig, ax = plt.subplots(1,1)
            ax.contour(coordinate0, coordinate1, projected_density)


