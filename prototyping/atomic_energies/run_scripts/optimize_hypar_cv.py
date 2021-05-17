#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:43:49 2019

@author: misa

performs hyperparameter optimization, the labels can be specified in pl
"""
import numpy as np
import glob
import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')

import qml_interface as qi
import qml_interface2 as qmi2

base='/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/'
pl = ['atomisation'] #['alch_pot', 'atomic', 'atomisation']
tr_set = 512
num_cv = 10
results = []

#for p in pl:
#    # load data into list, count number of atoms per molecule
#    paths=qi.wrapper_alch_data()
#    alchemy_data, molecule_size = qi.load_alchemy_data(paths)
#    labels = qi.generate_label_vector(alchemy_data, molecule_size.sum(), value=p)
#    
#    # load kernel
#    basepath = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/FCHL/'
#    sigma_path = glob.glob(basepath+'*alchoff*')
#    
#    for sigma_kernel in sigma_path:
#        print('Loading kernel {}'.format(sigma_kernel))
#        kernel = np.loadtxt(sigma_kernel)
#        print('Optimizing regularizer')
#        lam, error, std = qmi2.optimize_regularizer(kernel, labels, molecule_size, tr_size = tr_set, num_cross=num_cv)
#        store_array = np.array([lam, error,std]).T
#        np.savetxt(sigma_kernel+'_alchoff_lam', store_array)
    
##############################################################################
##                      OLD VERSION/not FCHL
##############################################################################

base='/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/'
pl = ['atomisation'] #['alch_pot', 'atomic', 'atomisation']
tr_set = 512
num_cv = 3
results = []
delta_learning = True

for p in pl:
    # load data into list, count number of atoms per molecule
    paths=qi.wrapper_alch_data()
    alchemy_data, molecule_size = qi.load_alchemy_data(paths)
    
    # local data
    local_reps = qi.generate_atomic_representations(alchemy_data, molecule_size)
    local_labels = qi.generate_label_vector(alchemy_data, molecule_size.sum(), value=p)
    
    if delta_learning:
        # delta Learning
        # divide indices in groups depending on charge
        charges = qi.generate_label_vector(alchemy_data, molecule_size.sum(), 'charge')
        part_charges = qi.partition_idx_by_charge(charges)
        # get mean atomisation energy per charge
        mean_atomisation = dict.fromkeys(list(set(charges)),0)
        for k in part_charges:
            mtmp = local_labels[part_charges[k]]
            mean_atomisation[k] = mtmp.mean()      
        delta_labels = qi.get_label_delta(mean_atomisation, np.arange(molecule_size.sum()), alchemy_data, molecule_size)
        local_labels = local_labels - delta_labels
    
    opt_sigma, opt_lambda, min_error, std = qi.optimize_hypar_cv(local_reps, local_labels, tr_set, molecule_size, sigmas = np.logspace(-1, 4, 12).tolist(), lams = np.logspace(-15, 0, 16).tolist(), num_cv=num_cv)
    
    together = [opt_sigma, opt_lambda, min_error, std]
    results.append(together)

file = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/optimized_hyperpar_'

for idx,el in enumerate(results):
    with open(file+pl[idx]+'_delta_mic.txt', 'w') as f:
        f.write('optimized hyperparameters for label: {}; training set size: {}; number of crossvalidations: {}\n'.format(pl[idx], tr_set, num_cv))
        f.write('opt_sigma \t opt_lambda \t mean_error \t std\n')
        f.write('{}\t{}\t{}\t{}'.format(el[0], el[1], el[2], el[3]))
        
        
###############################################################################
#                      OLD VERSION/not FCHL; learn per element
###############################################################################

#base='/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/'
#pl = ['alch_pot'] #['alch_pot', 'atomic', 'atomisation']
#tr_set = 512
#num_cv = 1
#results = []
#
## load data into list, count number of atoms per molecule
#paths=qi.wrapper_alch_data()
#alchemy_data, molecule_size = qi.load_alchemy_data(paths)
#
#charges = qi.generate_label_vector(alchemy_data, molecule_size.sum(), 'charge')
#part_charges = qi.partition_idx_by_charge(charges)
#
#
## local data
#local_reps = qi.generate_atomic_representations(alchemy_data, molecule_size)
#local_labels = qi.generate_label_vector(alchemy_data, molecule_size.sum(), value=pl[0])
## get one element
#element_idx = part_charges[1.0]
#local_reps = local_reps[element_idx]
#local_labels = local_labels[element_idx]
#    
#opt_sigma, opt_lambda, min_error, std = qi.optimize_hypar_cv(local_reps, local_labels, tr_set, molecule_size, num_cv=num_cv)
#
#together = [opt_sigma, opt_lambda, min_error, std]
#results.append(together)
#
#file = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/optimized_hyperpar_'
#
#for idx,el in enumerate(results):
#    with open(file+pl[idx]+'2_mic.txt', 'w') as f:
#        f.write('optimized hyperparameters for label: {}; training set size: {}; number of crossvalidations: {}\n'.format(pl[idx], tr_set, num_cv))
#        f.write('opt_sigma \t opt_lambda \t mean_error \t std\n')
#        f.write('{}\t{}\t{}\t{}'.format(el[0], el[1], el[2], el[3]))