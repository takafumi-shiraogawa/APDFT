import warnings
import numpy as np
from ase.optimize.optimize import Optimizer


class STEEPEST_DESCENT(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'alpha': 70.0}

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None):
        """BFGS optimizer.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.
        """
        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep

        if self.maxstep > 1.0:
            warnings.warn('You are using a *very* large value for '
                          'the maximum step size: %.1f Å' % maxstep)

        if alpha is None:
            self.alpha = self.defaults['alpha']
        else:
            self.alpha = alpha

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

    def step(self, f=None):
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()

        scale_grad = 0.1
        # Conversion factor from hartree to eV
        har_to_ev = 27.21162
        dr = scale_grad * f / har_to_ev
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths)
        atoms.set_positions(r + dr)

    def determine_step(self, dr, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength

            dr *= scale

        return dr