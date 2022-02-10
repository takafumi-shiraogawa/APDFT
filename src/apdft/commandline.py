#!/usr/bin/env python
import argparse
import enum

import numpy as np
import basis_set_exchange as bse

import apdft
import apdft.settings as aconf
import apdft.calculator as acalc
import apdft.physics as ap


def entry_cli():
    # load configuration
    parser = build_main_commandline()
    conf = aconf.Configuration()
    # try to read "apdft.conf"
    # If there is no "apdft.conf", conf becomes None.
    conf.from_file()
    # mode and modeshort are energies (or energies_geometries)
    # and *.xyz, respectively in the first invocation of apdft
    # (or cli.py)
    mode, modeshort, modeshort_2, conf = parse_into(parser, configuration=conf)

    # execute
    if mode == "energies":
        mode_energies(conf, modeshort)
    elif mode == "energies_geometries":
        mode_energies_geometries(conf, modeshort, modeshort_2)
    else:
        apdft.log.log("Unknown mode %s" % mode, level="error")

    # emphasize warnings
    warningcount = apdft.LOG_LEVEL_USAGE.get("warning", 0)
    if warningcount > 0:
        apdft.log.log(
            "This run had warnings. The results might be incomplete at worst. In absence of errors, existing results are ok.",
            warningcount=warningcount,
            level="warning",
        )

    # highlight errors
    errorcount = apdft.LOG_LEVEL_USAGE.get("error", 0)
    returncode = 0
    if errorcount > 0:
        apdft.log.log(
            "This run had errors. The results are not to be trusted.",
            errorcount=errorcount,
            level="error",
        )
        returncode = 1

    # persist configuration
    conf.to_file()
    return returncode


def parse_target_list(lines):
    """ Separates an explicit target list.

    Accepted values: - for a missing atom, labels, nuclear charges. One target per line, comma separated."""
    ret = []
    for lidx, line in enumerate(lines):
        res = []
        for part in line.strip().split(","):
            if part == "-":
                res.append(0)
                continue
            try:
                res.append(int(part))
                continue
            except:
                pass
            try:
                res.append(bse.lut.element_Z_from_sym(part))
            except:
                apdft.log.log(
                    "Unknown element label in target list. Skipping entry.",
                    elementlabel=part,
                    lineno=lidx + 1,
                    level="warning",
                )
                break
        if res == []:
            continue
        if ret != [] and len(ret[0]) != len(res):
            apdft.log.log(
                "Line with different number of atoms found. Skipping entry.",
                lineno=lidx + 1,
                level="warning",
            )
            continue
        ret.append(res)
    return np.array(ret)


def mode_energies(conf, modeshort=None):
    # select QM code
    calculator_options = conf.apdft_method, conf.apdft_basisset, conf.debug_superimpose
    calculator = conf.energy_code.get_calculator_class()(*calculator_options)

    # parse input
    try:
        nuclear_numbers, coordinates = apdft.read_xyz(conf.energy_geometry)
    except FileNotFoundError:
        apdft.log.log(
            'Unable to open input file "%s".' % conf.energy_geometry, level="error"
        )
        return

    # Parse optional targetlist
    if len(conf.apdft_targets) != 0:
        with open(conf.apdft_targets) as fh:
            targetlist = parse_target_list(fh.readlines())
    else:
        targetlist = None

    # call APDFT library
    derivatives = ap.APDFT(
        conf.apdft_maxorder,
        nuclear_numbers,
        coordinates,
        ".",
        calculator,
        conf.apdft_maxcharge,
        conf.apdft_maxdz,
        conf.apdft_includeonly,
        targetlist,
        conf.apdft_specifytargets,
        conf.apdft_targetatom,
        conf.apdft_targetpositions,
        conf.apdft_readtargetpath,
        conf.apdft_cartesian,
        conf.apdft_deltaz,
        conf.apdft_deltar,
        conf.apdft_lambda,
        conf.apdft_derivative
    )

    cost, coverage = derivatives.estimate_cost_and_coverage()
    if conf.debug_validation:
        cost += coverage
    apdft.log.log(
        "Cost estimated.",
        number_calculations=cost,
        number_predictions=coverage,
        level="RESULT",
    )
    if not conf.energy_dryrun:
        derivatives.prepare(conf.debug_validation)
        derivatives.analyse(conf.debug_validation)


def mode_energies_geometries(conf, modeshort=None, modeshort_2=None):
    apdft.log.log(
        'Mode "energies_geometries"', level="info"
    )

    # select QM code
    calculator_options = conf.apdft_method, conf.apdft_basisset, conf.debug_superimpose
    calculator = conf.energy_code.get_calculator_class()(*calculator_options)
    # In this mode, the default calculator is PYSCF.

    # parse input for a reference molecule
    try:
        nuclear_numbers, coordinates = apdft.read_xyz(conf.energy_geometry)
    except FileNotFoundError:
        apdft.log.log(
            'Unable to open input file "%s".' % conf.energy_geometry, level="error"
        )
        return

    # parse input for a target geometry
    try:
        target_nuclear_numbers, target_coordinates = \
            apdft.read_xyz(conf.energy_geometry2)
    except FileNotFoundError:
        apdft.log.log(
            'Unable to open input file "%s".' % conf.energy_geometry2, level="error"
        )
        return

    # If target molecule is different from the reference
    if (target_nuclear_numbers != nuclear_numbers).any():
        apdft.log.log(
            'Atoms are different in reference "%s" and target"%s".' % \
            (conf.energy_geometry, conf.energy_geometry2), level="error"
        )
        return

    # The lists of target atoms are not necessary.
    del target_nuclear_numbers

    # Parse optional targetlist
    if len(conf.apdft_targets) != 0:
        with open(conf.apdft_targets) as fh:
            targetlist = parse_target_list(fh.readlines())
    else:
        targetlist = None

    # call APDFT library
    derivatives = ap.APDFT(
        conf.apdft_maxorder,
        # e.g., [7, 7]
        nuclear_numbers,
        # e.g., [[0. 0. 0.] [0. 0. 1.]]
        coordinates,
        ".",
        calculator,
        conf.apdft_maxcharge,
        conf.apdft_maxdz,
        conf.apdft_includeonly,
        targetlist,
        conf.apdft_specifytargets,
        conf.apdft_targetatom,
        conf.apdft_targetpositions,
        conf.apdft_readtargetpath,
        conf.apdft_cartesian,
        conf.apdft_deltaz,
        conf.apdft_deltar,
        conf.apdft_lambda,
        conf.apdft_derivative
    )

    # cost is different from that of estimate_cost_and_coverage()
    # since the derivatives for the nuclear distribution are required.
    cost, coverage = derivatives.estimate_cost_and_coverage_general()
    if conf.debug_validation:
        cost += coverage
    apdft.log.log(
        "Cost estimated.",
        number_calculations=cost,
        number_predictions=coverage,
        level="RESULT",
    )

    # For the purpose of generating a list of target molecules (target_molecules.inp)
    # This is not same with
    # if conf.apdft_specifytargets and conf.energy_dryrun:
    # since target_molecules.inp is generated if conf.apdft_readtargetpath is not None.
    if conf.apdft_specifytargets and conf.apdft_readtargetpath == "None":
        exit()

    if not conf.energy_dryrun:
        derivatives.prepare_general(target_coordinates, conf.debug_validation)
        derivatives.analyse_general(target_coordinates, conf.debug_validation)


def build_main_commandline(set_defaults=True):
    """ Builds an argparse object of the user-facing command line interface."""

    c = apdft.settings.Configuration()

    parser = argparse.ArgumentParser(
        description="QM calculations on multiple systems at once.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # mode selection
    # "energies_geometries" is added to treat molecular
    # geometry changes
    modes = ["energies", "energies_geometries"]
    parser.add_argument(
        "mode",
        choices=modes,
        nargs=1,
        help="Mode to use. Supported: %s" % ", ".join(modes),
        metavar="mode",
    )

    # allow for shortcut where a mode gets one single argument of any kind
    parser.add_argument("modeshort", type=str, nargs="?",
                        help=argparse.SUPPRESS)

    parser.add_argument("modeshort_2", type=str, nargs="?",
                        help=argparse.SUPPRESS)

    # options
    for category in sorted(c.list_sections()):
        group = parser.add_argument_group(category)
        for option_name in c.list_options(category):
            option = c[option_name]
            choices = None
            try:
                if issubclass(option.get_validator(), enum.Enum):
                    choices = list(option.get_validator())
            except TypeError:
                pass
            default = None
            if set_defaults:
                default = option.get_value()
            group.add_argument(
                "--%s" % option.get_attribute_name(),
                type=option.get_validator(),
                help=option.get_description(),
                choices=choices,
                default=default,
                metavar="",
            )

    return parser


def parse_into(parser, configuration=None, cliargs=None):
    """ Updates the configuration with the values specified on the command line.

    Args:
        parser:         An argparse parser instance.
        configuration:  A :class:`apdft.settings.Configuration` instance. If `None`, a new instance will be returned.
        args:           List of split arguments from the command line.
    Returns:
        Mode of operation, single optional argument, updated configuration."""

    if configuration is None:
        configuration = apdft.settings.Configuration()

    # help specified?
    args = parser.parse_args(cliargs)
    nodefaultparser = build_main_commandline(set_defaults=False)
    args = nodefaultparser.parse_args(cliargs)
    valid_options = configuration.list_options()
    mode = None
    modeshort = None  # single argument for a mode for simplicity
    modeshort_2 = None
    for k, v in vars(args).items():
        if k in valid_options:
            if v is not None:
                configuration[k].set_value(v)
        else:
            if k == "mode":
                mode = v[0]
            elif k == "modeshort":
                modeshort = v
            elif k == "modeshort_2":
                modeshort_2 = v
            else:
                raise ValueError("Unknown argument found.")

    if mode == "energies":
        if modeshort is not None:
            configuration.energy_geometry = modeshort
    elif mode == "energies_geometries":
        if modeshort is not None:
            configuration.energy_geometry = modeshort
            configuration.energy_geometry2 = modeshort_2
    return mode, modeshort, modeshort_2, configuration
