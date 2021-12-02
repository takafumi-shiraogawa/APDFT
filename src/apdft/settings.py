#!/usr/bin/env python
""" Manages settings and config file parsing."""
import enum
import configparser
import basis_set_exchange as bse


class CodeEnum(enum.Enum):
    MRCC = "MRCC"
    G09 = "G09"
    PYSCF = "PYSCF"
    PSI4 = "PSI4"

    def __str__(self):
        return self.value

    def get_calculator_class(self):
        if self.value == CodeEnum.MRCC.value:
            from apdft.calculator.mrcc import MrccCalculator

            return MrccCalculator
        if self.value == CodeEnum.G09.value:
            from apdft.calculator.gaussian import GaussianCalculator

            return GaussianCalculator
        if self.value == CodeEnum.PYSCF.value:
            from apdft.calculator.pyscf import PyscfCalculator

            return PyscfCalculator
        if self.value == CodeEnum.PSI4.value:
            from apdft.calculator.psi4 import Psi4Calculator

            return Psi4Calculator


def intelementrange(val):
    if val is None:
        return val
    if type(val) == list:
        return val
    ret = []
    for part in val.split(","):
        if "-" in part:
            subparts = part.split("-")
            ret += list(range(int(subparts[0]), int(subparts[1]) + 1))
            continue
        try:
            ret.append(int(part))
        except ValueError:
            bse.lut.element_Z_from_sym(part)
            ret.append(part)
    return ret


def boolean(val):
    if val == "True":
        return True
    if val == "False":
        return False
    return bool(val)


class Option:
    """ Represents a single configuration option. """

    def __init__(self, category, name, validator, default, description):
        self._category = category
        self._name = name
        self._description = description
        self._validator = validator
        self._value = self._validator(default)

    def get_attribute_name(self):
        return "%s_%s" % (self._category, self._name)

    def get_value(self):
        return self._value

    def get_validator(self):
        return self._validator

    def get_description(self):
        return self._description

    def set_value(self, value):
        self._value = self._validator(value)


class Configuration:
    """ A complete set of configuration values. Merges settings and default values. 

    Settings are referred to as category.variablename. """

    def __init__(self):
        options = [
            # Section apdft: relevant for all invocations
            Option(
                "apdft",
                "maxdz",
                int,
                3,
                "Restricts target molecules to have at most this change in nuclear charge per atom",
            ),
            # apdft_maxorder = n - 1 means APDFTn
            # That is, apdft_maxorder = 2 corresponds to APDFT3,
            # and APDFT4 and higher-order ones can not be used,
            # possibly.
            Option("apdft", "maxorder", int, 2,
                   "Maximum alchemical expansion order"),
            # Sum of absolute values of nuclear charge changes at each atom
            Option(
                "apdft",
                "maxcharge",
                int,
                0,
                "Restricts target molecules to have at most this total molecular charge",
            ),
            Option("apdft", "basisset", str, "def2-TZVP",
                   "The basis set to be used"),
            Option("apdft", "method", str, "CCSD", "Method to be used"),
            Option(
                "apdft",
                "includeonly",
                intelementrange,
                None,
                "Include only these atom indices, e.g. 0,1,5,7 or these atom types, e.g. B,C,N. You can mix both.",
            ),
            Option(
                "debug",
                "validation",
                boolean,
                False,
                "Whether to perform validation calculations for all target molecules",
            ),
            Option(
                "debug",
                "superimpose",
                boolean,
                False,
                "Whether to superimpose atomic basis set functions from neighboring elements for fractional nuclear charges",
            ),
            # T.S.: default QM code changes from MRCC to PYSCF
            # Option("energy", "code", CodeEnum, "MRCC", "QM code to be used"),
            Option("energy", "code", CodeEnum, "PYSCF", "QM code to be used"),
            Option(
                "energy",
                "dryrun",
                boolean,
                False,
                "Whether to just estimate the number of targets",
            ),
            Option(
                "energy",
                "geometry",
                str,
                "inp.xyz",
                "XYZ file of the reference molecule",
            ),
            Option(
                "energy",
                "geometry2",
                str,
                "inp2.xyz",
                "XYZ file of the target molecule",
            ),
            Option(
                "apdft",
                "targets",
                str,
                "",
                "List of targets to be evaluated (one target per line, comma separated nuclear charges).",
            ),
            Option(
                "apdft",
                "cartesian",
                str,
                "z",
                "Cartesian axes along geometry changes. z for the Z axis and full for full-geometry changes",
            ),
            Option(
                "apdft",
                "deltaz",
                float,
                0.05,
                "Small difference of nuclear charge in finite differential for perturbed density.",
            ),
            Option(
                "apdft",
                "deltar",
                float,
                0.005,
                "Small difference of nuclear position in finite differential for perturbed density.",
            ),
            Option(
                "apdft",
                "lambda",
                float,
                1.0,
                "Alchemical interpolation between reference and target molecules (0 and 1, respectively).",
            ),
            Option(
                "apdft",
                "derivative",
                boolean,
                False,
                "Whether to perform calculations of analytical derivatives of potential energy with respect to nuclear coordinates.",
            ),
        ]
        self.__dict__["_options"] = {}
        for option in options:
            self.__dict__["_options"][option.get_attribute_name()] = option

    def __getattr__(self, attribute):
        """ Read access to configuration options."""
        return self.__dict__["_options"][attribute].get_value()

    def __setattr__(self, attribute, value):
        """ Write access to configuration options."""
        self.__dict__["_options"][attribute].set_value(value)

    def __getitem__(self, attribute):
        return self.__dict__["_options"][attribute]

    def list_options(self, section=None):
        """ Gives all configurable options for a given section."""
        options = [_ for _ in self.__dict__["_options"].keys()]
        if section is not None:
            options = [_ for _ in options if _.startswith("%s_" % section)]
        return options

    def list_sections(self):
        """ Returns a list of all sections."""
        return list(set([_.split("_")[0] for _ in self.__dict__["_options"].keys()]))

    def from_file(self):
        config = configparser.ConfigParser()
        config.read("apdft.conf")

        for section in config.sections():
            for option in config[section]:
                val = config[section][option]
                if val == "None":
                    val = None
                self[option].set_value(val)

    def to_file(self):
        config = configparser.ConfigParser()
        for section in sorted(self.list_sections()):
            vals = dict()
            for option in self.list_options(section):
                try:
                    vals[option] = self[option].get_value().name
                except AttributeError:
                    if type(self[option].get_value()) == list:
                        vals[option] = ",".join(
                            [str(_) for _ in self[option].get_value()]
                        )
                    else:
                        vals[option] = str(self[option].get_value())
            config[section] = vals
        with open("apdft.conf", "w") as configfile:
            config.write(configfile)
