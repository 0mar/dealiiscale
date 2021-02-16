# Parameter sets

This directory contains the parameter sets for the executables.
Each executable accepts a parameter file from which it reads the data for the PDE system under consideration.
The parameter sets in this directory are mainly stored for reproducibility of the paper results and/or debugging the code. If one wants to test new system, it is most likely easiest to generate new parameter sets.

## Manufactured solutions

The Python scripts use the SymPy library to generate parameter sets for the manufactured systems.
It is supposed to make the testing of different sets of parameters painless; just use the wizard to prescribe a set of solutions and the data automatically follows.
Specifically, `prepare_bio.py` can read `.ini` files (see the folder `solutions/`) which remove the need of typing new solutions in the wizard prompt.

Honestly, if you came this far, you must really want to use the code. In case you're stuck, just send me an email.
