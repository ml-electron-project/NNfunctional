NN-based functional
====

Parameters of NN-based functionals and example codes to call them (.py).

PySCF 1.6.2 and Pytorch 1.1.0 packages are required to run the example codes.

To call the NRA functional, (Where you installed PySCF)/pyscf/dft/numint.py should be replaced to the one in this folder.

Note that the numerical integration in the xc energy and potential of "NRA.py" is currently implemented in inefficient way. It will take a lot of computational time even for a small molecule. 

NOTICE:The distributed codes includes the work that is distributed in the Apache License 2.0.


21/07/2019 numint.py is modified for PySCF version 1.6.2.

## Reference
Cite the following paper in any related works.
@misc{1903.00238,
Author = {Ryo Nagai and Ryosuke Akashi and Osamu Sugino},
Title = {Completing density functional theory by machine-learning hidden messages from molecules},
Year = {2019},
Eprint = {arXiv:1903.00238},
}
