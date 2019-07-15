NN-based functional
====

Parameters of NN-based functionals and example codes to call them (.py).

PySCF and Pytorch packages are needed to run the example codes.

To call the NRA functional, (Where you installed PySCF)/PySCF/dft/numint.py should be replaced.
Please contact to ngrttt[at]gmail.com

Note that the numerical integration in the xc energy and potential of "NRA.py" is currently implemented in inefficient way. It will take a lot of computational time even for a small molecule. 

NOTICE:The distributed codes includes the work that is distributed in the Apache License 2.0.

## Reference
Cite the following paper in any related works.
@misc{1903.00238,
Author = {Ryo Nagai and Ryosuke Akashi and Osamu Sugino},
Title = {Completing density functional theory by machine-learning hidden messages from molecules},
Year = {2019},
Eprint = {arXiv:1903.00238},
}
