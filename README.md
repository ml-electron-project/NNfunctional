NN-based functional
====

Parameters of NN-based functionals and example codes to call them (.py).

PySCF and Pytorch packages are needed to run the example codes.

To call the NRA functional, a part of the source code of PySCF should be replaced. 
Please contact to ngrttt[at]gmail.com

Note that the numerical integration in the xc energy and potential of "NRA.py" is not implemented efficiently. It will take a lot of computational time even for a small molecule. 

## Reference

@misc{1903.00238,
Author = {Ryo Nagai and Ryosuke Akashi and Osamu Sugino},
Title = {Completing density functional theory by machine-learning hidden messages from molecules},
Year = {2019},
Eprint = {arXiv:1903.00238},
}
