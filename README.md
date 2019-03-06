NN-based functional
====

Weight parameters of NN-based functionals and example codes of using them (.py).

PySCF and Pytorch packages are needed to run the example codes.

In order to use the NR functional, please replace 
"(where you installed PySCF)/pyscf/dft/numint.py"
into numint.py in this folder.
Note that the numerical integration in the xc energy and potential of "NRA.py" is not implemented efficiently. It will take a long time even for a small molecule. 

## How to cite

Please cite the following work if you use or develop a functional related to this work.

@misc{1903.00238,
Author = {Ryo Nagai and Ryosuke Akashi and Osamu Sugino},
Title = {Completing density functional theory by machine-learning hidden messages from molecules},
Year = {2019},
Eprint = {arXiv:1903.00238},
}
