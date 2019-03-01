NN-based functional
====

Weight parameters of NN-based functionals and example codes of using them (.py).

PySCF and Pytorch packages are needed to run the example codes.

In order to use the NR functional, please replace 
"(where you installed PySCF)/pyscf/dft/numint.py"
into numint.py in this folder.
The numerical integration in the xc-potential of "NRA.py" is not implemented efficiently. It will take a long time even for a small molecule. 

## How to cite

Please cite the following work if you use or develop a functional related to this work.
