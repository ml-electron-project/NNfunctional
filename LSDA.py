# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np
from pyscf import gto, dft
import math
import torch 
from torch.autograd import Variable 
import torch.nn as nn 
import torch.nn.functional as F 


# definition of target molecule #
mol = gto.Mole()
mol.verbose = 4

mol.atom="""8            .000000     .000000     .119262
1            .000000     .763239    -.477047
1            .000000    -.763239    -.477047"""
mol.charge=0
mol.spin  =0
mol.basis = "6-31G"
mol.build()

# definition of NN structure #
hidden=100
print("hidden nodes= "+str(hidden))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 1)


    def forward(self, x):
        t=torch.empty((x.shape[0],2))
        unif=(x[:,1]+x[:,0]+1e-7)**(1.0/3)
        t[:,0]=unif
        t[:,1]=((1+torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(4.0/3)+(1-torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(4.0/3))*0.5

        #print(t)
        logt=torch.log(t)#/scale.float()
        g1=F.elu(self.fc1(logt))
        g2=F.elu(self.fc2(g1))
        g3=F.elu(self.fc3(g2))
        #g4=F.elu(self.fc4(g3))
        eunif=(x[:,1]+x[:,0]).view(-1,1)**(1.0/3)
        spinscale=(((1+torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(4.0/3)+(1-torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(4.0/3))*0.5).view(-1,1)
        g4=-(F.elu(self.fc4(g3))+1)*eunif*spinscale
        return g4


#loading NN weights
model = Net()
model.load_state_dict(torch.load("NNLSDA",map_location='cpu'))

# definition of functional #

def eval_xc(xc_code, rho, spin, relativity=0, deriv=2, verbose=None):
    if spin!=0:
        rho01=rho[0]
        rho02=rho[1]
        
    else:
        rho01=rho02=rho*0.5

    rho0=np.concatenate((rho01.reshape((-1,1)),rho02.reshape((-1,1))),axis=1)
    N=rho0.shape[0]
    x=Variable(torch.Tensor(rho0),requires_grad=True)
    pred_exc=model(x)
    exc=pred_exc.data[:,0].numpy()
    eneden=torch.dot(pred_exc[:,0],x[:,0]+x[:,1])
    eneden.backward()
    grad=x.grad.data.numpy()

    if spin!=0:
        vrho=np.hstack((grad[:,0].reshape((-1,1)),grad[:,1].reshape((-1,1))))

    else:
        vlapl=np.zeros(N)    
        vrho=(grad[:,0]+grad[:,1])/2
    vxc=(vrho, None, None, None)
    return exc, vxc, None, None

# DFT calculation #
if mol.spin==0:
    mfl = dft.RKS(mol)
else:
    mfl = dft.UKS(mol)

mfl = mfl.define_xc_(eval_xc, 'LDA')
mfl.kernel()
