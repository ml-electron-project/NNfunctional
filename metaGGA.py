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
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 1)
        #self.fc5 = nn.Linear(hidden, 1)


    def forward(self, x):
        t=torch.empty((x.shape[0],4))
        unif=(x[:,1]+x[:,0]+1e-7)**(1.0/3)
        t[:,0]=unif
        t[:,1]=((1+torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(4.0/3)+(1-torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(4.0/3))*0.5
        t[:,2]=((x[:,2]+x[:,4]+2*x[:,3])**0.5+1e-7)/unif**4
        ds=(1+torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(5.0/3)+(1-torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(5.0/3)
        t[:,3]=(x[:,5]+x[:,6]+1e-7)/(unif**5*ds)
        logt=torch.log(t)
        g1=F.elu(self.fc1(logt))
        g2=F.elu(self.fc2(g1))
        g3=F.elu(self.fc3(g2))
        eunif=(x[:,1]+x[:,0]).view(-1,1)**(1.0/3)
        spinscale=(((1+torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(4.0/3)+(1-torch.div((x[:,0]-x[:,1]),(x[:,1]+x[:,0]+1e-7)))**(4.0/3))*0.5).view(-1,1)
        g4=-(F.elu(self.fc4(g3))+1)*eunif*spinscale
        return g4

#loading NN weights
model = Net()
model.load_state_dict(torch.load("NNmGGA",map_location='cpu'))

# definition of functional #
def eval_xc(xc_code, rho, spin, relativity=0, deriv=2, verbose=None):
    if spin!=0:
        rho1=rho[0]
        rho2=rho[1]
        rho01, dx1, dy1, dz1, lapl1, tau1 = rho1[:6]
        rho02, dx2, dy2, dz2, lapl2, tau2 = rho2[:6]
        gamma1=dx1**2+dy1**2+dz1**2+1e-7
        gamma2=dx2**2+dy2**2+dz2**2+1e-7
        gamma12=dx1*dx2+dy1*dy2+dz1*dz2+1e-7
    else:
        rho0, dx, dy, dz, lapl, tau = rho[:6]
        gamma1=gamma2=gamma12=(dx**2+dy**2+dz**2)*0.25+1e-7
        rho01=rho02=rho0*0.5
        tau1=tau2=tau*0.5

    rho0=np.concatenate((rho01.reshape((-1,1)),rho02.reshape((-1,1)),gamma1.reshape((-1,1)),gamma12.reshape((-1,1)),gamma2.reshape((-1,1)),tau1.reshape((-1,1)),tau2.reshape((-1,1))),axis=1)
    
    N=rho0.shape[0]

    x=Variable(torch.Tensor(rho0),requires_grad=True)
    pred_exc=model(x)
    
    exc=pred_exc.data[:,0].numpy()#+0.015471944-0.002533411+0.001574637
    eneden=torch.dot(pred_exc[:,0],x[:,0]+x[:,1])
    eneden.backward()
    grad=x.grad.data.numpy()

    if spin!=0:
        vlapl=np.zeros((N,2))
        vrho=np.hstack((grad[:,0].reshape((-1,1)),grad[:,1].reshape((-1,1))))
        vgamma=np.hstack((grad[:,2].reshape((-1,1)),grad[:,3].reshape((-1,1)),grad[:,4].reshape((-1,1))))
        vtau=np.hstack((grad[:,5].reshape((-1,1)),grad[:,6].reshape((-1,1))))
    else:
        vlapl=np.zeros(N)    
        vrho=(grad[:,0]+grad[:,1])/2
        vgamma=(grad[:,2]+grad[:,4]+grad[:,3])/4
        vtau=(grad[:,5]+grad[:,6])/2
    vxc=(vrho,vgamma,vlapl,vtau)
    return exc, vxc, None, None


# DFT calculation #
if mol.spin==0:
    mfl = dft.RKS(mol)
else:
    mfl = dft.UKS(mol)

mfl = mfl.define_xc_(eval_xc, 'MGGA')
mfl.kernel()
