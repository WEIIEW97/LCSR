import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .regularizer import TDV
from typing import Optional, List, Dict

class Dataterm(nn.Module):
    """
    Basic dataterm function
    """

    def __init__(self):
        super(Dataterm, self).__init__()

    def forward(self, x, *args):
        raise NotImplementedError

    def energy(self):
        raise NotImplementedError

    def prox(self, x, *args):
        raise NotImplementedError

    def grad(self, x, *args):
        raise NotImplementedError
    
class L2Dataterm(Dataterm):
    def __init__(self):
        super(L2Dataterm, self).__init__()

    def energy(self, u, u0, lmda):
        return 0.5*lmda*(u-u0)**2

    def prox(self, u, u0, tau, lmda):
        return (u + tau*lmda*u0) / (1 + tau*lmda)                                                                                                                                                                                                                                                                                                                                                                                     

    def grad(self, u, u0, lmda):
        return lmda*(u-u0)

class WeightedL1Dataterm(Dataterm):
    def __init__(self):
        super(WeightedL1Dataterm, self).__init__()

    def energy(self, u, u0, gamma, w):
        return gamma*torch.sum(w * torch.abs(u - u0))

    def prox(self, u, u0, gamma, tau, w):
        return u0+torch.max(0, torch.abs(u-u0)-tau*gamma*w)*torch.sign(u-u0)

    def grad(self, u, u0, gamma, w):
        return gamma*w*torch.sign(u-u0)    
    
class L1Dataterm(Dataterm):
    def __init__(self):
        super(L1Dataterm, self).__init__()

    def energy(self, u, u0, mu):
        return mu*(u-u0)
    
    def prox(self, u, u0, mu, tau):
        return u0+torch.max(0, torch.abs(u-u0)-tau*mu)*torch.sign(u-u0)
    
    def grad(self, u, u0, mu):
        return mu*torch.sign(u-u0)
    

class CollaborativeDataFidelity(Dataterm):
    r"""
    This dataterm uses the definition defined in paper formula(3).
    It uses the approximation that candidates are independent from each
    other.

    D(u;\theta)=\frac{lambda}{2}|u^{rgb}-f^0|^2 + 
                \mu|u^{c}-c|_1 + 
                \nu|u^d-\hat{d}|)_{u^c, 1}
    """

class VNet(nn.Module):
    """
    Variational Network
    """
    def __init__(self):
        pass
    