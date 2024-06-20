import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

import numpy as np

from .tdv import TDV
from .utils import log_info
from typing import Optional, List, Dict

def add_projection_scalar(scalar_param, init=1., min=0, max=1000, lr_mul=None, name_tag=None, mode=''):
    """ sets the value of the parameter to the initial vlaue from the config and
        adds a projection function to the parameter.
        Requires Block Adam optimizer to work!
        Right after the GD step of ADAM the .proj function will be called by BlockAdam
        This allows to limit the data to an allowed range
    """
    assert type(scalar_param) == torch.nn.parameter.Parameter, f"parameter must be a pytorch parameter"
    scalar_param.data = torch.tensor(init, dtype=scalar_param.dtype)
    # add a positivity constraint
    scalar_param.proj = lambda: scalar_param.data.clamp_(min, max)
    if lr_mul is not None:
        scalar_param._lr_mul = lr_mul
    if name_tag is not None:
        scalar_param._name_tag = name_tag
    if mode:
        assert mode == 'learned', f"if constraints are used, the mode must be set to learned!"

def add_parameter(obj, name, config):
    assert name in config, f"Parameter '{name}' not found in config"
    assert 'mode' in config[name], f"Parameter '{name}' does not have a mode setting: '{config[name]}'"
    assert not hasattr(obj, name), f"A parameter '{name}' is already present on ojb:'{obj}''"
    if config[name]['mode'] == 'fixed':  # Fixed Mode:=> generate a buffer that will be saved with model
        scalar_param = torch.tensor(config[name]['init'])
        obj.register_buffer(name, torch.tensor(config[name]['init']))
    elif config[name]['mode'] == 'learned':  # Learned Mode:=> generate a Parameter
        scalar_param = torch.tensor(1.0) # must be floating point type to get gradients
        scalar_param = torch.nn.Parameter(scalar_param)
        add_projection_scalar(scalar_param, name_tag=name,  **config[name])
        setattr(obj, name, scalar_param)
    else:
        raise RuntimeError(f"mode {config[name]['mode']} unknown! for parameter {name} and modeconfig {config[name]}")
    scalar_param._is_custom_scalar_param = True

class Dataterm(nn.Module):
    """
    Basic dataterm function
    """

    def __init__(self, config):
        super(Dataterm, self).__init__()

    def forward(self, x, *args):
        raise NotImplementedError

    def energy(self):
        raise NotImplementedError

    def prox(self, x, *args):
        raise NotImplementedError

    def grad(self, x, *args):
        raise NotImplementedError


class L2DenoiseDataterm(Dataterm):
    def __init__(self, config):
        super(L2DenoiseDataterm, self).__init__(config)

    def energy(self, u, u0, lmda):
        return 0.5*lmda*(u-u0)**2

    def prox(self, u, u0, tau, lmda):
        return (u + tau*lmda*u0) / (1 + tau*lmda)                                                                                                                                                                                                                                                                                                                                                                                     

    def grad(self, u, u0, lmda):
        return lmda*(u-u0)
    

class WeightedL1DenoiseDataterm(Dataterm):
    def __init__(self, config):
        super(WeightedL1DenoiseDataterm, self).__init__(config)

    def energy(self, u, u0, gamma, w):
        return gamma*torch.sum(w * torch.abs(u - u0))

    def prox(self, u, u0, gamma, tau, w):
        return u0+torch.max(0, torch.abs(u-u0)-tau*gamma*w)*torch.sign(u-u0)

    def grad(self, u, u0, gamma, w):
        return gamma*w*torch.sign(u-u0) 
    

class FidelityDataterm(Dataterm):
    def __init__(self, config):
        super(FidelityDataterm, self).__init__(config)

    def energy(self): 
        pass

class VNet(torch.nn.Module):
    """
    Variational Network
    """

    def __init__(self, config, efficient=False):
        super(VNet, self).__init__()

        self.efficient = efficient
        
        self.S = config['S']

        # setup the stopping time
        add_parameter(self, 'T', config)
        # setup the regularization parameter * stopping time
        add_parameter(self, 'taulambda', config)

        # setup the regularization
        R_types = {
            'tdv': TDV,
        }
        self.R = R_types[config['R']['type']](config['R']['config'])

        # setup the dataterm
        self.use_prox = config['D']['config']['use_prox']
        D_types = {
            'denoise':L2DenoiseDataterm,         
        }
        self.D = D_types[config['D']['type']](config['D']['config'])
        self.D.S = 0                  # placeholders need to be updated before calling D.prox
        self.D.tau = torch.tensor(0)  # placeholders need to be updated before calling D.prox

        self.seperate_stages_count:Optional[int] = None

    def set_separate_stages_count(self, stages_count):
        """ Use this function to change the amount of refinement steps, but keeping T/S as trained"""
        self.seperate_stages_count = stages_count
        log_info(f"Using a stages count that is independent from S! S={self.S}, stages={self.seperate_stages_count}")


    def forward(self, x, z:Dict[str, torch.Tensor], get_grad_R: bool=False, get_gradR_L: bool=False, embedding: Optional[List[torch.Tensor]]=None, get_energy: bool=False, get_tau_Lgrad: bool=False):

        if self.seperate_stages_count is None:
            stages = self.S+1
        else:
            stages = self.seperate_stages_count+1 # Seperate stages from T for inference

        # x_all = x.new_empty((stages,*x.shape))
        x_all = [x.new_empty(x.shape),]*(stages)
        x_all[0] = x  # element 0 is initial image

        grad_R_all: Optional[torch.Tensor]   = None 
        if get_grad_R:
            grad_R_all = x.new_empty((stages-1,) +x.shape)

        # define the step size
        tau = self.T / self.S
        taulambda = self.taulambda / self.S
        L = []
        tau_Lgrad = []
        energies_reg : List[ torch.Tensor] = []
        energies_dat : List[Dict[str, torch.Tensor]] = []
        Lipschitz: Dict[str, List[torch.Tensor]] = {}
        
        lambda_tdv_mul = z['lambda_tdv_mul'] if 'lambda_tdv_mul' in z else None
        for s in range(1, stages):  # compute a single step
            # 1. Compute Gradient of Regularizer (wrap into gradient checkpointing if desired)
            energy_reg, grad_R = self.R.grad( x,
                            embedding=embedding,
                            lambda_mul=lambda_tdv_mul,
                            apply_lambda_mul=False, # Do not apply lambda_mul to the energy here, so that we can visualize it, ot it later
                            get_energy=get_energy)
                            
            if energy_reg is not None:
                energies_reg += [energy_reg]
            

            ################
            # 2. Compute update on Dataterm 
            if self.use_prox:
                self.D.S = self.S
                self.D.tau = tau
                x, en_dat = self.D.prox(x, tau , grad_R, z, taulambda, get_energy=True)
                energies_dat.append(en_dat)
            else:
                x = x - tau * grad_R - taulambda * self.D.grad(x, z)

            if get_grad_R and (grad_R_all is not None):
                grad_R_all[s-1] = grad_R
            x_all[s] = x

        
        if stages == 1:
            Lipschitz.clear()
        if stages == 2:
            L = [torch.zeros([x.shape[0],1,1,1])]
            Lipschitz.clear()
        if stages == 2:
            tau_Lgrad = [torch.zeros([x.shape[0],1,1,1])]

        return x_all, Lipschitz, energies_reg, energies_dat

    def set_end(self, s):
        assert 0 < s
        self.S = s

    def extra_repr(self):
        s = "S={S}"
        return s.format(**self.__dict__)