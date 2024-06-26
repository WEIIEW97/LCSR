import torch
import torch.nn as nn
import torch.nn.functional as F
from ddr.conv import *
from typing import Optional, List

class Regularizer(torch.nn.Module):
    """
    Basic regularization function
    """

    def __init__(self):
        super(Regularizer, self).__init__()

    def forward(self, x):
        return self.grad(x)

    def energy(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def get_theta(self):
        """
        return all parameters of the regularization
        """
        return self.named_parameters()

    def get_vis(self):
        raise NotImplementedError


class SoftPlus_fun2(torch.autograd.Function):
    """returns Soft-Plus as well as its derivative in the forward path
    """
    @staticmethod
    def forward(ctx, x):
        soft_x = F.softplus(x) # Numerically stable verison `torch.log( 1+ torch.exp(x) )`
        grad_soft_x_dx = F.sigmoid(x)
        ctx.save_for_backward(grad_soft_x_dx)
        return soft_x, grad_soft_x_dx
    
    @staticmethod
    def backward(ctx: torch.Any, grad_fx, grad_dfx) -> torch.Any:
        sigmoid_x = ctx.saved_tensors[0]
        dfx = sigmoid_x # 1st derivative
        d2fx = dfx * (1-dfx) #2nd derivative
        return grad_fx*dfx+grad_dfx*d2fx
    
class SoftPlus2(nn.Module):
    """returns Softplus and its derivative in the forward path"""
    def __init__(self):
        super(SoftPlus2, self).__init__()
    def forward(self, x):
        """returns ln(1+exp(x), sigmoid(x) in the forward path
            softplus2_mod = SoftPlus2()
            sp_x,sp_x_grad = softplus2(x)
        """
        return SoftPlus_fun2().apply(x)
    
class Tanh_fun2(torch.autograd.Function):
    """returns Tanh and its derivative in the forward path"""
    @staticmethod
    def forward(ctx, x):
        tanhx = torch.tanh(x)
        ctx.save_for_backward(tanhx)
        fx = tanhx        # forward function
        dfx = 1-tanhx**2  # 1st derivative 
        return fx, dfx

    @staticmethod
    def backward(ctx, grad_fx, grad_dfx):
        tanhx = ctx.saved_tensors[0]
        tanhx_sq = tanhx**2        
        dfx = 1-tanhx_sq          # 1st derivative 
        d2fx = -(dfx) * 2 * tanhx # 2nd derivative
        return grad_fx * dfx + grad_dfx * d2fx

class Tanh2(nn.Module):
    """returns Tanh and its derivative in the forward path"""
    def __init__(self):
        super(Tanh2, self).__init__()
    def forward(self, x):
        """returns Tanh(x) and d/dx(tanh(x)) in the forward path
            tanh2_mod = Tanh2()
            tanhx,tanhx_grad = tanh2_mod(x)
        """
        return Tanh_fun2().apply(x)


class Tanh2Inference(nn.Module):
    """returns Tanh and its derivative in the forward path"""
    def __init__(self):
        super(Tanh2Inference, self).__init__()
    def forward(self, x):
        """returns Tanh(x) and d/dx(tanh(x)) in the forward path
            tanh2_mod = Tanh2()
            tanhx,tanhx_grad = tanh2_mod(x)
        """
        tanhx = torch.tanh(x)
        fx = tanhx        # forward function
        dfx = 1-tanhx**2  # 1st derivative
        return fx, dfx


class StudentT_fun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        d = 1+alpha*x**2
        return torch.log(d)/(2*alpha), x/d

    @staticmethod
    def backward(ctx, grad_in1, grad_in2):
        x = ctx.saved_tensors[0]
        d = 1+ctx.alpha*x**2
        return (x/d) * grad_in1 + (1-ctx.alpha*x**2)/d**2 * grad_in2, None


class StudentT2(nn.Module):
    def __init__(self,alpha):
        super(StudentT2, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        return StudentT_fun2().apply(x, self.alpha)

