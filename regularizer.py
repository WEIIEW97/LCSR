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

def ActivationFactory(act_cfg):
    if act_cfg == "student-t":
        act = StudentT2(alpha=1)
    elif act_cfg == "tanh":
        act = Tanh2()
    elif act_cfg == "softplus":
        act = SoftPlus2()
    else:
        raise ValueError(f"wrong config for activation function! {act_cfg}")
    return act

class MicroBlock(nn.Module):
    act_prime:Optional[torch.Tensor]

    def __init__(self, num_features, bound_norm=False, invariant=False, act='student-t'):
        super(MicroBlock, self).__init__()

        self.conv1 = Conv2d(num_features, num_features, kernel_size=3, invariant=invariant, bound_norm=bound_norm, bias=False)
        self.act = ActivationFactory(act)
        self.conv2 = Conv2d(num_features, num_features, kernel_size=3, invariant=invariant, bound_norm=bound_norm, bias=False)

        # save the gradient of the the activation function for the backward path
        self.act_prime = None

    def forward(self, x):
        a, ap = self.act(self.conv1(x))
        self.act_prime = ap
        x = x+self.conv2(a)
        return x
    
    def backward(self, grad_out):
        act_prime = self.act_prime
        assert act_prime is not None, 'call forward before calling backward!' # required for TorchScript export

        out = grad_out + self.conv1.backward(act_prime*self.conv2.backward(grad_out))
        if not act_prime.requires_grad:
            self.act_prime = None
        return out
    
class MacroBlock(nn.Module):
    def __init__(self, num_features, num_scales=3, multiplier=1, bound_norm=False, invariant=False, act="student-t", act_fin="student-t"):
        super(MacroBlock, self).__init__()

        self.num_scales = num_scales

        # micro blocks
        self.mb = []
        # final micro block (scale=0) has different activation
        self.mb.append(nn.ModuleList([
            MicroBlock(num_features*multiplier**0, bound_norm=bound_norm, invariant=invariant, act=act),
            MicroBlock(num_features*multiplier**0, bound_norm=bound_norm, invariant=invariant, act=act_fin)
        ]))
        for i in range(1, num_scales-1):
            b = nn.ModuleList([
                MicroBlock(num_features*multiplier**i, bound_norm=bound_norm, invariant=invariant, act=act),
                MicroBlock(num_features*multiplier**i, bound_norm=bound_norm, invariant=invariant, act=act)
            ])
            self.mb.append(b)
        # the coarsest scale has only one microblock
        self.mb.append(nn.ModuleList([
            MicroBlock(num_features * multiplier**(num_scales-1), bound_norm=bound_norm, invariant=invariant, act=act)
        ]))
        self.mb = nn.ModuleList(self.mb)

        # down/up sample
        self.conv_down = []
        self.conv_up = []
        for i in range(1, num_scales):
            self.conv_down.append(
                ConvScale2d(num_features * multiplier**(i-1), num_features * multiplier**i, kernel_size=3, bias=False, invariant=invariant, bound_norm=bound_norm)
            )
            self.conv_up.append(
                ConvScaleTranspose2d(num_features * multiplier**(i-1), num_features * multiplier**i, kernel_size=3, bias=False, invariant=invariant, bound_norm=bound_norm)
            )
        self.conv_down = torch.nn.ModuleList(self.conv_down)
        self.conv_up = torch.nn.ModuleList(self.conv_up)

    def forward(self, x:List[torch.Tensor]):
        assert len(x) == self.num_scales
        
        # down scale and feature extraction
        for i, (micro_blocks, conv_down) in enumerate(zip(self.mb, self.conv_down)):
            # 1st micro block of scale
            x[i] = micro_blocks[0](x[i])
            # down sample for the next scale
            x_i_down = conv_down(x[i])
            # residual connection
            x[i+1] = x[i+1]+x_i_down

        # on the coarsest(smallest) scale we only have one micro block
        x[-1] = self.mb[-1][0](x[-1])

        # up scale the features
        return self._forward_std(x)
    
    @torch.jit.unused
    def _forward_std(self, x:List[torch.Tensor]) -> List[torch.Tensor]:
        for i in range(self.num_scales-1)[::-1]:
            # first upsample the next coarsest scale
            x_ip1_up = self.conv_up[i](x[i+1], x[i].shape)
            # skip connection
            x[i] = x[i] + x_ip1_up
            # 2nd micro block of scale
            x[i] = self.mb[i][1](x[i])
        return x
    
    def backward(self, grad_x: List[torch.Tensor]):
        # backward of up scale the features
        for i, (micro_blocks, conv_up) in enumerate(zip(self.mb, self.conv_up)):
            # 2nd micro block of scale
            grad_x[i] = micro_blocks[1].backward(grad_x[i])
            # first upsample the next coarsest scale
            grad_x_ip1_up = conv_up.backward(grad_x[i])
            grad_x[i+1] = grad_x[i+1] + grad_x_ip1_up

        # on the coarsest scale we only have one micro block
        grad_x[-1] = self.mb[-1][0].backward(grad_x[-1])

        # down scale and feature extraction
        # up scale the features
        return self._backward_std(grad_x)
    
    @torch.jit.unused
    def _backward_std(self, grad_x:List[torch.Tensor]) -> List[torch.Tensor]:
        # down scale and feature extraction
        for i in range(self.num_scales-1)[::-1]:
            # down sample for the next scale
            grad_x_i_down = self.conv_down[i].backward(grad_x[i+1], grad_x[i].shape)
            grad_x[i] = grad_x[i] + grad_x_i_down
            # 1st micro block of scale
            grad_x[i] = self.mb[i][0].backward(grad_x[i])
        return grad_x
    

class TDV(Regularizer):
    """
    total deep variation (TDV) regularizer
    """
    # potential referes to the negative log-likelihood
    @staticmethod
    def potential_linear(x):
        return x

    @staticmethod
    def potential_student_t(x):
        return 0.5*torch.log(1+x**2)

    @staticmethod
    def potential_tanh(x):
        return torch.log(torch.cosh(x))

    @staticmethod
    def activation_linear(x):
        return torch.ones_like(x)

    @staticmethod
    def activation_student_t(x):
        return x/(1+x**2)

    @staticmethod
    def activation_tanh(x):
        return torch.tanh(x)
    
    def __init__(self, in_channels, num_features, num_scales, multiplier, num_mb, act, act_fin, psi, zero_mean=True):
        self.in_channels = in_channels
        self.num_features = num_features
        self.num_scales = num_scales
        self.multiplier = multiplier
        self.num_mb = num_mb
        self.act = act
        self.act_fin = act_fin
        self.zero_mean = zero_mean
        self.psi = psi  # for the type of activation function

        self.tau = 1

        if self.psi == 'linear':
            self.out_pot = TDV.potential_linear
            self.out_act = TDV.activation_linear
        elif self.psi == 'student-t':
            self.out_pot = TDV.potential_student_t
            self.out_act = TDV.activation_student_t
        elif self.psi == 'tanh':
            self.out_pot = TDV.potential_tanh
            self.out_act = TDV.activation_tanh
        else:
            raise ValueError(f"Unknown value for psi:{self.psi}")
        
        # construct the regularizer
        self.K1 = Conv2d(self.in_channels, self.num_features, 3, zero_mean=self.zero_mean, invariant=False, bound_norm=True, bias=False)

        self.mb = torch.nn.ModuleList([MacroBlock(self.num_features, num_scales=self.num_scales, bound_norm=False, invariant=False, multiplier=self.multiplier, act=self.act,
                                                  act_fin=self.act_fin if i == self.num_mb-1 else self.act) 
                                        for i in range(self.num_mb)])

        self.KN = Conv2d(self.num_features, 1, 1, invariant=False, bound_norm=False, bias=False)

    def _transformation(self, x, embedding:Optional[List[torch.Tensor]]=None):
        # extract features
        x = self.K1(x)
        # apply mb
        x_list = [x,] + (self.num_scales-1)*[torch.zeros(1, device=x.device)]
        if embedding is not None:
            for s in range(self.num_scales):
                x_list[s] = embedding[s] + x_list[s]

        for macro_block in self.mb:
            x_list = macro_block(x_list)
        
        out = self.KN(x_list[0])

        return out
    
    def _activation(self, x, lambda_mul:Optional[torch.Tensor]=None):
        # scale by the number of features
        # return torch.ones_like(x) / self.num_features
        act = self.out_act(x) / self.num_features
        if lambda_mul is not None:
            act = act * lambda_mul
        return act
    
    def _potential(self, x):
        return self.out_pot(x) / self.num_features
    
    def _transformation_T(self, grad_out):
        # compute the output
        grad_x = self.KN.backward(grad_out)
        grad_x_list = [grad_x,] + (self.num_scales-1)*[torch.zeros(1, device=grad_x.device)]
        # apply mb
        return self._transformation_T_std(grad_x_list)
    
    @torch.jit.unused
    def _transformation_T_std(self, grad_x_list:List[torch.Tensor]):
        # apply mb
        for i in range(self.num_mb)[::-1]:
            grad_x_list = self.mb[i].backward(grad_x_list)

        # extract features
        grad_x = self.K1.backward(grad_x_list[0])
        return grad_x
    
    def energy(self, x, embedding=None, lambda_mul=None):
        x = self._transformation(x, embedding=embedding)
        if lambda_mul is not None:
            return self._potential(x) * lambda_mul
        else:
            return self._potential(x)
        
    def grad(self, x, get_energy:bool=False, embedding=None, lambda_mul=None, apply_lambda_mul=True):
        # compute the energy
        x = self._transformation(x, embedding=embedding)
        energy = None
        if get_energy:
            energy = self._potential(x)
            if  apply_lambda_mul and lambda_mul is not None:
                energy = energy * lambda_mul
        # and its gradient
        x = self._activation(x, lambda_mul)
        grad = self._transformation_T(x)
        return energy, grad