import sys

import torch.autograd as autograd
from torch.autograd.function import once_differentiable
from torch.nn import Module
import torch
from os.path import join, split, isfile

from typing import Tuple,List


# Torch Script compatible way: load via load_library into torch.ops.add_one_op.add_one_forward
# Will work inside the installed package only
def get_lib_path():
    """ returns the path to the binary library file"""
    if sys.platform.startswith("win"):
        lib_name = 'pad2d_op.dll'
    elif sys.platform.startswith("linux"):
        lib_name = 'libpad2d_op.so'

    lib_path = join(get_module_path(), lib_name)
    return lib_path

def get_include_path():
    """ returns the path to where the module is installed"""
    include_path = join(get_module_path(), "include")
    return include_path

def get_module_path():
    """ returns the path to where the module is installed"""
    module_path = split(__file__)[0]
    return module_path

lib_path = get_lib_path()
if not isfile(lib_path):
    if isfile( join(split(__file__)[0], 'source_dir_indicator.txt')  ):
        raise RuntimeError("Could not find {lib_path}\nThe library cannot be imported from within git repository directory! Change the directory!")
    else:
        raise RuntimeError("Could not find {lib_path}")
torch.ops.load_library(lib_path)
    
############################################################################################################################
############################################################################################################################  

def pad2d(data:torch.Tensor, padding:List[int], mode:str):
    """
    pad2d (data:Tensor, padding:List[int], mode:str)
    ---

    Custom 2D Padding function with accompanying transpose operator
    pad2d(tensor, padding=[1,1,1,1], mode='symmetric')

        padding: Padding [left,right,top,bottom]
        mode: ['symmetric','reflect','replicate']

        <Pad2d(x), y> = < x , Pad2dT(y) >  for all x,y of the respective domains
    """
    mode = conv_padding_mode(mode) # convert for legacy support
    return torch.ops.pad2d_op.pad2d_forward(data=data, padding=padding, mode=mode)

def pad2dT(data_padded:torch.Tensor, padding:List[int], mode:str):
    """
    pad2dT(data_padded:Tensor, padding:List[int], mode:str)
    ---

    Custom transpose 2D Padding function with accompanying transpose operator
    pad2dT(tensor_padded, padding=[1,1,1,1], mode='symmetric')

        padding: Padding [left,right,top,bottom]
        mode: ['symmetric','reflect','replicate']

        <Pad2d(x), y> = < x , Pad2dT(y) >  for all x,y of the respective domains
    """
    mode = conv_padding_mode(mode) # convert for legacy support
    return torch.ops.pad2d_op.pad2d_backward(data=data_padded, padding=padding, mode=mode)

class Pad2d(torch.nn.Module):
    def __init__(self, padding: List[int]=[1,1,1,1], mode:str='symmetric'):
        """
         Custom 2D Padding operator with accompanying transpose operator
           Pad2d(padding=[1,1,1,1], mode='symmetric')

        padding: Padding [left,right,top,bottom]
        mode: ['symmetric','reflect','replicate']

        <Pad2d(x), y> = < x , Pad2dT(y) >  for all x,y of the respective domains
        """
        super(Pad2d, self).__init__()
        self.padding: List[int] = padding
        self.mode: str = mode
    def forward(self, x):
        """ Apply predefined padding"""
        return pad2d(x, self.padding, self.mode)
    def extra_repr(self):
        return f"padding={self.padding},mode='{self.mode}'"

class Pad2dT(torch.nn.Module):
    def __init__(self, padding: List[int]=[1,1,1,1], mode:str="symmetric"):
        super(Pad2dT, self).__init__()
        self.padding: List[int] = padding
        self.mode: str = mode
    def forward(self, x):
        return pad2dT(x, self.padding, self.mode)
    def extra_repr(self):
        return f"padding={self.padding},mode='{self.mode}'"


def conv_padding_mode(padding_mode:str):
    """ Convert pytorch standard modes to our internal modes (legacy support)"""
    if padding_mode == 'border':
        padding_mode = 'replicate'
    elif padding_mode == 'reflection':
        padding_mode = 'reflect'
    return padding_mode

def pt_reflection_pad2d_backward_interface(data:torch.Tensor, padding:List[int]):
    """
        A direct interface to the Backward function of the padding operator in pytorch.
        TorchScript Jit-able but does not come with gradients.
        
        padding: Padding [left,right,top,bottom]
    """
    return torch.ops.pad2d_op.reflection_pad2d_backward_interface(data, padding)


def pt_replication_pad2d_backward_interface(data:torch.Tensor, padding:List[int]):
    """
        A direct interface to the Backward function of the padding operator in pytorch.
        TorchScript Jit-able but does not come with gradients.
        
        padding: Padding [left,right,top,bottom]
    """
    return torch.ops.pad2d_op.replication_pad2d_backward_interface(data, padding)
