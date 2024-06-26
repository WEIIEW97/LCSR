import copy
import glob
import os
import setuptools
import sys

import torch
import torch._appdirs
from torch.utils.file_baton import FileBaton
from torch.utils._cpp_extension_versioner import ExtensionVersioner
from torch.utils.cpp_extension import library_paths, include_paths
from torch.utils import cpp_extension
from torch.utils.hipify import hipify_python
from torch.utils.hipify.hipify_python import get_hip_file_path, GeneratedFileCleaner
from typing import List, Optional, Union

from setuptools.command.build_ext import build_ext
from pkg_resources import packaging  # type: ignore

if hasattr(cpp_extension, "BUILD_SPLIT_CUDA"):
    BUILD_SPLIT_CUDA = cpp_extension.BUILD_SPLIT_CUDA
else:
    BUILD_SPLIT_CUDA = False

def custom_CUDAExtension(name, sources, *args, is_python_module=True, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        >>> setup(
                name='cuda_extension',
                ext_modules=[
                    CUDAExtension(
                            name='cuda_extension',
                            sources=['extension.cpp', 'extension_kernel.cu'],
                            extra_compile_args={'cxx': ['-g'],
                                                'nvcc': ['-O2']})
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })

    Compute capabilities:

    By default the extension will be compiled to run on all archs of the cards visible during the
    building process of the extension, plus PTX. If down the road a new card is installed the
    extension may need to be recompiled. If a visible card has a compute capability (CC) that's
    newer than the newest version for which your nvcc can build fully-compiled binaries, Pytorch
    will make nvcc fall back to building kernels with the newest version of PTX your nvcc does
    support (see below for details on PTX).

    You can override the default behavior using `TORCH_CUDA_ARCH_LIST` to explicitly specify which
    CCs you want the extension to support:

    TORCH_CUDA_ARCH_LIST="6.1 8.6" python build_my_extension.py
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python build_my_extension.py

    The +PTX option causes extension kernel binaries to include PTX instructions for the specified
    CC. PTX is an intermediate representation that allows kernels to runtime-compile for any CC >=
    the specified CC (for example, 8.6+PTX generates PTX that can runtime-compile for any GPU with
    CC >= 8.6). This improves your binary's forward compatibility. However, relying on older PTX to
    provide forward compat by runtime-compiling for newer CCs can modestly reduce performance on
    those newer CCs. If you know exact CC(s) of the GPUs you want to target, you're always better
    off specifying them individually. For example, if you want your extension to run on 8.0 and 8.6,
    "8.0+PTX" would work functionally because it includes PTX that can runtime-compile for 8.6, but
    "8.0 8.6" would be better.

    Note that while it's possible to include all supported archs, the more archs get included the
    slower the building process will be, as it will build a separate kernel image for each arch.

    '''
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')

    #############################################################################################################
    # CUSTOM CHANGE => possibility to disable loading of torch_python libraries
    if is_python_module:
        libraries.append('torch_python')
    else:
        print("omitting library torch_python")
    #############################################################################################################

    libraries.append('cudart')
    libraries.append('c10_cuda')
    if BUILD_SPLIT_CUDA:
        libraries.append('torch_cuda_cu')
        libraries.append('torch_cuda_cpp')
    else:
        libraries.append('torch_cuda')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])

    include_dirs += include_paths(cuda=True)
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    return setuptools.Extension(name, sources, *args, **kwargs)

