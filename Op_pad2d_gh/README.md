# Padding transpose
### Overview

This library provides a TorchScrip compatible padding transpose operator to the padding operator in Pytorch. 
This means the padding operator fulfills the following scalar product

$$ < Pad (x) , y > = < x , Pad^T (y) > $$

and can also be exported into a binary TorchScript file, that can be directly loaded into a C++ Application.


This operator is important to build transpose convolutions with correct paddings, in the sense of a hermitian adjoint

$$ <(Conv \circ Pad )(x) , y > = < x , (Conv \circ Pad)^T (y) > = < x , (Pad ^T \circ Conv ^T) (y) > $$



### Structure:
Structure:
```
 - pad2d_op:          (module folder to be installed via pip)
   - __init__.py      (module header)
   - functions.py     (python frontend that loads the compiled C++ operator)
 - build:
   - *               (various build artifacts from setup.py will be created here)
 - tests:            
   - test_pad2d.py   (real test cases to test the operator)
   - mini_demo.py     (a few usage exampels)
 - test_cpp_application:
   - build           (an empty folder for the build artifacts)
   - CMakeLists.txt  (CMake build instructions)
   - example_app.cpp (An example app that uses our custom operator)
```


### Usage
#### PreRequirements:
This operator was built and tested with Pytorch 1.7 & 1.8 comming via Anaconda.
It is therefore recommended to have an anaconda environment with at least the following requirenments

- pytorch=1.7.1 (1.8 for windows)
- torchvision
- python=3.7
- cudatoolkit=10.2
- numpy
- pip    (! important since we install via pip!)
- parameterized (used for unittests)
- ipyhton (recommended)
```bash
conda create -n torch_op_demo   pytorch=1.7.1 torchvision python=3.7 cudatoolkit=10.1 pip numpy ipython parameterized -c pytorch
```
Verify the installation by checking if the paths truly match your newly created environment.
Otherwise open a new shell and retry before continuing!

```bash
conda activate torch_op_demo

which python

which pip
```
All following examples assume you run in that environment!

#### Building the operator with setuptools (Linux only)
```bash
# if not already activated (torch_op_demo)
# conda activate torch_op_demo

# Use Multiple processors:
# export MAX_JOBS=8

# Clean older artifacts:
python setup.py clean --all

# Build and install the operator
python setup.py install
```
This will install the operator via pip 

#### Building the operator with CMake (Linux/Windows)

```bash
# if not already activated (torch_op_demo)
# conda activate torch_op_demo

# Create build directory
mkdir build
cd build

# Build project/make files
cmake -DCMAKE_PREFIX_PATH=<path/to/torch> ..
```
Depending on the operation system and the tool chain you use this will create
the necessary project/make files. On Linux you can than build and install the
operator using ```make```. On Windows you can open the Visual Studio project
and build the INSTALL target. After this all the library and include files
will be copied into the ```install``` folder and pip is automatically used to install
the operator into your current python environment.

#### Running Unittests 
The operator cannot be imported from the root of this git repo. 
Please change the folder for using it or running tests
```bash
cd tests
python -m unittest
```
#### Usage in python
see for example tests/demo_pad2d.py
```python
import torch
from pad2d_op import pad2d,pad2dT
from pad2d_op import Pad2d,Pad2dT

inp = torch.arange(25).reshape(5,5).cuda().float()
padded = pad2d(inp, [2,2,2,2], "symmetric")
print(padded)

pad_mod = Pad2d([2,2,2,2],"symmetric")
padded = pad_mod(inp)
print(padded)

out = torch.ones(81).reshape(9,9).cuda().float()
padT_mod = Pad2dT([2,2,2,2],"symmetric")
inp = padT_mod(out)
print(inp)
```

#### Exporting a binary TorchScript Model (to be used in C++):
This operator was created with TorchScript support.
So Models using it can be exported as binary file unsing TorchScripts jit.script method.
This requires your python code to use Type Hinting on input data other than torch.Tensors.
See [demo_cpp_application/export_binary_torchscript.py](demo_cpp_application/export_binary_torchscript.py) for an example.

Exporting and building the C++ Demo:
   1. Activate the Environment (Anaconda) you created installed the operator into
      ```bash
      conda activate torch_demo
      ```
   2. Build the C++ demo application:
      ```bash
      cd demo_cpp_application
      mkdir -p build
      cd build
      # configure cmake (automatically searches for pytorch and the operator)
      cmake .. 
      # build the demo application 
      # links the custom operator against the application
      make -j
      ```
   3. run the python demo app to export the TorchScript binary model:<br>
      ```bash
      # assumes you are in: demo_cpp_application/build/
      python ../export_binary_torchscript.py
      ```
      This should generate a binary model representation under: `demo_cpp_application/build/torch_scrip_model_using_costume_op.pt`

      And the following screen output:
      ```bash
      Demo Output (python)
      tensor([[0.5000, 1.5000, 2.5000],
               [3.5000, 4.5000, 5.5000],
               [6.5000, 7.5000, 8.5000]], device='cuda:0')
      tensor([[0.5000, 0.5000, 1.5000, 2.5000, 2.5000],
               [0.5000, 0.5000, 1.5000, 2.5000, 2.5000],
               [3.5000, 3.5000, 4.5000, 5.5000, 5.5000],
               [6.5000, 6.5000, 7.5000, 8.5000, 8.5000],
               [6.5000, 6.5000, 7.5000, 8.5000, 8.5000]], device='cuda:0')
         ```
   4. run the C++ demo app with the exported binary model for comparison:<br>
      ```bash
      ./demo_app torch_scrip_model_using_costume_op.pt
      ```
      This should generate the following screen output:
      ```bash
      Demo Output C++
      0.5000  1.5000  2.5000
      3.5000  4.5000  5.5000
      6.5000  7.5000  8.5000
      [ CUDAFloatType{3,3} ]
      0.5000  0.5000  1.5000  2.5000  2.5000
      0.5000  0.5000  1.5000  2.5000  2.5000
      3.5000  3.5000  4.5000  5.5000  5.5000
      6.5000  6.5000  7.5000  8.5000  8.5000
      6.5000  6.5000  7.5000  8.5000  8.5000
      [ CUDAFloatType{5,5} ]
      ```

### Usefull commands for Debugging:
- Building CUDA with CMAKE syntax changed quite a few times.
  - before CMAKE 3.8 custom versions where common
  - with CMAKE 3.8 CUDA became std. macros like FindCUDA, cuda_select_nvcc_arch_flags and CUDA_NVCC_FLAGS became std.
  - with CMAKE3.18 a new std. was introduced and FindCUDA CUDA_NVCC_FLAGS became depricated - now use  CMAKE_CUDA_FLAGS instead and set the project type to CUDA
- Problems with CUDA Architecture / ARCH Flags (simplified):
  - NVCC can generate PTX (virtual intermediate representation/assembly) and SASS (real machine code) code. As PTX is an intermediate representation it can be JIT compiled into SASS machine code also for newer GPU generations but requieres extra startup time for that. Therefore one can generate fatbinaries that already contain PTX and SASS for different architectures at once.
    - Explicitly forcing the build system to use specific CUDA ARCH and CODE flags to be used within TORCHs version of the setuptools. This means this flag is only recognized by (setup.py). Here some examples:<br>
       `cmake -DTORCH_CUDA_ARCH_LIST=7.5`  Using more than one parameter seems not to be possible with older cmake versions 
       `cmake -DTORCH_CUDA_ARCH_LIST='5.2;7.5' ..`  Using more than one parameter seems not to be possible with older cmake versions 
       `cmake -DTORCH_CUDA_ARCH_LIST=ALL`  
    - Check which flags where used to build your precompiled pytorch:<br>
      `torch.__config__.show()` <br>
      `torch.cuda.get_arch_list()`
    - Investigate the libraries binary file, to see which architecturs PTX/ELF where integrated:<br>
      `cuobjdump <objfilename>`<br
      `cuobjdump <objfilename> -lelf -lptx`
  
  - Seeing calls to g++ and nvcc:
    - with python distutils:<br>
      `python  setup.py --verbose`
    - with cmake : <br>
      `make VERBOSE=1`
   - `CUDA error: no kernel image is available for execution on the device` indicates that the cuda kernel was not built for your graphics card

  - If while running the demo_app the following error occurs:
     `error while loading shared libraries: libc10_cuda.so: cannot open shared object file: No such file or directory`
     This means that the library path to libc10_cuda.so - a library provided by torch, was not found.
     This library is typically found in the pytorch library folder which can be retrieved using:
      ```bash
      python -c "from torch.utils.cpp_extension  import library_paths; print(':'.join(library_paths(True)))"
      ```

      Adding this path to the LD_LIBRARY_PATH or the LD_PRELOAD variable should fix this error.

      ```bash
      export TORCH_LIB_PATH=$(python -c "from torch.utils.cpp_extension  import library_paths; print(':'.join(library_paths(True)))") 
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_LIB_PATH       
      ./demo_app torch_scrip_model_using_costume_op.pt ```


### Linklist
This demo was built using information of these very good web sources:

[EXTENDING TORCHSCRIPT WITH CUSTOM C++ OPERATORS](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)<br>
[REGISTERING A DISPATCHED OPERATOR IN C++](https://pytorch.org/tutorials/advanced/dispatcher.html)<br>
[Custom C++ Autograd](https://pytorch.org/tutorials/advanced/cpp_autograd)<br>
[(old) Source Code for this tutorial ](https://github.com/pytorch/extension-script/)<br>
[TorchScript intro](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)<br>
[TorchScript Jit inof](https://pytorch.org/docs/stable/jit.html)<br>
[PyTorch C++ API](https://pytorch.org/cppdocs/)<br>
[OptoX - our previous framework - currently not TorchScriptable](https://github.com/VLOGroup/optox)<br>

[Pytorch C10::ArrayRef References](https://pytorch.org/cppdocs/api/classc10_1_1_array_ref.html)
[Pytorch c10::IValue Reference](https://pytorch.org/cppdocs/api/structc10_1_1_i_value.html)
[CUDA NVCC Compiler Docu (PDF)](https://docs.nvidia.com/cuda/archive/10.1/pdf/CUDA_Compiler_Driver_NVCC.pdf)
