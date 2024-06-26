cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(pad2d_op CXX)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install"
       CACHE PATH "default install path" FORCE)
    # message(${CMAKE_INSTALL_PREFIX})
    message (STATUS "Overwriting default install path to ./install, as the actuall installation happens with a python script")
endif()

find_package(CUDA)



#####################################################################################
# Setting Up Automatic finding of TORCH Libraries
message("-- Looking for TORCH Libraries:")
message("    HINT: Call this command to setup the Torch Library Path, if autodetedct does not work!")
message("         export Torch_DIR=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')\n")
message("    CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
message("    Torch_DIR=${Torch_DIR}")
message("    _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}")
message("    CMAKE_INTERNAL_PLATFORM_ABI=${CMAKE_INTERNAL_PLATFORM_ABI}")

exec_program("python -c \"import torch;print(torch.utils.cmake_prefix_path)\""
            OUTPUT_VARIABLE TORCH_CMAKE_PATH_FROM_PYTHON
            RETURN_VALUE    RETVAL)
message("    Path Found: ")
message("      TORCH_CMAKE_PATH_FROM_PYTHON=${TORCH_CMAKE_PATH_FROM_PYTHON}")
find_package(Torch REQUIRED
             HINTS "${TORCH_CMAKE_PATH_FROM_PYTHON}/" )
message( "\n-- TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(" TORCH_CXX_FLAGS= ${TORCH_CXX_FLAGS}")
# message(" TORCH_CXX_FLAGS= ${TORCH_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(" TORCH_INSTALL_PREFIX= ${TORCH_INSTALL_PREFIX}")
message(" TORCH_SO_LIB_PATH= ${TORCH_INSTALL_PREFIX}/lib")

####################################################################################

####################################################################################
# Checking C++11 ABI
exec_program("python -c \"import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))\""
                OUTPUT_VARIABLE TORCH_CXX_ABI)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX_ABI}")
message(" CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")
#####################################################################################


cuda_add_library(pad2d_op SHARED src/pad2d.cpp src/pad2d_cuda.cu)

target_include_directories(pad2d_op PUBLIC "src/include/")

# Setup To use CUDAs own FP16 types and not those of pytorch
# https://discuss.pytorch.org/t/error-more-than-one-operator-matches-these-operands-in-thcnumerics-cuh/89935
target_compile_definitions(pad2d_op  PUBLIC
    CUDA_HAS_FP16=1
    __CUDA_NO_HALF_OPERATORS__
    __CUDA_NO_HALF_CONVERSIONS__
    __CUDA_NO_HALF2_OPERATORS__)

target_compile_features(pad2d_op PRIVATE cxx_std_17)
set_target_properties(pad2d_op PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(pad2d_op PROPERTIES PUBLIC_HEADER src/include/pad2d.hpp)

target_link_libraries(pad2d_op "${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Setting up RPATH => allowing library to keep path to system libraries from pytorch
# set(CMAKE_SKIP_RPATH FALSE)
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(pad2d_op PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
# These libs typically can be found under the following path
# set_target_properties(pad2d_op PROPERTIES INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")

configure_file(setup.py.cmake_in setup.py)


install(TARGETS pad2d_op
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	PUBLIC_HEADER DESTINATION include
)

install(CODE "execute_process(COMMAND python ${CMAKE_CURRENT_BINARY_DIR}/setup.py install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
