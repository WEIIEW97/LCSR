cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(demo_app)

set(CMAKE_CXX_STANDARD 17)

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

####################################################################################
# Checking C++11 ABI
exec_program("python -c \"import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))\""
                OUTPUT_VARIABLE TORCH_CXX_ABI)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX_ABI}")
message(" CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")
#####################################################################################

# The main Application Target
add_executable(demo_app demo_cpp_app_using_custom_op.cpp)
# Link main Application with Torch Libraries
# target_link_libraries(demo_app "${TORCH_LIBRARIES}")
# Setup to use C++ 14 standard
set_property(TARGET demo_app PROPERTY CXX_STANDARD 17)

#####################################################################################
# Trying To automatically find the Custom Operator
message("\n-- Automatically finding the custom Operator: pad2d_op (/pad2d_op/libpad2d_op.so")

exec_program("python -c \"import pad2d_op; print(pad2d_op.get_module_path())\""
			 OUTPUT_VARIABLE PAD2D_MODULE_PATH)
find_library(PAD2D_LIB_PATH pad2d_op
             HINTS "${PAD2D_MODULE_PATH}"
			 PATH_SUFFIXES "lib")

exec_program("python -c \"import pad2d_op; print(pad2d_op.get_include_path())\""
	OUTPUT_VARIABLE PAD2D_INCLUDE_PATH)
include_directories("${PAD2D_INCLUDE_PATH}")

message("    PAD2D_LIB_PATH=${PAD2D_LIB_PATH}")

target_link_libraries(demo_app  "${TORCH_LIBRARIES}" "${PAD2D_LIB_PATH}" )

target_compile_features(demo_app PRIVATE cxx_range_for)

target_compile_features(demo_app PRIVATE cxx_std_17)

