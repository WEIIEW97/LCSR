//////////////////////////////////////////////////////////////////////////////////////////////////v
///@file pad2d_cuda.cuh
///@brief Operator that pads an image given with symmetric boundary conndition
///@authors Erich Kobler <erich.kobler@icg.tugraz.at>
///         Markus Hofinger <markus.hofinger@icg.tugraz.at>
///@date 05.2021

#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "include/pad2d.hpp"
