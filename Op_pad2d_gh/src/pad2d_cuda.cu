//////////////////////////////////////////////////////////////////////////////////////////////////v
///@file pad2d_cuda.cu
///@brief Operator that pads an image given with symmetric boundary conndition
///@authors Erich Kobler <erich.kobler@icg.tugraz.at>
///         Markus Hofinger <markus.hofinger@icg.tugraz.at>
///@date 05.2021


#include <ATen/cuda/detail/IndexUtils.cuh>
// #include <ATen/Dispatch.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAApplyUtils.cuh>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/NativeFunctions.h>
// #include <ATen/TensorUtils.h>
// #include <ATen/Utils.h>
#include <THC/THCAtomics.cuh> // keeping THC headers for gpuAtomicAdd
#include <torch/script.h> // One-stop header.

#include <thrust/pair.h>


#include "pad2d_cuda.cuh"
#include "include/pad2d.hpp"
#include "include/CUDAException.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enum Tag Dispatch Macro
// Concept:
//  - The implementation variants reside each in shallow templated classes/structs that only serve
//    as containers for the inlined static methods
//  - These classes define a type, which can be selected via 'using typename_t  = TypeStruct<Enum::EnumVal>'
//  - Write a user Macro that switches the dynamic enum variable and enters scopes where the default
//    type of the class to use is chosen via the aformentioned 'using typname...'
//    Then the macro calls the the actuall payload - a lambda-function containing the actual function
//    call that already uses the typename_t argument to select the correct templated function

template<typename index_t, PaddingMode paddingmode_t> struct PadmodeType;

template <typename index_t> struct PadmodeType<index_t, PaddingMode::symmetric>{
    static __device__ __forceinline__ index_t getPixel(index_t x, index_t width){
        if (x < 0) return abs(x) - 1;
        else if (x >= width) return 2 * width - x - 1;
        else return x;
    }
};
template <typename index_t> struct PadmodeType<index_t, PaddingMode::reflect>{
    static __device__ __forceinline__  index_t getPixel(index_t x, index_t width){
        if (x < 0)  return abs(x);
        else if (x >= width)  return 2 * width - x - 2;
        else return x;
    }
};
template <typename index_t> struct PadmodeType<index_t, PaddingMode::replicate>{
    static __device__ __forceinline__  index_t getPixel(index_t x, index_t width){
        if (x < 0) return  0;
        else if (x >= width) return  width - 1;
        else return x;
    }
};

#define SELECT_PADDINGMODE_TYPE(index_t, padmode_enum)                                \
  using padmode_t = PadmodeType<index_t, padmode_enum>;

#define DISPATCH_PADDING_MODE(PADMODE_ENUM, INDEX_TYPE, ...) [&] {                    \
    using index_t = INDEX_TYPE;                                                       \
    switch (PADMODE_ENUM) {                                                           \
        case PaddingMode::symmetric : {                                               \
            SELECT_PADDINGMODE_TYPE(index_t,  PaddingMode::symmetric)                 \
            return __VA_ARGS__();                                                     \
        }                                                                             \
        case PaddingMode::reflect : {                                                 \
            SELECT_PADDINGMODE_TYPE(index_t,  PaddingMode::reflect)                   \
            return __VA_ARGS__();                                                     \
        }                                                                             \
        case PaddingMode::replicate : {                                               \
            SELECT_PADDINGMODE_TYPE(index_t,  PaddingMode::replicate)                 \
            return __VA_ARGS__();                                                     \
        }                                                                             \
        default: {                                                                    \
            AT_ERROR("Unsupported type '" #PADMODE_ENUM "'");                         \
        }                                                                             \
    }                                                                                 \
}() 


static inline unsigned int divUp(unsigned int a, unsigned int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

using at::cuda::detail::canUse32BitIndexMath;

template<typename scalar_t, typename index_t, typename padmode_t>
__global__ void pad2d_forward_kernel(
          at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> padded_out,
    const at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> data_in,
    index_t left, index_t top)
{
    index_t x = threadIdx.x + blockIdx.x * blockDim.x;
    index_t y = threadIdx.y + blockIdx.y * blockDim.y;
    index_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < padded_out.size(2) && y < padded_out.size(1) && z < padded_out.size(0))
    {
        // compute the corresponding index 
        const index_t x_in = padmode_t::getPixel(x - left, data_in.size(2));
        const index_t y_in = padmode_t::getPixel(y - top,  data_in.size(1));
        padded_out[z][y][x] = data_in[z][y_in][x_in];
    }
}


torch::Tensor pad2d_forward_cuda_interface(const torch::Tensor& data_in, torch::IntArrayRef padding, PaddingMode mode)
{
    TORCH_CHECK( (data_in.dim() >= 2),
      "Tensor input needs to be at least 2-Dimensional!\n data_in: ", data_in.sizes() );

    // Compute old and new H,W shapes
    const auto H = data_in.size(-2);
    const auto W = data_in.size(-1);    
    const auto Hpad = H + padding[2]+ padding[3];
    const auto Wpad = W + padding[0]+ padding[1];
        
    // Do a shape conversion (Flatten input to [-1,H,W] )
    std::vector<int64_t> new_sizes_vec = data_in.sizes().vec();  // Convert input shape from torch::IntArrayRef to std::vec<int64_t>
    // std::cout<< "data_in:" << data_in.sizes() << new_sizes_vec << std::endl;
    new_sizes_vec.resize(new_sizes_vec.size()-2); // remove old H&W
    new_sizes_vec.push_back(Hpad);
    new_sizes_vec.push_back(Wpad);
    torch::IntArrayRef new_sizes(new_sizes_vec);  //convert vector back to torch::IntArrayRef - standard type for sizes
    // std::cout <<" data_out_proposed shape " << new_sizes_vec  << new_sizes << std::endl;
    // Now create a flat representation
    torch::Tensor data_in_flat = data_in.reshape({-1,H,W});
    const auto N = data_in_flat.size(0);


    auto padded_out = at::zeros({N,Hpad,Wpad} , data_in_flat.options());

    TORCH_CHECK(canUse32BitIndexMath(padded_out),
    "padded input tensor must fit into 32-bit index math");

    // TORCH_CHECK((Hpad>0) && (Wpad),
    // "Padding larger than image! This is undefined and not implemented!");
    
    TORCH_CHECK( ((W>padding[0]) && (W>padding[1]) && (H>padding[2]) && (H>padding[3])),
      "Padding values need to be smaller than the actual image size, but at least one value is larger!\n" 
      "Image: [",N,",",H,",",W,"] vs. padding [",padding[0],",",padding[1],",",padding[2],",",padding[3],"]   ");

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid = dim3(divUp(Wpad, dim_block.x),
                         divUp(Hpad, dim_block.y),
                         divUp(N, dim_block.z));

    TORCH_CHECK(  dim_grid.x < 65536 , "Width is too large - does not fit cuda kernel - maximum limit is 65535*32!\n","data_shape=",data_in_flat.sizes());
    TORCH_CHECK(  dim_grid.y < 65536 , "Heigh is too large - does not fit cuda kernel - maximum limit is 65535*32!\n","data_shape=",data_in_flat.sizes());
    TORCH_CHECK(  dim_grid.z < 65536 , "Flattened first dimensions are too large - does not fit cuda kernel - maximum limit is 65535*32!\n","data_shape=",data_in_flat.sizes());

    AT_DISPATCH_FLOATING_TYPES(data_in.scalar_type(), "pad2d_forward" , ([&]{
        DISPATCH_PADDING_MODE(mode, int32_t, ([&]{
            pad2d_forward_kernel<scalar_t, index_t, padmode_t> <<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                /*padded_out=*/ padded_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                /*data_in=*/ data_in_flat.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                /*left=*/ padding[0], 
                /*top= */ padding[2]
            );
        }));
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    padded_out = padded_out.reshape(new_sizes);
    return (padded_out);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Backward Kernel
template <typename scalar_t, typename index_t, typename paddingmode_t>
__global__ void pad2d_backward_kernel(
          at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_in,
    const at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_padded_out,
    index_t left, index_t top)
{
    index_t x = threadIdx.x + blockIdx.x * blockDim.x;
    index_t y = threadIdx.y + blockIdx.y * blockDim.y;
    index_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < grad_padded_out.size(2) && y < grad_padded_out.size(1) && z < grad_padded_out.size(0))
    {
        // compute the corresponding index 
        const index_t x_in = paddingmode_t::getPixel(x - left, grad_in.size(2));
        const index_t y_in = paddingmode_t::getPixel(y - top,  grad_in.size(1));
        // atomicAdd(&grad_in[z][y_in][x_in], grad_padded_out[z][y][x]);  // original CUDA version (has trouble with double in older compute capabilities)
        gpuAtomicAdd(&grad_in[z][y_in][x_in], grad_padded_out[z][y][x]);  // Pytorch's version which is compatible with more datatypes
    }
}



torch::Tensor pad2d_backward_cuda_interface(
    const torch::Tensor &grad_padded_out,
    const torch::IntArrayRef padding,
    const PaddingMode mode){


    TORCH_CHECK( (grad_padded_out.dim() >= 2),
    "Tensor input needs to be at least 2-Dimensional!\n grad_padded_out: ", grad_padded_out.sizes() );

    const auto Hpad = grad_padded_out.size(-2);
    const auto Wpad = grad_padded_out.size(-1);
    const auto H = Hpad - padding[2]- padding[3];
    const auto W = Wpad - padding[0]- padding[1];

    // Do a shape conversion (Flatten input to [-1,H,W] )
    std::vector<int64_t> new_sizes_vec = grad_padded_out.sizes().vec();  // Convert input shape from torch::IntArrayRef to std::vec<int64_t>
    // std::cout<< "grad_padded_out:" << grad_padded_out.sizes() << new_sizes_vec << std::endl;
    new_sizes_vec.resize(new_sizes_vec.size()-2); // remove old H&W
    new_sizes_vec.push_back(H);
    new_sizes_vec.push_back(W);
    torch::IntArrayRef new_sizes(new_sizes_vec);  //convert vector back to torch::IntArrayRef - standard type for sizes
    // std::cout <<" data_out_proposed shape " << new_sizes_vec  << new_sizes << std::endl;
    // Now create a flat representation
    torch::Tensor grad_added_out_flat = grad_padded_out.reshape({-1,Hpad,Wpad});
    const auto N = grad_added_out_flat.size(0);

    TORCH_CHECK((Hpad>0) && (Wpad>0),
    "Padding larger than image! This is undefined and not implemented in the backward path! \n"
    "Image padded: [",N,",",Hpad,",",Wpad,"] vs. padding [",padding[0],",",padding[1],",",padding[2],",",padding[3],"]   ");

    TORCH_CHECK(canUse32BitIndexMath(grad_padded_out),
    "padded input tensor must fit into 32-bit index math");

    
    auto grad_unpadded_in = at::zeros({N,H,W} , grad_added_out_flat.options());

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid = dim3(divUp(Wpad, dim_block.x),
                         divUp(Hpad, dim_block.y),
                         divUp(N, dim_block.z));

    TORCH_CHECK(  dim_grid.x < 65536 , "Width is too large - does not fit cuda kernel - maximum limit is 65535*32!\n","data_shape=",grad_padded_out.sizes());
    TORCH_CHECK(  dim_grid.y < 65536 , "Heigh is too large - does not fit cuda kernel - maximum limit is 65535*32!\n","data_shape=",grad_padded_out.sizes());
    TORCH_CHECK(  dim_grid.z < 65536 , "Flattened first dimensions are too large - does not fit cuda kernel - maximum limit is 65535*32!\n","data_shape=",grad_padded_out.sizes());
                     

    AT_DISPATCH_FLOATING_TYPES(grad_added_out_flat.scalar_type(), "pad2d_backward" , ([&]{
        DISPATCH_PADDING_MODE(mode, int32_t, ([&]{
            pad2d_backward_kernel<scalar_t, index_t, padmode_t> <<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                /*grad_in=*/grad_unpadded_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                /*grad_padded_out=*/grad_added_out_flat.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                /*left=*/ padding[0], 
                /*top= */ padding[2]
            );
        }));
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    grad_unpadded_in = grad_unpadded_in.reshape(new_sizes);
    return(grad_unpadded_in);
}