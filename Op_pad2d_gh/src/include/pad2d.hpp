#pragma once
#include <torch/script.h>

#if defined(_MSC_VER)
	#ifdef pad2d_op_EXPORTS
		#define PAD2D_OP_API __declspec(dllexport)
	#else
		#define PAD2D_OP_API __declspec(dllimport)
	#endif
#elif defined(__GNUC__)
	#ifdef pad2d_op_EXPORTS
		#define PAD2D_OP_API __attribute__((visibility("default")))
	#else
		#define PAD2D_OP_API
	#endif
#endif

// Since we don't directly use the pad2d operator library
// the linker would normally discard the library because
// it thinks we don't use it at all. When this happens
// PyTorch won't be able to find the operator. The compilers
// on Linux support the flag "-Wl,--no-as-needed" which prevents
// the library from being discarded. The MSVC doesn't have a flag
// which allows you to do this. To workaround this there is an initPad2d
// dummy function which needs to be called so that the linker doesn't discard the
// library.
PAD2D_OP_API void initPad2d();

enum class PaddingMode {symmetric, reflect, replicate};

torch::Tensor pad2d_forward_cuda_interface(
    const torch::Tensor &data_in,
    const torch::IntArrayRef padding,
    const PaddingMode mode
);

torch::Tensor pad2d_backward_cuda_interface(
    const torch::Tensor &grad_padded,
    const torch::IntArrayRef padding,
    const PaddingMode mode
);


torch::Tensor pad2d_autograd( const torch::Tensor &,
              const torch::IntArrayRef ,
              const std::string& );

torch::Tensor pad2dT_autograd( const torch::Tensor &,
              const torch::IntArrayRef ,
              const std::string& );

////////////////////////////////////////////
// Interfaces to the original pytorch padding backward passes
torch::Tensor reflection_pad2d_backward_interface(
    const torch::Tensor &grad_padded,
    torch::IntArrayRef padding
);

torch::Tensor replication_pad2d_backward_interface(
    const torch::Tensor &grad_padded,
    torch::IntArrayRef padding
);
