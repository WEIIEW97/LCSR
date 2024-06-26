//////////////////////////////////////////////////////////////////////////////////////////////////v
///@file pad2d.cpp
///@brief Operator that pads an image given with symmetric boundary conndition
///       Interface between cuda and pytorch
///@authors Erich Kobler <erich.kobler@icg.tugraz.at>
///         Markus Hofinger <markus.hofinger@icg.tugraz.at>
///@date 05.2021


#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

#include <ATen/NamedTensorUtils.h>

#include "include/pad2d.hpp"

using torch::Tensor;
using torch::DeviceType;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;

void initPad2d() {}

torch::Tensor pad2d_forward_cuda(
    const torch::Tensor &data_in,
    const torch::IntArrayRef padding,
    const std::string& mode){
      
    PaddingMode pad_mode;
    if (mode == "symmetric"){
      // std::cout << "OP:symmetric" << std::endl;
      pad_mode = PaddingMode::symmetric;
    }else if (mode == "reflect"){
      pad_mode = PaddingMode::reflect;
      // std::cout << "OP:reflect" << std::endl;
    }else if (mode == "replicate"){
      pad_mode = PaddingMode::replicate;
      // std::cout << "OP:replicate" << std::endl;
    }else{
      TORCH_CHECK(false , "Unkown padding mode '", mode,"'! Must be 'symmetric', 'reflect', 'replicate'");
    } 
    return pad2d_forward_cuda_interface(data_in, padding, pad_mode);
}

torch::Tensor pad2d_backward_cuda(
    const torch::Tensor &grad_padded,
    const torch::IntArrayRef padding,
    const std::string& mode){
      
    PaddingMode pad_mode;
    if (mode == "symmetric"){
      // std::cout << "OP:symmetric" << std::endl;
      pad_mode = PaddingMode::symmetric;
    }else if (mode == "reflect"){
      pad_mode = PaddingMode::reflect;
      // std::cout << "OP:reflect" << std::endl;
    }else if (mode == "replicate"){
      pad_mode = PaddingMode::replicate;
      // std::cout << "OP:replicate" << std::endl;
    }else{
      TORCH_CHECK(false , "Unkown padding mode '", mode,"'! Must be 'symmetric', 'reflect', 'replicate'");
    } 
    return pad2d_backward_cuda_interface(grad_padded, padding, pad_mode);
}


  // Advanced Way with Argument names and default values
  TORCH_LIBRARY(pad2d_op, m) {
    // v1) Add a single operator for all backends
    // m.def("pad2d_forward(Tensor data, int[4] padding, str mode) -> Tensor", pad2d_forward_cuda);
    // m.def("pad2d_backward(Tensor data, int[4] padding, str mode) -> Tensor",pad2d_backward_cuda);

    // v2) Just define the operator and use different backends (Cuda,CPU,...)
    m.def("pad2d_forward(Tensor data, int[4] padding, str mode) -> Tensor");  
    m.def("pad2d_backward(Tensor data, int[4] padding, str mode) -> Tensor");

    // Additional interfaces for original pytorch padding functions
    m.def("reflection_pad2d_backward_interface(Tensor grad_padded, int[4] padding) -> Tensor", reflection_pad2d_backward_interface);
    m.def("replication_pad2d_backward_interface(Tensor grad_padded, int[4] padding) -> Tensor", replication_pad2d_backward_interface);
  }

  // Advanced Way with Argument names and default values
  TORCH_LIBRARY_IMPL(pad2d_op, CUDA, m) {
    // Registe a specific backend (here CUDA), requires previous definition m.def(OPERATOR_STR_SCHEME) without a function name passed
    m.impl("pad2d_forward", pad2d_forward_cuda);
    m.impl("pad2d_backward", pad2d_backward_cuda);
  }

////////////////////////////////////////////////////////////////////////////////
// BEGIN: Adding Autograd Support for our Custom Operator
// Our Custom Op Functions are registered with Pytorchs Dynamic Dispatcher.
// Here we get a dynamic interface to these functions by invoking the dispatcher (as we would do in python)
Tensor pad2d_forward( const torch::Tensor &data_in,
                      const torch::IntArrayRef padding,
                      const std::string& mode) {
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("pad2d_op::pad2d_forward", "")
    .typed<decltype(pad2d_forward)>();
  return op.call(data_in, padding, mode);
}
Tensor pad2d_backward(const torch::Tensor &grad_padded,
                      const torch::IntArrayRef padding,
                      const std::string& mode) {
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("pad2d_op::pad2d_backward", "")
    .typed<decltype(pad2d_backward)>();
  return op.call(grad_padded, padding, mode);
}

class Pad2dFunction : public torch::autograd::Function<Pad2dFunction> {
 public:
  static Tensor forward(AutogradContext *ctx, 
                        const torch::Tensor &data_in,
                        const torch::IntArrayRef padding,
                        const std::string& mode) {
    ctx->saved_data["padding_vec"] = c10::IValue(padding.vec());
    ctx->saved_data["mode"       ] = c10::IValue(mode);
    
    // Add a guard that deactivates autograd (we are already in the Autograd function), to prevent infinity loops
    // https://github.com/pytorch/pytorch/blob/master/docs/cpp/source/notes/inference_mode.rst
    #if (defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR>=1) && (TORCH_VERSION_MINOR>=9))
      // AutoDispatchBelowADInplaceOrView was introduced in Pytorch 1.9 and  TORCH_VERSION_MAJOR/MINOR was introduced in 1.8
      at::AutoDispatchBelowADInplaceOrView guard; // Pytorch < 1.9
    #else
      at::AutoNonVariableTypeMode guard; // Pytorch < 1.9
    #endif
    return pad2d_forward(data_in, padding, mode);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto grad_padded = grad_outputs[0];
    // torch::IntArrayRef padding = {1,1,1,1};
    // std::string mode = "reflect";

    // Extracting saved values from context (internally saved as c10::IValue)
    // See  https://pytorch.org/cppdocs/api/structc10_1_1_i_value.html
    //// Inspect context
    // std::cout << "mode:"    << ctx->saved_data["mode"];
    // std::cout << "padding: "<< ctx->saved_data["padding0"] <<"|"<< ctx->saved_data["padding1"] <<"|"<< ctx->saved_data["padding2"] <<"|"<<std::endl;

    // extract mode (std::string)
    std::string mode = ctx->saved_data["mode"].toStringRef();
    // std::cout << "mode:" << mode << std::endl;

    // extract padding => (torch::IntArrayRef (std::vec))
    auto  padding_vec = ctx->saved_data["padding_vec"].toIntVector();     // 1) For some reason this only works as two
    torch::IntArrayRef padding(padding_vec ) ;                            // 2) separate lines
    // std::cout << "padding:" << ctx->saved_data["padding_vec"] << "  " << padding_vec  << padding << std::endl;

    torch::Tensor grad_unpadded_in = pad2d_backward(grad_padded, padding, mode);
    // torch::Tensor grad_unpadded_in = pad2dT_autograd(grad_padded, padding, mode);
    return {grad_unpadded_in, torch::Tensor() , torch::Tensor()};
  }
};

Tensor pad2d_autograd( const torch::Tensor &data_in,
              const torch::IntArrayRef padding,
              const std::string& mode) {
  return Pad2dFunction::apply(data_in, padding, mode);
}

///////
//The Pad Transpose Operator Pad2dT
class Pad2dTFunction : public torch::autograd::Function<Pad2dTFunction> {
 public:
  static Tensor forward(AutogradContext *ctx, 
                        const torch::Tensor &grad_padded,
                        const torch::IntArrayRef padding,
                        const std::string& mode) {
    ctx->saved_data["padding_vec"] = c10::IValue(padding.vec());
    ctx->saved_data["mode"       ] = c10::IValue(mode);

    // Add a guard that deactivates autograd (we are already in the Autograd function), to prevent infinity loops
    // https://github.com/pytorch/pytorch/blob/master/docs/cpp/source/notes/inference_mode.rst
    #if (defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR>=1) && (TORCH_VERSION_MINOR>=9))
      // AutoDispatchBelowADInplaceOrView was introduced in Pytorch 1.9 and  TORCH_VERSION_MAJOR/MINOR was introduced in 1.8
      at::AutoDispatchBelowADInplaceOrView guard; // Pytorch < 1.9
    #else
      at::AutoNonVariableTypeMode guard; // Pytorch < 1.9
    #endif
    return pad2d_backward(grad_padded, padding, mode);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto grad_unpadded = grad_outputs[0];
    // torch::IntArrayRef padding = {1,1,1,1};
    // std::string mode = "reflect";

    // Extracting saved values from context (internally saved as c10::IValue)
    // See  https://pytorch.org/cppdocs/api/structc10_1_1_i_value.html
    //// Inspect context
    // std::cout << "mode:"    << ctx->saved_data["mode"];
    // std::cout << "padding: "<< ctx->saved_data["padding0"] <<"|"<< ctx->saved_data["padding1"] <<"|"<< ctx->saved_data["padding2"] <<"|"<<std::endl;

    // extract mode (std::string)
    std::string mode = ctx->saved_data["mode"].toStringRef();
    // std::cout << "mode:" << mode << std::endl;

    // extract padding => (torch::IntArrayRef (std::vec))
    auto  padding_vec = ctx->saved_data["padding_vec"].toIntVector();     // 1) For some reason this only works as two
    torch::IntArrayRef padding(padding_vec ) ;                            // 2) separate lines
    // std::cout << "padding:" << ctx->saved_data["padding_vec"] << "  " << padding_vec  << padding << std::endl;

    torch::Tensor grad_padded_in = pad2d_forward(grad_unpadded, padding, mode);
    // torch::Tensor grad_padded_in = pad2d_autograd(grad_unpadded, padding, mode);
    return {grad_padded_in, torch::Tensor() , torch::Tensor()};
  }
};

Tensor pad2dT_autograd( const torch::Tensor &grad_padded,
              const torch::IntArrayRef padding,
              const std::string& mode) {
  return Pad2dTFunction::apply(grad_padded, padding, mode);
}


TORCH_LIBRARY_IMPL(pad2d_op, Autograd, m) {
  m.impl("pad2d_forward", pad2d_autograd);
  m.impl("pad2d_backward", pad2dT_autograd);
}
// END myadd
////////////////////////////////////////////////////////////////////////////////











////////////////////////////////////////////////////////////////////////////
// Adding interfaces to pytorch standard operators:

torch::Tensor reflection_pad2d_backward_interface(
    const torch::Tensor &grad_padded,
    torch::IntArrayRef padding
){
  auto N = grad_padded.size(0);
  auto C = grad_padded.size(1);
  auto H_pad = grad_padded.size(2);
  auto W_pad = grad_padded.size(3);
  std::cout << "[N,C,H,W] = [" << N <<"," << C <<"," << H_pad <<","<< W_pad <<"]" << std::endl;

  auto H = H_pad - padding[2] - padding[3];
  auto W = W_pad - padding[0] - padding[1];

  // TORCH_CHECK( (H > 0) && (W>0),
  //   "Input is too small to apply pad transpose!"
  //   " padding = [", padding ,"] \n"
  //   " padded:  [NCHW] = [", N,",",C,",",H_pad,",",W_pad,"] \n"
  //   " unpadded: [NCHW] = [", N,",",C,",",H,",",W,"]");

  auto grad_unpadded = torch::zeros({N,C,H,W}, grad_padded.options());
  // std::cout << "Reflection Fwd:\n" << x_pad_repl << std::endl;
  // inp  =  torch::ones({1,1,7,7});
  // grad =  torch::zeros({1,1,5,5});
  auto grad_unpadded_out = torch::reflection_pad2d_backward( grad_padded, grad_unpadded, padding);
  // std::cout << "Reflection Bwd:\n" << out2 << std::endl;
  // std::cout << inp.sizes() << "; " << grad.sizes() << "; " << out2.sizes()<<  std::endl;
  return grad_unpadded_out;
}


torch::Tensor replication_pad2d_backward_interface(
    const torch::Tensor &grad_padded,
    torch::IntArrayRef padding
){
  auto N = grad_padded.size(0);
  auto C = grad_padded.size(1);
  auto H_pad = grad_padded.size(2);
  auto W_pad = grad_padded.size(3);
  std::cout << "[N,C,H,W] = [" << N <<"," << C <<"," << H_pad <<","<< W_pad <<"]" << std::endl;

  auto H = H_pad - padding[2] - padding[3];
  auto W = W_pad - padding[0] - padding[1];

  // TORCH_CHECK( (H > 0) && (W>0),
  //   "Input is too small to apply pad transpose!"
  //   " padding = [", padding ,"] \n"
  //   " padded:  [NCHW] = [", N,",",C,",",H_pad,",",W_pad,"] \n"
  //   " unpadded: [NCHW] = [", N,",",C,",",H,",",W,"]");

  auto grad_unpadded = torch::zeros({N,C,H,W}, grad_padded.options());
  // std::cout << "Reflection Fwd:\n" << x_pad_repl << std::endl;
  // inp  =  torch::ones({1,1,7,7});
  // grad =  torch::zeros({1,1,5,5});
  auto grad_unpadded_out = torch::replication_pad2d_backward( grad_padded, grad_unpadded, padding);
  // std::cout << "Reflection Bwd:\n" << out2 << std::endl;
  // std::cout << inp.sizes() << "; " << grad.sizes() << "; " << out2.sizes()<<  std::endl;
  return grad_unpadded_out;
}
