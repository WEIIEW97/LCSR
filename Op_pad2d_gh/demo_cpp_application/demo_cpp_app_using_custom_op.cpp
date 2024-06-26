#include <torch/script.h> // One-stop header.

#include <pad2d.hpp>

#include <iostream>
#include <memory>


int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example_app_using_costum_op.cpp <path-to-exported-script-module>\n";
    return -1;
  }

  initPad2d();

  // 1.st Load the Module from the script
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // create some dummy input
  auto input = torch::arange(9).reshape({3,3}).to(torch::kFloat32).cuda();
  // convert the data to a vector of IValue (Interpreter Values) a generic dataformat to hold our input data:
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( input );  //impllicit conversion from torch::Tensor to torch::jit::IValue

  // run the TorchScript module
  torch::jit::IValue outputs  = module.forward(  std::move(inputs)  );
  auto output_tuple = outputs.toTuple();  //convert generic outputs to Tuple (as we used a Tuple of Two Tensors in our example)
  torch::Tensor out         = output_tuple->elements()[0].toTensor();
  torch::Tensor out_paddeed = output_tuple->elements()[1].toTensor();
 
  std::cout << "Demo Output C++" << std::endl;
  std::cout << out << std::endl;
  std::cout << out_paddeed << std::endl;

}