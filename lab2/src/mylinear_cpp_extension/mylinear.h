/*mylinear.h*/
#include<torch/extension.h>
#include<vector>

// forward propagation
torch::Tensor mylinear_forward(const torch::Tensor &inputA, const torch::Tensor &inputB);
// backward propagation
std::vector<torch::Tensor> mylinear_backward(const torch::Tensor &gradOutput,const torch::Tensor & x,const torch::Tensor & w);
