/*mylinear.cpp*/
#include "mylinear.h"

// forward propagation
torch::Tensor mylinear_forward(const torch::Tensor &w,const torch::Tensor &x)
{
    auto output = torch::mm(x, w.transpose(0, 1));
    return output;
}

// backward propagation
std::vector<torch::Tensor> mylinear_backward(const torch::Tensor &gradOutput,const torch::Tensor & x,const torch::Tensor & w)
{
        torch::Tensor grad_w = torch::mm(gradOutput.transpose(0, 1), x);
        torch::Tensor grad_x = torch::mm(gradOutput, w);
        return {grad_w, grad_x};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("forward", &mylinear_forward, "mylinear forward");
    m.def("backward", &mylinear_backward, "mylinear backward");
}