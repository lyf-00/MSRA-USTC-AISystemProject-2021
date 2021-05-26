## <center>Lab2 实验报告</center>

### 0 文件内容说明

```
| -- lab2
	 | -- images 	# 图片
     | -- src	# 代码
          | -- mnist_custom_linear.py		# 基于Python API实现定制化张量运算Linear
          | -- mnist_custom_linear_cpp.py	# 基于C++ API实现定制化张量运算Linear
          | -- mylinear_cpp_extension		# 基于C++ API实现定制化张量运算Linear的C++文件
               | -- myalinear.h				# 头文件
               | -- mylinear.cpp
               | -- setup.py				# 编译C++文件，建立链接的 setup 脚本
     | -- resources
     	  | -- profile 						# 一些 profile 示例文件
     | -- README							# 实验报告
```

### 1 实验环境

|          |                                    |                                                    |
| -------- | ---------------------------------- | -------------------------------------------------- |
| 硬件环境 | CPU（vCPU数目）                    | Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz (6 core) |
|          | GPU(型号，数目)                    | Nvidia GeForce MX250                               |
| 软件环境 | OS版本                             | Ubuntu 18.04                                       |
|          | 深度学习框架<br>python包名称及版本 | pytorch 1.5.0                                      |
|          | CUDA版本                           | 未使用                                             |
|          |                                    |                                                    |

### 2 实验流程

lab2说明文档给出的实验流程图如下：

![](.\images\Lab2-flow.png)

我的实验步骤为：

**step1：基础知识学习**

由于我之前只了解一些传统的机器学习方法，没有接触过神经网络与深度学习，因此在项目一开始花费了很长时间学习深度学习的基础知识。查阅资料包括但不限于：[吴恩达深度学习课程第一课 — 神经网络与深度学习](https://www.bilibili.com/video/BV164411m79z?t=276&p=33)，《神经网络与深度学习》（邱锡鹏著），《深度学习框架PyTorch入门与实践》（陈云著）等。

在对神经网络有一定基本了解后，正式开始实验。

**step2：产生思路，理解参考代码**

参考lab2的说明文档，确定本次实验需要实现的主要有以下几个部分：

- Python API
  - 实现自定义的类 mylinearFunction，继承自 torch.autograd.Function，实现前向计算和反向传播函数
  - 实现自定义的类 Linear ，继承自 nn.Module，调用上述 Function 
  - 修改 Net 的 init 函数，使用自定义的 Linear 层
- C++ 扩展
  - 使用 C++ 实现自定义的 forward 和 backward 函数
  - 将代码生成为 pytorch 的 C++ 扩展
  - 在基于 Python API 实现的 mylinearFunction 模块调用上述 C++ 扩展进行前向计算、反向传播

**step3：学习 lab2 需要的相关知识，写代码**

学习参考资料：[pytorch的C++extension写法](https://zhuanlan.zhihu.com/p/100459760)

- Python API 实现

   ```python
   # custom linear function
   class  mylinearFunction(torch.autograd.Function):
       @staticmethod
       def forward(ctx,x,w,x_requires_grad = True):
           ctx.x_requires_grad = x_requires_grad
           ctx.save_for_backward(w, x)             # keep middle result
           output = x.mm(w.t())                    # y = x * w^T
           return output
   
       @staticmethod
       def backward(ctx,grad_output):
           w,x = ctx.saved_tensors
           grad_w = grad_output.t().mm(x)          # dw = dy ^ T * x
           if ctx.x_requires_grad:
               grad_x = grad_output.mm(w)          # dx = dy * w
           else:
               grad_x = None
           return grad_x,grad_w,None
   
   class myLinear(nn.Module):
       def __init__(self,in_features,out_features):
           super(myLinear, self).__init__()
           self.in_features = in_features
           self.out_features = out_features
           self.w = nn.Parameter(torch.Tensor(out_features,in_features))
           self.w.data.uniform_(-0.1, 0.1)         # 参数初始化
   
       def forward(self,x):
           return mylinearFunction.apply(x,self.w) # call function
   ```

- C++ 扩展实现

  ```C++
  /*mylinear.cpp*/
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
  ```

  ```python
  import mylinear_cpp
  # forward 关键步
  output = mylinear_cpp.forward(w,x) 
  
  # backward 关键步
  grad_w,grad_x = mylinear_cpp.backward(grad_output,x,w)       
  ```

**step4：正确性验证，性能测试**

在 lab1 的profiler基础上，进行修改：

- 原 profiler 每次只执行1次计算，将其修改为每次执行 64 次计算（默认 batch size），取平均值
- 将 profile 结果输出到文件中

```python
# Add profile function
def profile(model, device, train_loader,f):
    # f 为 profile 结果输出文件
    dataiter = iter(train_loader)
    data, target = dataiter.next()
    data, target = data.to(device), target.to(device)
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        for i in range(64):	# 执行 64 次计算
            model(data[i].reshape(1,1,28,28))
    print(prof.key_averages(), file=f)
```

性能测试结果见 `3 实验结果`

### 3 实验结果

一些 profile 的测试结果放在了resources/profile 文件夹下，取平均值得到如下结果：

| 实现方式（Linear层为例）             | &nbsp; &nbsp; &nbsp; 性能评测                         |
| ------------------------------------ | ----------------------------------------------------- |
| PyTorch原有张量运算&nbsp;            | &nbsp; 88 μs / call&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| 基于Python API的定制化张量运算&nbsp; | 93 μs / call                                          |
| 基于C++的定制化张量运算              | 106 μs / call                                         |

