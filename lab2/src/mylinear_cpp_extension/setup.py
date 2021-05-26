from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 头文件目录
include_dirs = os.path.dirname(os.path.abspath(__file__))
#源代码目录 
source_file = glob.glob(os.path.join( 'mylinear.cpp'))

setup(
    name='mylinear_cpp',  # 模块名称
    ext_modules=[CppExtension('mylinear_cpp', sources=source_file, include_dirs=[include_dirs])],
    cmdclass={
        'build_ext': BuildExtension
    }
)