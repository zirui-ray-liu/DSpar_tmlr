from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='dspar',
      ext_modules=[
          cpp_extension.CppExtension('dspar.cpp_extension.sampler', 
                                     ['dspar/cpp_extension/edge_sample.cc'],
                                     extra_compile_args=['-fopenmp'])
          
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)