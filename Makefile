nvcc := /usr/local/cuda/bin/nvcc
cudalib := /usr/local/cuda/lib64
cudainclude := /usr/local/cuda/include
tensorflow := $(HOME)/.local/lib/python3.8/site-packages/tensorflow/include
eigen := /usr/include/eigen3
core_path := ./src/chamfer_distance/core/

TF_CFLAGS := $$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') 
TF_LFLAGS := $$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')


all: $(core_path)tf_nndistance_so.so
clean:
	rm -rf $(core_path)*.o $(core_path)*.so
.PHONY : all clean

$(core_path)tf_nndistance_so.so: $(core_path)tf_nndistance_g.o $(core_path)tf_nndistance.cpp
	g++ -std=c++14 -shared $(core_path)tf_nndistance.cpp $(core_path)tf_nndistance_g.o -o $(core_path)tf_nndistance_so.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2

$(core_path)tf_nndistance_g.o: $(core_path)tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -c -o $(core_path)tf_nndistance_g.o $(core_path)tf_nndistance_g.cu -I $(eigen) -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

