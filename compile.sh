TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_INC="$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')"
TF_LIB="$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')"
g++ -std=c++11 -shared batcher.cc -o batcher.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -ltensorflow_framework ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
