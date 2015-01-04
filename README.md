Sparse Gibbs sampler for LDA
====

This is a minimalistic C++11 implementation of [Sparse sampling for
LDA](http://people.cs.umass.edu/~lmyao/papers/fast-topic-model10.pdf).

This package depends on `glog`, `gflags`, and `gperftools`(optional). To build,

    cd some_directory
    git clone https://github.com/xunzheng/third_party
    cd third_party
    ./install.sh

Third party libraries will be installed at `some_directory/third_party/`.

Now we can build the project:

    git clone https://github.com/xunzheng/topic
    cd topic
    ln -s some_directory/third_party third_party
    make

Toy dataset `20news.train` is included in the `exp/` directory. Try

    cd exp
    ./run.sh

to get a sample run. Since LDA is an unsupervised model, label information in
LIBSVM format is ignored.

To see all the available flags, run

    ./gibbs

without any flags.

Reference
----
Limin Yao, David Mimno, and Andrew McCallum. Efficient Methods for Topic Model Inference on Streaming Document Collections. In *International Conference on Knowledge Discovery and Data mining (SIGKDD)*, 2009.
