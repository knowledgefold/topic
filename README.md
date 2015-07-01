Sparse Gibbs sampler for LDA
====

This is a minimalistic C++11 implementation of [Sparse sampling for
LDA](http://people.cs.umass.edu/~lmyao/papers/fast-topic-model10.pdf).

To compile, simply type

    make

Toy dataset `20news.train` is included in the `exp/` directory. Try

    cd exp
    ./run.sh

to get a sample run. Since LDA is an unsupervised model, label information in
LIBSVM format is ignored.

To see all the available flags, type

    ./sparselda -h


Reference
----
Limin Yao, David Mimno, and Andrew McCallum. Efficient Methods for Topic Model Inference on Streaming Document Collections. In *International Conference on Knowledge Discovery and Data mining (SIGKDD)*, 2009.
