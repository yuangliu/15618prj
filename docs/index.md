Yuang Liu (yuangl) and Shuyao Bi (shuyaob)

# Proposal

## Summary
We are going to implement a Domain-Specific Language (DSL) for defining Long Short-Term Memory (LSTM). The DSL will be able to generate CUDA code based from Python, making use of cuDNN or cuBLAS. By developing a series of scheduling mechanisms, it will support efficient execution for different LSTM variants.

## Background

## Challenges

## Resources
GHC machines with NVIDIA GeForce GTX 1080.

Probably start with [CUDA code optimized for LSTM](https://github.com/parallel-forall/code-samples/tree/master/posts/rnn)

## Goals and Deliverables

## Platform
CUDA.

Why?

## Schedule

| Week | Plan | Note |
| :--- |:---| :---|
| Apr 9 |   |   |
| Apr 16 |   |   |
| Apr 23 |   |   |
| Apr 30 |   |   |
| May 7 |   |   |

## References
- Christopher Olah, Understanding LSTM Networks. [Link](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Klaus Greff, Rupesh K. Srivastava, Jan Koutník, Bas R. Steunebrink, and Jürgen Schmidhuber. "LSTM: A search space odyssey." IEEE transactions on neural networks and learning systems. 2016. [PDF](https://arxiv.org/pdf/1503.04069.pdf)
- Jeremy Appleyard, Optimizing Recurrent Neural Networks in cuDNN 5. [Link](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/)
- Jeremy Appleyard, Tomáš Kociský, and Phil Blunsom. "Optimizing Performance of Recurrent Neural Networks on GPUs." arXiv preprint arXiv:1604.01946 (2016). [PDF](https://arxiv.org/pdf/1604.01946.pdf)
