---
theme:
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
# Project Proposal

## Summary
<!--Summarize your project in no more than 2-3 sentences. Describe what you plan to do and what parallel systems you will be working with. -->

We are going to implement a Domain-Specific Language (DSL) for Long Short-Term Memory (LSTM). The DSL will be able to translate Python code to CUDA blocks. By developing a series of scheduling mechanisms, it will support efficient execution for different LSTM variants.

## Background
<!--If your project involves accelerating a compute-intensive application, describe the application or piece of the application you are going to implement in more detail. This description need only be a few paragraphs. It might be helpful to include a block diagram or pseudocode of the basic idea. An important detail is what aspects of the problem might benefit from parallelism? and why?-->
As a recurrent neural network (RNN) architecture, Long short-term memory (LSTM) is excel at learning from past experience to classify, process, and predict time series. The LSTM units are able to remember values for either long or short durations of time, due to no activation function within the recurrent components. Containing several LSTM units, a LSTM block contains three or four "gates" controlling the flow of information. Figure 1 shows the chain of repeating LSTM blocks with four LSTM units.

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" style="background-color:#666;"/>  
**Figure 1:** *The chain of LSTM blocks consisting of  four LSTM units.*

Despite similar ideas, different LSTM variants have individual network structures and formulas. Common variants include vanilla LSTM [[Graves 2005](http://www.sciencedirect.com/science/article/pii/S0893608005001206)], traditional LSTM [[Hochreiter 1997](http://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)], Peephole LSTM [[Gers 2000](http://ieeexplore.ieee.org/abstract/document/861302/)], etc. The following formulas show the operations for traditional LSTM in each iteration.  
<!--![{\displaystyle {\begin{aligned}f_{t}&=\sigma _{g}(W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\\\\i_{t}&=\sigma _{g}(W_{i}x_{t}+U_{i}h_{t-1}+b_{i})\\\\o_{t}&=\sigma _{g}(W_{o}x_{t}+U_{o}h_{t-1}+b_{o})\\\\c_{t}&=f_{t}\circ c_{t-1}+i_{t}\circ \sigma _{c}(W_{c}x_{t}+U_{c}h_{t-1}+b_{c})\\\\h_{t}&=o_{t}\circ \sigma _{h}(c_{t})\end{aligned}}}](eqn.png)-->
$$\begin{aligned}f_{t}&=\sigma _{g}(W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\\i_{t}&=\sigma _{g}(W_{i}x_{t}+U_{i}h_{t-1}+b_{i})\\o_{t}&=\sigma _{g}(W_{o}x_{t}+U_{o}h_{t-1}+b_{o})\\c_{t}&=f_{t}\circ c_{t-1}+i_{t}\circ \sigma _{c}(W_{c}x_{t}+U_{c}h_{t-1}+b_{c})\\h_{t}&=o_{t}\circ \sigma _{h}(c_{t})\end{aligned}$$

The training for LSTM involves a series of matrix-matrix multiplications (GEMMs) and lots of point-wise operations on vectors.  Therefore, it is both necessary and natural to execute it in parallel. However, writing efficient CUDA code is troublesome for some users and maching learning researchers. Our goal is develop a Python library that generates CUDA code automatically. By identifying the pattern of LSTM variants, our library could schedule them with different schemes, and achieve a good performance in most cases.

## Challenges
<!--Describe why the problem is challenging. What aspects of the problem might make it difficult to parallelize? In other words, what to you hope to learn by doing the project?

- Describe the workload: what are the dependencies, what are its memory access characteristics? (is there locality? is there a high communication to computation ratio?), is there divergent execution?
- Describe constraints: What are the properties of the system that make mapping the workload to it challenging?
-->
* LSTM training involves multiple layers, each with multiple iterations performed in a cell. In each iteration, eight GEMMs need to be performed with input from former iteration and former layer. Moreover, lots of point-wise operations are to be performed. Since there exist dependencies between tasks within iteration, among iterations and layers, it is challenging to exploit parallism that can fully utilize hardware resources while maintaining correctness of result and generality of translation.
* The output of previous iteration will be used as the input of each iteration. The data transfer can therefore generate a lot of communicationS between kernel and device memory. 
* LSTM has many variants with different network structures and formulas. Identifying and finding a generic solution suitable for all variants of LSTMs is difficult.


## Resources
<!--Describe the resources (type of computers, starter code, etc.) you will use. What code base will you start from? Are you starting from scratch or using an existing piece of code? Is there a book or paper that you are using as a reference (if so, provide a citation)? Are there any other resources you need, but haven't figured out how to obtain yet? Could you benefit from access to any special machines?-->

* GHC machines with NVIDIA GeForce GTX 1080.  
* [Appleyard](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/) has proposed several optimizations for traditional LSTM. The code is publicly available on [GitHub](https://github.com/parallel-forall/code-samples/tree/master/posts/rnn), which is a good source to start with.
* [Iamtrask](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/) posts tutorial and code for programming LSTM with Python, which can be helpful to get familiar with LSTM implentation.
* Papers and articles listed in the [references](#references) are great materials to get us on track.


## Goals and Deliverables
<!--Describe the deliverables or goals of your project.-->
### Plan to achieve:
* Implement Python-based DSL that can schedule LSTM tasks on GPU and ensure correctness.
* Identify primitives that can provide generic performance speedups for variants of LSTMs.
* Implement LSTM with NVIDIA Deep Learning SDK (e.g. cuDNN, cnBLAS) which acts as the baseline of evaluation. 
* Evaluate implementations for LSTM variants on LSTM benchmark problems (e.g. acoustic modeling, handwriting recognition and polyphinic music modeling).

### Hope to achieve:
* Provide user-friendly interface.
* Achieve comparable implementation with Tensorflow and Theano.


## Platform
<!--Describe why the platform (computer and/or language) you have chosen is a good one for your needs. Why does it make sense to use this parallel system for the workload you have chosen?-->

CUDA, due to the property that tons of GEMMs and point point-wise operations are performed in LSTM. It makes sense to parallelize those computations and achieve high-performance on GPU which is good at SPMD execution.

## Schedule
<!--Produce a schedule for your project. Your schedule should have at least one item to do per week. List what you plan to get done each week from now until the parallelism competition in order to meet your project goals. Keep in mind that due to other classes, you'll have more time to work some weeks than others (work that into the schedule). You will need to re-evaluate your progress at the end of each week and update this schedule accordingly. Note the intermediate checkpoint deadline is April 25th. In your schedule we encourage you to be precise as precise as possible. It's often helpful to work backward in time from your deliverables and goals, writing down all the little things you'll need to do (establish the dependencies!).-->

| Week | Plan | Note |
| :--- |:---| :---|
| Apr 9 | Paper and code research; correct code implementation | **Proposal** |
| Apr 16 | CUDA code implementation and optimization |   |
| Apr 23 | Code auto-generation; exploration on LSTM variants | **Checkpoint** |
| Apr 30 | Scheduling optimizations for LSTM variants  | _Exam week_ |
| May 7 | Catch-up; performance tuning; wrap up | **Parallelism Competition** |

## References
- Christopher Olah. "Understanding LSTM Networks." [Link](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Klaus Greff, Rupesh K. Srivastava, Jan Koutník, Bas R. Steunebrink, and Jürgen Schmidhuber. "LSTM: A Search Space Odyssey." IEEE transactions on neural networks and learning systems (2016). [PDF](https://arxiv.org/pdf/1503.04069.pdf)
- Jeremy Appleyard. "Optimizing Recurrent Neural Networks in cuDNN 5." [Link](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/)
- Jeremy Appleyard, Tomáš Kociský, and Phil Blunsom. "Optimizing Performance of Recurrent Neural Networks on GPUs." arXiv preprint arXiv:1604.01946 (2016). [PDF](https://arxiv.org/pdf/1604.01946.pdf)
