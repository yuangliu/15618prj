# Proposal

## Summary
<!--Summarize your project in no more than 2-3 sentences. Describe what you plan to do and what parallel systems you will be working with. -->

We are going to implement a Domain-Specific Language (DSL) for defining Long Short-Term Memory (LSTM). The DSL will be able to generate CUDA code based from Python, making use of cuDNN or cuBLAS. By developing a series of scheduling mechanisms, it will support efficient execution for different LSTM variants.

## Background
<!--If your project involves accelerating a compute-intensive application, describe the application or piece of the application you are going to implement in more detail. This description need only be a few paragraphs. It might be helpful to include a block diagram or pseudocode of the basic idea. An important detail is what aspects of the problem might benefit from parallelism? and why?-->
As a recurrent neural network (RNN) architecture, Long short-term memory (LSTM) is excel at learning from past experience to classify, process, and predict time series. The LSTM units are able to remember values for either long or short durations of time, due to no activation function within the recurrent components. Containing several LSTM units, a LSTM block contains three or four "gates" controlling the flow of information.  $W$
![LSTM chains](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
(E=mc^2)，$$x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}.$$

**Theorem**: The translation $[\![e]\!]$ given by

Let $\text{S}_1(N) = \sum_{p=1}^N \text{E}(p)$



## Challenges
<!--Describe why the problem is challenging. What aspects of the problem might make it difficult to parallelize? In other words, what to you hope to learn by doing the project?

- Describe the workload: what are the dependencies, what are its memory access characteristics? (is there locality? is there a high communication to computation ratio?), is there divergent execution?
- Describe constraints: What are the properties of the system that make mapping the workload to it challenging?
-->

## Resources
<!--Describe the resources (type of computers, starter code, etc.) you will use. What code base will you start from? Are you starting from scratch or using an existing piece of code? Is there a book or paper that you are using as a reference (if so, provide a citation)? Are there any other resources you need, but haven't figured out how to obtain yet? Could you benefit from access to any special machines?-->

GHC machines with NVIDIA GeForce GTX 1080.  
Probably start with [CUDA code optimized for LSTM](https://github.com/parallel-forall/code-samples/tree/master/posts/rnn)

## Goals and Deliverables
<!--Describe the deliverables or goals of your project.-->

## Platform
<!--Describe why the platform (computer and/or language) you have chosen is a good one for your needs. Why does it make sense to use this parallel system for the workload you have chosen?-->

CUDA.  
Why?

## Schedule
<!--Produce a schedule for your project. Your schedule should have at least one item to do per week. List what you plan to get done each week from now until the parallelism competition in order to meet your project goals. Keep in mind that due to other classes, you'll have more time to work some weeks than others (work that into the schedule). You will need to re-evaluate your progress at the end of each week and update this schedule accordingly. Note the intermediate checkpoint deadline is April 25th. In your schedule we encourage you to be precise as precise as possible. It's often helpful to work backward in time from your deliverables and goals, writing down all the little things you'll need to do (establish the dependencies!).-->

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
