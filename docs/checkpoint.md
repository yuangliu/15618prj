---
layout: null
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
# Project Checkpoint

## Summary
<!--One to two paragraphs, summarize the work that you have completed so far. (This should be easy if you have been maintaining this information on your project page.)-->

In the past two weeks, we first investigated related papers and codes about parallelizing LSTM on GPUs. Based on [a serial code implemented in Python](http://nicodjimenez.github.io/2014/08/08/lstm.html), we added multiple layer LSTM training, and did a few experiment on toy data to make sure our implementation was correct. Besides, we also supported the Peephole variant of LSTM, based on the formulas in [LSTM: A Search Space Odyssey](https://arxiv.org/pdf/1503.04069.pdf).

For the parallel part, we ran the [starter CUDA code](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/) on GHC machines. The code only did the forward process of training, so we added the back propagation of LSTM. We were trying to run LSTM on [cuDNN](https://developer.nvidia.com/cudnn) and [TensorFlow](https://www.tensorflow.org/tutorials/recurrent) as baselines, but we have not finished yet because we were trying to get familiar the environment on GHC machines. 

## Goals and Deliverables
<!--Describe how you are doing with respect to the goals and deliverables stated in your proposal. Do you still believe you will be able to produce all your deliverables? If not, why? What about the "nice to haves"? In your checkpoint writeup we want a new list of goals that you plan to hit for the Parallelism competition.-->
We think most of our priliminary goals could be achieved before the parallelism competition. Although we are not halfway through the project, the approaches towards our goals are clear. We should be able to finish at least the correctness part, and the performance part will depend on the efforts and time we put in the end. 

<!--What do you plan to show at the parallelism competition? Will it be a demo? Will it be a graph?-->
**Our plan is to show a graph of the speedup of our design.**
### Plan to achieve:
* Implement Python-based DSL that can schedule LSTM tasks on GPU and ensure correctness.
* Identify primitives that can provide generic performance speedups for variants of LSTMs.
* Implement LSTM with NVIDIA Deep Learning SDK (e.g. cuDNN, cnBLAS) which acts as the baseline of evaluation. 
* Evaluate implementations for LSTM variants on LSTM benchmark problems (e.g. acoustic modeling, handwriting recognition and polyphinic music modeling).

### Hope to achieve:
* Provide user-friendly interface.
* Achieve comparable implementation with Tensorflow and Theano.

# Issues
<!--List the issues that concern you the most. Are there any remaining unknowns (things you simply don't know how to solve, or resource you don't know how to get) or is it just a matter of coding and doing the work? If you do not wish to put this information on a public web site you are welcome to email the staff directly.-->
* One concern is that we did not figure out how to run cuDNN or TensorFlow on GHC machine environments, due to limited permissions. We will try them in next a few days. If we still fail, we will contact TAs as soon as possible or try to switch to other platforms.
* Another one is that we are not sure that our code could run faster than other designs, since for basic matrix operations, we are planning make use of cuDNN library that is also used by other platforms. But we could definitely develop other features, such as LSTM variants supports, if we could not beat others.

## Schedule
<!--Make sure your project schedule on your main project page is up to date with work completed so far, and well as with a revised plan of work for the coming weeks. As by this time you should have a good understanding of what is required to complete your project, I want to see a very detailed schedule for the coming weeks. I suggest breaking time down into half-week increments. Each increment should have at least one task, and for each task put a person's name on it.-->

| (Sub)Week | Plan | Note | By |
| :--- |:---| :---| :---|
| Apr 9 -- Apr 15 | Paper and code research; correct code implementation | ✅ | Yuang |
| Apr 16 -- Apr 22 | CUDA code implementation and optimization | ✅ | Shuyao |
| Apr 23 -- Apr 26 | Exploration on LSTM variants | | Shuyao |
| Apr 27 -- Apr 29 | Code auto-generation | | Yuang|
| Apr 30 -- May 3 | Scheduling optimizations for LSTM variants  | | Shuyao |
| May 4 -- May 6 | _Cont'd_  | | Yuang |
| May 7 -- May 10 | Catch-up | | Yuang |
|  | Performance tuning | | Shuyao |
| May 11 -- May 12 | Wrap up; presentation | | Both |
