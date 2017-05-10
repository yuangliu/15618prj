---
layout: null
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
# Final Writeup

## Summary
<!--A short (no more than a paragraph) project summary. If applicable, the summary should list your project deliverables (including what you plan to show at the parallelism competition) and what machines they ran on.-->

In this project, we implemented **CuLSTM**, a _Domain-Specific Language_ (DSL) for _Long Short-Term Memory_ (LSTM). The DSL is able to generate CUDA code based on LSTM network definition and specification in Python. CuLSTM supports multiple LSTM variants, and allows great flexibility, such as user-defined loss function and parameters. The outcome is a productive Python interface, as well as a performant program running on GPU. The evaluations on GHC machines (with NVIDIA GeForce GTX 1080) showed that our implementation achieved a XXx speedup compared to the sequential version, and a XXx speedup compared to TensorFlow running on GPU.

## Background
<!--Describe the algorithm, application, or system you parallelized in computer science terms. (Recall our discussion from the last day of class.) Figure(s) would be really useful here.-->
As a _recurrent neural network_ (RNN) architecture, _long short-term memory_ (LSTM) is excel at learning from past experience to classify, process, and predict time series. The LSTM units are able to remember values for either long or short durations of time, due to no activation function within the recurrent components. Containing several LSTM units, a LSTM block contains three or four "gates" controlling the flow of information. Figure 1 shows the chain of repeating LSTM blocks with four LSTM units.

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" style="background-color:#666;"/>  
**Figure 1:** *The chain of LSTM blocks consisting of four LSTM units.*

Despite similar ideas, different LSTM variants have individual network structures and formulas. Common variants include vanilla LSTM [[Graves 2005](http://www.sciencedirect.com/science/article/pii/S0893608005001206)], traditional LSTM [[Hochreiter 1997](http://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)], Peephole LSTM [[Gers 2000](http://ieeexplore.ieee.org/abstract/document/861302/)], etc. The following formulas show the operations for traditional LSTM in each iteration.  
<!--![{\displaystyle {\begin{aligned}f_{t}&=\sigma _{g}(W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\\\\i_{t}&=\sigma _{g}(W_{i}x_{t}+U_{i}h_{t-1}+b_{i})\\\\o_{t}&=\sigma _{g}(W_{o}x_{t}+U_{o}h_{t-1}+b_{o})\\\\c_{t}&=f_{t}\circ c_{t-1}+i_{t}\circ \sigma _{c}(W_{c}x_{t}+U_{c}h_{t-1}+b_{c})\\\\h_{t}&=o_{t}\circ \sigma _{h}(c_{t})\end{aligned}}}](eqn.png)-->
$$\begin{aligned}f_{t}&=\sigma _{g}(W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\\i_{t}&=\sigma _{g}(W_{i}x_{t}+U_{i}h_{t-1}+b_{i})\\o_{t}&=\sigma _{g}(W_{o}x_{t}+U_{o}h_{t-1}+b_{o})\\c_{t}&=f_{t}\circ c_{t-1}+i_{t}\circ \sigma _{c}(W_{c}x_{t}+U_{c}h_{t-1}+b_{c})\\h_{t}&=o_{t}\circ \sigma _{h}(c_{t})\end{aligned}$$

The training for LSTM involves a series of _matrix-matrix multiplications_ (GEMMs) and lots of point-wise operations on vectors.  Therefore, it is both necessary and natural to execute it in parallel. However, writing efficient CUDA code is troublesome for some users and machine learning researchers. Our goal is develop a Python library that generates CUDA code automatically. By identifying the pattern of LSTM variants, our library could schedule them with different schemes, and achieve a good performance in most cases.

## Approach
<!--Tell us how your implementation works. Your description should be sufficiently detailed to provide the course staff a basic understanding of your approach. Again, it might be very useful to include a figure here illustrating components of the system and/or their mapping to parallel hardware.-->

## Results
<!--How successful were you at achieving your goals? We expect results sections to differ from project to project, but we expect your evaluation to be very thorough (your project evaluation is a great way to demonstrate you understood topics from this course).-->
### Data Movement Cost
Data movement is a main source of time and energy cost for GPU applications. We recorded the time of data initialization and memory free, and showed the result in Fig. 2. The absolute time (in ms) and percentage of each part are shown in the figure. We can see from Fig. 2(a) that, when we train the network for 10 iterations, the cost of initialization is relatively high. However, in Fig. 2(b), as we train for more iterations, it remains constant, and becomes trivial relative to running time.

<img src="data1.png" style="background-color:#666;"/>  
**(a)** *10 iterations.*
<img src="data2.png" style="background-color:#666;"/>  
**(b)** *1,000 iterations.*  
**Figure 2:** *Cost of data movement. (ms)*


## References
<!--Please provide a list of references used in the project.-->
- Christopher Olah. "Understanding LSTM Networks." [Link](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Klaus Greff, Rupesh K. Srivastava, Jan Koutník, Bas R. Steunebrink, and Jürgen Schmidhuber. "LSTM: A Search Space Odyssey." IEEE transactions on neural networks and learning systems (2016). [PDF](https://arxiv.org/pdf/1503.04069.pdf)
- Jeremy Appleyard. "Optimizing Recurrent Neural Networks in cuDNN 5." [Link](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/)
- Jeremy Appleyard, Tomáš Kociský, and Phil Blunsom. "Optimizing Performance of Recurrent Neural Networks on GPUs." arXiv preprint arXiv:1604.01946 (2016). [PDF](https://arxiv.org/pdf/1604.01946.pdf)
- Szymon Sidor, "Simple Implementation of LSTM in Tensorflow." [Link](https://gist.github.com/nivwusquorum/b18ce332bde37e156034e5d3f60f8a23)

## Work by Each Student
<!--If your project is a team project, please list the work performed by each partner. If you do not feel comfortable placing this information on a public web page, you may email the course staff this information directly. Alternatively, you can simply state: "equal work was performed by both project members."-->
Equal work was performed by both project members.
