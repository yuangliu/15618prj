1) Environment
The code was tested on GHC machines. Run "source env.sh" to enable CUDA and TensorFlow environment.
2) Performance
Modify the settings in makefile. Use "make demo" to run sequential code, TensorFlow, and CuLSTM. Use "make seqlstm/tflstm/cu" to each of them. The result is in "res.txt".
3) CuLSTM Interface
In "culstm" directory, modify "test.py" to test different LSTM settings. Use "make" to generate and run CUDA code.
4) Directories
- seq: Sequential code in python
- tf: TensorFlow code
- rnn: Plain CUDA code
- culstm: Auto-generator library