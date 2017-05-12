import numpy as np

from culstm import LstmInput, LstmOutput, LstmNetwork, LstmConfig, LstmNode
import culstm

def example_0():
    # parameters for input data dimension and lstm cell count 
    hiddenSize = 32
    inputSize = 32
    numLayers = 4
    seqLength = 20
    miniBatch = 64
    lstm_input = LstmInput(inputSize, seqLength)
    for e in range(miniBatch):
        lstm_input.add([[0.2 for x in range(inputSize)] for x in range(seqLength)])
    lstm_output = LstmOutput(hiddenSize, seqLength)
    for e in range(miniBatch):
        lstm_output.add([[1 for x in range(1)] for x in range(seqLength)])

    lstm_config = LstmConfig(inputSize, hiddenSize, numLayers, seqLength, miniBatch, loss_func = culstm.square)
    lstm_node = LstmNode(gates = culstm.vanilla, Peepholes=False)
    lstm_net = LstmNetwork(lstm_config, lstm_node)
    lstm_net.run(lstm_input, lstm_output)
    lstm_net.clean()

if __name__ == "__main__":
    example_0()

