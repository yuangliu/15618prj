import numpy as np

from multilstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count 
    mem_cell_ct = 100
    x_dim = 50
    lstm_param = LstmParam(mem_cell_ct, x_dim) 
    lstm_net = LstmNetwork(lstm_param)
    y_list = [-0.5,0.2,0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(100):
        print("cur iter: " + str(cur_iter))
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            print("y_pred[" + str(ind) + "] : " + str(lstm_net.lstm_node_list[ind].state.h[0]))

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss: " + str(loss))
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

def example_0_multi():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count 
    mem_cell_ct = 100
    x_dim = 50
    lstm_param = LstmParam(mem_cell_ct, x_dim) 
    lstm_hidden_param = LstmParam(mem_cell_ct, mem_cell_ct) 
    lstm_net = LstmNetwork(lstm_param, lstm_hidden_param, 2)
    y_list = [-0.5,0.2,0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(10000):
        print("cur iter: " + str(cur_iter))
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            print("y_pred[" + str(ind) + "] : " + str(lstm_net.lstm_hidden_node_list[ind].state.h[0]))

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss: " + str(loss))
        lstm_net.apply_diff(0.1, 0.1)
        lstm_net.x_list_clear()

def example_1():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)
    # training dataset generation
    int2binary = {}
    binary_dim = 8

    largest_number = pow(2,binary_dim)
    binary = np.unpackbits(
        np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
    for i in range(largest_number):
        int2binary[i] = binary[i]

    # parameters for input data dimension and lstm cell count 
    mem_cell_ct = 100
    x_dim = 2
    lstm_param = LstmParam(mem_cell_ct, x_dim) 
    lstm_net = LstmNetwork(lstm_param)
    #y_list = [-0.5,0.2,0.1, -0.5]
    #input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(10000):
        print("cur iter: " + str(cur_iter))
        # generate a simple addition problem (a + b = c)
        a_int = np.random.randint(largest_number/2) # int version
        a = int2binary[a_int] # binary encoding
        b_int = np.random.randint(largest_number/2) # int version
        b = int2binary[b_int] # binary encoding
        # true answer
        e_int = a_int + b_int
        e = int2binary[e_int]
        # where we'll store our best guess (binary encoded)
        d = np.zeros_like(e)
        y_list = [e[binary_dim - position - 1] for position in range(binary_dim)]
        input_val_arr = [[a[binary_dim - position - 1],b[binary_dim - position - 1]] for position in range(binary_dim)]

        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            #print("y_pred[" + str(ind) + "] : " + str(lstm_net.lstm_node_list[ind].state.h[0]))

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss: " + str(loss))
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

def example_1_multi():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)
    # training dataset generation
    int2binary = {}
    binary_dim = 8

    largest_number = pow(2,binary_dim)
    binary = np.unpackbits(
        np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
    for i in range(largest_number):
        int2binary[i] = binary[i]

    # parameters for input data dimension and lstm cell count 
    mem_cell_ct = 100
    x_dim = 2
    lstm_param = LstmParam(mem_cell_ct, x_dim) 
    lstm_hidden_param = LstmParam(mem_cell_ct, mem_cell_ct) 
    lstm_net = LstmNetwork(lstm_param, lstm_hidden_param, 2)
    #y_list = [-0.5,0.2,0.1, -0.5]
    #input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(10000):
        print("cur iter: " + str(cur_iter))
        # generate a simple addition problem (a + b = c)
        a_int = np.random.randint(largest_number/2) # int version
        a = int2binary[a_int] # binary encoding
        b_int = np.random.randint(largest_number/2) # int version
        b = int2binary[b_int] # binary encoding
        # true answer
        e_int = a_int + b_int
        e = int2binary[e_int]
        # where we'll store our best guess (binary encoded)
        d = np.zeros_like(e)
        y_list = [e[binary_dim - position - 1] for position in range(binary_dim)]
        input_val_arr = [[a[binary_dim - position - 1],b[binary_dim - position - 1]] for position in range(binary_dim)]

        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            print("y_pred[" + str(ind) + "] : " + str(lstm_net.lstm_hidden_node_list[ind].state.h[0]) + " " + str(y_list[ind]))

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss: " + str(loss))
        lstm_net.apply_diff(0.1, 0.1)
        lstm_net.x_list_clear()

if __name__ == "__main__":
    example_0_multi()

