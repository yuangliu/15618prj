import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def tanh(x):
    output = np.tanh(x)
    return output

def tanh_output_to_derivative(output):
    return 1 - output * output

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


# input variables
alpha = 0.1
input_dim = 2
#hidden_dim = 16
output_dim = 1


# initialize neural network weights
W_i = 2*np.random.random((output_dim,input_dim)) - 1
W_f = 2*np.random.random((output_dim,input_dim)) - 1
W_o = 2*np.random.random((output_dim,input_dim)) - 1
W_c = 2*np.random.random((output_dim,input_dim)) - 1
R_i = 2*np.random.random((output_dim,output_dim)) - 1
R_f = 2*np.random.random((output_dim,output_dim)) - 1
R_o = 2*np.random.random((output_dim,output_dim)) - 1
R_c = 2*np.random.random((output_dim,output_dim)) - 1
b_i = 2*np.random.random((output_dim,1)) - 1
b_f = 2*np.random.random((output_dim,1)) - 1
b_o = 2*np.random.random((output_dim,1)) - 1
b_c = 2*np.random.random((output_dim,1)) - 1
#synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
#synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

W_i_update = np.zeros_like(W_i)
W_f_update = np.zeros_like(W_f)
W_o_update = np.zeros_like(W_o)
W_c_update = np.zeros_like(W_c)
R_i_update = np.zeros_like(R_i)
R_f_update = np.zeros_like(R_f)
R_o_update = np.zeros_like(R_o)
R_c_update = np.zeros_like(R_c)
b_i_update = np.zeros_like(b_i)
b_f_update = np.zeros_like(b_f)
b_o_update = np.zeros_like(b_o)
b_c_update = np.zeros_like(b_c)
#synapse_1_update = np.zeros_like(synapse_1)
#synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10000):
    
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

    overallError = 0
    
    #layer_2_deltas = list()
    i_list = list()
    f_list = list()
    o_list = list()
    cp_list = list()
    c_list = list()
    c_list.append(np.zeros(output_dim))
    h_list = list()
    h_list.append(np.zeros(output_dim))
    error_list = list()

    #layer_2_deltas = list()
    #layer_1_values = list()
    #layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]]).T
        y = np.array([[e[binary_dim - position - 1]]]).T

        i = sigmoid(np.dot(W_i,X) + np.dot(R_i,h_list[-1]) + b_i)
        f = sigmoid(np.dot(W_f,X) + np.dot(R_f,h_list[-1]) + b_f)
        o = sigmoid(np.dot(W_o,X) + np.dot(R_o,h_list[-1]) + b_o)
        cp = tanh(np.dot(W_c,X) + np.dot(R_c,h_list[-1]) + b_c)
        c = f * c_list[-1] + i * cp
        h = o * tanh(c)

        error = y - h
        #layer_2_deltas.append((error)*tanh_output_to_derivative(h))
        overallError += np.abs(error[0])

        d[binary_dim - position - 1] = np.round(h[0][0])

        i_list.append(copy.deepcopy(i))
        f_list.append(copy.deepcopy(f))
        o_list.append(copy.deepcopy(o))
        cp_list.append(copy.deepcopy(cp))
        c_list.append(copy.deepcopy(c))
        h_list.append(copy.deepcopy(h))
        error_list.append(copy.deepcopy(error))

        # # hidden layer (input ~+ prev_hidden)
        # layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # # output layer (new binary representation)
        # layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # # did we miss?... if so, by how much?
        # layer_2_error = y - layer_2
        # layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        # overallError += np.abs(layer_2_error[0])
    
        # # decode estimate so we can print it out
        # d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # # store hidden layer so we can use it in the next timestep
        # layer_1_values.append(copy.deepcopy(layer_1))
    
    prev_c_delta = np.zeros_like(b_c)
    next_h_delta = np.zeros_like(b_c)
    #future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]]).T

        error = error_list[-position-1] + next_h_delta
        i = i_list[-position-1]
        f = f_list[-position-1]
        o = o_list[-position-1]
        cp = cp_list[-position-1]
        h = h_list[-position-1]
        prev_h = h_list[-position-2]
        c = c_list[-position-1]
        prev_c = c_list[-position-2]

        o_delta = error * tanh(c)
        prev_c_delta += error * o * tanh_output_to_derivative(tanh(c))
        i_delta = prev_c_delta * cp
        f_delta = prev_c_delta * prev_c
        cp_delta = prev_c_delta * i
        prev_c_delta = prev_c_delta * f
        b_i_delta = i_delta * sigmoid_output_to_derivative(i)
        b_f_delta = f_delta * sigmoid_output_to_derivative(f)
        b_o_delta = o_delta * sigmoid_output_to_derivative(o)
        b_c_delta = cp_delta * tanh_output_to_derivative(cp)

        W_i_update += b_i_delta.dot(X.T)
        W_f_update += b_f_delta.dot(X.T)
        W_o_update += b_o_delta.dot(X.T)
        W_c_update += b_c_delta.dot(X.T)
        R_i_update += b_i_delta.dot(prev_h.T)
        R_f_update += b_f_delta.dot(prev_h.T)
        R_o_update += b_o_delta.dot(prev_h.T)
        R_c_update += b_c_delta.dot(prev_h.T)
        b_i_update += b_i_delta
        b_f_update += b_f_delta
        b_o_update += b_o_delta
        b_c_update += b_c_delta
        next_h_delta = R_i.T.dot(b_i_delta) + R_f.T.dot(b_f_delta) + R_o.T.dot(b_o_delta) + R_c.T.dot(b_c_delta)

        # layer_1 = layer_1_values[-position-1]
        # prev_layer_1 = layer_1_values[-position-2]
        
        # # error at output layer
        # layer_2_delta = layer_2_deltas[-position-1]
        # # error at hidden layer
        # layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # # let's update all our weights so we can try again
        # synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        # synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        # synapse_0_update += X.T.dot(layer_1_delta)
        
        # future_layer_1_delta = layer_1_delta
    
    W_i += W_i_update * alpha
    W_f += W_f_update * alpha
    W_o += W_o_update * alpha
    W_c += W_c_update * alpha
    R_i += R_i_update * alpha
    R_f += R_f_update * alpha
    R_o += R_o_update * alpha
    R_c += R_c_update * alpha
    b_i += b_i_update * alpha
    b_f += b_f_update * alpha
    b_o += b_o_update * alpha
    b_c += b_c_update * alpha

    W_i_update *= 0
    W_f_update *= 0
    W_o_update *= 0
    W_c_update *= 0
    R_i_update *= 0
    R_f_update *= 0
    R_o_update *= 0
    R_c_update *= 0
    b_i_update *= 0
    b_f_update *= 0
    b_o_update *= 0
    b_c_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(e)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"

        