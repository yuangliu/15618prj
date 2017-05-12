import random

import numpy as np
import math
import copy

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    return values*(1-values)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(values): 
    return 1. - values ** 2

def linear(x):
    return x

def one(values):
    return values * 0 + 1.

def derivative(func):
    if func == linear:
        return one
    elif func == tanh:
        return tanh_derivative
    elif func == sigmoid:
        return sigmoid_derivative


# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args): 
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct) 
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct) 

    def apply_diff(self, lr = 1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff

        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wf_diff = np.zeros_like(self.wf) 
        self.wo_diff = np.zeros_like(self.wo) 
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo) 

class LstmParamPeephole(LstmParam):
    def __init__(self, mem_cell_ct, x_dim, ):
        LstmParam.__init__(self, mem_cell_ct, x_dim)
        self.pi = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.pf = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.po = rand_arr(-0.1, 0.1, mem_cell_ct)

        self.pi_diff = np.zeros_like(self.pi) 
        self.pf_diff = np.zeros_like(self.pf) 
        self.po_diff = np.zeros_like(self.po) 

    def apply_diff(self, lr = 1):
        LstmParam.apply_diff(self, lr)

        self.po -= lr * self.po_diff
        self.pi -= lr * self.pi_diff
        self.pf -= lr * self.pf_diff
        # reset diffs to zero
        self.pi_diff = np.zeros_like(self.pi) 
        self.pf_diff = np.zeros_like(self.pf) 
        self.po_diff = np.zeros_like(self.po)

class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.hs = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        self.bottom_diff_x = np.zeros(x_dim) 

class LstmNode:
    def __init__(self, lstm_param, lstm_state, g_func, h_func):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None
        self.g_func = g_func
        self.h_func = h_func
        self.g_func_derivative = derivative(g_func)
        self.h_func_derivative = derivative(h_func)

    def bottom_data_is(self, x, s_prev = None, h_prev = None):
        # if this is the first lstm node in the network
        if s_prev == None: s_prev = np.zeros_like(self.state.s)
        if h_prev == None: h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  h_prev))
        self.state.g = self.g_func(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.hs = self.h_func(self.state.s)
        self.state.h = self.state.hs * self.state.o

        self.xc = xc
    
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.h_func_derivative(self.state.hs) * self.state.o * top_diff_h + top_diff_s
        do = self.state.hs * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di 
        df_input = sigmoid_derivative(self.state.f) * df 
        do_input = sigmoid_derivative(self.state.o) * do 
        dg_input = self.g_func_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]
        self.state.bottom_diff_x = dxc[:self.param.x_dim]

class LstmNodePeephole(LstmNode):
    def __init__(self, lstm_param, lstm_state, g_func, h_func):
        LstmNode.__init__(self, lstm_param, lstm_state, g_func, h_func)

    def bottom_data_is(self, x, s_prev = None, h_prev = None):
        # if this is the first lstm node in the network
        if s_prev == None: s_prev = np.zeros_like(self.state.s)
        if h_prev == None: h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  h_prev))
        self.state.g = self.g_func(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.pi * s_prev + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.pf * s_prev + self.param.bf)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.po * self.state.s + self.param.bo)
        self.state.hs = self.h_func(self.state.s)
        self.state.h = self.state.hs * self.state.o

        self.xc = xc
    
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        do = self.state.hs * top_diff_h
        do_input = sigmoid_derivative(self.state.o) * do 
        ds = self.h_func_derivative(self.state.hs) * self.state.o * top_diff_h + self.param.po * do_input + top_diff_s
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di 
        df_input = sigmoid_derivative(self.state.f) * df 
        dg_input = self.g_func_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input
        self.param.pi_diff += self.s_prev * di_input
        self.param.pf_diff += self.s_prev * df_input
        self.param.po_diff += self.state.s * do_input

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f + self.param.pi * di_input + self.param.pf * df_input
        self.state.bottom_diff_h = dxc[self.param.x_dim:]
        self.state.bottom_diff_x = dxc[:self.param.x_dim]

class LstmNetwork():
    def __init__(self, lstm_param, lstm_hidden_param = None, layer = 0, g_func = tanh, h_func = tanh, peephole = False):
        self.layer = layer
        self.lstm_param = [copy.deepcopy(lstm_hidden_param) for x in range(layer)]
        self.lstm_param[0] = lstm_param
        self.lstm_node_list = [copy.deepcopy([]) for x in range(layer)]
        # input sequence
        self.x_list = []
        self.g_func = g_func
        self.h_func = h_func
        self.peephole = peephole

    def y_list_is(self, y_list, loss_layer):
        """
        Updates diffs by setting target sequence 
        with corresponding loss layer. 
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        for l in range(self.layer):
            ll = self.layer - l - 1
            idx = len(self.x_list) - 1
            # first node only gets diffs from label ...
            if l == 0:
                loss = loss_layer.loss(self.lstm_node_list[ll][idx].state.h, y_list[idx])
                diff_h = loss_layer.bottom_diff(self.lstm_node_list[ll][idx].state.h, y_list[idx])
            else:
                diff_h = self.lstm_node_list[ll + 1][idx].state.bottom_diff_x
            # here s is not affecting loss due to h(t+1), hence we set equal to zero
            diff_s = np.zeros(self.lstm_param[ll].mem_cell_ct)
            self.lstm_node_list[ll][idx].top_diff_is(diff_h, diff_s)
            idx -= 1

            ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
            ### we also propagate error along constant error carousel using diff_s
            while idx >= 0:
                if l == 0:
                    loss += loss_layer.loss(self.lstm_node_list[ll][idx].state.h, y_list[idx])
                    diff_h = loss_layer.bottom_diff(self.lstm_node_list[ll][idx].state.h, y_list[idx])
                else:
                    diff_h = self.lstm_node_list[ll + 1][idx].state.bottom_diff_x
                diff_h += self.lstm_node_list[ll][idx + 1].state.bottom_diff_h
                #print diff_h
                diff_s = self.lstm_node_list[ll][idx + 1].state.bottom_diff_s
                self.lstm_node_list[ll][idx].top_diff_is(diff_h, diff_s)
                idx -= 1 

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list[0]):
            # need to add new lstm node, create new state mem
            for l in range(self.layer):
                lstm_state = LstmState(self.lstm_param[l].mem_cell_ct, self.lstm_param[l].x_dim)
                if self.peephole:
                    self.lstm_node_list[l].append(LstmNodePeephole(self.lstm_param[l], lstm_state, self.g_func, self.h_func))
                else:
                    self.lstm_node_list[l].append(LstmNode(self.lstm_param[l], lstm_state, self.g_func, self.h_func))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            for l in range(self.layer):
                if l == 0:
                    self.lstm_node_list[l][idx].bottom_data_is(x)
                else:
                    self.lstm_node_list[l][idx].bottom_data_is(self.lstm_node_list[l - 1][idx].state.h)
        else:
            for l in range(self.layer):
                s_prev = self.lstm_node_list[l][idx - 1].state.s
                h_prev = self.lstm_node_list[l][idx - 1].state.h
                if l == 0:
                    self.lstm_node_list[l][idx].bottom_data_is(x, s_prev, h_prev)
                else:
                    self.lstm_node_list[l][idx].bottom_data_is(self.lstm_node_list[l - 1][idx].state.h, s_prev, h_prev)

    def apply_diff(self, lr = 0.1):
        for l in range(self.layer):
            self.lstm_param[l].apply_diff(lr)
