import cudafun

tanh = "tanhf(x)"
sigmoid = "sigmoidf(x)"
linear = "linearf(x)"
square = "squaree(x,y)"
entropy = "entropye(x,y)"
vanilla = "vanilla"
nog = "nog"
nfg = "nfg"
nig = "nig"
cifg = "cifg"

class LstmInput():
    def __init__(self, inputSize = 512, seqLength = 100):
        self.size = inputSize
        self.seqLength = seqLength
        self.examples = 0
        self.input = [[] for _ in range(seqLength)]

    def add(self, lst):
        assert len(lst) == self.seqLength
        self.examples += 1
        for i in range(self.seqLength):
            assert len(lst[i]) == self.size
            self.input[i].append(lst[i])

class LstmOutput():
    def __init__(self, hiddenSize = 512, seqLength = 100):
        self.size = hiddenSize
        self.seqLength = seqLength
        self.examples = 0
        self.output = [[] for _ in range(seqLength)]

    def add(self, lst):
        assert len(lst) == self.seqLength
        self.examples += 1
        for i in range(self.seqLength):
            if len(lst[i]) == self.size:
                self.output[i].append(lst[i])
            else:
                self.output[i].append(lst[i] + [0] * (self.size - len(lst[i])))

class LstmConfig():
    def __init__(self, inputSize = 512, hiddenSize = 512, numLayers = 4, seqLength = 100, miniBatch = 64, iterations = 10, learningRate = 0.001, loss_func = square, de_loss_func = None):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.seqLength = seqLength
        self.miniBatch = miniBatch
        self.iterations = iterations
        self.learningRate = learningRate
        self.loss_func = loss_func
        self.de_loss_func = de_loss_func

class LstmNode():
    def __init__(self, gates = vanilla, g_func = tanh, de_g_func = None, h_func = tanh, de_h_func = None, Peepholes = False):
        self.gates = gates
        self.g_func = g_func
        self.de_g_func = de_g_func
        self.h_func = h_func
        self.de_h_func = de_h_func
        self.Peepholes = Peepholes

class LstmNetwork():
    def __init__(self, config = LstmConfig(), node = LstmNode(), file = "LSTM.cu"):
        self.file = open(file, 'w')
        self.config = config
        self.node = node

    def run(self, input, output):
        self.file.write(cudafun.header) 
        self.file.write('#define GFUNC(x) (%s)\n' % (self.node.g_func))
        if self.node.de_g_func == None:
            self.file.write('#define DEGFUNC(x) (de_%s)\n' % (self.node.g_func))
        else:
            self.file.write('#define DEGFUNC(x) (%s)\n' % (self.node.de_g_func))
        self.file.write('#define HFUNC(x) (%s)\n' % (self.node.h_func))
        if self.node.de_h_func == None:
            self.file.write('#define DEHFUNC(x) (de_%s)\n' % (self.node.h_func))
        else:
            self.file.write('#define DEHFUNC(x) (%s)\n' % (self.node.de_h_func))
        if self.node.Peepholes:
            self.file.write('#define PEEPHOLES\n')

        self.file.write('#define LOSSFUNC(x,y) (%s)\n' % (self.config.loss_func))
        if self.config.de_loss_func == None:
            self.file.write('#define DELOSSFUNC(x,y) (de_%s)\n' % (self.config.loss_func))
        else:
            self.file.write('#define DELOSSFUNC(x,y) (%s)\n' % (self.config.de_loss_func))

        if self.node.gates == nog:
            self.file.write(r'''#define GATE_NUM (3)
#define PEEP_NUM (2)
#define I_INDEX (0)
#define F_INDEX (1)
#define G_INDEX (2)''')
        elif self.node.gates == nfg:
            self.file.write(r'''#define GATE_NUM (3)
#define PEEP_NUM (2)
#define I_INDEX (0)
#define G_INDEX (1)
#define O_INDEX (2)''')
        elif self.node.gates == nig:
            self.file.write(r'''#define GATE_NUM (3)
#define PEEP_NUM (2)
#define F_INDEX (0)
#define G_INDEX (1)
#define O_INDEX (2)''')
        elif self.node.gates == cifg:
            self.file.write(r'''#define GATE_NUM (3)
#define PEEP_NUM (2)
#define I_INDEX (0)
#define G_INDEX (1)
#define O_INDEX (2)
#define CIFG''')
        else:
            self.file.write(r'''#define GATE_NUM (4)
#define PEEP_NUM (3)
#define I_INDEX (0)
#define F_INDEX (1)
#define G_INDEX (2)
#define O_INDEX (3)''')

        self.file.write(cudafun.common) 
        assert input.size == self.config.inputSize
        assert input.seqLength == self.config.seqLength
        assert input.examples % self.config.miniBatch == 0
        self.file.write('float input_data_auto['+str(input.examples / self.config.miniBatch)+']['+str(self.config.inputSize*self.config.seqLength*self.config.miniBatch)+']=')
        self.file.write('{'+','.join('{'+','.join(','.join(','.join(str(_) for _ in input.input[t][b * self.config.miniBatch + e]) for e in range(self.config.miniBatch) if b * self.config.miniBatch + e < input.examples) for t in range(self.config.seqLength))+'}' for b in range(input.examples / self.config.miniBatch))+'};\n')
        assert output.size == self.config.hiddenSize
        assert output.seqLength == self.config.seqLength
        self.file.write('float label_data_auto['+str(output.examples / self.config.miniBatch)+']['+str(self.config.hiddenSize*self.config.seqLength*self.config.miniBatch)+']=')
        self.file.write('{'+','.join('{'+','.join(','.join(','.join(str(_) for _ in output.output[t][b * self.config.miniBatch + e]) for e in range(self.config.miniBatch) if b * self.config.miniBatch + e < output.examples) for t in range(self.config.seqLength))+'}' for b in range(output.examples / self.config.miniBatch))+'};\n')
        self.file.write(cudafun.scheduler)

        self.file.write('\n\
        for (int i = 0; i < %d; i++) {\n\
          for(int j = 0; j < %d; ++j) {\n\
            scheduler.clearStates(input_data_auto[j], label_data_auto[j]);\n\
            elapsedTime = scheduler.Forward(&loss);\n\
            printf("Forward time is %%f, loss is %%f\\n", elapsedTime, loss);\n\
            if (TRAINING) {\n\
              elapsedTime = scheduler.Backward(%f);\n\
              printf("Backward time is %%f\\n", elapsedTime);\n\
            }\n\
          }\n\
        }\n\
        ' % (self.config.iterations, input.examples / self.config.miniBatch, self.config.learningRate))

        self.file.write(cudafun.scheduler_2)
        self.file.write('int seqLength = %d;\nint numLayers = %d;\nint hiddenSize = %d;\nint miniBatch = %d;\nint inputSize = %d;\n'
            % (self.config.seqLength, self.config.numLayers, self.config.hiddenSize, self.config.miniBatch, self.config.inputSize))
        self.file.write(cudafun.others)

    def clean(self):
        self.file.close()
