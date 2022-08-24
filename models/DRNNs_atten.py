import torch
import torch.nn as nn
from TorchsRNN import contextwin_2
from attention import SelfDotProductAttention


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DRNN(nn.Module):
    """
    主试验网路
    （已经证明是有效的）
    """
    def __init__(self, inputsize, inputsize1, hiddensize1, hiddensize2, inchanle, outchanle1, outchanle2, batchsize, embw):
        super(DRNN, self).__init__()
        self.hiddensize = hiddensize1
        self.batchsize = batchsize
        self.RNN1 = nn.RNN(inputsize1, hiddensize1, batch_first=True, bidirectional=False)
        self.RNN2 = nn.RNN(inputsize, hiddensize2, batch_first=True, bidirectional=False)
        self.Linear1 = nn.Linear(in_features=inchanle, out_features=outchanle1)
        self.Linear2 = nn.Linear(in_features=inchanle, out_features=outchanle2)
        self.dropout = nn.Dropout(0.5)
        self.embw = nn.Parameter(embw)
        self.atten = SelfDotProductAttention(
            input_size=hiddensize1,    #数据维度
            k_size=512,
            q_size=512,
            v_size=512,
            drop_rate=0.5
        )
        self.weight_init()

    def forward(self, inputs):
        """前向计算"""
        state1 = self.init_state(self.batchsize)
        state2 = self.init_state(self.batchsize)
        if isinstance(inputs, tuple):
            x = nn.functional.embedding(torch.tensor(contextwin_2(inputs[0], 3), dtype=torch.long).to(device),
                                        self.embw).flatten(2)
            rnnout1, state1 = self.RNN1(x)
            rnnout1 = self.atten(rnnout1)
            rnnout2, state2 = self.RNN2(rnnout1)
            y = self.Linear1(rnnout1)
            z = self.Linear2(rnnout2)
            return y, z
        else:
            x = nn.functional.embedding(torch.tensor(contextwin_2(inputs, 3), dtype=torch.int32).to(device),
                                        self.embw).flatten(2)
            x = self.atten(x)
            rnnout1, state1 = self.RNN1(x)
            rnnout2, state2 = self.RNN2(rnnout1)
            y = self.Linear1(rnnout1)
            z = self.Linear2(rnnout2)
            return y, z

    def init_state(self, batchsize):
        """提供零初始化"""
        return torch.zeros((1, batchsize, self.hiddensize))

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1)


