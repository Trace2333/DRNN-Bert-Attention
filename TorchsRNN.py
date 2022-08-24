import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import numpy as np
from data_process_for_bert import dict_to_list


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
        self.embw = nn.Parameter(embw)    # self.register_parameter("wemb", nn.Parameter(self.embw))
        self.weight_init()

    def forward(self, inputs):
        """前向计算"""
        state1 = self.init_state(self.batchsize)
        state2 = self.init_state(self.batchsize)
        if isinstance(inputs, tuple):
            x = nn.functional.embedding(torch.tensor(contextwin_2(inputs[0], 3), dtype=torch.long).to(device),
                                        self.embw).flatten(2)
            x = self.dropout(x)
            rnnout1, state1 = self.RNN1(x)
            rnnout2, state2 = self.RNN2(rnnout1)
            y = self.Linear1(rnnout1)
            z = self.Linear2(rnnout2)
            return y, z
        else:
            x = nn.functional.embedding(torch.tensor(contextwin_2(inputs, 3), dtype=torch.int32).to(device),
                                        self.embw).flatten(2)
            x = self.dropout(x)
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



class Selectitem(nn.Module):
    def __init__(self, index):
        super(Selectitem, self).__init__()
        self.index = index

    def forward(self, inputs):
        return inputs[self.index]


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, inputs):
        inputsX = self.dropout(inputs[0])
        if isinstance(inputs, tuple):
            return (self.rnn(inputsX, torch.zeros(list(inputs[1].size())).to(device)))
        else:
            return self.rnn(inputs)    # 默认0初始化


class SrnnNet(nn.Module):
    """
    未修改的网络，目前无效
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, embw):
        super(SrnnNet, self).__init__()
        self.shared_layer = nn.Sequential(
            RNNLayer(input_size=input_size, hidden_size=hidden_size1)
        )
        self.tow2 = nn.Sequential(
            RNNLayer(input_size=input_size, hidden_size=hidden_size2),
            # neuros related to 2 rnns will be used for sequense classification
            Selectitem(0),  # get the RNN output
            nn.Linear(in_features=hidden_size2, out_features=5),
        )  # for sequence
        self.tow1 = nn.Sequential(
            Selectitem(0),  # get the RNN output
            nn.Linear(in_features=hidden_size1, out_features=2),
        )  # for sentence
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Parameter(embw)

    def forward(self, x):
        """返回的是y_pred， z_pred"""
        x = nn.functional.embedding(torch.tensor(x[0], dtype=torch.long).to(device), self.embedding)
        out = self.shared_layer(x)  # out is a tuple
        model_out1 = self.tow1(out)
        model_out2 = self.tow2(out)
        return model_out1, model_out2


class RNNdataset(Dataset):
    """
    dataset
    """
    def __init__(self, input_data):
        super(RNNdataset, self).__init__()
        self.data = (input_data)

    def __getitem__(self, item):
        """标准gettiem格式"""
        y1 = list(self.data[1][item])
        y1_tensor = torch.tensor(y1)
        y2 = list(self.data[2][item])
        y2_tensor = torch.tensor(y2)
        return torch.tensor(list(self.data[0][item])), (y1_tensor, y2_tensor)

    def __len__(self):
        """标准len格式"""
        return len(self.data[0])


class lossfun(nn.Module):
    """废用函数"""
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        loss = torch.sum((out * torch.log(y)))
        return loss


def sen_process(sen_list, embedding_model):
    """
    将句子列表处理成embedding，没有涉及到nn.embedding

    Note:
        embedding_file:contains vectors--->vectors:[1,1,1,1,1...]
    """
    if sen_list[0] in embedding_model:
        sen_embedding = torch.tensor(embedding_model[sen_list[0]]).unsqueeze(0)
    else:
        sen_embedding = torch.randn([1, 300])
    for i in sen_list[1:]:
        if i in embedding_model:  # For the trained word2vec model
            sen_embedding = torch.cat((sen_embedding, torch.tensor(embedding_model[i]).unsqueeze(0)), dim=0)
        else:
            sen_embedding = torch.cat((sen_embedding, torch.randn([1, 300])), dim=0)  # For the trained word2vec model
    return sen_embedding


def padding_sentence(inputs):
    """
    Padding句子并输出

    Args:
        :params inputList:list of lists
        :params forced_length:none if no input,if the parameter is not given,we will pad tne sentences with the maxlength of the list
    Return:
        padding后的句子对
    """
    inputListX = []
    inputListY = []
    inputListZ = []
    for i in inputs:
        inputListX.append(list(i)[0])
        inputListY.append(list(i[1])[0])
        inputListZ.append(list(i[1])[1])
    max_length = max(len(list(x)) for x in inputListY)
    return padding(inputListX, max_length), (padding(inputListY, max_length), padding(inputListZ, max_length))

embedding_model = open("./data/embedding.pkl", "rb")
matrix = pickle.load(embedding_model)
embedding_model.close()
matrix = np.array(dict_to_list(matrix))
matrix = torch.tensor(matrix)

# When using given embedding
def collate_fun1(batch_data):
    """
     自定义collate方法1，返回已经完成embedding后的batch

     Args:
         batch_data:list
     Return:
         完成处理的batch
    """
    # NO EXTRA PARAMETERS.....THE lENGTH of PADDED SHOULD BE SET IN FUN:padding_sentence()
    padded_list = padding_sentence(batch_data)
    # batched_dataX = []
    # for listX in padded_list[0]:
    #    batched_dataX.append(sen_process(listX, embedding_model))
    # When using the self-trained word2vec embedding, it is available
    batched_dataX = nn.functional.embedding(torch.tensor(contextwin_2(padded_list[0], 3), dtype=torch.int32),
                                            matrix).flatten(2)
    # When using given embedding
    batched_dataY = padded_list[1][0]
    batched_dataZ = padded_list[1][1]
    dataX = batched_dataX[0].unsqueeze(0)
    dataY = torch.tensor(batched_dataY[0]).unsqueeze(0)
    dataZ = torch.tensor(batched_dataZ[0]).unsqueeze(0)
    for i in batched_dataX[1:]:
        dataX = torch.cat((dataX, i.unsqueeze(0)), dim=0)
    for i in batched_dataY[1:]:
        dataY = torch.cat((dataY, torch.tensor(i).unsqueeze(0)), dim=0)
    for i in batched_dataZ[1:]:
        dataZ = torch.cat((dataZ, torch.tensor(i).unsqueeze(0)), dim=0)
    dataX.to(device)
    dataY.to(device)
    dataZ.to(device)
    return (dataX, (dataY, dataZ))


def collate_fun2(batch_data):
    """
    自定义的collate方法2，得到未经过embedding的x输入和经过cat的标签对

    Args：
        :params batch_data:由dataloader自动输出的batch data
    Return:
        元组，（未embedding的x， （tensorY， tensorZ））
    """
    padded_list = padding_sentence(batch_data)
    batched_dataX = padded_list[0]
    batched_dataY = padded_list[1][0]
    batched_dataZ = padded_list[1][1]
    dataY = (torch.tensor(batched_dataY[0]).to(device)).unsqueeze(0)
    dataZ = (torch.tensor(batched_dataZ[0]).to(device)).unsqueeze(0)
    for i in batched_dataY[1:]:
        dataY = torch.cat((dataY, (torch.tensor(i).to(device)).unsqueeze(0)), dim=0)
    for i in batched_dataZ[1:]:
        dataZ = torch.cat((dataZ, (torch.tensor(i).to(device)).unsqueeze(0)), dim=0)
    return (batched_dataX, (dataY, dataZ))


def padding(inputList, max_length, forced_length=None):
    """
    padding句子

    Args:
        :params inputList:输入的嵌套列表
        :params max_length:最大的长度，用于在指定长度时使用
        :params forced_length:是否强制长度
    Return:
        padding后的嵌套列表
    """
    if forced_length is None:
        num_padded_length = max_length  # padding to the curant max length
        padded_list = []
        for sentence in inputList:
            padded_sentence = sentence.tolist()
            while len(padded_sentence) < num_padded_length:
                padded_sentence.append(0)  # is the max length indefinitely
            padded_list.append(padded_sentence)
        return padded_list
    else:
        if max_length < forced_length:
            num_padded_length = forced_length
            padded_list = []
            for sentence in inputList:
                padded_sentence = sentence.tolist()
                while len(padded_sentence) < num_padded_length:
                    padded_sentence.append(0)  # is the max length indefinitely
                    padded_list.append(padded_sentence)
            return padded_list
        else:
            num_padded_length = forced_length    # some sentences shoule be cut.
            padded_list = []
            for sentence in inputList:
                padded_sentence = sentence.tolist()
                while len(padded_sentence) < num_padded_length:
                    padded_sentence.append(0)  # is the max length indefinitely
                    padded_list.append(padded_sentence)
                while len(padded_sentence) > num_padded_length:
                    padded_sentence.pop()
                    padded_list.append(padded_sentence)
            return padded_list


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = int(win / 2) * [0] + l + int(win / 2) * [0]
    out = [lpadded[i:i + win] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def contextwin_2(ls, win):
    assert (win % 2) == 1
    assert win >= 1
    outs = []
    for l in ls:
        outs.append(contextwin(l, win))
    return outs


def getKeyphraseList(l):
    res, now = [], []
    for i in range(len(l)):
        if l[i] != 0:
            now.append(str(i))
        if l[i] == 0 or i == len(l) - 1:
            if len(now) != 0:
                res.append(' '.join(now))
            now = []
    return set(res)


def load_config(model, target_path, para_name, if_load_or_not):
    """
    读取对应的模型并装载
    Args:
        model: 模型
        tagert_path: 目标路径，指的是在check_points之后的路径，
        如"/RNN_attention/"
        if_load: True/False
    """
    if not os.path.exists("./check_points"):
        os.mkdir("./check_points")
    check_point = './check_points'
    if_load = if_load_or_not
    if os.path.exists(check_point + target_path+ para_name) and if_load is True:    # 参数加载
        model.load_state_dict(torch.load(check_point + target_path+ para_name))
        print(para_name + " Parms loaded!!!")
