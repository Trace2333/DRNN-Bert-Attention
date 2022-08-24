import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DRNNBert(nn.Module):
    """
    主试验网路
    （已经证明是有效的）
    """
    def __init__(self, inputsize, hiddensize1, hiddensize2, inchanle, outchanle1, outchanle2, batchsize):
        super(DRNNBert, self).__init__()
        self.hiddensize = hiddensize1
        self.batchsize = batchsize
        self.RNN1 = nn.RNN(768, hiddensize1, batch_first=True, bidirectional=False)
        self.RNN2 = nn.RNN(inputsize, hiddensize2, batch_first=True, bidirectional=False)
        self.Linear1 = nn.Linear(in_features=inchanle, out_features=outchanle1)
        self.Linear2 = nn.Linear(in_features=inchanle, out_features=outchanle2)
        self.dropout = nn.Dropout(0.5)
        self.embw = Embw("/data1/trace/Projects/bert-base-uncased")
        # self.register_parameter("wemb", nn.Parameter(self.embw))

    def forward(self, inputs):
        """前向计算"""
        # inputs中的x更换为已经完成split的句子，不再是id
        state1 = self.init_state(self.batchsize)
        state2 = self.init_state(self.batchsize)
        if isinstance(inputs, tuple):
            x = self.embw.compute(inputs[0])
            x = self.dropout(x)
            rnnout1, state1 = self.RNN1(x)
            rnnout2, state2 = self.RNN2(rnnout1)
            y = self.Linear1(rnnout1)
            z = self.Linear2(rnnout2)
            return y, z
        else:
            x = self.embw.compute(inputs)
            x = self.dropout(x)
            rnnout1, state1 = self.RNN1(x)
            rnnout2, state2 = self.RNN2(rnnout1)
            y = self.Linear1(rnnout1)
            z = self.Linear2(rnnout2)
            return y, z

    def init_state(self, batchsize):
        """提供零初始化"""
        return torch.zeros((1, batchsize, self.hiddensize))


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


class Embw:
    def __init__(self, bert_path):
        self.model = BertModel.from_pretrained(bert_path).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.sen_embw = torch.zeros([1, 768]).to(device)

    def compute(self, input_data):
        """Note：
            为什么这里没有直接将一个batch的句子送进去来进行encode再计算？
            这里是一个NER任务，如果让tokenizer来自己encode的话，最后产生的序列长度会不同，因此不能使用这种方法，
            还是需要手动来padding
        """
        self.output = torch.zeros([1, len(input_data[0]), 768]).to(device)
        for sen in input_data:
            for token in sen:
                encoded_token = self.tokenizer.batch_encode_plus(
                    [token],
                    return_tensors='pt',
                )['input_ids'].to(device)
                out = self.model(encoded_token)['last_hidden_state'].mean(1)
                self.sen_embw = torch.cat((self.sen_embw, out), dim=0)
            self.output = torch.cat((self.output, self.sen_embw[1:].unsqueeze(0))).to(device)
            self.sen_embw = torch.zeros([1, 768]).to(device)
        return self.output[1:]

    def model_to_eval(self):
        self.model.eval()

def padding_Y_Z(x):
    output = []
    max_length = max(len(list(i)) for i in x)
    for sentence in x:
        while len(sentence) < max_length:
            sentence.append(0)
        output.append(sentence)
    tensor_out = torch.tensor(output[0], dtype=torch.int8).unsqueeze(0)
    for i in output[1:]:
        t = torch.tensor(i).unsqueeze(0)
        tensor_out = torch.cat((tensor_out, t), dim=0)
    return tensor_out.to(device)



def collate_fn_for_bert(batch):
    batchX = [i[0] for i in batch]
    batchY = [i[1][0] for i in batch]
    batchZ = [i[1][1] for i in batch]
    output = []
    max_length = max(len(list(x)) for x in batchX)
    for sentence in batchX:
        while len(sentence) < max_length:
            sentence.append("[PAD]")
        output.append(sentence)
    Y = padding_Y_Z(batchY)
    Z = padding_Y_Z(batchZ)
    return (output, Y, Z)


class RNNdatasetForBert(Dataset):
    """
    dataset
    """
    def __init__(self, input_data):
        super(RNNdatasetForBert, self).__init__()
        self.data = input_data

    def __getitem__(self, item):
        """标准gettiem格式"""
        y1 = list(self.data[1][item])
        y1_tensor = torch.tensor(y1)
        y2 = list(self.data[2][item])
        y2_tensor = torch.tensor(y2)
        return list(self.data[0][item]), (y1, y2)

    def __len__(self):
        """标准len格式"""
        return len(self.data[0])