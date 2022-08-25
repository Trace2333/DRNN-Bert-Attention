import torch
import math
import torch.nn as nn


class SelfDotProductAttention(nn.Module):
    """基础Attention，无多头机制"""
    def __init__(self, input_size, v_size, k_size, q_size, drop_rate):
        super(SelfDotProductAttention, self).__init__()
        self.k_fn = nn.Linear(input_size, k_size, bias=False)
        self.q_fn = nn.Linear(input_size, q_size, bias=False)
        self.v_fn = nn.Linear(input_size, v_size, bias=False)
        self.out_fn = nn.Linear(v_size, input_size)
        # 利用全连接层参数矩阵作为权值矩阵
        self.in_size = input_size
        self.k_size = k_size
        self.q_size = q_size
        self.v_size = v_size
        self.dropout = nn.Dropout(drop_rate)

        self.init_weights()

    def init_weights(self):
        """初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1)

    def forward(self, input_data):
        """计算，该attention模块会参数更新"""
        q = self.q_fn(input_data)
        k = self.k_fn(input_data)
        v = self.v_fn(input_data)    # 计算当前的KQV矩阵，三个fn即是权值矩阵
        if input_data.ndim is 3:
            attention_score = torch.matmul(k, q.permute(0, 2, 1)) / math.sqrt(self.k_size)
        else:
            attention_score = torch.matmul(k, q.permute(1, 0)) / math.sqrt(self.k_size)
        matrix = torch.softmax(attention_score, dim=-1)
        sum_v = torch.matmul(matrix, v)
        output_atten = self.out_fn(sum_v)
        output_atten = self.dropout(output_atten)
        return output_atten


class SelfDotMultiHeadAttention(nn.Module):
    """多头注意力机制：主要内容即将一个大的注意力矩阵分为多个小矩阵，即多头"""
    def __init__(self, input_size, k_size, q_size, v_size, h, dropout_rate):
        """多头的每一个头保持相同的大小
        NOTE：
            权值矩阵的大小需要能够被head数量整除，输入的维度尽量能够开方
        """
        super(SelfDotMultiHeadAttention, self).__init__()
        self.h = h   # h表示头的数量
        self.ks = []
        self.vs = []
        self.qs = []
        self.k_size = k_size
        self.v_size = v_size
        self.q_size = q_size
        for i in range(h):
            self.ks.append(nn.Linear(input_size, int(k_size / h), bias=False))
            self.vs.append(nn.Linear(input_size, int(v_size / h), bias=False))
            self.qs.append(nn.Linear(input_size, int(q_size / h), bias=False))
        self.out_fn = nn.Linear(v_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_data):
        k_matrixs = []
        q_matrixs = []
        v_matrixs = []
        sum_matrix = []
        attention_score = []
        for i in range(self.h):    # 分头
            k_matrixs.append(self.ks[i](input_data))
            q_matrixs.append(self.qs[i](input_data))
            v_matrixs.append(self.vs[i](input_data))
        for i in range(self.h):
            if input_data.ndim == 2:
                attention_score.append(torch.matmul(k_matrixs[i], q_matrixs[i].permute(1, 0)))
                attention_score[i] = attention_score[i] / math.sqrt(input_data.size()[-1])
            if input_data.ndim == 3:
                attention_score.append(torch.matmul(k_matrixs[i], q_matrixs[i].permute(0, 2, 1)))
                attention_score[i] = attention_score[i] / math.sqrt(input_data.size()[-1])
        for i in range(self.h):
            sum_matrix.append(torch.matmul(torch.softmax(attention_score[i], dim=-1), v_matrixs[i]))
        if input_data.ndim == 3:
            output = torch.zeros([input_data.size()[0], input_data.size()[1], int(self.v_size / self.h)])
        else:
            output = torch.zeros([input_data.size()[0], int(self.v_size / self.h)])
        for i in range(self.h):
            output = torch.cat((output, sum_matrix[i]), dim=-1)
        if output.ndim == 3:
            output = output.permute(2, 1, 0)[int(self.v_size / self.h):].permute(2, 1, 0)
        else:
            output = output.permute(1, 0)[int(self.v_size / self.h):].permute(1, 0)
        output = self.out_fn(output)
        output = self.dropout(output)
        return output

    def weight_init(self):
        for module_no in self.ks:
            if isinstance(module_no, nn.Linear):
                nn.init.xavier_normal_(module_no.weight, gain=1)
        for module_no in self.vs:
            if isinstance(module_no, nn.Linear):
                nn.init.xavier_normal_(module_no.weight, gain=1)
        for module_no in self.qs:
            if isinstance(module_no, nn.Linear):
                nn.init.xavier_normal_(module_no.weight, gain=1)


if __name__ == '__main__':    # 测试
    inputs = torch.randn([16, 300])
    atten = SelfDotProductAttention(
        input_size=300,
        v_size=100,
        k_size=100,
        q_size=100,
        drop_rate=0.5
    )
    output = atten(inputs)
    print(output.size())
    multi_inputs = torch.randn([16, 20, 300])
    multi_atten = SelfDotMultiHeadAttention(
        input_size=300,
        v_size=512,
        k_size=512,
        q_size=512,
        dropout_rate=0.5,
        h=8
    )
    multi_output = multi_atten(multi_inputs)
    print(multi_output.size())