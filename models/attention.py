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
        output = self.out_fn(sum_v)
        output = self.dropout(output)
        return output




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
