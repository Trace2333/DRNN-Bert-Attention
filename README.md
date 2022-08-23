# DRNN-Bert-Attention
双层RNN做关键词提取的改进
已经完成：
  1.将Bert作为第一个embedding层在训练过程中实时计算词嵌入，相当于多加一层，取消参数层
  2.利用bert生成的词嵌入矩阵来替代之前的随机数嵌入矩阵
