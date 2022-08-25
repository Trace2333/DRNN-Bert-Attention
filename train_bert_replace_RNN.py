import logging
import wandb
import torch
import pickle
import dill
from tqdm import tqdm
from torch.utils.data import DataLoader
from TorchsRNN import RNNdataset, collate_fun2
from RNN_bert import DRNNBertReplaceRNN, RNNdatasetForBertReplaceRNN, collate_fun_for_bert_replace_rnn
from evalTools import acc_metrics, recall_metrics, f1_metrics
from data_process_for_bert import get_list



wandb.login(host="http://47.108.152.202:8080",
            key="local-86eb7fd9098b0b6aa0e6ddd886a989e62b6075f0")
wandb.init(project="DRNN-Bert-embw")
wandb.config = {
  "learning_rate": 1e-3,
  "epochs": 1,
  "batch_size": 64
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 32    # 基本参数
input_size = 768
hiden_size1 = 768
hiden_size2 = 768
loss1 = 0
loss2 = 0
loss = 0
epochs = 1
evaluation_epochs = 1
lr = 1e-3

model = DRNNBertReplaceRNN(inputsize=input_size,
                           hiddensize1=hiden_size1,
                           hiddensize2=hiden_size2,
                           inchanle=hiden_size2,
                           outchanle1=2,
                           outchanle2=5,
                           batchsize=batch_size,
                           ).to(device)

with open("./data/train_add.pkl", "rb") as f:
    train_data_pre = pickle.load(f)
with open("./data/test_add.pkl", "rb") as f:
    test_data_pre = pickle.load(f)

train_data = []
test_data = []
for tokens in train_data_pre:
    temp = tokens[0]
    for token in tokens[1:]:
        temp += (" " + token)
    train_data.append(temp)
    temp = ""

for tokens in test_data_pre:
    temp = tokens[0]
    for token in tokens[1:]:
        temp += (" " + token)
    test_data.append(temp)
    temp = ""

dataset_file = open("data_set.pkl", 'rb')
train_origin, test_origin, dict = dill.load(dataset_file)
train = [train_data, train_origin[1], train_origin[2]]
test = [test_data, test_origin[1], test_origin[2]]

dataset = RNNdatasetForBertReplaceRNN(train)
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          drop_last=True,
                          collate_fn=collate_fun_for_bert_replace_rnn
                          )

evaluation_dataset = RNNdatasetForBertReplaceRNN(test)
evaluation_loader = DataLoader(dataset=evaluation_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=0,
                               drop_last=True,
                               collate_fn=collate_fun_for_bert_replace_rnn)

lossfunction = torch.nn.CrossEntropyLoss()    # 优化器、损失函数选择
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

logging.info("Start Iteration")
for epoch in range(epochs):  # the length of padding is 128
    iteration = tqdm(train_loader, desc=f"TRAIN on epoch {epoch}")
    model.train()
    for step, inputs in enumerate(iteration):
        output1, output2 = model(
            (inputs[0], torch.randn([1, batch_size, hiden_size1])))  # 模型计算
        sentence_preds = output1.argmax(axis=2)
        sequence_preds = output2.argmax(axis=2)

        sen_acc = acc_metrics(sentence_preds, inputs[1])    #  指标计算
        seq_acc = acc_metrics(sequence_preds, inputs[2])
        sen_recall = recall_metrics(sentence_preds, inputs[1])
        seq_recall = recall_metrics(sentence_preds, inputs[2])
        sen_f1 = f1_metrics(sen_acc, sen_recall)
        seq_f1 = f1_metrics(seq_acc, seq_recall)

        wandb.log({"Train Sentence Precision": sen_acc})    #  指标可视化
        wandb.log({"Train Sequence Precision": seq_acc})
        wandb.log({"Train Sentence Recall": sen_recall})
        wandb.log({"Train Sequence Recall": seq_recall})
        wandb.log({"Train Sentence F1 Score": sen_f1})
        wandb.log({"Train Sequence F1 Score": seq_f1})

        loss1 = lossfunction(output1.permute(0, 2, 1), inputs[1])    # loss计算,按照NER标准
        loss2 = lossfunction(output2.permute(0, 2, 1), inputs[2])
        loss = loss2 * 0.5 + loss1 * 0.5

        iteration.set_postfix(loss1='{:.4f}'.format(loss1), loss2='{:.4f}'.format(loss2))
        wandb.log({"train loss1": loss1})
        wandb.log({"train loss2": loss2})
        wandb.log({"train Totalloss": loss})
        wandb.log({"lr:": optimizer.state_dict()['param_groups'][0]['lr']})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "./check_points/RNN_Bert_first_layer/epoch_1.pth")

for epoch in range(evaluation_epochs):
    evaluation_iteration = tqdm(evaluation_loader, desc=f"EVALUATION on epoch {epoch + 1}")
    model.eval()
    for step, evaluation_input in enumerate(evaluation_iteration):
        with torch.no_grad():
            output1, output2 = model((evaluation_input[
                0]))  # 模型计算
            sentence_preds = output1.argmax(axis=2)
            sequence_preds = output2.argmax(axis=2)

            sen_acc = acc_metrics(sentence_preds, evaluation_input[1])    # 参数计算
            seq_acc = acc_metrics(sequence_preds, evaluation_input[2])

            wandb.log({"Sentence Precision": sen_acc})
            wandb.log({"Sequence Precision": seq_acc})
