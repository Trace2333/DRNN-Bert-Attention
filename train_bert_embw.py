import logging
import wandb
import torch
import os
import dill
import pickle
from tqdm import tqdm
import numpy as np
from data_process_for_bert import dict_to_list
from torch.utils.data import DataLoader
from TorchsRNN import RNNdataset, collate_fun2, DRNN, load_config
from evalTools import acc_metrics, recall_metrics, f1_metrics
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
wandb.init()
wandb.login("0be0c1cf934799d42fd8ab6bd84f1caf9ff9f890")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 64    # 基本参数
input_size = 768
hiden_size1 = 768
hiden_size2 = 768
loss1 = 0
loss2 = 0
loss = 0
epochs = 1
evaluation_epochs = 1
lr = 2e-3

embedding_model = open("./data/embedding.pkl", "rb")
matrix = pickle.load(embedding_model)
embedding_model.close()
matrix = np.array(dict_to_list(matrix))
matrix = torch.tensor(matrix).to(device)

model = DRNN(inputsize=input_size,
             inputsize1=768 * 3,
             hiddensize1=hiden_size1,
             hiddensize2=hiden_size2,
             inchanle=hiden_size2,
             outchanle1=2,
             outchanle2=5,
             batchsize=batch_size,
             embw=matrix).to(device)


load_config(
    model, 
    target_path="/RNN_Bert_as_Embw/",
    para_name="parameters_epoch_1.pth",
    if_load=True
)

dataset_file = open("data_set.pkl", 'rb')
train, test, dict = dill.load(dataset_file)

dataset = RNNdataset(train)
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True,
                          collate_fn=collate_fun2
                          )

evaluation_dataset = RNNdataset(test)
evaluation_loader = DataLoader(dataset=evaluation_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=0,
                               drop_last=True,
                               collate_fn=collate_fun2)

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

        sen_acc = acc_metrics(sentence_preds, inputs[1][0])    #  指标计算
        seq_acc = acc_metrics(sequence_preds, inputs[1][1])
        sen_recall = recall_metrics(sentence_preds, inputs[1][0])
        seq_recall = recall_metrics(sentence_preds, inputs[1][0])
        sen_f1 = f1_metrics(sen_acc, sen_recall)
        seq_f1 = f1_metrics(seq_acc, seq_recall)

        wandb.log({"Train Sentence Precision": sen_acc})    #  指标可视化
        wandb.log({"Train Sequence Precision": seq_acc})
        wandb.log({"Train Sentence Recall": sen_recall})
        wandb.log({"Train Sequence Recall": seq_recall})
        wandb.log({"Train Sentence F1 Score": sen_f1})
        wandb.log({"Train Sequence F1 Score": seq_f1})

        loss1 = lossfunction(output1.permute(0, 2, 1), inputs[1][0])    # loss计算,按照NER标准
        loss2 = lossfunction(output2.permute(0, 2, 1), inputs[1][1])
        loss = loss2 * 0.7 + loss1 * 0.3

        iteration.set_postfix(loss1='{:.4f}'.format(loss1), loss2='{:.4f}'.format(loss2))
        wandb.log({"train loss1": loss1})
        wandb.log({"train loss2": loss2})
        wandb.log({"train Totalloss": loss})
        wandb.log({"lr:": optimizer.state_dict()['param_groups'][0]['lr']})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """
        for name, parms in model.named_parameters():    #debug时使用，可视化每一个层的grad与weight
            wandb.log({f"{name} Weight:" : torch.mean(parms.data)})
            if parms.grad is not None:
                wandb.log({f"{name} Grad_Value:" : torch.mean(parms.grad)})
        """

torch.save(model.state_dict(), "./check_points/RNN_Bert_as_Embw/parameters_epoch_1.pth")

for epoch in range(evaluation_epochs):
    evaluation_iteration = tqdm(evaluation_loader, desc=f"EVALUATION on epoch {epoch + 1}")
    model.eval()
    for step, evaluation_input in enumerate(evaluation_iteration):
        with torch.no_grad():
            output1, output2 = model((evaluation_input[
                0]))  # 模型计算
            sentence_preds = output1.argmax(axis=2)
            sequence_preds = output2.argmax(axis=2)

            sen_acc = acc_metrics(sentence_preds, evaluation_input[1][0])    # 参数计算
            seq_acc = acc_metrics(sequence_preds, evaluation_input[1][1])
            sen_recall = recall_metrics(sentence_preds, evaluation_input[1][0])
            seq_recall = recall_metrics(sentence_preds, evaluation_input[1][0])
            sen_f1 = f1_metrics(sen_acc, sen_recall)
            seq_f1 = f1_metrics(seq_acc, seq_recall)

            wandb.log({"Sentence Precision": sen_acc})
            wandb.log({"Sequence Precision": seq_acc})
            wandb.log({"Sentence Recall": sen_recall})
            wandb.log({"Sequence Recall": seq_recall})
            wandb.log({"Sentence f1-score": sen_f1})
            wandb.log({"Sequence f1-score": seq_f1})
