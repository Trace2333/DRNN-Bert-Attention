import os
import pickle
import torch
import logging
import dill
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')
logging.basicConfig(level=logging.INFO,    # 输出INFO
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename='my.log',
                    filemode='w')


def get_list(filename):
    with open(filename, encoding='utf8') as f:
        datalist, taglist = [], []
        for line in f:
            line = line.strip()
            datalist.append(line.split('\t')[0])
            taglist.append(line.split('\t')[1])

    return datalist, taglist


def get_labels_list(filename):
    """
    Args：
        filename: dataset.pkl文件路径
    Return:
        train_labels: train数据集的标签，为一个包含list的list
        test_labels: test数据集的标签，为一个包含list的list
    """
    with open(filename, "rb") as dataset_file:
        train, test, dict = dill.load(dataset_file)
        train_labels = train[1:]
        test_labels = test[1:]
    return train_labels, test_labels


def sen_to_list(input_data):
    """
    返回一个经过split的列表
    Args:
        input_data: 输入的sentence列表
    Return:
        tokens: 完成split的tokens列表
    """
    tokens = []
    for i in input_data:
        token_list = i.split()
        tokens.append(token_list)
    return tokens


def create_dataset(tokens):
    """
    Args:
        tokens: token列表，由split后的tokens列表
    Return:
        dataset: 没有重复元素的列表
    """
    tokens_list = []
    for i in tokens:
        tokens_list += i
    dataset = list(set(tokens_list))
    return dataset


def generate_embedding(bert_filename, dataset):
    """
    建立词嵌入词典并进行保存
    Args:
        bert_filename: bert配置和参数文件的文件路径
        dataset: 初步建立的dataset，经过create_dataset函数创建，是一个列表形式
    Return:
        无
    """
    model = BertModel.from_pretrained(bert_filename).to(device)
    model.eval()
    embedding = {}
    tokenizer = BertTokenizer.from_pretrained(bert_filename)
    iteration = tqdm(dataset)
    out_dataset = {}
    for i, j in zip(dataset, range(len(dataset) + 1)[1:]):
        out_dataset[i] = j
    out_dataset["[O]"] = 0
    for token, i in zip(iteration, range(len(dataset) + 1)[1:]):
        encoded_token = tokenizer.batch_encode_plus(
            [token],
            return_tensors='pt',
        )['input_ids'].to(device)
        embw = model(encoded_token)['last_hidden_state']
        embw = np.array(torch.mean(embw, dim=1).cpu().detach(), dtype=np.float32)
        embedding[i] = embw
    embedding["[O]"] = np.array(torch.zeros([1, 768]).cpu().detach(), dtype=np.float32)
    if os.path.exists(".\\data"):
        with open(".\\data\\embedding.pkl", "wb") as f1:
            pickle.dump(embedding, f1)
            print("embedding.pkl Created!")
        with open(".\\data\\dataset.pkl", "wb") as f2:
            pickle.dump(out_dataset, f2)
            print("dataset.pkl, Created!")


def sentence_to_ids(data, dataset, targetname):
    """
    句子转为id并存储
    Args:
        data: 输入的经过split的句子列表
        dataset: dataset，是一个字典
    Return:
        无
    """
    ids = []
    sen_id = []
    for sen in data:
        for token in sen:
            sen_id.append(dataset[token])
        ids.append(sen_id)
    with open(".\\data\\" + targetname, "wb") as f:
        pickle.dump(ids, f)
        print(targetname, "Created!")


def dict_to_list(input_dict):
    out_list = []
    for i in input_dict:
        out_list.append(input_dict[i].squeeze(0))
    return out_list


if __name__ == '__main__':
    train_data, tag1 = get_list(".\\data\\trnTweet")
    test_data, tag2 = get_list(".\\data\\testTweet")
    splitted_train_sen = sen_to_list(train_data)
    splitted_test_sen = sen_to_list(test_data)
    splited_data = splitted_train_sen + splitted_test_sen
    dataset_file = open("data_set.pkl", 'rb')
    train, test, dict = dill.load(dataset_file)
    dict = dict['words2idx']
    dataset = list(dict.keys())
    dataset_file.close()
    generate_embedding("C:\\Users\\Trace\\Desktop\\Projects\\bert-base-uncased",
                       dataset)
    with open(".\\data\\dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    sentence_to_ids(splitted_train_sen, dataset, "train.pkl")
    sentence_to_ids(splitted_test_sen, dataset, "test.pkl")