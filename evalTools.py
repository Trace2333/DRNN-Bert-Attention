import torch
def acc_metrics(sentence_preds, y_labels):
    """
    output from the network
    sentence_preds:2 dimension tensor, size=[batch_size, padden_sentence_length]
    evaluation target
    y:2 dimension tensor, size=[batch_size, padden_sentence_length]
    Precision = (True_preds)/(True_preds+ False_preds)
    """
    Titems = 0
    Fitems = 0
    Totalitems = 0
    if sentence_preds.ndim != 2:
        raise RuntimeError("Size Error!The input shoule be [prediction, y] and both of them should be 3 dimensions")
    if y_labels.ndim != 2:
        raise RuntimeError("Size Error!The input shoule be [prediction, y] and both of them should be 3 dimensions")
    if sentence_preds.size()[1] != y_labels.size()[1]:
        raise RuntimeError("Both seqtence should have the same padding length")
    for sentence, y in zip(sentence_preds, y_labels):
        if sentence.size()[0] != y.size()[0]:
            raise RuntimeError("Both data should have the same batch_size")
        if torch.equal(sentence, y):
            Titems = Titems +1
        else:
            Fitems = Fitems +1
        Totalitems = Titems + Fitems
        """for i,j in zip(sentence, y):
            if i == j:
                Titems = Titems + 1
            else:
                Fitems = Titems + 1
        Totalitems = Titems + Fitems"""
    return Titems/Totalitems

def recall_metrics(sentence_preds, y_labels):
    Titems = 0
    Fitems = 0
    Totalitems = 0
    Totalyitems = 0
    Totalpitems = 0
    if sentence_preds.ndim != 2:
        raise RuntimeError("Size Error!The input shoule be [prediction, y] and both of them should be 3 dimensions")
    if y_labels.ndim != 2:
        raise RuntimeError("Size Error!The input shoule be [prediction, y] and both of them should be 3 dimensions")
    if sentence_preds.size()[1] != y_labels.size()[1]:
        raise RuntimeError("Both seqtence should have the same padding length")
    for sentence, y in zip(sentence_preds, y_labels):
        if sentence.size()[0] != y.size()[0]:
            raise RuntimeError("Both data should have the same batch_size")
        for i , j in zip(sentence, y):
            if j != 0:
                Totalyitems += 1
            if i == j and j != 0:
                Titems += 1
            else:
                Fitems += 1
    return Titems/Totalyitems

def f1_metrics(precision, recall):
    return (2*precision*recall)/(precision + recall + 0.01)