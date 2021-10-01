import torch
import math


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def all(output, target):
    tn = tp = fn = fp = 0
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        #print("Metric:", output[:5])
        #print("argmax:", y_pred[:5])
        # pred = torch.max(output, 1)[1]  # .numpy().squeeze()
        #print("max:", pred[:5])
        assert pred.shape[0] == len(target)

        for (i, j) in zip(pred, target):
            i = int(i)
            j = int(j)
            #print(i, j)
            if i == 0 and j == 0:
                tn += 1
            elif i == 1 and j == 1:
                tp += 1
            elif i == 0 and j == 1:
                fn += 1
            else:
                fp += 1
        #print(tn, tp, fn, fp)
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        try:
            F1 = 2 * (recall * precision) / (recall + precision)
        except ZeroDivisionError:
            F1 = 0
        try:
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp)
                                                  * (tp + fn) * (tn + fp) * (tn + fn))
        except ZeroDivisionError:
            mcc = 0

    return recall, precision, accuracy, F1, mcc


def top_k_acc(output, target, k=2):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def recall(output, target):
    return all(output, target)[0]


def precision(output, target):
    return all(output, target)[1]


def F1(output, target):
    return all(output, target)[2]


def mcc(output, target):
    return all(output, target)[3]
