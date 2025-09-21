


def recall(data, true_label):


    return 3

def precision(data, true_label):
    return 2

def f1(data, true_label):
    pre = precision(data, true_label)
    rec = recall(data, true_label)
    return pre * rec * 2 / (pre + rec)