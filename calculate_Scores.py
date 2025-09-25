
def silhouette_Score():

def purity():

def accuracy():

def recall(confusion_matrix):
    tp =
    tn =
    fp =
    fn =

    return tp / (tp + fn)

def precision(confusion_matrix):
    tp =
    tn =
    fp =
    fn =

    return tp / (tp + fp)

def f1(confusion_matrix):
    pre = precision(confusion_matrix)
    rec = recall(confusion_matrix)
    return pre * rec * 2 / (pre + rec)

