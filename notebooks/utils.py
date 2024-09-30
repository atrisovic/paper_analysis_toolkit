import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


labels = ['context', 'not_context']

def plot_cm(true, pred, xlabel = None, ylabel = None):
    labels = sorted(list(set(true).union(set(pred))))
    cm = confusion_matrix(true, pred)
    plt.xlabel(xlabel or 'pred')
    plt.ylabel(ylabel or 'true')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar = False, annot_kws={"size": 25})