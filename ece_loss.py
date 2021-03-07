import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def compute_ece_loss(classifier_output, labels, n_gaps):
        gap_boundaries = torch.linspace(0, 1, n_gaps + 1)
        gap_lowers = gap_boundaries[:-1]
        gap_uppers = gap_boundaries[1:]
        confidences, predictions = torch.max(classifier_output, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1)
        confidences_pl = []
        accuracies_pl = []
        for gap_lower, gap_upper in zip(gap_lowers, gap_uppers):
            relevant_samples = confidences.gt(gap_lower.item()) * confidences.le(gap_upper.item())
            prop_relevant_samples = relevant_samples.float().mean()
            if prop_relevant_samples.item() > 0:
                accuracy_relevant_samples = accuracies[relevant_samples].float().mean()
                avg_confidence_relevant_samples = confidences[relevant_samples].mean()
                ece += torch.abs(avg_confidence_relevant_samples - accuracy_relevant_samples) * prop_relevant_samples
                confidences_pl.append(confidences[relevant_samples].mean())
                accuracies_pl.append(accuracies[relevant_samples].float().mean())
        return ece, confidences_pl, accuracies_pl

def plot_con_acc(con, acc):
    fig, ax = plt.subplots()
    ax.scatter(acc, con)
    x = np.arange(0.0, 1.0, 0.01)
    s = x
    ax.plot(x,s, 'r--')
    ax.set(xlabel='accuracy (%)', ylabel='confidence (%)',
       title='accuracy/confidence plot')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    #ax.grid()


    plt.show()
    
