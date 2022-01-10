import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

#
# class_2 = pd.read_csv("../experiments/predictions_bin.csv").to_numpy()
class_3 = pd.read_csv("../experiments/finetuning-hparams/block-4_ds-4_seq-1024-lr5e-5-f8/predictions.csv").to_numpy()

# class_2_p, class_2_l = class_2[:, 1], class_2[:, 2]
class_3_p, class_3_l = class_3[:, 1], class_3[:, 2]

# print(classification_report(class_2_l, class_2_p, labels=[0, 1]))
# print(confusion_matrix(class_2_l, class_2_p, labels=[0, 1]))


print(classification_report(class_3_l, class_3_p, labels=[0, 1, 2]))
print(confusion_matrix(class_3_l, class_3_p, labels=[0, 1, 2]))