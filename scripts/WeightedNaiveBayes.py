import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, plot_confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from information_gain import information_gain

'''''''''
1. IMPORT DATASET
'''''''''
# train_df: CHAGOK1; test_df: LOUNH91
train_df = pd.read_csv("Data-Gene-Essentiality\\Clean\\Full_Training_CHAGOK1.csv")
test_df = pd.read_csv("Data-Gene-Essentiality\\Clean\\Full_Testing_LOUNH91.csv")
# print(train_df.info())
# print(test_df.info())


'''''''''
2. EXTRACT GAIN CALCULATION VALUES:
'''''''''
# Subset rows based on the presence of gene within a cluster.
cluster_meta = [] # list of each n functional cluster dataframe.
for i in range(2,32): # 30 functional clusters
    subset_df = train_df[train_df.iloc[:,i] == 1]
    cluster_meta.append(subset_df)
# print(len(cluster_meta))

# get information gain value in each n functional cluster.
cluster_gain = []
for i in range(0,30):
    score = pd.Series(cluster_meta[i]['Disc_Scores'])
    exp_split = pd.Series(cluster_meta[i]['Disc_Exp'])
    clust_gain = information_gain(score, exp_split)
    cluster_gain.append(clust_gain)
# print(cluster_gain)


'''''''''
3. GET WEIGHT VECTOR FROM ALL N GENES
'''''''''
# boolean matrix (0/1): gene vs cluster:
cluster_matrix = train_df.drop(columns=['Unnamed: 0','Gene', 'Disc_Scores', 'Exp_Data','Disc_Exp','Score']).values
# print(cluster_matrix)

# weight vector for n genes. To input into Naive Bayes model
weights = []
for gene in cluster_matrix:
    sum = 0
    for i in range(0, len(gene)):
        sum += gene[i]*cluster_gain[i]
    weights.append(sum)
# print(len(weights))

'''''''''
4. TRAINING & TESTING THE MODEL: WEIGHTED NAIVE BAYES
'''''''''
X_train = np.array(train_df['Exp_Data']).reshape(-1,1)
y_train = np.array(train_df['Disc_Scores'])
X_test = np.array(test_df['Exp_Data']).reshape(-1,1)
y_test = np.array(test_df['Disc_Scores'])

le = preprocessing.LabelEncoder()
y_train = le.fit(y_train).transform(y_train)
y_test = le.fit(y_test).transform(y_test)


# 4a. WITHOUT WEIGHTS (GAUSSIAN NAIVE BAYES)
gnb_gaussian = GaussianNB()
gnb_gaussian.fit(X_train, y_train)


# 4b. WITH WEIGHTS (WEIGHTED GAUSSIAN NAIVE BAYES)
gnb_weighted = GaussianNB()
gnb_weighted.fit(X_train, y_train, sample_weight= weights)

# get predicted values
predictions_gaussian = gnb_gaussian.predict(X_test)
predictions_weighted = gnb_weighted.predict(X_test)
# print((predictions_gaussian, predictions_weighted))

# OBTAIN PREDICTED GENES (FROM WEIGHTED GAUSSIAN NAIVE BAYES)
gene_list = test_df['Gene']
essentiality_predicted = []
non_essentiality_predicted = []

for i in range(0, len(gene_list)):
    if predictions_weighted[i] == 0:
        essentiality_predicted.append(gene_list[i])
    if predictions_weighted[i] == 1:
        non_essentiality_predicted.append(gene_list[i])

# print(len(essentiality_predicted))
# print(len(non_essentiality_predicted))

# export predicted gene list
# pd.Series(essentiality_predicted).to_csv('Results-Gene-Essentiality\\essentiality_predicted.csv')
# pd.Series(non_essentiality_predicted).to_csv('Results-Gene-Essentiality\\non_essentiality_predicted.csv')

'''''''''
4. EVALUATING THE MODEL
'''''''''
# ACCURACY
accuracy_gaussian = accuracy_score(y_test, predictions_gaussian)
accuracy_weighted = accuracy_score(y_test, predictions_weighted, sample_weight=weights)
# print((accuracy_gaussian, accuracy_weighted))

# ROC & AUC
fpr_g, tpr_g, thresholds_c = roc_curve(y_test, predictions_gaussian, pos_label=1)
fpr_w, tpr_w, thresholds_w = roc_curve(y_test, predictions_weighted, pos_label=1)
auc_g = auc(fpr_g, tpr_g)
auc_w = auc(fpr_w, tpr_w)
# print((auc_g,auc_w))

'''''''''
5. VISUALIZATION: CONFUSION MATRIX, ROC CURVE
'''''''''
# CONFUSION MATRIX
gaussian_disp = plot_confusion_matrix(gnb_gaussian,\
                                 X_test, y_test,\
                                 display_labels=['Essential','Non-Essential'],\
                                 cmap=plt.cm.Blues,\
                                 normalize='all',\
                                 xticks_rotation='horizontal')
gaussian_disp.ax_.set_title('Gaussian Naive Bayes')
# print(gaussian_disp.confusion_matrix)

weighted_disp = plot_confusion_matrix(gnb_weighted,\
                                 X_test, y_test,\
                                 sample_weight= weights,\
                                 display_labels=['Essential','Non-Essential'],\
                                 cmap=plt.cm.Blues,\
                                 normalize='all',\
                                 xticks_rotation='horizontal')
weighted_disp.ax_.set_title('Weighted Naive Bayes')
# print(weighted_disp.confusion_matrix)

# ROC CURVE
plt.figure()
lw = 2
plt.plot(fpr_g, tpr_g, color='darkorange',\
        lw=lw, label='ROC Gaussian (area = %0.2f)' % auc_g)
plt.plot(fpr_w, tpr_w, color='green',\
        lw=lw, label='ROC Weighted (area = %0.2f)' %auc_w)
plt.plot([0,1],[0,1],color='navy', lw=lw, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: Gaussian vs Weighted Naive Bayes')
plt.legend(loc='lower right')
# plt.show()
