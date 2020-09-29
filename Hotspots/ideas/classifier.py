#!/usr/bin/env python

### HEADER ###
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        word pairs (target and context word), IDs generated in skip_gram_NN_1
# output:       embedded tokens (weights and their IDs)
# author:       HR


import os
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import *

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve


os.chdir('/home/hanna/Documents/QuantSysBios/ProtTransEmbedding/Hotspots/Classification')

### INPUT ###
# load data
extS = pd.read_csv('ext_substr.csv', header = 0)
minS = pd.read_csv('min_substr.csv', header = 0)
prots = pd.read_csv('proteins.csv', header = 0)
labels = pd.read_csv('labels.csv', header = 0)

# hyperparameters
VAL = 0.5

### MAIN PART ###
# transform into numpy arrays
extS = np.array(extS, dtype = 'float64')
minS = np.array(minS, dtype = 'float64')
prots = np.array(prots, dtype = 'float64')

# features and labels
X = np.concatenate((extS, minS, prots), axis = 1)
y = np.array(labels, dtype = 'int64').flatten()


# remove highly correlated features
# https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
df = pd.DataFrame(X)
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(f"found  highly correlated dimensions {to_drop} and is removing them")

# Drop features 
df = df.drop(df[to_drop], axis=1)
X = np.array(df, dtype = 'float64')

# split data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = VAL, random_state = 42)

# compare different classifiers
# copied from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


figure = plt.figure(figsize=(27, 3))


i = 1

# plot data
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, len(classifiers) + 1, i)

ax.set_title("Input data")

# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
           edgecolors='k')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1


res = []

for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    y_pred= clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    #prec, recall, _ = precision_recall_curve(clf, X_test, y_test)
    #pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    
    cm = confusion_matrix(y_test, y_pred)
    
    res.append([name, acc, cm])
    

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
               
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
    ax.set_title(name)

    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1


plt.tight_layout()
plt.show()

plt.savefig('classifiers.png')






































## deep approach

# batch generator

class BatchGenerator(keras.utils.Sequence):

     def __init__(self, extS, minS, prots, labels, batch_size):
         self.extS, self.minS, self.prots, self.labels = extS, minS, prots, labels
         self.batch_size = batch_size
         self.indices = np.arange(self.labels.shape[0])

     def __len__(self):
         return int(np.ceil(len(self.labels) / float(self.batch_size)))

     def __getitem__(self, idx):

         batch_extS = self.extS[idx*self.batch_size : (idx + 1)*self.batch_size,:]
         batch_minS = self.minS[idx*self.batch_size : (idx + 1)*self.batch_size,:]
         batch_prots = self.prots[idx*self.batch_size : (idx + 1)*self.batch_size,:]
         batch_labels = self.labels[idx*self.batch_size : (idx + 1)*self.batch_size,:]


         #print(batch_extS)

         return [batch_extS, batch_minS, batch_prots], batch_labels

     def on_epoch_end(self):
         np.random.shuffle(self.indices)



### model ###
# input
input_min = keras.Input((100,), name = 'input_min')
input_ext = keras.Input((100,), name = 'input_ext')
input_prot = keras.Input((100,), name = 'input_prots')

#input_min = layers.Reshape((100,1))(input_min)
#input_ext = layers.Reshape((100,1))(input_ext)
#input_prot = layers.Reshape((100,1))(input_prot)


conc = layers.Concatenate()([input_min, input_ext])


x = layers.Dense(256, activation = 'relu')(conc)
x = layers.Dropout(0.5)(x)
x = layers.Dense(100, activation = 'linear')(x)
x = layers.Dropout(0.5)(x)

conc2 = layers.Multiply()([x, input_prot])

x2 = layers.Dense(256, activation = 'linear')(conc2)
x2 = layers.Dropout(0.5)(x2)
x2 = layers.Dense(128, activation = 'tanh')(x2)
x2 = layers.Dropout(0.5)(x2)
x2 = layers.Dense(64, activation = 'tanh')(x2)
x2 = layers.Dropout(0.5)(x2)
output = layers.Dense(1, activation = 'sigmoid')(x2)


# build model
model = keras.Model(inputs=[input_min, input_ext, input_prot], outputs=output)

model.compile(loss = 'binary_crossentropy',
              optimizer = 'Adam',
              metrics = ['accuracy', 'mean_squared_error'])

print(model.summary())


# split data
t_ext, v_ext, t_min, v_min, t_prots, v_prots, t_labels, v_labels = train_test_split(extS, minS, prots, labels, test_size = VAL)

# apply batch generator
train_data = BatchGenerator(t_ext, t_min, t_prots, t_labels, BatchSize)

# fit model
hist = model.fit_generator(generator = train_data,
                           epochs = 10,
                           verbose = 2,
                           shuffle = True,
                           max_queue_size = 1,
                           workers = 16,
                           use_multiprocessing = False)

# predict
label_probs = model.predict([v_ext, v_min, v_prots, v_labels], verbose = 2)
label_probs = np.array(np.round(label_probs, decimals = 0), dtype = 'int64')


# evaluate
precision = precision_score(v_labels, label_probs)
recall = recall_score(v_labels, label_probs)
F1 = f1_score(v_labels, label_probs)

print('precision: {} - recall: {} - F1: {}'.format(precision, recall, F1))

matrix = confusion_matrix(v_labels, label_probs)
print(matrix)

# classification report
nm = ['non-hotspot','hotspot']
cl = classification_report(v_labels, label_probs, target_names = nm)
print(cl)


### OUTPUT ###
