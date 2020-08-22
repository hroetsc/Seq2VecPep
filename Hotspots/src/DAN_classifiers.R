### HEADER ###
# HOTSPOT REGIONS
# description: try out different classifiers
# input: sequence repres after DAN
# output: -
# author: HR

library(dplyr)
library(caret)
library(MLeval)
library(plotROC)
library(mlbench)
library(pROC)

### INPUT ###
weights = read.csv("DAN/DAN_ext_substr_w5_d100_seq2vec-TFIDF.csv", stringsAsFactors = F)

#### MAIN PART ###
# preprocessing and splitting into training and testing

labels = rep(NA, nrow(weights))
for (r in 1:nrow(weights)){
  if (weights$label[r] == "hotspot") { labels[r] = 1 } else { labels[r] = 0 }
}

X = weights[, c(5:104)]

train.idx = sample(nrow(weights), ceiling(nrow(weights)*0.8))
test.idx = rownames(weights)[which(!rownames(weights) %in% train)] %>% as.numeric()

########## remove highly correlated features ##########

crl = cor(X)
hc = findCorrelation(crl, cutoff = 0.5) %>% sort()
X = X[, -c(hc)]

X_train = X[train.idx, ]
Y_train = labels[train.idx]

X_test = X[test.idx, ]
Y_test = labels[test.idx]

########## different built-in classifiers ##########
# 10-fold cross valiadation
control = trainControl(method = "cv", number = 10,
                       savePredictions = T)
metric = "Accuracy"

train.df = cbind(X_train, Y_train)
test.df = cbind(Y_train, Y_test)

# linear discriminant analysis
set.seed(42)
fit.lda = train(as.factor(Y_train)~., data = train.df,
                 method="lda", metric=metric, trControl=control)

# CART
set.seed(42)
fit.cart = train(as.factor(Y_train)~., data = train.df,
                 method="lda", metric=metric, trControl=control)

# kNN
set.seed(42)
fit.knn = train(as.factor(Y_train)~., data = train.df,
                method="knn", metric=metric, trControl=control)

# SVM
set.seed(42)
fit.svm = train(as.factor(Y_train)~., data = train.df,
                method="svmRadial", metric=metric, trControl=control)

# random forest
set.seed(42)
fit.rf = train(as.factor(Y_train)~., data = train.df,
                method="rf", metric=metric, trControl=control)

########## perform predictions ##########

results = resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)


eval = MLeval::evalm(fit.rf)
eval$roc


### OUTPUT ###

