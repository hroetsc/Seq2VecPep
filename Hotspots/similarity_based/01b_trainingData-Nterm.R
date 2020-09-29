### HEADER ###
# HOTSPOT REGIONS
# description: generate training data for hotspot prediction - N-terminal extension
# input: hotspot/non-hotspot regions (extended/minimal version), hotspot density
# output: training data set and its embedding
# author: HR

library(dplyr)
library(stringr)
library(seqinr)

library(future)
library(foreach)
library(doParallel)
registerDoParallel(cores=availableCores())

# requires functions from 01_trainingData.R, 02_computeScores.R and 03_makePrediction.R
# regions with extended substring

#### INPUT ### 
training = read.csv("data/classifier/training_DATA.csv", stringsAsFactors = F)
testing = read.csv("data/classifier/testing_DATA.csv", stringsAsFactors = F)

# add pure seq2vec embeddings in case of joint vectors
sqs.train.PURE = read.csv("data/classifier/pure_training_EMBEDDING.csv", stringsAsFactors = F)
sqs.test.PURE = read.csv("data/classifier/pure_testing_EMBEDDING.csv", stringsAsFactors = F)


### MAIN PART ###

# N terminal extension
train.NTERM = extensions(tbl = training, direction = "N", by = 25)
test.NTERM = extensions(tbl = testing, direction = "N", by = 25)

# sequence representations
sqs.train.NTERM.joint = get_seq_repres(train.NTERM[["joint"]], out = "train_NTERM_joint.csv")
sqs.train.NTERM.only_ext = get_seq_repres(train.NTERM[["only_ext"]], out = "train_NTERM_onlyExt.csv")

sqs.test.NTERM.joint = get_seq_repres(test.NTERM[["joint"]], out = "test_NTERM_joint.csv")
sqs.test.NTERM.only_ext = get_seq_repres(test.NTERM[["only_ext"]], out = "test_NTERM_onlyExt.csv")

# add biophysical properties
sqs.train.PROP.joint = getPropMatrix(train.NTERM[["joint"]])
sqs.test.PROP.joint = getPropMatrix(test.NTERM[["joint"]])

sqs.train.NTERM_PROP.joint = left_join(sqs.train.NTERM.joint,
                                       sqs.train.PROP.joint) %>% na.omit()
sqs.test.NTERM_PROP.joint = left_join(sqs.test.NTERM.joint,
                                       sqs.test.PROP.joint) %>% na.omit()


# compute similarities
# 2 versions: joint core region and N-terminal extension, two separate, appended vectors
# joint
dimRange = c(8:ncol(sqs.train.NTERM.joint))
res.NTERM1 = Prediction(train = sqs.train.NTERM.joint, 
                        test = sqs.test.NTERM.joint,
                        dimRange = dimRange)
res.NTERM1_PROP = Prediction(train = sqs.train.NTERM_PROP.joint, 
                        test = sqs.test.NTERM_PROP.joint,
                        dimRange = dimRange)

# appended
sqs.train.NTERM.appended = cbind(sqs.train.PURE, sqs.train.NTERM.only_ext[, dimRange])
sqs.test.NTERM.appended = cbind(sqs.test.PURE, sqs.test.NTERM.only_ext[, dimRange])

# add biophys
sqs.train.NTERM_PROP.appended = left_join(sqs.train.NTERM.appended,
                                       sqs.train.PROP.joint) %>% na.omit()
sqs.test.NTERM_PROP.appended = left_join(sqs.test.NTERM.appended,
                                      sqs.test.PROP.joint) %>% na.omit()

res.NTERM2 = Prediction(train = sqs.train.NTERM.appended, 
                        test = sqs.test.NTERM.appended,
                        dimRange = c(7:ncol(sqs.train.NTERM.appended)))
res.NTERM2_PROP = Prediction(train = sqs.train.NTERM_PROP.appended, 
                             test = sqs.test.NTERM_PROP.appended,
                             dimRange = c(7:ncol(sqs.train.NTERM.appended)))

# evaluate performance
MakePred(res.NTERM1, "NTERM1")
MakePred(res.NTERM1_PROP, "NTERM1_PROP")

MakePred(res.NTERM2, "NTERM2")
MakePred(res.NTERM2_PROP, "NTERM2_PROP")

#### OUTPUT ### 
write.csv(sqs.train.NTERM.joint, "data/classifier/nterm1_training_EMBEDDING.csv", row.names = F)
write.csv(sqs.test.NTERM.joint, "data/classifier/nterm1_testing_EMBEDDING.csv", row.names = F)

write.csv(sqs.train.NTERM.appended, "data/classifier/nterm1_training_EMBEDDING.csv", row.names = F)
write.csv(sqs.test.NTERM.appended, "data/classifier/nterm1_testing_EMBEDDING.csv", row.names = F)

write.csv(res.NTERM1, "data/classifier/nterm1_PREDICTION.csv", row.names = F)
write.csv(res.NTERM2, "data/classifier/nterm2_PREDICTION.csv", row.names = F)

