### HEADER ###
# HOTSPOT REGIONS
# description: generate training data for hotspot prediction - C-terminal extension
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
train.CTERM = extensions(tbl = training, direction = "C", by = 25)
test.CTERM = extensions(tbl = testing, direction = "C", by = 25)

# sequence representations
sqs.train.CTERM.joint = get_seq_repres(train.CTERM[["joint"]], out = "train_CTERM_joint.csv")
sqs.train.CTERM.only_ext = get_seq_repres(train.CTERM[["only_ext"]], out = "train_CTERM_onlyExt.csv")

sqs.test.CTERM.joint = get_seq_repres(test.CTERM[["joint"]], out = "test_CTERM_joint.csv")
sqs.test.CTERM.only_ext = get_seq_repres(test.CTERM[["only_ext"]], out = "test_CTERM_onlyExt.csv")

# add biophysical properties
sqs.train.PROP.joint = getPropMatrix(train.CTERM[["joint"]])
sqs.test.PROP.joint = getPropMatrix(test.CTERM[["joint"]])

sqs.train.CTERM_PROP.joint = left_join(sqs.train.CTERM.joint,
                                       sqs.train.PROP.joint) %>% na.omit()
sqs.test.CTERM_PROP.joint = left_join(sqs.test.CTERM.joint,
                                      sqs.test.PROP.joint) %>% na.omit()


# compute similarities
# 2 versions: joint core region and N-terminal extension, two separate, appended vectors
# joint
dimRange = c(8:ncol(sqs.train.CTERM.joint))
res.CTERM1 = Prediction(train = sqs.train.CTERM.joint, 
                        test = sqs.test.CTERM.joint,
                        dimRange = dimRange)
res.CTERM1_PROP = Prediction(train = sqs.train.CTERM_PROP.joint, 
                             test = sqs.test.CTERM_PROP.joint,
                             dimRange = dimRange)

# appended
sqs.train.CTERM.appended = cbind(sqs.train.PURE, sqs.train.CTERM.only_ext[, dimRange])
sqs.test.CTERM.appended = cbind(sqs.test.PURE, sqs.test.CTERM.only_ext[, dimRange])

# add biophys
sqs.train.CTERM_PROP.appended = left_join(sqs.train.CTERM.appended,
                                          sqs.train.PROP.joint) %>% na.omit()
sqs.test.CTERM_PROP.appended = left_join(sqs.test.CTERM.appended,
                                         sqs.test.PROP.joint) %>% na.omit()

res.CTERM2 = Prediction(train = sqs.train.CTERM.appended, 
                        test = sqs.test.CTERM.appended,
                        dimRange = c(7:ncol(sqs.train.CTERM.appended)))
res.CTERM2_PROP = Prediction(train = sqs.train.CTERM_PROP.appended, 
                             test = sqs.test.CTERM_PROP.appended,
                             dimRange = c(7:ncol(sqs.train.CTERM.appended)))

# evaluate performance
MakePred(res.CTERM1, "CTERM1")
MakePred(res.CTERM1_PROP, "CTERM1_PROP")

MakePred(res.CTERM2, "CTERM2")
MakePred(res.CTERM2_PROP, "CTERM2_PROP")


#### OUTPUT ### 
write.csv(sqs.train.CTERM.joint, "data/classifier/cterm1_training_EMBEDDING.csv", row.names = F)
write.csv(sqs.test.CTERM.joint, "data/classifier/cterm1_testing_EMBEDDING.csv", row.names = F)

write.csv(sqs.train.CTERM.appended, "data/classifier/cterm1_training_EMBEDDING.csv", row.names = F)
write.csv(sqs.test.CTERM.appended, "data/classifier/cterm1_testing_EMBEDDING.csv", row.names = F)

write.csv(res.CTERM1, "data/classifier/cterm1_PREDICTION.csv", row.names = F)
write.csv(res.CTERM2, "data/classifier/cterm1_PREDICTION.csv", row.names = F)

