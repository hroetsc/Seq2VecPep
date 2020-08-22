### HEADER ###
# HOTSPOT REGIONS
# description: generate training data for hotspot prediction - shrunken hotspots
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
train.RED = extensions(tbl = training, direction = "R", by = 20)
test.RED = extensions(tbl = testing, direction = "R", by = 20)

# sequence representations
sqs.train.RED.joint = get_seq_repres(train.RED[["joint"]], out = "train_RED_joint.csv")
sqs.train.RED.only_ext = get_seq_repres(train.RED[["only_ext"]], out = "train_RED_onlyExt.csv")

sqs.test.RED.joint = get_seq_repres(test.RED[["joint"]], out = "test_RED_joint.csv")
sqs.test.RED.only_ext = get_seq_repres(test.RED[["only_ext"]], out = "test_RED_onlyExt.csv")

# add biophysical properties
sqs.train.PROP.joint = getPropMatrix(train.RED[["joint"]])
sqs.test.PROP.joint = getPropMatrix(test.RED[["joint"]])

sqs.train.RED_PROP.joint = left_join(sqs.train.RED.joint,
                                       sqs.train.PROP.joint) %>% na.omit()
sqs.test.RED_PROP.joint = left_join(sqs.test.RED.joint,
                                      sqs.test.PROP.joint) %>% na.omit()


# compute similarities
# 2 versions: joint core region and N-terminal extension, two separate, appended vectors
# joint
dimRange = c(7:ncol(sqs.train.RED.joint))
res.RED1 = Prediction(train = sqs.train.RED.joint, 
                        test = sqs.test.RED.joint,
                        dimRange = dimRange)
res.RED1_PROP = Prediction(train = sqs.train.RED_PROP.joint, 
                             test = sqs.test.RED_PROP.joint,
                             dimRange = dimRange)

# appended
sqs.train.RED.appended = sqs.train.RED.only_ext
sqs.test.RED.appended = sqs.test.RED.only_ext

# add biophys
sqs.train.RED_PROP.appended = left_join(sqs.train.RED.appended,
                                          sqs.train.PROP.joint) %>% na.omit()
sqs.test.RED_PROP.appended = left_join(sqs.test.RED.appended,
                                         sqs.test.PROP.joint) %>% na.omit()

res.RED2 = Prediction(train = sqs.train.RED.appended, 
                        test = sqs.test.RED.appended,
                        dimRange = c(7:ncol(sqs.train.RED.appended)))
res.RED2_PROP = Prediction(train = sqs.train.RED_PROP.appended, 
                             test = sqs.test.RED_PROP.appended,
                             dimRange = c(7:ncol(sqs.train.RED.appended)))

# evaluate performance
MakePred(res.RED1, "RED1")
MakePred(res.RED1_PROP, "RED1_PROP")

MakePred(res.RED2, "RED2")
MakePred(res.RED2_PROP, "RED2_PROP")

#### OUTPUT ### 
write.csv(sqs.train.RED.joint, "data/classifier/red1_training_EMBEDDING.csv", row.names = F)
write.csv(sqs.test.RED.joint, "data/classifier/red1_testing_EMBEDDING.csv", row.names = F)

write.csv(sqs.train.RED.appended, "data/classifier/red1_training_EMBEDDING.csv", row.names = F)
write.csv(sqs.test.RED.appended, "data/classifier/red1_testing_EMBEDDING.csv", row.names = F)

write.csv(res.RED1, "data/classifier/red1_PREDICTION.csv", row.names = F)
write.csv(res.RED2, "data/classifier/red2_PREDICTION.csv", row.names = F)

