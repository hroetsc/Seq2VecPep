### HEADER ###
# HOTSPOT REGIONS
# description: generate training data for hotspot prediction - "pure" regions
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

# requires functions from 01_trainingData.R and 02_computeScores.R
# regions with extended substring

#### INPUT ### 

training = read.csv("data/classifier/training_DATA.csv", stringsAsFactors = F)
testing = read.csv("data/classifier/testing_DATA.csv", stringsAsFactors = F)


### MAIN PART ###
# sequence representations
sqs.train.PURE = get_seq_repres(sequences.master = training,
                                out = "train_PURE.csv")
sqs.test.PURE = get_seq_repres(sequences.master = testing,
                                out = "test_PURE.csv")

# add biophys
sqs.train.PROP = getPropMatrix(training)
sqs.test.PROP = getPropMatrix(testing)

sqs.train.PURE_PROP = left_join(sqs.train.PURE, sqs.train.PROP) %>% na.omit()
sqs.test.PURE_PROP = left_join(sqs.test.PURE, sqs.test.PROP) %>% na.omit()

# compute similarities
res.PURE = Prediction(train = sqs.train.PURE, test = sqs.test.PURE,
                      dimRange = c(7, ncol(sqs.train.PURE)))
res.PURE_PROP = Prediction(train = sqs.train.PURE_PROP, test = sqs.test.PURE_PROP,
                      dimRange = c(7, ncol(sqs.train.PURE_PROP)))


# evaluate performance
MakePred(res.PURE, "PURE")
MakePred(res.PURE_PROP, "PURE_PROP")


#### OUTPUT ###
write.csv(sqs.train.PURE, "data/classifier/pure_training_EMBEDDING.csv", row.names = F)
write.csv(sqs.test.PURE, "data/classifier/pure_testing_EMBEDDING.csv", row.names = F)

write.csv(sqs.train.PURE_PROP, "data/classifier/pureProp_training_EMBEDDING.csv", row.names = F)
write.csv(sqs.test.PURE_PROP, "data/classifier/pureProp_testing_EMBEDDING.csv", row.names = F)

write.csv(res.PURE, "data/classifier/pure_PREDICTION.csv", row.names = F)

