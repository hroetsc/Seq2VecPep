### HEADER ###
# HOTSPOT REGIONS
# description: generate training data for hotspot prediction - concatenate everything with biophysical
#               properties
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

### INPUT ###
# biophys properties calculated in 01_trainingData.R
sqs.train.PURE = read.csv()


### MAIN PART ###
# concatenation


# run prediction


# evaluation



#### OUTPUT ####