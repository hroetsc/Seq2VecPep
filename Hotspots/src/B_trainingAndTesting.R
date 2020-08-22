### HEADER ###
# HOTSPOT PREDICTION
# description: split features into training and testing data set
# input: features from A_featureGeneration.R, protein accessions
# output: training and testing data set
# author: HR

library(plyr)
library(dplyr)
library(stringr)

### INPUT ###
load("HOTSPOTS/accU.RData")
windowTokens = read.csv("results/windowTokens.csv", stringsAsFactors = F)
TFIDF = read.csv("../RUNS/HumanProteome/TF_IDF.csv", stringsAsFactors = F)

### MAIN PART ###
testSize = 0.2

n_train = ceiling(length(accU) * (1-testSize))
n_test = floor(length(accU) * testSize)

training.acc = accU[sample(length(accU), n_train)]
testing.acc = accU[sample(length(accU), n_test)]

# no isoforms of training data in testing data
testing.acc = testing.acc %>%
  as.character()
testing.acc = testing.acc[-which(str_split_fixed(testing.acc, coll("-"), Inf)[,1] %in% training.acc)]

# split data sets
training = windowTokens[windowTokens$Accession %in% training.acc, ]
testing = windowTokens[windowTokens$Accession %in% testing.acc, ]

TFIDF$token = toupper(TFIDF$token)
TFIDF = TFIDF[, -which(names(TFIDF) %in% c("n", "total", "tf", "idf")), ]

TFIDF_training = TFIDF[which(TFIDF$Accession %in% training.acc), ]
TFIDF_testing = TFIDF[which(TFIDF$Accession %in% testing.acc), ]


### OUTPUT ###
write.csv(training, "results/windowTokens_training.csv", row.names = F)
write.csv(testing, "results/windowTokens_testing.csv", row.names = F)

write.csv(TFIDF_training, "results/TFIDF_training.csv", row.names = F)
write.csv(TFIDF_testing, "results/TFIDF_testing.csv", row.names = F)

