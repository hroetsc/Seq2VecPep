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
load("../HOTSPOTS/accU.RData")
windowTokens = read.csv("data/windowTokens.csv", stringsAsFactors = F)
# TFIDF = read.csv("../../RUNS/HumanProteome/v_50k/TF_IDF_hp_v50k.csv", stringsAsFactors = F)


### MAIN PART ###
### characterize count distributions
counts = log2(windowTokens$counts+1)
summary(counts)

zero = which(counts == 0)
length(zero) / length(counts)

plot(density(counts[-zero]))
# lines(density(2^counts[-zero] - 1), type = 'l', col = 'red')
summary(counts[-zero])


### sample
testSize = 0.4

n_train = ceiling(length(accU) * (1-testSize))
n_test = floor(length(accU) * testSize)

training.acc = accU[sample(length(accU), n_train)]
testing.acc = accU[sample(length(accU), n_test)]


#### clean data 
# no isoforms of training data in testing data
testing.acc = testing.acc %>%
  as.character()
testing.acc = testing.acc[-which(str_split_fixed(testing.acc, coll("-"), Inf)[,1] %in% training.acc)]

training.acc = training.acc %>%
  as.character()

# split data sets
training = windowTokens[windowTokens$Accession %in% training.acc, ]
testing = windowTokens[windowTokens$Accession %in% testing.acc, ]

# remove sliding windows that occur in training data set from testing data set
intersect(training$tokens, testing$tokens) %>% length()
rm = which(testing$tokens %in% training$tokens)
testing = testing[-rm, ]


#### split TF-IDF scores
# TFIDF$token = toupper(TFIDF$token)
# TFIDF = TFIDF[, -which(names(TFIDF) %in% c("n", "total", "tf", "idf")), ]
# 
# TFIDF_training = TFIDF[which(TFIDF$Accession %in% training.acc), ]
# TFIDF_testing = TFIDF[which(TFIDF$Accession %in% testing.acc), ]


### OUTPUT ###
write.csv(training, "data/windowTokens_training.csv", row.names = F)
write.csv(testing, "data/windowTokens_testing.csv", row.names = F)

# write.csv(TFIDF_training, "data/TFIDF_training.csv", row.names = F)
# write.csv(TFIDF_testing, "data/TFIDF_testing.csv", row.names = F)


### OPTIMIZATION DATA SETS ###
# training = read.csv("data/windowTokens_training.csv", stringsAsFactors = F)
# testing = read.csv("data/windowTokens_testing.csv", stringsAsFactors = F)
# 
# tmp, for benchmarking
# write.csv(training[sample(nrow(training), 1e03), ], "data/windowTokens_benchmark.csv", row.names = F)
#
# training.acc = training$Accession %>% unique()
# testing.acc = testing$Accession %>% unique()
# 
# training.sub = training[training$Accession %in% sample(training.acc, ceiling(length(training.acc)*0.05)), ]
# testing.sub = testing[testing$Accession %in% sample(testing.acc, ceiling(length(testing.acc)*0.05)), ]
# 
# write.csv(training.sub, "data/windowTokens_OPTtraining.csv")
# write.csv(testing.sub, "data/windowTokens_OPTtesting.csv")


### N- AND C-TERMINAL EXTENSIONS ###
# use smaller data sets for optimization
OPTtraining = read.csv("data/windowTokens_OPTtraining.csv", stringsAsFactors = F)
OPTtesting = read.csv("data/windowTokens_OPTtesting.csv", stringsAsFactors = F)

load_and_extract_features = function(extension = ""){
  
  print(extension)
  windowT = read.csv(paste0("data/", extension, "_windowTokens.csv"), stringsAsFactors = F)
  
  # split to get same train/test data as in data set
  windowT.train = windowT[windowT$tokens %in% OPTtraining$tokens, ] %>%
    unique()
  print(paste0("same training data: ", nrow(windowT.train) == nrow(OPTtraining)))
  
  
  windowT.test = windowT[windowT$tokens %in% OPTtesting$tokens, ] %>%
    unique()
  print(paste0("same training data: ", nrow(windowT.test) == nrow(OPTtesting)))
  
  # save
  write.csv(windowT.train, file = paste0("data/", extension, "_windowTokens_OPTtraining.csv"),
            row.names = F)
  write.csv(windowT.test, file = paste0("data/", extension, "_windowTokens_OPTtesting.csv"),
            row.names = F)
}

load_and_extract_features(extension = "Next")
load_and_extract_features(extension = "Cext")
load_and_extract_features(extension = "NandCext")

