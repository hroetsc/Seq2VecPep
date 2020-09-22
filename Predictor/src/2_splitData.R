### SPLICING PREDICTOR ###
# description: split substrates into training and testing data set (all, 4 hrs and 20 hrs)
#               make sure that no substrates from the same proteins occur in both data sets
# input: proteasome DB, all possible products
# output: list of substrates for every category
# author: HR

library(plyr)
library(dplyr)
library(stringr)

split = .3

### INPUT ###
DB = read.csv("data/ProteasomeDB.csv", stringsAsFactors = F)
DB_4hrs = DB[DB$digestTime == 4, ]
DB_20hrs = DB[DB$digestTime != 4, ]

load("data/allPossible.RData")

### MAIN PART ###
# split substrates into training and testing
getSubstrates = function(DB = ""){
  
  # make suure that all substrates from the same protein are in the same data set
  SUBS = DB[, c("substrateID", "substrateSeq", "substrateOrigin")] %>% unique()
  SUBS$protein = str_split_fixed(SUBS$substrateOrigin, coll("|"), Inf)[, 1]
  no.dup = SUBS$substrateID[-which(duplicated(SUBS$protein))]
  
  k.test = no.dup[sample(length(no.dup), floor(length(no.dup)*split))]
  k.train = no.dup[!no.dup %in% k.test]
  
  for (i in 1:length(k.test)){
    d = SUBS$substrateID[SUBS$protein == SUBS$protein[SUBS$substrateID == k.test[i]]]
    if (length(d) > 0) {
      k.test = c(k.test, d)
    }
  }
  
  for (j in 1:length(k.train)){
    d = SUBS$substrateID[SUBS$protein == SUBS$protein[SUBS$substrateID == k.train[j]]]
    if (length(d) > 0) {
      k.train = c(k.train, d)
    }
  }
  
  substrates = data.frame(substrateID = c(k.test %>% unique(),
                                        k.train %>% unique()),
                          group = c(rep("test", length(k.test %>% unique())),
                                    rep("train", length(k.train %>% unique()))))
  
  substrates = left_join(substrates, SUBS)
  substrates$group = as.character(substrates$group)
  return(substrates)
}

substrates = getSubstrates(DB)
substrates_4hrs = getSubstrates(DB_4hrs)
substrates_20hrs = getSubstrates(DB_20hrs)

# apply to all-possible data set
getDataSets = function(substrates = "", col_name = "", fname = "") {
  
  train = master[names(master) %in% substrates$substrateSeq[substrates$group == "train"]] %>%
    ldply()
  names(train)[1] = "substrateSeq"
  train = left_join(train, substrates)
  train = train[, c("substrateID", "substrateSeq", "product",
                    "tokens", "positions", "type", col_name)]
  names(train)[names(train) == col_name] = "label"
  write.csv(train, paste0("data/TRAIN_", fname, ".csv"), row.names = F)
  
  test = master[names(master) %in% substrates$substrateSeq[substrates$group == "test"]] %>%
    ldply()
  names(test)[1] = "substrateSeq"
  test = left_join(test, substrates)
  test = test[, c("substrateID", "substrateSeq", "product",
                    "tokens", "positions", "type", col_name)]
  names(test)[names(test) == col_name] = "label"
  write.csv(test, paste0("data/TEST_", fname, ".csv"), row.names = F)
}


### OUTPUT ###
getDataSets(substrates, col_name = "label_all", fname = "allTP")
getDataSets(substrates, col_name = "label_4hrs", fname = "4hrs")
getDataSets(substrates, col_name = "label_20hrs", fname = "20hrs")
