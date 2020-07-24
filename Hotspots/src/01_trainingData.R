### HEADER ###
# HOTSPOT REGIONS
# description: generate training data for hotspot prediction
# input: hotspot/non-hotspot regions (extended/minimal version)
# output: training data set and its embedding
# author: HR

library(dplyr)
library(stringr)

### INPUT ###
# extended version
reg = read.csv("data/regions_ext_substr.csv", stringsAsFactors = F)
TFIDF = read.csv("data/ext_substr_TFIDF.csv", stringsAsFactors = F)


### MAIN PART ###
# sliding window should contain 3 to 5 tokens (equivalent to ~ 8-15 aa)
# how many hotspot/non-hotspot regions do have this size?

reg[which(nchar(reg$region) >= 8 &
            nchar(reg$region) <= 15), ] %>% nrow()

reg[which(nchar(reg$region) >= 8 &
            nchar(reg$region) <= 15 &
            reg$label == "hotspot"), ] %>% nrow()

reg[which(nchar(reg$region) <= 15), ] %>% nrow()

# majority of short sequences are non-hotspots
# select training data with same length distributions of hotspots/non-hotspots

summary(nchar(reg$region))
summary(nchar(reg$region[which(reg$label == "hotspot")]))
summary(nchar(reg$region[which(reg$label == "non_hotspot")]))

rg = seq(8, 117)
k = 8
# how many training samples ??
# for each length, sample 2 hotspots and 2 non_hotspots

hsp.reg = reg[which(reg$label == "hotspot"), ]
n.hsp.reg = reg[which(reg$label == "non_hotspot"), ]

training.hsp = matrix(ncol = ncol(reg), nrow = length(rg)*2) %>% as.data.frame()
training.n.hsp = matrix(ncol = ncol(reg), nrow = length(rg)*2)  %>% as.data.frame()

pb = txtProgressBar(min = 0, max = length(rg), style = 3)

counter = 1

for (i in rg){
  
  setTxtProgressBar(pb, i)
  
  tmp.hsp = hsp.reg[which(nchar(hsp.reg$region) == i), ]
  tmp.n.hsp = n.hsp.reg[which(nchar(n.hsp.reg$region) == i), ]
  
  if (nrow(tmp.hsp) > 0 & nrow(tmp.n.hsp) > 0){
    
    hsp.idx = sample(nrow(tmp.hsp), k)
    n.hsp.idx = sample(nrow(tmp.n.hsp), k)
    
    training.hsp[counter:(counter+k-1), ] = tmp.hsp[hsp.idx, ]
    training.n.hsp[counter:(counter+k-1), ] = tmp.n.hsp[n.hsp.idx, ]
    
  } else {
    
    training.hsp[counter:(counter+k-1), ] = NA
    training.n.hsp[counter:(counter+k-1), ] = NA
    
  }
  
  counter = counter + k
  
}

# merge to get training data
training = rbind(training.hsp, training.n.hsp) %>% na.omit()
colnames(training) = colnames(reg)
# shuffle
training = training[sample(nrow(training)), ]


# build testing data set
# remove all proteins and isoforms that occur in the training data

testing = reg[which(!str_split_fixed(reg$Accession, coll("-"), Inf)[,1] %in%
                      str_split_fixed(training$Accession, coll("-"), Inf)[,1]), ]

print(paste0("size of training data set: ", nrow(training)))
print(paste0("size of testing data set: ", nrow(testing)))

# embedd the training data set
# referring to sequence_repres.R in Seq2Vec/src/ directory

# tmp
# training = read.csv("data/classifier/training_DATA.csv",
#                     stringsAsFactors = F)
# 
# testing = read.csv("data/classifier/testing_DATA.csv",
#                    stringsAsFactors = F)

{
  TF_IDF = TFIDF
  sequences.master = training[, c("Accession", "region", "tokens", "label")]
  sequences.master$Accession = paste0(sequences.master$Accession, "_",
                                      seq(1, nrow(sequences.master)))
  out = "training_extSubstr_w5_d100_seq2vec.csv"
  out.tfidf = "training_extSubstr_w5_d100_seq2vec-TFIDF.csv"
  out.sif = "training_extSubstr_w5_d100_seq2vec-SIF.csv"
  out.ccr = "training_extSubstr_w5_d100_seq2vec_CCR.csv"
  out.tfidf.ccr = "training_extSubstr_w5_d100_seq2vec-TFIDF_CCR.csv"
  out.sif.ccr = "training_extSubstr_w5_d100_seq2vec-SIF_CCR.csv"
  
}



### OUTPUT ###
write.csv(training, "data/classifier/training_DATA.csv", row.names = F)
write.csv(testing, "data/classifier/testing_DATA.csv", row.names = F)

