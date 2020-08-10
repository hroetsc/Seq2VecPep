### HEADER ###
# HOTSPOT PREDICTION
# description: generate table with token embeddings
# input: weights file and indices
# output: token embeddings
# author: HR

library(plyr)
library(dplyr)
library(stringr)
library(rhdf5)


### INPUT ###
indices = read.csv("../RUNS/HumanProteome/v_5k/ids_hp_w5_new.csv", stringsAsFactors = F, header = F)
weight_matrix = h5read("../RUNS/HumanProteome/v_5k/word2vec_model/hp_model_w5_d100/weights.h5", "/embedding/embedding")

embeddingDim = 100
tokPerWindow = 8

### MAIN PART ###
# assign tokens to weight matrix
{
  if (ncol(indices) == 3){
    indices$V1 = NULL
  }
  colnames(indices) = c("subword", "word_ID")
  indices$subword = toupper(indices$subword)
  
  # extract weights
  weight_matrix = plyr::ldply(weight_matrix)
  weight_matrix = t(weight_matrix) %>% as.data.frame()
  
  weight_matrix = weight_matrix[-1,]
  colnames(weight_matrix) = seq(1, ncol(weight_matrix))
  
  weight_matrix["word_ID"] = seq(0, (nrow(weight_matrix)-1), 1)
  weight_matrix$word_ID = weight_matrix$word_ID + 1
  
  
  # merge indices and weights
  weights = full_join(weight_matrix, indices) %>% na.omit() %>% unique()
}

weights$word_ID = NULL
weights = weights[, c("subword", seq(1, embeddingDim))]


### OUTPUT ###
write.csv(weights, "results/token_embeddings.csv", row.names = F)


