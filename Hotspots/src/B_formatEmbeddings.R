### HEADER ###
# HOTSPOT PREDICTION
# description: merge TF-IDF scores with table of token embeddings and weight embeddings by TF-IDF scores
#               to facilitate neural net training downstream
# input: TF-IDF scores, weights file and indices
# output: huge table with weighted token embeddings
# author: HR

library(plyr)
library(dplyr)
library(stringr)
library(rhdf5)


### INPUT ###
indices = read.csv("../RUNS/HumanProteome/ids_hp_w5_new.csv", stringsAsFactors = F, header = F)
weight_matrix = h5read("../RUNS/HumanProteome/word2vec_model/hp_model_w5_d100/weights.h5", "/embedding/embedding")

TF_IDF = read.csv("../RUNS/HumanProteome/TF_IDF.csv", stringsAsFactors = F, header = T)
load("HOTSPOTS/accU.RData")

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

# merge tokens and weights
TF_IDF = TF_IDF[TF_IDF$Accession %in% accU, ]
TF_IDF$token = toupper(TF_IDF$token)
names(weights)[names(weights) == "subword"] = "token"

tfidf.weights = left_join(TF_IDF, weights)
tfidf.weights = tfidf.weights[, -which(names(tfidf.weights) %in% c("n", "total", "tf", "idf", "word_ID"))]

# multiply token embeddings by tf-idf score and save
# do it in chunks (proteins) due to memory

weightsRange = c(4:ncol(tfidf.weights))

out = "results/tfidf-weights.csv"
system(paste0("rm ", out))


acc = tfidf.weights$Accession %>% as.character %>% unique()
pb = txtProgressBar(min = 0, max = length(acc), style = 3)

for (c in 1:length(acc)) {
  
  setTxtProgressBar(pb, c)
  
  cnt_tfidf.weights = tfidf.weights[tfidf.weights$Accession == acc[c], ] %>%
    as.matrix()
  
  for (i in 1:nrow(cnt_tfidf.weights)) {
    
    cnt_tfidf.weights[i, weightsRange] = as.numeric(cnt_tfidf.weights[i, weightsRange]) * 
      as.numeric(cnt_tfidf.weights[i, "tf_idf"])
    
  }
  
  if(file.exists(out)) {
    write.table(cnt_tfidf.weights, out, sep = ",", row.names = F, append = T, col.names = F)
    
  } else {
    write.table(cnt_tfidf.weights, out, sep = ",", row.names = F, append = F, col.names = T)
    
  }
  
}

### OUTPUT ###

