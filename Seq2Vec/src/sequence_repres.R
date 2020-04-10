### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  concatenate embedded tokens to numerical sequence matrix
# input:        weight matrix from seq2vec_model_training, word table (Accession, tokens), TF-IDF scores,
#               table that maps word-IDs to tokens
# output:       sequence matrices
# author:       HR


print("### RETRIEVE PROTEIN REPRESENATION ###")

library(seqinr)
library(protr)
library(Peptides)

library(plyr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)

library(ggplot2)
library(ggthemes)
library(ggpubr)
library(uwot)
library(reshape2)

library(foreach)
library(doParallel)
library(doMC)


# sequences = read.csv("../../ProtTransEmbedding/files/ProteasomeDB.csv", stringsAsFactors = F, header = T)
# words = read.csv("database-embedding/words.csv", stringsAsFactors = F, header = T)
# weight_matrix = read.csv("database-embedding/seq2vec_weights.csv", stringsAsFactors = F, header = F)
# indices = read.csv("database-embedding/seq2vec_ids.csv", stringsAsFactors = F, header = F)
# TF_IDF = read.csv("database-embedding/TF_IDF.csv", stringsAsFactors = F, header = T)
# params = read.csv("database-embedding/hyperparams.csv", stringsAsFactors = F, header = T)

### INPUT ###
print("LOAD DATA")
# sequences information
params = read.csv(snakemake@input[["params"]], stringsAsFactors = F, header = T)
sequences = read.csv(file = params[which(params$parameter == "Seqinput"), "value"],
                     stringsAsFactors = F, header = T)
# TF-IDF scores
TF_IDF = read.csv(file = snakemake@input[["TF_IDF"]], stringsAsFactors = F, header = T)
# sequences and encoded sequences
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)
# weight matrix
weight_matrix = read.csv(file = snakemake@input[["weights"]], stringsAsFactors = F, header = F)
# corresponding subwords
indices = read.csv(file = snakemake@input[["ids"]], stringsAsFactors = F, header = F)


### MAIN PART ###
# multiprocessing
cl <- makeCluster(as.numeric(params[which(params$parameter == "threads"), "value"]))
registerDoParallel(cl)
registerDoMC(as.numeric(params[which(params$parameter == "threads"), "value"]))

# cl <- makeCluster(16)
# registerDoParallel(cl)
# registerDoMC(16)


TF_IDF$token = toupper(TF_IDF$token)

if (ncol(indices) == 3){
  indices$V1 = NULL
}
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)

# merge indices and weights
colnames(weight_matrix)[1] = "word_ID"
weight_matrix$word_ID = weight_matrix$word_ID + 1

weights = full_join(weight_matrix, indices)
weights = na.omit(weights)
weights = unique(weights)


# combine sequences table with tokens file
sequences.master = left_join(sequences, words)
sequences.master = na.omit(sequences.master)

# define function (that allows parallel searching) for tokens in weight matrix
find_tokens = function(token = ""){
  if (token %in% weights$subword) {
    return(weights[which(token == weights$subword)[1], c(2:(ncol(weights)-1))])[1, ]
  } else {
    print(paste0("no embedding found for token: ", token))
    return(rep(NA, length(c(2:(ncol(weights)-1)))))
  }
}

# define function that finds TF-IDF score for token in correct sequence
find_TF.IDF = function(tokens = "", sequence = "") {
  scores = rep(NA, length(tokens))
  tmp = TF_IDF[which(TF_IDF$Accession == sequence), ]
  
  for (t in 1:length(tokens)) {
    if (tokens[t] %in% tmp$token) {
      scores[t] = tmp[which(tmp$token == tokens[t]), "tf_idf"]
    } else {
      scores[t] = 1
    }
  }
  return(scores)
}


# create matrix that will contain average of token embeddings
sequence.repres = as.data.frame(matrix(ncol = ncol(sequences.master)+ncol(weights)-2,
                                       nrow = nrow(sequences.master)))
colnames(sequence.repres) = c(colnames(sequences.master), seq(1,ncol(weights)-2,1))
dim_range = c(ncol(sequences.master)+1, ncol(sequence.repres))
sequence.repres[, c(1:(dim_range[1]-1))] = sequences.master

# iterate sequences in master table to get their representation

print("RETRIEVING NUMERIC REPRESENTATION OF EVERY SEQUENCE")

sequence.repres = foreach(i = 1:nrow(sequences.master), .combine = "rbind") %dopar% {
  
  # build temporary table that contains all tokens and weights for the current sequences
  current_tokens = t(str_split(sequences.master$tokens[i], pattern = " ", simplify = T))
  tmp = as.data.frame(matrix(ncol = ncol(weights)-2, nrow = nrow(current_tokens)))
  tmp[, "token"] = current_tokens
  
  # find embeddings for every token in tmp
  for (r in 1:nrow(tmp)) {
    tmp[r,c(1:(ncol(weights)-2))] = find_tokens(paste(tmp[r, ncol(tmp)]))
  }
  
  # extract TF-IDF scores for all tokens
  tmp[, "TF_IDF"] = find_TF.IDF(tokens = tmp$token, sequence = sequences.master$Accession[i])
  
  # multiply token embeddings by their TF-IDF scores
  tmp[, c(1:(ncol(tmp)-2))] = tmp[, c(1:(ncol(tmp)-2))] * tmp$TF_IDF
  tmp$TF_IDF = NULL
  tmp$token = NULL
  
  # only proceed if embeddings for every token are found, otherwise discard sequence
  if (!any(is.na(tmp))) {
    # calculate mean of every token dimension to get sequence dimension
    for (c in 1:ncol(tmp)) {
      tmp[,c] = as.numeric(as.character(tmp[,c]))
    }
    sequence.repres[i, c(dim_range[1]:dim_range[2])] = colSums(tmp[,c(1:(ncol(weights)-2))]) / nrow(tmp)
    
  } else {
    sequence.repres[i, c(dim_range[1]:dim_range[2])] = rep(NA, length(c(dim_range[1]:dim_range[2])))
  }
}

print("DONE")

sequence.repres = cbind(sequences.master, sequence.repres)

sequence.repres = na.omit(sequence.repres)
sequence.repres = unique(sequence.repres)

### OUTPUT ###
# vector representation of sequences
# write.csv(sequence.repres, file = "database-embedding/sequence_repres_TF-IDF.csv", row.names = F)

write.csv(sequence.repres, file = unlist(snakemake@output[["sequence_repres"]]), row.names = F)