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


### INPUT ###
print("LOAD DATA")
# sequences information
params = read.csv(snakemake@input[["params"]], stringsAsFactors = F, header = T)
sequences = read.csv(file = params[which(params$parameter == "Seqinput"), "value"],
                     stringsAsFactors = F, header = T)

# TF-IDF scores
TF_IDF = read.csv(file = snakemake@input[["TF_IDF"]], stringsAsFactors = F, header = T)
TF_IDF$token = toupper(TF_IDF$token)

# sequences and encoded sequences
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)

# weight matrix
weight_matrix = read.csv(file = snakemake@input[["weights"]], stringsAsFactors = F, header = F)

# corresponding subwords
indices = read.csv(file = snakemake@input[["ids"]], stringsAsFactors = F, header = F)
if (ncol(indices) == 3){
  indices$V1 = NULL
}
colnames(indices) = c("subword", "word_ID")


### MAIN PART ###
# merge indices and weights
colnames(weight_matrix)[1] = "word_ID"
indices$subword = toupper(indices$subword)

weights = full_join(weight_matrix, indices)
weights = na.omit(weights)
weights = unique(weights)

# combine sequences table with tokens file
sequences.master = left_join(sequences, words)
sequences.master = na.omit(sequences.master)

# add embeddings to master table and calculate sequence representation
# find tokens for every unique sequence (Accession) and sum up all dimensions of the tokens
# to get the respective embedding dimension
# thereby, normalize token embeddings by TF-IDF score

print("CALCULATE NUMERIC REPRESENTATION OF EVERY SEQUENCE")

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
find_TF.IDF = function(sequence = "", tokens = "") {
  scores = rep(NA, length(tokens))
  tmp = TF_IDF[which(TF_IDF$Accession == sequence), ]

  for (t in 1:length(tokens)) {
    if (tokens[t] %in% tmp$token) {
      scores[t] = tmp[which(tmp$token == tokens[t]), "tf_idf"]
    }
  }
  return(scores)
}

sequence.repres = as.data.frame(matrix(ncol = ncol(sequences.master)+ncol(weights)-2,
                                      nrow = nrow(sequences.master))) # contains vector representation for every sequence
colnames(sequence.repres) = c(colnames(sequences.master), seq(1,ncol(weights)-2,1))
dim_range = c(ncol(sequences.master)+1, ncol(sequence.repres))
sequence.repres[, c(1:(dim_range[1]-1))] = sequences.master

# iterate sequences in master table to get their representation
progressBar = txtProgressBar(min = 0, max = nrow(sequences.master), style = 3)
for (i in 1:nrow(sequences.master)) {
  setTxtProgressBar(progressBar, i)

  # build temporary table that contains all tokens and weights for the current sequences
  current_tokens = t(str_split(sequences.master$tokens[i], pattern = " ", simplify = T))
  tmp = as.data.frame(matrix(ncol = ncol(weights)-2, nrow = nrow(current_tokens)))
  tmp[, "token"] = current_tokens

  # find embeddings for every token in tmp
  for (r in 1:nrow(tmp)) {
    tmp[r,c(1:(ncol(weights)-2))] = find_tokens(paste(tmp[r, ncol(tmp)]))
  }

  # extract TF-IDF scores for all tokens
  tmp[, "TF_IDF"] = find_TF.IDF(sequence = sequences.master$Accession[i],
                                tokens = tmp$token)

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

print("FORMAT OUTPUT")
sequence.repres = na.omit(sequence.repres)
sequence.repres = unique(sequence.repres)

### OUTPUT ###
# vector representation of sequences
write.csv(sequence.repres, file = unlist(snakemake@output[["sequence_repres"]]), row.names = F)