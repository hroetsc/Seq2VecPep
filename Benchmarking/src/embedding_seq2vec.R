### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  embedding based on trained weights
#               (sequence representation based on average token weights)
# input:        sequences, word-IDs, weights
# output:       seq2vec embedding
# author:       HR

library(seqinr)
library(protr)
library(Peptides)

library(plyr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)

### INPUT ###
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)
# sequences and encoded sequences
words = read.csv(snakemake@input[["words"]], stringsAsFactors = F, header = T)
# weight matrix
weight_matrix = read.csv(snakemake@input[["weights"]], stringsAsFactors = F, header = F)
# corresponding subwords
indices = read.csv(snakemake@input[["ids"]], stringsAsFactors = F, header = F)


### MAIN PART ###
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

# create matrix that will contain average of token embeddings
sequence.repres = as.data.frame(matrix(ncol = ncol(sequences.master)+ncol(weights)-2,
                                       nrow = nrow(sequences.master)))
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

sequence.repres = na.omit(sequence.repres)
sequence.repres = unique(sequence.repres)

### OUTPUT ###
# vector representation of sequences
write.csv(sequence.repres, file = unlist(snakemake@output[["embedding_seq2vec"]]), row.names = F)