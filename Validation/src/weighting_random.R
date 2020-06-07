### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  weight random embeddings by TF-IDF and SIF
# input:        sequence embeddings
# output:       embeddings weighted by TF-IDF, embeddings weighted by SIF
# author:       HR

library(plyr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)

print("### WEIIGHTING OF RANDOM EMBEDDINGS ###")

### INPUT ###
sequences = read.csv(file = snakemake@input[["formatted_sequence"]],
                     stringsAsFactors = F, header = T)
TF_IDF = read.csv(file = snakemake@input[["TF_IDF"]], stringsAsFactors = F, header = T)
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)
indices = read.csv(file = snakemake@input[["ids"]], stringsAsFactors = F, header = F)

embeddingDim = 100

### MAIN PART ###
TF_IDF$token = toupper(TF_IDF$token)

if (ncol(indices) == 3){
  indices$V1 = NULL
}
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)


# combine sequences table with tokens file
sequences.master = left_join(sequences, words)
sequences.master = sequences.master[order(sequences.master$Accession), ]

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

# calculate smooth inverse frequency
a = 0.001
find_SIF = function(tokens = "", sequence = "") {
  scores = rep(NA, length(tokens))
  tmp = TF_IDF[which(TF_IDF$Accession == sequence), ]
  
  for (t in 1:length(tokens)) {
    if (tokens[t] %in% tmp$token) {
      scores[t] = tmp[which(tmp$token == tokens[t]), "idf"]
    } else {
      scores[t] = NaN
    }
  }
  # get document frequency from inverse frequency
  scores = exp(log(nrow(sequences), scores))
  # smooth inverse frequency
  scores = a/(a + scores)
  scores[which(!is.finite(scores))] = 1
  return(scores)
}



# matrix for TF-IDF weighting
repres.TFIDF = as.data.frame(matrix(ncol = ncol(sequences.master)+embeddingDim,
                                    nrow = nrow(sequences.master))) # contains vector representation for every sequence
colnames(repres.TFIDF) = c(colnames(sequences.master), seq(1,embeddingDim,1))
dim_range = c(ncol(sequences.master)+1, ncol(repres.TFIDF))
repres.TFIDF[, c(1:(dim_range[1]-1))] = sequences.master

# matrix for SIF weighting
repres.SIF = as.data.frame(matrix(ncol = ncol(sequences.master)+embeddingDim,
                                  nrow = nrow(sequences.master))) # contains vector representation for every sequence
colnames(repres.SIF) = c(colnames(sequences.master), seq(1,embeddingDim,1))
dim_range = c(ncol(sequences.master)+1, ncol(repres.SIF))
repres.SIF[, c(1:(dim_range[1]-1))] = sequences.master


# he_uniform
# fan_in equals input units to weight tensor (roughly 5000)
fan_in = 5000
# calculate limit for uniform distribution
limit = sqrt(6 / fan_in)


# iterate sequences in master table to get their representation
progressBar = txtProgressBar(min = 0, max = nrow(sequences.master), style = 3)
for (i in 1:nrow(sequences.master)) {
  setTxtProgressBar(progressBar, i)
  
  # build temporary table that contains all tokens and weights for the current sequences
  current_tokens = t(str_split(sequences.master$tokens[i], pattern = " ", simplify = T))
  
  tfidf = as.data.frame(matrix(ncol = embeddingDim, nrow = nrow(current_tokens)))
  tfidf[, "token"] = current_tokens
  
  # random embeddings for every token in tfidf
  for (r in 1:nrow(tfidf)) {
    tfidf[r,c(1:embeddingDim)] = runif(length(c(1:embeddingDim)),
                                            min = -limit, max = limit)
  }
  
  sif = tfidf
  
  # extract TF-IDF scores for all tokens
  tfidf[, "TF_IDF"] = find_TF.IDF(tokens = tfidf$token, sequence = sequences.master$Accession[i])
  sif[, "SIF"] = find_SIF(tokens = sif$token, sequence = sequences.master$Accession[i])
  
  # multiply token embeddings by their TF-IDF scores
  tfidf[, c(1:(ncol(tfidf)-2))] = tfidf[, c(1:(ncol(tfidf)-2))] * tfidf$TF_IDF
  tfidf$TF_IDF = NULL
  tfidf$token = NULL
  
  # multiply token embeddings by their SIF scores
  sif[, c(1:(ncol(sif)-2))] = sif[, c(1:(ncol(sif)-2))] * sif$SIF
  sif$SIF = NULL
  sif$token = NULL
  
  # only proceed if embeddings for every token are found, otherwise discard sequence
  if (!any(is.na(tfidf) & is.na(sif))) {
    # calculate mean of every token dimension to get sequence dimension
    for (c in 1:ncol(tfidf)) {
      tfidf[,c] = as.numeric(as.character(tfidf[,c]))
      sif[,c] = as.numeric(as.character(sif[,c]))
    }
    repres.TFIDF[i, c(dim_range[1]:dim_range[2])] = colSums(tfidf[,c(1:embeddingDim)]) / nrow(tfidf)
    repres.SIF[i, c(dim_range[1]:dim_range[2])] = colSums(sif[,c(1:embeddingDim)]) / nrow(sif)
    
  } else {
    repres.TFIDF[i, c(dim_range[1]:dim_range[2])] = rep(NA, length(c(dim_range[1]:dim_range[2])))
    repres.SIF[i, c(dim_range[1]:dim_range[2])] = rep(NA, length(c(dim_range[1]:dim_range[2])))
  }
}

repres.TFIDF = na.omit(repres.TFIDF)
repres.TFIDF = unique(repres.TFIDF)
repres.TFIDF$tokens = NULL

repres.SIF = na.omit(repres.SIF)
repres.SIF = unique(repres.SIF)
repres.SIF$tokens = NULL


### OUTPUT ###
write.csv(repres.TFIDF, file = unlist(snakemake@output[["random_TFIDF"]]), row.names = F)
write.csv(repres.SIF, file = unlist(snakemake@output[["random_SIF"]]), row.names = F)