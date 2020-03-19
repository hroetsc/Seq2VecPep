### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  concatenate embedded tokens to numerical protein matrix
# input:        weight matrix from seq2vec_model_training, word table (UniProtID, tokens and their TF-IDF scores)
# output:       protein matrices
# author:       HR


#tmp !!!
setwd("Documents/QuantSysBios/ProtTransEmbedding/Snakemake/")
weight_matrix = read.csv(file = "results/embedded_proteome/opt_seq2vec_weights_10000.csv", stringsAsFactors = F, header = F)
indices = read.csv(file = "results/embedded_proteome/opt_seq2vec_ids_1000.csv", stringsAsFactors = F, header = F)
proteome = read.csv(file = "data/peptidome/formatted_proteome.csv", stringsAsFactors = F, header = T)
words = read.csv(file = "results/encoded_proteome/opt_words_10000.csv", stringsAsFactors = F, header = T)

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
library(reshape)

#library(parallel)
library(foreach)
library(doParallel)
#cluster = parallel::makeCluster(12)
#doParallel::registerDoParallel(cluster)

### INPUT ###
print("LOAD DATA")
# proteome information
proteome = read.csv(file = snakemake@input[["formatted_proteome"]], stringsAsFactors = F, header = T)
tokens = read.csv(file = snakemake@input[["TF_IDF"]], stringsAsFactors = F, header = T)
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)

# weight matrix
weight_matrix = read.csv(file = snakemake@input[["weights"]], stringsAsFactors = F, header = F)

# corresponding subwords
indices = read.csv(file = snakemake@input[["ids"]], stringsAsFactors = F, header = F)
colnames(indices)[2] = "subword"

# merge indices and weights
colnames(weight_matrix)[1] = "word_ID"
indices$V1 = NULL
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)

weights = full_join(weight_matrix, indices)
weights = na.omit(weights)
weights = unique(weights)

### MAIN PART ###
# combine proteome table with tokens file
proteins.master = left_join(proteome, words)
proteins.master = na.omit(proteins.master)

# add embeddings to master table and calculate protein representation
# find tokens for every unique protein (UniProtID) and sum up all dimensions of the tokens
# to get the respective embedding dimension
# thereby, normalize token embeddings by TF-IDF score

print("CALCULATE NUMERIC REPRESENTATION OF EVERY PROTEIN")

# define function (that allows parallel searching) for tokens in weight matrix
find_tokens = function(token = ""){
  if (token %in% weights$subword) {
    return(weights[which(token == weights$subword)[1], c(2:(ncol(weights)-1))])[1, ]
  } else {
    print(paste0("no embedding found for token: ", token))
    return(rep(NA, length(c(2:(ncol(weights)-1)))))
  }
}

protein.repres = as.data.frame(matrix(ncol = ncol(proteins.master)+ncol(weights)-2,
                                      nrow = nrow(proteins.master))) # contains vector representation for every protein
colnames(protein.repres) = c(colnames(proteins.master), seq(1,ncol(weights)-2,1))
dim_range = c(ncol(proteins.master)+1, ncol(protein.repres))
protein.repres[,c(1:(dim_range[1]-1))] = proteins.master

# iterate proteins in master table to get their representation
progressBar = txtProgressBar(min = 0, max = nrow(proteins.master), style = 3)
for (i in 1:nrow(proteins.master)) {
  setTxtProgressBar(progressBar, i)
  
  # build temporary table that contains all tokens and weights for the current proteins
  current_tokens = t(str_split(proteins.master$tokens[i], pattern = " ", simplify = T))
  tmp = as.data.frame(matrix(ncol = ncol(weights)-2, nrow = nrow(current_tokens)))
  tmp[, "token"] = current_tokens
  
  # find embeddings for every token in tmp
  for (r in 1:nrow(tmp)) {
    tmp[r,c(1:(ncol(weights)-2))] = find_tokens(paste(tmp[r, ncol(tmp)]))
  }
  
  # only proceed if embeddings for every token are found, otherwise discard protein
  if (!is.na(tmp)) {
    # calculate mean of every token dimension to get protein dimension
    for (c in 1:(ncol(weights)-2)) {
      tmp[,c] = as.numeric(as.character(tmp[,c]))
    }
    protein.repres[i, c(dim_range[1]:dim_range[2])] = colSums(tmp[,c(1:(ncol(weights)-2))]) / nrow(tmp)
    
  } else {
    protein.repres[i, c(dim_range[1]:dim_range[2])] = rep(NA, length(c(dim_range[1]:dim_range[2])))
  }
}

print("FORMAT OUTPUT")
protein.repres = na.omit(protein.repres)
protein.repres = unique(protein.repres)

### EVALUATION ###
print("EVALUATION")
# generate a random weights matrix to evaluate embeddings
vocab_size = nrow(weight_matrix) # input dimension of embedding layer
embedding_dim = dim_range[2] - dim_range[1] +1 # output dimension of embedding layer

# calculate limits for sampling from uniform distribution
limit = as.numeric(sqrt(6/(vocab_size + embedding_dim)))

# function that generates random protein representation given the limit and the number of tokens
random_protein = function(limit = "", no_tokens = "") {
  tmp = matrix(ncol = embedding_dim, nrow = no_tokens)
  for (j in 1:nrow(tmp)) {
    tmp[j, ] = runif(embedding_dim, -1*limit, limit)
  }
  return(colSums(tmp) / no_tokens)
}

# apply to generate random protein embeddings
protein.repres.random = protein.repres
protein.repres.random[, "no_tokens"] = ncol(str_split(protein.repres$tokens, pattern = " ", simplify = T))
for (i in 1:nrow(protein.repres.random)) {
  protein.repres.random[i, c(dim_range[1]:dim_range[2])] = random_protein(limit = limit,
                                                                          no_tokens = protein.repres.random[i, "no_tokens"])
}

# get statistics summary
# for (c in 1:nrow(protein.repres)) {
#   ln = as.numeric(protein.repres[c, c(dim_range[1]:ncol(protein.repres))])
#   print(summary(ln))
#   dens = density(ln)
#   plot(dens, main = paste0("protein: ", protein.repres[c,"UniProtID"]))
# }

for (p in dim_range[1]:ncol(protein.repres)){
  protein.repres[,p] = as.numeric(as.character(protein.repres[,p]))
  protein.repres.random[,p] = as.numeric(as.character(protein.repres.random[,p]))
}

### OUTPUT ###
# tmp!!
write.csv(protein.repres, file = "results/embedded_proteome/opt_proteome_repres.csv", row.names = F)
write.csv(protein.repres.random, file = "results/embedded_proteome/opt_proteome_repres_random.csv", row.names = F)

# vector representation of proteins
write.csv(protein.repres, file = unlist(snakemake@output[["proteome_repres"]]), row.names = F)
write.csv(protein.repres.random, file = unlist(snakemake@output[["proteome_repres_random"]]), row.names = F)
