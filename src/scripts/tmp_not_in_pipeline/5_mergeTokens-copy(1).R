### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  concatenate embedded tokens to numerical antigen matrix
# input:        weight matrix from the 3_word2vec.py script, encoded antigens table
# output:       antigen matrices
# author:       HR
setwd("/home/hroetsc/Documents/ProtTransEmbedding/Snakemake/")

library(dplyr)
library(plyr)
library(uwot)

### INPUT ###
# tmp!!!!!!!!!!!
antigens.Encoded = read.csv("results/encoded_antigens/encoded_antigens.csv", stringsAsFactors = F, header = T)
antigens.Encoded = na.omit(antigens.Encoded)
# weight matrix
weights = read.csv("results/embedded_antigens/seq2vec_weights_batch.csv", stringsAsFactors = F, header = F)
weights$V1 = NULL
# corresponding subwords
indices = read.csv("results/embedded_antigens/seq2vec_ids_batch.csv", stringsAsFactors = F, header = F)
indices$V1 = NULL
indices[nrow(indices)+1, 1] = ""
# merge indices and weights
weights = cbind(indices, weights)
colnames(weights)[1] = "subword"
weights = na.omit(weights)
weights = unique(weights)

### MAIN PART ###
# add rPCP
antigens.Encoded[,"subword"] = antigens.Encoded$SegmentedSeq
antigens.Encoded$SegmentedSeq = NULL
weights$subword = toupper(weights$subword)
weights = inner_join(antigens.Encoded[,c("subword", "rPCP")], weights)

# convert weights into numeric
for (p in 2:ncol(weights)){
  weights[,p] = as.numeric(as.character(weights[,p]))
}
weights = na.omit(weights)
head(weights)

# add seq vectors to antigens.Encoded
# memory saving (but slower) way to merge the dataframes
antigens.Encoded.list = split.data.frame(antigens.Encoded, antigens.Encoded$UniProtID)
antigens.Encoded.weights = list()

for (i in 1:length(antigens.Encoded.list)) {
  tmp = inner_join(antigens.Encoded.list[[i]], weights)
  antigens.Encoded.weights[[i]] = tmp
}
antigens.Encoded.weights = ldply(antigens.Encoded.weights, rbind)

# find tokens for every unique antigen (UniProtID) and sum up all dimensions of the tokens
# to get the respective embedding dimension
UniProtIDs = unique(as.character(antigens.Encoded.weights$UniProtID))
progressBar = txtProgressBar(min = 0, max = length(UniProtIDs), style = 3)

antigen.repres = matrix(ncol = ncol(antigens.Encoded.weights), nrow = length(UniProtIDs)) # contains vector representation for every antigen
colnames(antigen.repres) = colnames(antigens.Encoded.weights)
dim_range = c(ncol(antigens.Encoded)+1, ncol(antigen.repres))

for (u in 1:length(UniProtIDs)) {
  setTxtProgressBar(progressBar, u)
  # get tokens for current antigen
  tmp = antigens.Encoded.weights[which(antigens.Encoded.weights$UniProtID == UniProtIDs[u]),]
  # only take the 1st accession (some antigens occur within different sets of accessions)
  if (length(levels(as.factor(tmp$Accession))) > 1) {
    tmp = tmp %>% group_split(Accession)
    tmp = as.data.frame(tmp[[1]])
  }
  for (c in seq(dim_range[1], dim_range[2])) {
    tmp[,c] = as.numeric(tmp[,c])
  }
  # calculate colsums
  c_sum = colSums(tmp[,c(dim_range[1]:dim_range[2])])
  # add to df
  ln = antigens.Encoded.weights[which(antigens.Encoded.weights$UniProtID == UniProtIDs[u])[1],c(1:dim_range[1]-1)]
  antigen.repres[u, seq(1,dim_range[1]-1)] = as.character(ln[1,])
  antigen.repres[u, c(dim_range[1]:dim_range[2])] = c_sum
  rm(tmp)
}

### EVALUATION ###
# get statistics summary
for (c in 1:nrow(antigen.repres)) {
  ln = as.numeric(antigen.repres[c, c(dim_range[1]:ncol(antigen.repres))])
  print(summary(ln))
  dens = density(ln)
  plot(dens, main = paste0("peptide: ", antigen.repres[c,1]))
}
antigen.repres = na.omit(antigen.repres)
antigen.repres = unique(antigen.repres)

# UMAP and biophysical properties
mode(antigen.repres[, c(dim_range[1]:ncol(antigen.repres))]) = "numeric"

set.seed(42)
dims_UMAP = umap(antigen.repres,
                  n_neighbors = 10,
                  min_dist = 0.0,
                  spread = 2,
                  #n_trees = 50,
                  verbose = T,
                  approx_pow = T,
                  ret_model = T,
                 # metric = list("categorical" = colnames(antigen.repres[,c(1:dim_range[1]-1)]),
                  #              "cosine" = colnames(antigen.repres[,c(dim_range[1]:ncol(antigen.repres))])),
                  scale = "none",
                  n_epochs = 500,
                  n_threads = 11)

### OUTPUT ###
# merged antigen table and token vectors
save(file = "./seq2vec/5_merge_antigensWeights.RData", antigens.Encoded.weights)
write.csv(antigens.Encoded.weights, "./seq2vec/5_merge_antigensWeights.csv")
load("./seq2vec/5_merge_antigensWeights.RData")
# vector representation of antigens
save(file = "./seq2vec/5_merge_antigen_repres.RData", antigen.repres)
write.csv(antigen.repres, "./seq2vec/5_merge_antigen_repres.csv")
