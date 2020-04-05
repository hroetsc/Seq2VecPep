### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  predicted similarity between protein sequences based on embedding and post-processing
# input:        sequence embeddings
# output:       similarity matrix
# author:       HR

library(parallel)
library(foreach)
library(doParallel)
library(doMC)
library(plyr)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
registerDoMC(detectCores())

input = snakemake@input[["embedding"]]

foreach(i = 1:length(input)) %dopar% {
  print(snakemake@input[["embedding"]][i])
  
  ### INPUT ###
  emb = read.csv(snakemake@input[["embedding"]][i], stringsAsFactors = F, header = T)
  
  ### MAIN PART ###
  emb = emb[order(emb$Accession), ]
  proteins = emb$Accession
  
  # remove metainformation
  emb$Accession = NULL
  emb$seqs = NULL
  if("tokens" %in% colnames(emb)){
    emb$tokens = NULL
  }
  
  emb[, which(!is.finite(colSums(emb)))] = NULL
  
  # initialize matrix for similarity scores
  sim = matrix(nrow = length(proteins), ncol = length(proteins))
  colnames(sim) = proteins
  rownames(sim) = proteins
  
  # function that returns cosine of angle between two vectors
  dot_product = function(v1 = "", v2 = ""){
    p = sum(v1 * v2)/(sqrt(sum(v1^2)) * sqrt(sum(v2^2)))
    return(p)
  }
  
  progressBar = txtProgressBar(min = 0, max = length(proteins), style = 3)
  for (s in 1:length(proteins)){
    setTxtProgressBar(progressBar, s)
    
    for (p in 1:length(proteins)){
      sim[s,p] = dot_product(v1 = as.numeric(as.character(emb[s,])),
                             v2 = as.numeric(as.character(emb[p,])))
    }
  }
  
  
  # add protein information
  res = matrix(ncol = ncol(sim)+1, nrow = nrow(sim))
  res[, 1] = proteins
  res[, c(2:ncol(res))] = sim
  colnames(res) = c("Accession", seq(1, ncol(sim)))
  
  ### OUTPUT ###
  write.csv(res, file = unlist(snakemake@output[["similarity"]][i]), row.names = T)
  
}
