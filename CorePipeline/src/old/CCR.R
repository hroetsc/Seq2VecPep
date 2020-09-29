### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  common component removal to retrieve a better (?) sequence embedding
# input:        sequence embeddings
# output:       processed sequence embeddings
# author:       HR

library(WGCNA)

library(foreach)
library(doParallel)
library(doMC)
library(future)


print("### COMMON COMPONENT REMOVAL ###")


### INPUT ###
seq2vec = read.csv(snakemake@input[["sequence_repres_seq2vec"]], stringsAsFactors = F, header = T)
seq2vec.tfidf = read.csv(snakemake@input[["sequence_repres_seq2vec_TFIDF"]],
                         stringsAsFactors = F, header = T)
seq2vec.sif = read.csv(snakemake@input[["sequence_repres_seq2vec_SIF"]],
                        stringsAsFactors = F, header = T)

# tmp!!
# setwd("/home/hanna/Documents/QuantSysBios/ProtTransEmbedding/RUNS/HumanProteome/results")
# seq2vec = read.csv("hp_sequence_repres_w5_d100_seq2vec.csv", stringsAsFactors = F, header = T)
# seq2vec.tfidf = read.csv("hp_sequence_repres_w5_d100_seq2vec-TFIDF.csv", stringsAsFactors = F, header = T)
# seq2vec.sif = read.csv("hp_sequence_repres_w5_d100_seq2vec-SIF.csv", stringsAsFactors = F, header = T)


### MAIN PART ###

threads = availableCores()


CCR = function(emb = ""){
  
  cl = makeCluster(threads)
  registerDoParallel(cl)
  registerDoMC(threads)
  
  print("preprocessing")
  
  emb = emb[order(emb$Accession), ]
  
  # get metainformation to add it again later
  proteins = emb$Accession
  seq = emb$seqs
  tokens = emb$tokens
  
  # remove metainformation
  emb$Accession = NULL
  emb$seqs = NULL
  emb$tokens = NULL
  
  for (e in 1:ncol(emb)){
    emb[,e] = as.numeric(emb[,e])
  }
  
  emb[, which(!is.finite(colSums(emb)))] = NULL
  
  
  # calculate the mean vector and remove it
  print("remove mean vector")
  
  mu = colSums(emb) / nrow(emb)
  
  
  emb = foreach(e = 1:nrow(emb), .combine = "rbind") %dopar% {
    emb[e,] = emb[e,] - mu
  }
  
  # returns the residuals of a linear regression of each column on the principal components
  
  print("linear regression on PC1")
  
  emb = as.matrix(emb)
  emb = removePrincipalComponents(x = emb, n = 1)
  
  # add protein information
  
  print("format output")
  
  res = matrix(ncol = ncol(emb)+3, nrow = nrow(emb))
  res[, 1] = proteins
  res[, 2] = seq
  res[, 3] = tokens
  res[, c(4:ncol(res))] = emb
  colnames(res) = c("Accession", "seqs", "tokens", seq(1, ncol(emb)))
  
  return(as.data.frame(res))
  
  stopImplicitCluster()
  stopCluster(cl)
  
}

seq2vec = CCR(seq2vec)
#write.csv(seq2vec, file = "hp_sequence_repres_seq2vec_CCR.csv", row.names = F)

seq2vec.tfidf = CCR(seq2vec.tfidf)
#write.csv(seq2vec.sif, file = "hp_sequence_repres_seq2vec-TFIDF_CCR.csv", row.names = F)

seq2vec.sif = CCR(seq2vec.sif)
# write.csv(seq2vec.tfidf, file = "hp_sequence_repres_seq2vec-SIF_CCR.csv", row.names = F)


### OUTPUT ###
write.csv(seq2vec, file = unlist(snakemake@output[["sequence_repres_seq2vec_CCR"]]), row.names = F)
write.csv(seq2vec.sif, file = unlist(snakemake@output[["sequence_repres_seq2vec_CCR"]]), row.names = F)
write.csv(seq2vec.tfidf, file = unlist(snakemake@output[["sequence_repres_seq2vec_CCR"]]), row.names = F)
