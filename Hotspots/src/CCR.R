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
# tmp!!
seq2vec = read.csv("repres_min_regions_w3_d100_seq2vec.csv", stringsAsFactors = F, header = T)
seq2vec.tfidf = read.csv("repres_min_regions_w3_d100_seq2vec-TFIDF.csv", stringsAsFactors = F, header = T)
seq2vec.sif = read.csv("repres_min_regions_w3_d100_seq2vec-SIF.csv", stringsAsFactors = F, header = T)


### MAIN PART ###

threads = availableCores()


CCR = function(emb = ""){
  
  cl = makeCluster(threads)
  registerDoParallel(cl)
  registerDoMC(threads)
  
  print("preprocessing")
  
  emb = emb[order(emb$Accession), ]
  
  # get metainformation to add it again later
  Accession = emb$Accession
  region = emb$region
  label = emb$label
  start = emb$start
  end = emb$end
  tokens = emb$tokens
  
  # remove metainformation
  emb$Accession = NULL
  emb$region = NULL
  emb$label = NULL
  emb$start = NULL
  emb$end = NULL
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
  
  res = matrix(ncol = ncol(emb)+6, nrow = nrow(emb))
  res[, 1] = Accession
  res[, 2] = region
  res[, 3] = label
  res[, 4] = start
  res[, 5] = end
  res[, 6] = tokens
  res[, c(7:ncol(res))] = emb
  colnames(res) = c("Accession", "region", "label", "start", "end", "tokens", seq(1, ncol(emb)))
  
  return(as.data.frame(res))
  
  stopImplicitCluster()
  stopCluster(cl)
  
}

seq2vec = CCR(emb = seq2vec)
seq2vec.tfidf = CCR(seq2vec.tfidf)
seq2vec.sif = CCR(seq2vec.sif)


### OUTPUT ###
# tmp
write.csv(seq2vec, file = "repres_min_regions_w3_d100_seq2vec_CCR.csv", row.names = F)
write.csv(seq2vec.sif, file = "repres_min_regions_w3_d100_seq2vec-TFIDF_CCR.csv", row.names = F)
write.csv(seq2vec.tfidf, file = "repres_min_regions_w3_d100_seq2vec-SIF_CCR.csv", row.names = F)

