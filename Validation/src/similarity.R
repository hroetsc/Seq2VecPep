### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  predicted similarity between protein sequences based on embedding and post-processing
# input:        sequence embeddings
# output:       similarity matrix
# author:       HR

library(foreach)
library(doParallel)
library(doMC)
library(future)
library(plyr)

#library(emdist)

library(plyr)
library(dplyr)

print("### CALCULATE PAIRWISE EMBEDDING SIMILARITIES ###")

# parallel computing
cl <- makeCluster(availableCores())
registerDoParallel(cl)
registerDoMC(availableCores())

input = snakemake@input[["embedding"]]

accessions = read.csv(snakemake@input[["batch_accessions"]], stringsAsFactors = F, header = T)


# function that returns cosine of angle between two vectors
matmult = function(v1 = "", v2 = ""){
  return(as.numeric(v1) %*% as.numeric(v2))
}

dot_product = function(v1 = "", v2 = ""){
  p = matmult(v1, v2)/(sqrt(matmult(v1, v1)) * sqrt(matmult(v2, v2)))
  return(p)
}


foreach(i = 1:length(input)) %dopar% {
  print(snakemake@input[["embedding"]][i])

  ### INPUT ###
  emb = read.csv(snakemake@input[["embedding"]][i], stringsAsFactors = F, header = T)
  # emb = read.csv("postprocessing/QSO.csv", stringsAsFactors = F, header = T)
  
  ### MAIN PART ###
  emb = emb[order(emb$Accession), ]
  proteins = emb$Accession

  # remove metainformation
  emb$seqs = NULL
  if("tokens" %in% colnames(emb)){
    emb$tokens = NULL
  }

  #emb[, which(!is.finite(colSums(emb)))] = NULL

  sim = accessions %>% as.data.frame()
  sim$similarity = NULL
  
  pb = txtProgressBar(min = 0, max = nrow(accessions), style = 3)
  
  for(a in 1:nrow(accessions)) {
    setTxtProgressBar(pb, a)
    
    v1 = emb[which(emb$Accession == accessions$acc1[a]), c(2:ncol(emb))] %>% as.numeric()
    v2 = emb[which(emb$Accession == accessions$acc2[a]), c(2:ncol(emb))] %>% as.numeric()
    
    # dot product similarity
    sim[a, "dot"] = dot_product(v1 = v1, v2 = v2)
    #euclidean distance
    sim[a, "euclidean"] = dist(rbind(v1,v2), method = "euclidean")
    # earth mover's distance
    # sim[a, "emd"] = emd2d(as.matrix(v1), as.matrix(v2))
  }


  ### OUTPUT ###
  print(paste0("done with: ",snakemake@input[["embedding"]][i]))
  write.csv(sim, file = unlist(snakemake@output[["similarity"]][i]), row.names = F)

}
