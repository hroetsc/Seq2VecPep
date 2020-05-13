### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  common component removal to retrieve a better (?) sequence embedding
# input:        sequence embeddings
# output:       processed sequence embeddings
# author:       HR

library(WGCNA)

print("### COMMON COMPONENT REMOVAL ###")

input = snakemake@input[["embedding"]]

for (i in 1:length(input)){
  
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
  
  for (e in 1:ncol(emb)){
    emb[,e] = as.numeric(emb[,e])
  }
  
  emb[, which(!is.finite(colSums(emb)))] = NULL
  
  # calculate the mean vector and remove it
  mu = colSums(emb) / nrow(emb)
  
  for (e in 1:nrow(emb)){
    emb[e,] = emb[e,] - mu
  }
  
  # returns the residuals of a linear regression of each column on the principal components
  emb = as.matrix(emb)
  emb = removePrincipalComponents(x = emb, n = 1)
  
  # add protein information
  res = matrix(ncol = ncol(emb)+1, nrow = nrow(emb))
  res[, 1] = proteins
  res[, c(2:ncol(res))] = emb
  colnames(res) = c("Accession", seq(1, ncol(emb)))
  
  ### OUTPUT ###
  write.csv(res, file = unlist(snakemake@output[["CCR_emb"]][i]), row.names = T)
}

