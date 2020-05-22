### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  evaluate embeddings based on ability to capture similarity between sequences
# input:        sequence embeddings similarity matrices, "true" similarities (syntax and semantics)
# output:       heatmaps showing the differences, similarity scores (the smaller the better)
# author:       HR

library(stringr)
library(dplyr)
library(plyr)
library(tidyr)

library(reshape2)

library(foreach)
library(doParallel)
library(doMC)
library(future)

print("### COMPUTE SCORES ###")

# parallel computing
cl <- makeCluster(availableCores())
registerDoParallel(cl)
registerDoMC(availableCores())

### INPUT ###
# overall input
syntax = read.csv(snakemake@input[["true_syntax"]], stringsAsFactors = F, header = T)
semantics_MF = read.csv(snakemake@input[["true_semantics_MF"]], stringsAsFactors = F, header = T)
semantics_BP = read.csv(snakemake@input[["true_semantics_BP"]], stringsAsFactors = F, header = T)
semantics_CC = read.csv(snakemake@input[["true_semantics_CC"]], stringsAsFactors = F, header = T)

# tmp !!!
# syntax = read.csv("similarity/true_syntax.csv", stringsAsFactors = F, header = T)
# semantics_MF = read.csv("similarity/true_semantics_MF.csv", stringsAsFactors = F, header = T)
# pred = read.csv("similarity/seq2vec_TFIDF_CCR.csv", stringsAsFactors = F, header = T)


# functions
# preprocessing
prots = function(tbl = ""){
  # sort in protein order
  if ("Accession" %in% colnames(tbl)){
    tbl = tbl[order(tbl[, "Accession"]), ]
    return(tbl[, "Accession"])
  } else {
    tbl = tbl[order(tbl[,1]), ]
    return(tbl[,1])
  }
}

cleaning = function(tbl = ""){
  # sort in protein order
  if ("Accession" %in% colnames(tbl)){
    tbl = tbl[order(tbl[, "Accession"]), ]
    tbl$Accession = NULL
  } else {
    tbl = tbl[order(tbl[,1]), ]
    tbl[,1] = NULL
  }

  # remove redundant columns
  if ("seqs" %in% colnames(tbl)){
    tbl$seqs = NULL
  }

  if ("X" %in% colnames(tbl)){
    tbl$X = NULL
  }

  return(as.matrix(tbl))
}


# calculate scores
compare = function(true = "", predicted = "", prot_pred = "", prot_true = ""){
  # same proteins
  if(!dim(true)[1] == dim(predicted)[1]){
    k = which(prot_pred %in% prot_true)
    
    predicted = predicted[, k]
    predicted = predicted[k, ]
    
    l = which(prot_true %in% prot_pred)
    
    true = true[, l]
    true = true[l, ]
    
  }
  
  # scale between 0 and 1
  predicted = (predicted - min(predicted)) / (max(predicted) - min(predicted))
  true = (true - min(true)) / (max(true) - min(true))
  
  # z-transformation
  predicted = (predicted - mean(predicted)) / sd(predicted)
  true = (true - mean(true)) / sd(true)
  
  # transform into p-values
  predicted = pnorm(predicted)
  true = pnorm(true)
  
  
  # score: absolute squared difference between true and random
  tbl = (predicted - true)^2

  
  # mean of difference between matrices and standard deviation
  diff = mean(tbl)
  SD = sd(tbl)

  
  # pearson correlation between true and predicted similarity scores
  corr = cor(melt(true)$value, melt(predicted)$value, method = "spearman")


  return(c(diff, SD, corr))
}



# actually start computing

input = snakemake@input[["predicted"]]

foreach(i = 1:length(input)) %dopar% {

  print(snakemake@input[["predicted"]][i])

  ### INPUT ###
  pred = read.csv(snakemake@input[["predicted"]][i], stringsAsFactors = F, header = T)

  ### MAIN PART ###
 
  # preprocessing
  prot_syn = prots(tbl = syntax)
  prot_sem_MF = prots(tbl = semantics_MF)
  prot_sem_BP = prots(tbl = semantics_BP)
  prot_sem_CC = prots(tbl = semantics_CC)
  prot_pred = prots(tbl = pred)
  
  
  syntax = cleaning(tbl = syntax)
  semantics_MF = cleaning(tbl = semantics_MF)
  semantics_BP = cleaning(tbl = semantics_BP)
  semantics_CC = cleaning(tbl = semantics_CC)
  pred = cleaning(tbl = pred)
  
  
  # calculate scores
  syn = compare(true = syntax, predicted = pred,
                prot_pred = prot_pred, prot_true = prot_syn)

  sem_MF = compare(true = semantics_MF, predicted = pred,
                   prot_pred = prot_pred, prot_true = prot_sem_MF)

  sem_BP = compare(true = semantics_BP, predicted = pred,
                   prot_pred = prot_pred, prot_true = prot_sem_BP)

  sem_CC = compare(true = semantics_CC, predicted = pred,
                   prot_pred = prot_pred, prot_true = prot_sem_CC)

  scores = data.frame(syntax = syn,
                      semantics_MF = sem_MF,
                      semantics_BP = sem_BP,
                      semantics_CC = sem_CC)
  
  ### OUTPUT ###
  write.table(x = scores, file = unlist(snakemake@output[["scores"]][i]), sep = " ", row.names = F)

  print(paste0("done with ", snakemake@input[["predicted"]][i]))
}
