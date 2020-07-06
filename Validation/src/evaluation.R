### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  evaluate embeddings based on ability to capture similarity between sequences
# input:        sequence embeddings similarity matrices, "true" similarities (syntax and semantics)
# output:       heatmaps showing the differences, similarity scores (the smaller the better)
# author:       HR

library(stringr)
library(plyr)
library(dplyr)
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
# syntax = read.csv("postprocessing/similarity_true_syntax.csv", stringsAsFactors = F, header = T)
# semantics_MF = read.csv("postprocessing/similarity_true_semantics_MF.csv", stringsAsFactors = F, header = T)
# pred = read.csv("postprocessing/similarity_seq2vec.csv", stringsAsFactors = F, header = T)


# functions
# plot distributions
dist_plot = function(df = "", name = "", state = ""){
  
  png(filename = paste0(name, "_", state, "_dist.png"), type = "png")
  plot(density(df),
       main = name,
       sub = state)
  dev.off()
  
}

# remove metainformation and transform into matrix
cleaning = function(df = ""){
  df$acc1 = NULL
  df$acc2 = NULL
  
  df = as.matrix(df)
  
  return(df)
}

# calculate scores
compare = function(true = "", predicted = "", n_true = "", n_pred = ""){
  # same proteins
  if(!dim(true)[1] == dim(predicted)[1]){
    k = which(predicted[, c("acc1", "acc2")] %in% true[, c("acc1", "acc2")])
    
    predicted = predicted[, k]
    predicted = predicted[k, ]
    
    l = which(true[, c("acc1", "acc2")] %in% predicted[, c("acc1", "acc2")])
    
    true = true[, l]
    true = true[l, ]
    
  }
  
  true = cleaning(true)
  predicted = cleaning(predicted)
  
  ### plot
  dist_plot(df = true, name = n_true, state = "pre")
  dist_plot(df = predicted, name = n_pred, state = "pre")
  
  # similarities on log scale
  predicted = log(predicted)
  true = log(true)
  
  # scale between 0 and 1
  predicted = (predicted - min(predicted)) / (max(predicted) - min(predicted))
  true = (true - min(true)) / (max(true) - min(true))
  
  # z-transformation
  predicted = (predicted - mean(predicted)) / sd(predicted)
  true = (true - mean(true)) / sd(true)
  
  # transform into p-values
  predicted = pnorm(predicted)
  true = pnorm(true)
  
  ### plot
  dist_plot(df = true, name = n_true, state = "post")
  dist_plot(df = predicted, name = n_pred, state = "post")
  
  
  # score: absolute squared difference between true and predicted
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
  
  nm = str_split(snakemake@input[["predicted"]][i], coll("."), simplify = T)[,1] %>% 
    as.character()
  
  # calculate scores
  syn = compare(true = syntax, predicted = pred,
                n_true = "syntax", n_pred = nm)

  sem_MF = compare(true = semantics_MF, predicted = pred,
                   n_true = "semantics_MF", n_pred = nm)

  sem_BP = compare(true = semantics_BP, predicted = pred,
                   n_true = "semantics_BP", n_pred = nm)

  sem_CC = compare(true = semantics_CC, predicted = pred,
                   n_true = "semantics_CC", n_pred = nm)
  
  # concatenate scores
  scores = data.frame(syntax = syn,
                      semantics_MF = sem_MF,
                      semantics_BP = sem_BP,
                      semantics_CC = sem_CC)
  
  
  ### OUTPUT ###
  write.table(x = scores, file = unlist(snakemake@output[["scores"]][i]), sep = " ", row.names = F)

  print(paste0("done with ", snakemake@input[["predicted"]][i]))
}
