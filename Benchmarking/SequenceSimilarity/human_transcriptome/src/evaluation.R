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

library(RColorBrewer)
library(grDevices)
library(lattice)
library(reshape2)

library(parallel)
library(foreach)
library(doParallel)
library(doMC)
library(plyr)

# tmp!
# syntax = read.csv("proteome/similarity/true_syntax.csv", stringsAsFactors = F, header = T)
# semantics_MF = read.csv("proteome/similarity/true_semantics.csv", stringsAsFactors = F, header = T)
# pred = read.csv("proteome/similarity/seq2vec_TFIDF_CCR.csv", stringsAsFactors = F, header = T)

# parallel computing
cl <- makeCluster(detectCores())
registerDoParallel(cl)
registerDoMC(detectCores())

### INPUT ###
# overall input
syntax = read.csv(snakemake@input[["true_syntax"]], stringsAsFactors = F, header = T)
semantics_MF = read.csv(snakemake@input[["true_semantics_MF"]], stringsAsFactors = F, header = T)
semantics_BP = read.csv(snakemake@input[["true_semantics_BP"]], stringsAsFactors = F, header = T)
semantics_CC = read.csv(snakemake@input[["true_semantics_CC"]], stringsAsFactors = F, header = T)

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

# viz
spectral <- brewer.pal(11, "Spectral")
spectralRamp <- colorRampPalette(spectral)
spectral5000 <- spectralRamp(5000)

# plotting
plotting = function(tbl = "", file = ""){

  scale = seq(from = 1, to = nrow(tbl), by = round(nrow(tbl)*0.1, 0))

  png(filename = file,
      height = 2000, width = 2000, res = 300)

  print(levelplot(tbl,
                  pretty = T,
                  col.regions = spectral5000,
                  main = "squared difference between predicted and true similarity",
                  xlab = "transcripts",
                  ylab = "transcripts",
                  cex.lab = 0.1,
                  scales=list(x= scale, y= scale)))

  dev.off()
}


# calculate scores
compare = function(true = "", predicted = "", prot_pred = "", prot_true = "",
                   plot_file = "", out_file = ""){
  # same proteins
  if(!dim(true)[1] == dim(predicted)[1]){
    k = which(prot_pred %in% prot_true)
    predicted = predicted[, k]
    predicted = predicted[k, ]
  }

  # z-transformation
  predicted = (predicted - mean(predicted)) / sd(predicted)
  true = (true - mean(true)) / sd(true)

  # cumulated fraction of variance explained by PC1 and PC2 in predicted matrix
  PC = summary(prcomp(predicted, center = F, scale. = F))$importance[3,2]

  # score: squared difference between true and random, weighted by amount of variance explained
  tbl = ((predicted - true)^2) / PC
  tbl = (tbl - mean(tbl)) / sd(tbl)
  tbl = abs(tbl)

  # mean of difference between matrices divided by variance explained by PC1
  score = mean(tbl)
  SD = sd(tbl)

  # output
  write.csv(tbl, file = out_file, row.names = F)
  plotting(tbl = tbl/PC, file = plot_file)

  return(c(score, SD))
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
  syn = compare(true = syntax, predicted = pred, prot_pred = prot_pred, prot_true = prot_syn,
                plot_file = unlist(snakemake@output[["syntax_heatmap"]][i]),
                out_file = unlist(snakemake@output[["syntax_diff"]][i]))

  sem_MF = compare(true = semantics_MF, predicted = pred, prot_pred = prot_pred, prot_true = prot_sem_MF,
                plot_file = unlist(snakemake@output[["semantics_heatmap_MF"]][i]),
                out_file = unlist(snakemake@output[["semantics_diff_MF"]][i]))

  sem_BP = compare(true = semantics_BP, predicted = pred, prot_pred = prot_pred, prot_true = prot_sem_BP,
                   plot_file = unlist(snakemake@output[["semantics_heatmap_BP"]][i]),
                   out_file = unlist(snakemake@output[["semantics_diff_BP"]][i]))

  sem_CC = compare(true = semantics_CC, predicted = pred, prot_pred = prot_pred, prot_true = prot_sem_CC,
                   plot_file = unlist(snakemake@output[["semantics_heatmap_CC"]][i]),
                   out_file = unlist(snakemake@output[["semantics_diff_CC"]][i]))

  scores = data.frame(syntax = syn,
                      semantics_MF = sem_MF,
                      semantics_BP = sem_BP,
                      semantics_CC = sem_CC)

  ### OUTPUT ###
  write.table(x = scores, file = unlist(snakemake@output[["scores"]][i]), sep = " ", row.names = F)

  print(paste0("done with ", snakemake@input[["predicted"]][i]))
}
