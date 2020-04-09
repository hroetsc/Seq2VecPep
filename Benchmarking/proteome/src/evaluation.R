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
# semantics = read.csv("proteome/similarity/true_semantics.csv", stringsAsFactors = F, header = T)
# pred = read.csv("proteome/similarity/random.csv", stringsAsFactors = F, header = T)

# parallel computing
cl <- makeCluster(detectCores())
registerDoParallel(cl)
registerDoMC(detectCores())

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
                  main = "predicted protein similarity",
                  xlab = "proteins",
                  ylab = "proteins",
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
  
  # plot heatmap
  plotting(tbl = predicted, file = plot_file)
  
  # z-transformation
  predicted = (predicted - mean(predicted)) / sd(predicted)
  true = (true - mean(true)) / sd(true)
  
  tbl = abs(predicted - true)
  
  write.csv(predicted, file = out_file, row.names = F)

  # mean of difference between matrices
  score = mean(tbl)
  SD = sd(tbl)

  return(c(score, SD))
}

# actually start computing

input = snakemake@input[["predicted"]]

foreach(i = 1:length(input)) %dopar% {
  
  print(snakemake@input[["predicted"]][i])
  
  ### INPUT ###
  syntax = read.csv(snakemake@input[["true_syntax"]], stringsAsFactors = F, header = T)
  semantics = read.csv(snakemake@input[["true_semantics"]], stringsAsFactors = F, header = T)
  pred = read.csv(snakemake@input[["predicted"]][i], stringsAsFactors = F, header = T)
  
  ### MAIN PART ###
  # preprocessing
  prot_syn = prots(tbl = syntax)
  prot_sem = prots(tbl = semantics)
  prot_pred = prots(tbl = pred)
  
  syntax = cleaning(tbl = syntax)
  semantics = cleaning(tbl = semantics)
  pred = cleaning(tbl = pred)
  
  # calculate scores
  syn = compare(true = syntax, predicted = pred, prot_pred = prot_pred, prot_true = prot_syn,
                plot_file = unlist(snakemake@output[["syntax_heatmap"]][i]),
                out_file = unlist(snakemake@output[["syntax_diff"]][i]))
  sem = compare(true = semantics, predicted = pred, prot_pred = prot_pred, prot_true = prot_sem,
                plot_file = unlist(snakemake@output[["semantics_heatmap"]][i]),
                out_file = unlist(snakemake@output[["semantics_diff"]][i]))
  
  scores = data.frame(syntax = syn,
                      semantics = sem)
  
  ### OUTPUT ###
  write.table(x = scores, file = unlist(snakemake@output[["scores"]][i]), sep = " ", row.names = F)
  
  print(paste0("done with ", snakemake@input[["predicted"]][i]))
}

