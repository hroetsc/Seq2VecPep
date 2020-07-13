### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  plot model metrics
# input:        model metrics (loss and accuracy for training and validation data set)
# output:       accuracy and loss plots
# author:       HR

library(stringr)
library(tibble)
library(ggplot2)
library(dplyr)

### INPUT ###
metrics = read.table(snakemake@input[["metrics"]], stringsAsFactors = F)

# tmp!
metrics = read.table("hp_model_metrics_w5_d100.txt", sep = ",", stringsAsFactors = F)

### MAIN PART ###
# clean input table
metrics$V2 = str_split_fixed(metrics$V2, coll("["), Inf)[,2]
metrics$V2 = str_split_fixed(metrics$V2, coll("]"), Inf)[,1]

var = metrics$V1
val = str_split_fixed(metrics$V2, coll(","), Inf) %>% as.data.frame()
metrics = cbind(var, val)

metrics = t(metrics) %>% as.data.frame()
metrics = metrics[-1,]

epochs = as.numeric(seq(1, nrow(metrics)))

rownames(metrics) = epochs
colnames(metrics) = var

# convert factors into numeric
for (c in 1:ncol(metrics)){
  metrics[,c] = as.numeric(as.character(metrics[,c]))
}

# plotting function
plotting = function(col1 = "", col2 = "", name = "", path = "results/model_metrics_"){
  
  if (max(col1, col2) <=1){
    upper = 1
  } else {
    upper = max(col1, col2)
  }
  
  png(filename = paste0(path, name, ".png"),
      width = 2000, height = 2000, res = 300)
  
  plot(col1 ~ epochs,
       main = paste0(name, " for train and test data set"),
       pch = 20,
       cex = 0.8,
       col = "blue",
       ylim = c(0, upper),
       xlab = "epoch",
       ylab = name)
  points(col2 ~ epochs,
         pch = 1,
         cex = 1,
         col = "green")
  legend("bottomright", cex = 1,
         legend = c("train", "validation"),
         col = c("blue", "green"),
         pch = c(20, 1),
         box.lty = 1,
         pt.cex = 1)
  
  dev.off()
}


### OUTPUT ###
# plot and save

for (i in 1:(ncol(metrics)/2)){
  plotting(col1 = metrics[,i],
           col2 = metrics[, (i + (ncol(metrics)/2))],
           name = colnames(metrics)[i],
           path = "../metrics/model_metrics_hp_w5_d100_") # remove path in future!!
}

