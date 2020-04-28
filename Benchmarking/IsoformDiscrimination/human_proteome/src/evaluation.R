### HEADER ###
# ISOFORM DISCRIMINATION ABILITY
# description:  evaluate ability to discriminate between isoforms with different window/embedding sizes
# input:        sequence embeddings, similarity matrix
# output:       nice plots
# author:       HR

library(stringr)
library(reshape2)
library(pheatmap)
library(lattice)
library(RColorBrewer)
library(grid)
library(gridExtra)
library(grDevices)

### INPUT ###
# get all files in this directory
fs = list.files(path = "results", pattern = "similarity_", full.names = T)

# true similarity
true = read.csv("data/true_syntax.csv", stringsAsFactors = F)

### MAIN PART ###
#---- true and predicted similarity ----
# mean similarity between isoform vectors
scores = matrix(ncol = length(fs), nrow = 51)
scores[1,] = c(fs)

for (f in 1:length(fs)){
  m = read.csv(file = fs[f], stringsAsFactors = F)
  m = as.matrix(m)
  
  if("gene" %in% colnames(m)){
    col = which(colnames(m) == "gene")
    acc = unique(m[,col])
    for (i in 1:length(acc)){
      k = which(str_split_fixed(m[,col], coll("-"), Inf)[,1] == acc[i])
      
      tmp = m[k,-c(1:col)]
      tmp = tmp[,k] %>% as.matrix() %>% as.numeric()
      
      # z-transform
      #tmp = (tmp - mean(tmp)) / sd(tmp)
      
      scores[c(1+i),f] = mean(tmp)
      
    }
    rownames(scores) = c("hyperparams", acc)
    
  } else {
    col = which(colnames(m) == "Accession")
    acc = unique(str_split_fixed(m[,col], coll("-"), Inf)[,1])
    for (i in 1:length(acc)){
      k = which(str_split_fixed(m[,col], coll("-"), Inf)[,1] == acc[i])
      
      tmp = m[k,-c(1:col)]
      tmp = tmp[,k] %>% as.matrix() %>% as.numeric()
      
      # z-transform
      #tmp = (tmp - mean(tmp)) / sd(tmp)
      
      scores[c(1+i),f] = mean(tmp)
      
    }
    rownames(scores) = c("hyperparams", acc)
  }
  
}


# true similarity as reference
ref = matrix(ncol = 1, nrow = 50)

# for transcriptome, merge true with sequences master table!
# for proteome, just take accession
col = which(colnames(true) == "Accession")
acc = unique(str_split_fixed(true[,col], coll("-"), Inf)[,1])

rownames(ref) = acc

for (i in 1:length(acc)){
  k = which(str_split_fixed(true[,col], coll("-"), Inf)[,1] == acc[i])
  
  tmp = true[k,-c(1:col)]
  tmp = tmp[,k] %>% as.matrix() %>% as.numeric()
  
  # z-transform
  #tmp = (tmp - mean(tmp)) / sd(tmp)
  
  ref[i,1] = mean(tmp)
  
}

# extract hyperparameters
hyp =  str_split_fixed(scores[1,],pattern = coll("/"), Inf)[,-1]

keep = str_split_fixed(hyp, coll("_"), Inf)[,3]

window = str_split_fixed(hyp, coll("_"), Inf)[,4]
window = str_split_fixed(window, pattern = coll("."), n = Inf)[,1]

# build matrix
scores = scores[-1,] %>% as.data.frame()
SD = rep(NA, length(keep))

scores.true = scores

for (s in 1:ncol(scores)){
  scores[,s] = as.numeric(as.character(scores[,s]))
  SD[s] = sd(as.numeric(as.character(scores[,s])))
  
  scores.true[,s] = as.numeric(as.character(scores.true[,s]))
  scores.true[,s] = scores.true[,s] - ref[,1]
}


transform = function(data = ""){
  pred = cbind(keep, window, colMeans(data)) %>% as.data.frame()
  names(pred) = c("keep", "window", "mean_isoform_similarity")
  
  # transform data to matrix
  pred = dcast(data = pred, formula = keep ~ window)
  rownames(pred) = as.numeric(as.character(pred[, "keep"])) / 10
  pred[, "keep"] = NULL
  
  colnames(pred) = as.numeric(colnames(pred))
  pred = pred[,paste(seq(5,15))]
  
  pred = pred[paste(seq(0.7,1,0.1)),]
  
  pred = as.matrix(pred)
  
  
  for (p in 1:ncol(pred)){
    pred[,p] = as.numeric(as.character(pred[,p]))
  }
  
  pred = pred[,-c(14,15)]
  return(pred)
}


pred = transform(data = scores)
pred.true = transform(data = scores.true)

# plot heatmaps
spectral <- brewer.pal(9, "Blues")
spectralRamp <- colorRampPalette(spectral)
spectral5000 <- spectralRamp(5000)

postscript("plots/heatmap_pred.ps")
levelplot(pred, col.regions = spectral5000,
          xlab = "downsampling size (fraction of kept skip-grams)",
          ylab = "window size",
          main = "mean predicted sequence similarity")
dev.off()

postscript("plots/heatmap_pred-true.ps")
levelplot(pred.true, col.regions = spectral5000,
          xlab = "downsampling size (fraction of kept skip-grams)",
          ylab = "window size",
          main = "mean predicted - true sequence similarity")
dev.off()

# correlation between true and predicted
#x11()
Corr = data.frame(keep = keep,
                  window = window,
                  spearman_corr = rep(NA, length(keep)),
                  slope = rep(NA, length(keep)),
                  y_intercept = rep(NA, length(keep)))
for (s in 1:ncol(scores)) {
  
  R = cor(scores[,s], ref, method = "spearman")
  S = lm(scores[,s] ~ ref)$coefficients[2]
  Y = lm(scores[,s] ~ ref)$coefficients[1]
  
  Corr[s, "spearman_corr"] = R
  Corr[s, "slope"] = S
  Corr[s, "y_intercept"] = Y
  
  # postscript(paste0("plots/corr_true_pred_", keep[s], "_", window[s], ".ps"))
  # plot(scores[,s] ~ ref,
  #      xlab = "true isoform similarity (sequence alignment)",
  #      ylab = "predicted isoform similarity (dot product)",
  #      xlim = c(0,1),
  #      ylim = c(0,1),
  #      main = paste0("keep: ", as.numeric(keep[s])/10 , ", window size: ", window[s]),
  #      sub = paste0("Spearman correlation: ", round(R,4), ", slope: ", round(S, 4)))
  # dev.off()
  
}

# take slope
{
c = dcast(data = Corr[,c("keep", "window", "slope")], formula = keep ~ window)
rownames(c) = as.numeric(as.character(c[, "keep"])) / 10
c[, "keep"] = NULL

colnames(c) = as.numeric(colnames(c))
c = c[,paste(seq(5,15))]

c = c[paste(seq(0.7,1,0.1)),]

c = as.matrix(c)


for (p in 1:ncol(c)){
  c[,p] = as.numeric(as.character(c[,p]))
}

c = c[,-c(14,15)]

postscript("plots/corr_pred_true.ps")
levelplot(c, col.regions = spectral5000,
          xlab = "downsampling size (fraction of kept skip-grams)",
          ylab = "window size",
          main = "slope between true and predicted isoform similarity")
dev.off()
}

#---- relation between accuracy and isoform discrimination ---

accur = list.files("results", pattern = "model_metrics_", full.names = T)
accuracy = rep(NA, length(accur))

for (a in 1:length(accur)){
  metrics = read.table(accur[a], stringsAsFactors = F)
  metrics = t(metrics)
  colnames(metrics) = metrics[1,]
  metrics = metrics[-1,]
  
  for (i in 1:ncol(metrics)) {
    metrics[1,i] = str_split(metrics[1,i], coll("["), simplify = T)[,2]
    metrics[nrow(metrics),i] = str_split(metrics[nrow(metrics),i], coll("]"), simplify = T)[,1]
    metrics[c(2:nrow(metrics)-1),i] = str_split(metrics[c(2:nrow(metrics)-1),i], coll(","), simplify = T)[,1]
  }
  
  accuracy[a] = max(metrics[,"val_accuracy"])
  
}

hyp =  str_split_fixed(accur,pattern = coll("/"), Inf)[,-1]

keep = str_split_fixed(hyp, coll("_"), Inf)[,4]

window = str_split_fixed(hyp, coll("_"), Inf)[,5]
window = str_split_fixed(window, pattern = coll("."), n = Inf)[,1]

accuracy = cbind(keep, window, accuracy) %>% as.data.frame()
colnames(accuracy) = c("keep", "window", "value")

{
  c = dcast(data = accuracy, formula = keep ~ window)
  rownames(c) = as.numeric(as.character(c[, "keep"])) / 10
  c[, "keep"] = NULL
  
  colnames(c) = as.numeric(colnames(c))
  c = c[,paste(seq(5,15))]
  
  c = c[paste(seq(0.7,1,0.1)),]
  
  c = as.matrix(c)
  
  
  for (p in 1:ncol(c)){
    c[,p] = as.numeric(as.character(c[,p]))
  }
  
  c = c[,-c(14,15)]
  
  postscript("plots/max_val_acc.ps")
  levelplot(c, col.regions = spectral5000,
            xlab = "downsampling size (fraction of kept skip-grams)",
            ylab = "window size",
            main = "maximum validation accuracy in model training")
  dev.off()
}


### OUTPUT ###
