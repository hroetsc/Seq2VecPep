### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  plot model metrics
# input:        model metrics (loss and accuracy for training and validation data set)
# output:       accuracy and loss plots
# author:       HR

# tmp!!!
# setwd("/home/hanna/Documents/QuantSysBios/ProtTransEmbedding/Snakemake/")
# metrics = read.table("results/embedded_proteome/model_metrics.txt", stringsAsFactors = F)

library(stringr)
library(tibble)
library(ggplot2)

### INPUT ###
metrics = read.table(snakemake@input[["metrics"]], stringsAsFactors = F)

### MAIN PART ###
# clean input table
metrics = t(metrics)
colnames(metrics) = metrics[1,]
metrics = metrics[-1,]

for (i in 1:ncol(metrics)) {
  metrics[1,i] = str_split(metrics[1,i], coll("["), simplify = T)[,2]
  metrics[nrow(metrics),i] = str_split(metrics[nrow(metrics),i], coll("]"), simplify = T)[,1]
  metrics[c(2:nrow(metrics)-1),i] = str_split(metrics[c(2:nrow(metrics)-1),i], coll(","), simplify = T)[,1]
}

metrics = as.data.frame(metrics)
epochs = as.numeric(seq(1, nrow(metrics)))

# transform data
acc = as.numeric(levels(metrics$accuracy))[metrics$accuracy]
val_acc = as.numeric(levels(metrics$val_accuracy))[metrics$val_accuracy]

loss = as.numeric(levels(metrics$loss))[metrics$loss]
val_loss = as.numeric(levels(metrics$val_loss))[metrics$val_loss]

# plot accuracy
accuracy = data.frame(val = c(acc, val_acc),
                      label = c(rep("training", nrow(metrics)), rep("validation", nrow(metrics))),
                      epoch = c(epochs, epochs))
a = ggplot(accuracy, aes(x = epoch, y = val, group = label)) +
  geom_line(aes(linetype = label, color = label)) +
  geom_point(aes(color = label))+
  ggtitle('accuracy of skip-gram NN') +
  ylab('accuracy') +
  ylim(c(0,1)) +
  theme_minimal()
a


# plot loss
loss = data.frame(val = c(loss, val_loss),
                      label = c(rep("training", nrow(metrics)), rep("validation", nrow(metrics))),
                      epoch = c(epochs, epochs))
l = ggplot(loss, aes(x = epoch, y = val, group = label)) +
  geom_line(aes(linetype = label, color = label)) +
  geom_point(aes(color = label))+
  ggtitle('loss of skip-gram NN') +
  ylab('loss') +
  #ylim(c(0,1)) +
  theme_minimal()
l

### OUTPUT ###
ggsave(filename = unlist(snakemake@output[["acc"]]), plot = a, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["loss"]]), plot = l, device = "png", dpi = "retina")
