### HEADER ###
# HOTSPOT PREDICTION
# description: download results from cluster, plot performance curves, evaluate prediction
# input: -
# output: evaluation results
# author: HR

library(dplyr)
library(stringr)
library(rhdf5)
library(ggplot2)
library(tidyr)


### INPUT ###
# download results
system("scp -rp hroetsc@transfer.gwdg.de:/usr/users/hroetsc/Hotspots/results/* results/")

# open them
metrics = read.table("results/model_metrics.txt", sep = ",", stringsAsFactors = F)
prediction = read.csv("results/model_predictions.csv", stringsAsFactors = F)


### MAIN PART ###

########## metrics ##########
# clean input table
{
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
}

# plotting function
plotting = function(col1 = "", col2 = "", name = "", path = "results/Conv1D_v2_model_metrics_"){
  
  out.path = str_split(path, coll("/"), simplify = T)[,1]
  
  if(! dir.exists(out.path)){
    dir.create(out.path)
  }
  
  if (max(col1, col2) <=1){
    upper = 1
  } else {
    upper = max(col1, col2)
  }
  
  png(filename = paste0(path, name, ".png"),
      width = 2000, height = 2000, res = 300)
  
  plot(col1 ~ epochs,
       main = paste0(str_replace_all(name, "_", " "), " for training and validation data set"),
       pch = 20,
       cex = 0.8,
       col = "darkblue",
       ylim = c(0, upper),
       xlab = "epoch",
       ylab = str_replace_all(name, "_", " "))
  points(col2 ~ epochs,
         pch = 1,
         cex = 0.8,
         col = "firebrick")
  legend("topright", cex = 1,
         legend = c("training", "validation"),
         col = c("darkblue", "firebrick"),
         pch = c(20, 1),
         box.lty = 1,
         pt.cex = 1)
  
  dev.off()
}

# plot and save
for (i in 1:(ncol(metrics)/2)){
  plotting(col1 = metrics[,i],
           col2 = metrics[, (i + (ncol(metrics)/2))],
           name = colnames(metrics)[i])
}


########## prediction ##########
# prediction$count = log(prediction$count + 1)
# prediction$count = prediction$count + 0.65
# prediction = prediction[-which(prediction$count < 1), ]

# general
summary(prediction$count)
summary(prediction$prediction)

# visual
prediction %>% gather() %>%
  ggplot(aes(x = value, color = key)) +
  geom_density() +
  ggtitle("true and predicted hotspot counts") +
  theme_bw()
ggsave("results/Conv1D_v2_trueVSpredicted-dens.png", plot = last_plot(),
       device = "png", dpi = "retina")

ggplot(prediction, aes(x = count, y = prediction)) +
  geom_point() +
  xlim(c(-0.2, 5.5)) +
  ylim(c(-0.2, 5.5)) +
  ggtitle("true and predicted hotspot counts") +
  theme_bw()
ggsave("results/Conv1D_v2_trueVSpredicted-scatter.png", plot = last_plot(),
       device = "png", dpi = "retina")


# linear model --> R^2 and adjusted R^2
pred.lm = lm(prediction ~ count, data = prediction)
summary(pred.lm)

# correlation coefficients
pc = cor(prediction$count, prediction$prediction, method = "pearson")
sm = cor(prediction$count, prediction$prediction, method = "spearman")

# mean squared error
mse = (prediction$count - prediction$prediction)^2 %>% mean()
# root mean squared error
rmse = sqrt(mse)
# mean absolute deviation
mae = (prediction$count - prediction$prediction) %>% abs() %>% mean()

# sum up
all.metrics = c(summary(pred.lm)$r.squared, pc, mse, rmse, mae) %>% round(4)
names(all.metrics) = c("Rsquared", "PCC", "MSE", "RMSE", "MAE")
all.metrics

# some ideas
plot(density(log10((prediction$count - prediction$prediction)^2)),
     main = "log10 squared error distribution")

plot(density(log10((prediction$count - prediction$prediction) %>% abs())),
     main = "log10 absolute error distribution")

# accuracy = bias
# precision = inverse of variance

########## actual model ##########

ls = h5ls("results/model/model.h5")
h5read("results/model/model.h5",
       "/optimizer_weights/Adam/output/kernel")
