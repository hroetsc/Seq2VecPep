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
library(tidymodels)
library(DescTools)


JOBID = "5205164-12"
no_ranks = 1

### INPUT ###
# download results
system("scp -rp hroetsc@transfer.gwdg.de:/usr/users/hroetsc/Hotspots/results/model_metrics.txt results/")
system("scp -rp hroetsc@transfer.gwdg.de:/usr/users/hroetsc/Hotspots/results/model_predictions.csv results/")

# open them
metrics = read.table("results/model_metrics.txt", sep = ",", stringsAsFactors = F)
prediction = read.csv("results/model_predictions.csv", stringsAsFactors = F)

# prediction.fs = list.files("results", pattern = "model_predictions_rank",
#                            full.names = T, recursive = T)
# 
# for (p in 0:(no_ranks-1)){
#   if (p == 0){
#     prediction = read.csv(paste0("results/model_predictions_rank", p, ".csv"), stringsAsFactors = F)
#   } else {
#     
#     prediction = rbind(prediction,
#                       read.csv(paste0("results/model_predictions_rank", p, ".csv"), stringsAsFactors = F))
#     
#   }
# }


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
plotting = function(col1 = "", col2 = "", name = "", path = paste0("results/plots/", JOBID, "_model_metrics_")){
  
  out.path = str_split(path, coll("/"), simplify = T)[,1]
  
  if(! dir.exists(out.path)){
    dir.create(out.path)
  }
  
  
  if (min(col1, col2) < 0){
    col1 = log2(col1 + abs(lower)+1)
    col2 = log2(col2 + abs(lower)+1)
    
  } else {
    col1 = log2(col1)
    col2 = log2(col2)
  }
  
  lower = min(col1, col2) - 1
  upper = max(col1, col2) + 1
  
  
  png(filename = paste0(path, name, ".png"),
      width = 2000, height = 2000, res = 300)
  
  plot(col1 ~ epochs,
       main = paste0(str_replace_all(name, "_", " "), " for training and validation data set"),
       pch = 20,
       cex = 0.8,
       col = "darkblue",
       ylim = c(lower, upper),
       xlab = "epoch",
       ylab = paste0('log2 ', str_replace_all(name, "_", " ")))
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

# generalization loss
gl = 100 * ((metrics$val_loss/min(metrics$val_loss)) - 1)

png(filename = paste0("results/plots/", JOBID, "_generalisation.png"),
    width = 2000, height = 2000, res = 300)
plot(log2(gl), col='seagreen',
     ylab = 'log2 generalisation loss',
     xlab = 'epoch',
     main = 'generalisation loss during training')
dev.off()


########## regression ##########
# prediction$count = log(prediction$count + 1)
# prediction$count = prediction$count + 0.65
# prediction = prediction[-which(prediction$count < 1), ]

# tmp !!
# prediction$count = 2^(prediction$count) - 1
# prediction$pred_count = 2^(prediction$pred_count) - 1

# general
summary(prediction$count)
summary(prediction$pred_count)


prediction$Accession = NULL
prediction$window = NULL


# linear model --> R^2 and adjusted R^2
pred.lm = lm(pred_count ~ count, data = prediction)
summary(pred.lm)


# correlation coefficients
pc = cor(prediction$count, prediction$pred_count, method = "pearson")
sm = cor(prediction$count, prediction$pred_count, method = "spearman")

# mean squared error
mse = (prediction$count - prediction$pred_count)^2 %>% mean() %>% round(4)
# root mean squared error
rmse = sqrt(mse) %>% round(4)
# mean absolute deviation
mae = (prediction$count - prediction$pred_count) %>% abs() %>% mean() %>% round(4)

# sumarise
all.metrics = c(JOBID, summary(pred.lm)$r.squared, pc, mse, rmse, mae)
names(all.metrics) = c("JOBID", "Rsquared", "PCC", "MSE", "RMSE", "MAE")
all.metrics

start = min(prediction) - 0.1
stop = max(prediction) + 0.1


# visual
prediction[, c("count", "pred_count")] %>% gather() %>%
  ggplot(aes(x = value, color = key)) +
  geom_density() +
  ggtitle("true and predicted hotspot counts") +
  theme_bw()
ggsave(paste0("results/plots/", JOBID, "_trueVSpredicted-dens.png"), plot = last_plot(),
       device = "png", dpi = "retina")

ggplot(prediction, aes(x = count, y = pred_count)) +
  geom_point(alpha = 0.3, size = 0.1) +
  xlim(c(start, stop)) +
  ylim(c(start, stop)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dotted") +
  coord_equal() +
  ggtitle("true and predicted hotspot counts",
          subtitle = paste0("PCC: ", pc %>% round(4), ", R^2: ", summary(pred.lm)$r.squared %>% round(4))) +
  theme_bw()
ggsave(paste0("results/plots/", JOBID, "_trueVSpredicted-scatter.png"), plot = last_plot(),
       device = "png", dpi = "retina")




### OUTPUT ###
out = "results/performance.csv"
if(file.exists(out)) {
  write(all.metrics, file = out, ncolumns = length(all.metrics),
        append = T, sep = ",")
  
} else {
  
  write(all.metrics, file = out, ncolumns = length(all.metrics),
        append = F, sep = ",")
  
}



########## binary classification ##########
PRECISION = function(TP = "", FP = "") {
  return(as.numeric(TP) / (as.numeric(TP) + as.numeric(FP)))
}

RECALL = function(TP = "", P = "") {
  return(as.numeric(TP) / as.numeric(P))
}

SENSITIVITY = function(TP = "", P = "") {
  return(as.numeric(TP) / as.numeric(P))
}

SPECIFICITY = function(TN = "", N = "") {
  return(as.numeric(TN) / as.numeric(N))
}


roc.pr.CURVE = function(df = "") {
  
  th_range = c(-Inf, seq(min(df$pred_label), max(df$pred_label), length.out = 300), Inf)
  
  df$pred_count = NULL
  
  
  sens = rep(NA, length(th_range))
  spec = rep(NA, length(th_range))
  
  prec = rep(NA, length(th_range))
  rec = rep(NA, length(th_range))
  
  for (t in seq_along(th_range)) {
    
    cnt_dat = df %>% mutate(pred = ifelse(pred_label > th_range[t], 1, 0))
    
    # P, N, TP, TN, FP
    P = cnt_dat[cnt_dat$label == 1, ] %>% nrow()
    N = cnt_dat[cnt_dat$label == 0, ] %>% nrow()
    
    TP = cnt_dat[cnt_dat$pred == 1 & cnt_dat$label == 1, ] %>% nrow()
    TN = cnt_dat[cnt_dat$pred == 0 & cnt_dat$label == 0, ] %>% nrow()
    
    FP = cnt_dat[cnt_dat$pred == 1 & cnt_dat$label == 0, ] %>% nrow()
    
    
    sens[t] = SENSITIVITY(TP, P)
    spec[t] = SPECIFICITY(TN, N)
    
    prec[t] = PRECISION(TP, FP)
    rec[t] = RECALL(TP, P)
  }
  
  curve = data.frame(score = th_range,
                     precision = prec,
                     recall = rec,
                     sensitivity = sens,
                     specificity = spec)
  
  return(curve)
}
curve = roc.pr.CURVE(df = prediction)

# AUC
pr.na = which(! is.na(curve$precision | curve$recall))
pr.auc = AUC(curve$recall[pr.na],
             curve$precision[pr.na])

roc.na = which(! is.na(curve$sensitivity | curve$specificity))
roc.auc = AUC(curve$specificity[pr.na],
              curve$sensitivity[pr.na])

theme_set(theme_bw())
roc.curve = curve %>%
  ggplot() +
  geom_path(aes(1 - specificity, sensitivity)) + 
  geom_abline(intercept = 0, slope = 1, linetype = "dotted") +
  xlim(c(0,1)) +
  ylim(c(0,1)) +
  ggtitle("ROC",
          subtitle = paste0("AUC: ", roc.auc %>% round(4)))

roc.curve

pr.curve = curve %>%
  ggplot() +
  geom_path(aes(recall, precision)) + 
  xlim(c(0,1)) +
  ylim(c(0,1)) +
  ggtitle("PR",
          subtitle = paste0("AUC: ", pr.auc %>% round(4)))

pr.curve

ggsave(paste0("results/plots/", JOBID, "_ROC.png"),
       plot = roc.curve, device = "png", dpi = "retina")
ggsave(paste0("results/plots/", JOBID, "_PR.png"),
       plot = pr.curve, device = "png", dpi = "retina")



########## estimating weight decay parameter ##########

ls = h5ls("results/model/best_model.h5")

weights = h5read("results/model/best_model.h5"
                 , "/model_weights/output/output")
summary(weights[["kernel:0"]] %>% as.numeric())

# A Simple Trick for Estimating the Weight Decay Parameter (Roegnvaldsson 2006
# weight decay for regression
reg_weights = h5read("results/model/best_model.h5",
                     "/model_weights/regression/regression")[[2]]

est_reg_lambda = abs(2 * reg_weights) %>% sum()

# weight decay for regression
class_weights = h5read("results/model/best_model.h5",
                     "/model_weights/classification/classification")[[2]]

est_class_lambda = abs(2 * class_weights) %>% sum()

