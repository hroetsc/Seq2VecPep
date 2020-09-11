### HEADER ###
# HOTSPOT PREDICTION
# description: assemble token scores to sequence and compare true/predicted count distributions
# input:  predicted counts, tokenized proteome
# output: sequences with counts
# author: HR


library(dplyr)
library(stringr)
library(zoo)


### INPUT ###
prediction = read.csv("results/model_predictions.csv", stringsAsFactors = F)
# tmp
# prediction$pred_count = prediction$pred_label*prediction$pred_count

prots = read.csv("../../files/proteome_human.csv", stringsAsFactors = F, header = T)
words = read.csv("../../RUNS/HumanProteome/v_50k/words_hp_v50k.csv", stringsAsFactors = F, header = T)


### MAIN PART ###
# keep only proteins with predictions
prots = prots[which(prots$Accession %in% prediction$Accession), ]
prots = left_join(prots, words)


# get counts for every aa position from token counts
countsTrue = list()
countsPred = list()
countsPred.roll = list()

pb = txtProgressBar(min = 0, max = nrow(prots), style = 3)

for (i in 1:nrow(prots)){
  
  setTxtProgressBar(pb, i)
  
  cnt.Data = prediction[prediction$Accession == prots$Accession[i], ]
  cnt.Data = left_join(cnt.Data, prots[i, ])
  
  cnt.True = rep(NA, nchar(prots$seqs[i]))
  cnt.Pred = rep(NA, nchar(prots$seqs[i]))

  
  for (j in 1:nrow(cnt.Data)) {
    
    substr = str_replace_all(cnt.Data$window[j], coll(" "), coll(""))
    idx = str_locate(prots$seqs[i], substr) %>% as.numeric()
    
    cnt.True[idx[1]:idx[2]] = rep(cnt.Data$count[j], idx[2]-idx[1]+1)
    cnt.Pred[idx[1]:idx[2]] = rep(cnt.Data$pred_count[j], idx[2]-idx[1]+1)
    
  }
  
  countsTrue[[i]] = cnt.True
  countsPred[[i]] = cnt.Pred
  countsPred.roll[[i]] = rollmean(cnt.Pred, k = 9)
  
  names(countsTrue)[i] = prots$Accession[i]
  names(countsPred)[i] = prots$Accession[i]
  names(countsPred.roll[[i]]) = prots$Accession[i]
  
}


# plot

pdf(paste0("results/",JOBID,"_Counts_trueVSpred.pdf"), width = 12, height = 8)
par(mfrow = c(2,2))

for (k in 1:length(countsTrue)) {
  
  y_true = countsTrue[[k]]
  y_pred = countsPred[[k]]
  y_pred.roll = countsPred.roll[[k]]
  
  x = seq(length(y_true))
  
  plot(x, y_true, type = "l", col="red",
       xlab = prots$Accession[k], axes = F, ylab = "counts")
  points(x, y_pred, type = "l", col="blue")
  points(seq(1, length(y_pred.roll), ceiling(length(y_pred.roll)/length(y_pred))),
         y_pred.roll, type = "l", col="black")
  axis(1)
  axis(2)
  
}

dev.off()



