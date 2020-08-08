### HEADER ###
# HOTSPOT REGIONS
# description: load similarity results, convert them into a prediction and evaluate performance
# input: testing data set with distance scores
# output: performance metrics
# author: HR

library(dplyr)
library(tidymodels)
library(DescTools)

n = 500
th = 1.36 / sqrt(n)


### INPUT ###
# see respective scripts

### MAIN PART ###

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
  
  th_range = c(-Inf, seq(min(df$score), max(df$score), length.out = 300), Inf)
  
  df$pred_label = NULL
  df = df %>% mutate(label = ifelse(label == "hotspot", 1, 0))
  
  sens = rep(NA, length(th_range))
  spec = rep(NA, length(th_range))
  
  prec = rep(NA, length(th_range))
  rec = rep(NA, length(th_range))
  
  for (t in seq_along(th_range)) {
    
      cnt_dat = df %>% mutate(pred_label = ifelse(score > th_range[t], 1, 0))
      
      # P, N, TP, TN, FP
      P = cnt_dat[cnt_dat$label == 1, ] %>% nrow()
      N = cnt_dat[cnt_dat$label == 0, ] %>% nrow()
      
      TP = cnt_dat[cnt_dat$pred_label == 1 & cnt_dat$label == 1, ] %>% nrow()
      TN = cnt_dat[cnt_dat$pred_label == 0 & cnt_dat$label == 0, ] %>% nrow()
      
      FP = cnt_dat[cnt_dat$pred_label == 1 & cnt_dat$label == 0, ] %>% nrow()
      
      
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


MakePred = function(pred = "", nm = "") {
  
  print(toupper(nm))
  
  dat = pred %>%
    mutate(pred_label = if_else(score > th, "hotspot", "non_hotspot")) %>%
    mutate_at(vars(label, pred_label), list(. %>% factor() %>% forcats::fct_relevel("hotspot")))
  
  curve = roc.pr.CURVE(dat)
  
  # tidyr approach
  # sensitivity
  dat %>%
    sens(label, pred_label)
  
  # AUC
  pr.na = which(! is.na(curve$precision | curve$recall))
  pr.auc = AUC(curve$recall[pr.na],
               curve$precision[pr.na])
  
  roc.na = which(! is.na(curve$sensitivity | curve$specificity))
  roc.auc = AUC(curve$specificity[pr.na],
               curve$sensitivity[pr.na])
  
  # plot curves
  
  title = paste0(nm)
  
  theme_set(theme_bw())
  
  roc.curve = curve %>%
    ggplot() +
    geom_path(aes(1 - specificity, sensitivity)) + 
    geom_abline(intercept = 0, slope = 1, linetype = "dotted") +
    xlim(c(0,1)) +
    ylim(c(0,1)) +
    ggtitle(title,
            subtitle = paste0("AUC: ", roc.auc %>% round(4)))
  
  roc.curve
  
  pr.curve = curve %>%
    ggplot() +
    geom_path(aes(recall, precision)) + 
    xlim(c(0,1)) +
    ylim(c(0,1)) +
    ggtitle(title,
            subtitle = paste0("AUC: ", pr.auc %>% round(4), " - inverted predictor"))
  
  pr.curve
  
  if(! dir.exists("hotspot_similarity")){
    dir.create("hotspot_similarity")
  }
  
  nm = gsub("[[:punct:]]+",'_', nm)
  
  ggsave(paste0("results/classifier/hotspot_similarity/ROC_", nm, ".png"), plot = roc.curve, device = "png", dpi = "retina")
  ggsave(paste0("results/classifier/hotspot_similarity/PR_", nm, ".png"), plot = pr.curve, device = "png", dpi = "retina")
  
  
}


### OUTPUT ###
# see respective scripts

