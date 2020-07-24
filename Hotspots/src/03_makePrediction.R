### HEADER ###
# HOTSPOT REGIONS
# description: load similarity results, convert them into a prediction and evaluate performance
# input: testing data set with distance scores
# output: performance metrics
# author: HR

library(dplyr)
library(tidymodels)


### INPUT ###
pred = read.csv("data/classifier/testing_PREDICTION.csv",
                stringsAsFactors = F)

n = 500

### MAIN PART ###

th = 1.36 / sqrt(n)

pred$score[which(pred$label == "hotspot")] %>% summary()
pred$score[which(pred$label == "non_hotspot")] %>% summary()

# th = 0.045

# make prediction
# is the distance to a hotspot or to a non-hotspot smaller?

# tidyr approach
theme_set(theme_bw())

dat = pred %>%
  mutate(pred_class = if_else(score >= th, "hotspot", "non_hotspot")) %>%
  mutate_at(vars(label, pred_class), list(. %>% factor() %>% forcats::fct_relevel("hotspot")))

# sensitivity
dat %>%
  sens(label, pred_class)

# ROC
roc_dat = dat %>%
  roc_curve(label, score)
roc_dat

# PR
pr_dat = dat %>%
  pr_curve(label, score)

pr_dat


# AUC
roc.auc = dat %>%
  roc_auc(label, score)

pr.auc = dat %>%
  pr_auc(label, score)


# plot curves

title = "1k sampled regions, extended substring,\nseq2vec + TFIDF + CCR"

roc.curve = roc_dat %>%
  arrange(.threshold) %>%
  ggplot() +
  geom_path(aes(1 - specificity, sensitivity)) + 
  geom_abline(intercept = 0, slope = 1, linetype = "dotted") +
  coord_equal() +
  ggtitle(title,
          subtitle = paste0("AUC: ", roc.auc$.estimate %>% round(4)))

roc.curve

pr.curve = pr_dat %>%
  arrange(.threshold) %>%
  ggplot() +
  geom_path(aes(recall, precision)) +
  coord_equal() +
  ggtitle(title,
          subtitle = paste0("AUC: ", pr.auc$.estimate %>% round(4)))

pr.curve

### OUTPUT ###

if(! dir.exists("hotspot_similarity")){
  dir.create("hotspot_similarity")
}

ggsave("hotspot_similarity/ROC.png", plot = roc.curve, device = "png", dpi = "retina")
ggsave("hotspot_similarity/PR.png", plot = pr.curve, device = "png", dpi = "retina")


