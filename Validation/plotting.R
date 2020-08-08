### visualize results of validation pipeline ###

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)


### INPUT ###
scores = read.csv("scores_hp_200807_dot.csv", stringsAsFactors = F)


### MAIN PART ###
# tmp: remove Bhattacharyya and Jensen-Shannon
scores$Jensen_Shannon_divergence = NULL
scores$Bhattacharyya = NULL


metrics = c("Wasserstein_metric", "euclidean",
            "KS_pvalue", "cosine")

# remove entries where equal metrics are compared
pairs = data.frame(p1 = c("syn", "sem_MF", "sem_BP", "sem_CC"),
                   p2 = c("true_syntax", "true_semantics_MF", "true_semantics_BP", "true_semantics_CC"))

for (p in 1:nrow(pairs)){
  scores[which(scores$ground_truth == pairs$p1[p] & scores$embedding == pairs$p2[p]), ] = NA
}
scores = na.omit(scores)

# subsets: scores calculated prior to and after z-transformation of data
scores.pre = scores[scores$state == "pre", ]
scores.post = scores[scores$state == "post", ]


# plots for different ontologies/ground truth data sets
# for different similarity metrics
# for different states


extract_metric = function(df = "", metric = ""){
  
  df = df[, c("embedding", metric)]
  names(df) = c("embedding", "metric")
  df$colour_group = str_split_fixed(df$embedding, "_", Inf)[,1]
  
  return(df)
}


# violin plots
violin = function(df = "", sim_metric = "", ground_truth = "", state = ""){
  
  set.seed(42)
  
  title = paste0("comparison of true and predicted sequence similarity",
                 "\n (based on cosine similarity between vectors)")
  
  df = extract_metric(df, sim_metric)
  
  p = ggplot(df, aes(factor(embedding), metric, fill = colour_group)) +
    geom_violin(scale = "width", trim = F,
                draw_quantiles = c(0.5)) +
    scale_fill_viridis_d(option = "inferno", direction = -1) +
    stat_summary(fun=mean, geom="point", size=1, color="red")
  
  p = p +
    ggtitle(title,
            subtitle = paste0(ground_truth, ", ", state, " z-transformation")) +
    ylab(sim_metric) +
    xlab("embedding") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90),
          legend.position = "none")
  
  # ggsave
  ggsave(filename = paste0("results/hp_dot_", state, "_", ground_truth, "_", sim_metric, ".png"),
         plot = p, device = "png", dpi = "retina",
         height = 3.93*2, width = 5.56*2)
  
}

### OUTPUT ###


# plots for different ontologies/ground truth data sets
# for different similarity metrics
# for different states

gt = scores$ground_truth %>% unique()

for (g in gt) {
  
  cnt_scores.pre = scores.pre[scores.pre$ground_truth == g, ]
  cnt_scores.post = scores.post[scores.post$ground_truth == g, ]
  
  for (m in metrics) {
    
    violin(df = cnt_scores.pre, sim_metric = m, ground_truth = g, state = "pre")
    violin(df = cnt_scores.post, sim_metric = m, ground_truth = g, state = "post")
    
    
  }
  
}




########## old ##########
merge_data = function(fs = ""){
  for (f in 1:length(fs)){
    
    if (f == 1){
      df = read.csv(fs[f], stringsAsFactors = F)
      
      
    } else {
      
      tmp = read.csv(fs[f], stringsAsFactors = F)
      
      tmp$iteration = rep(f, nrow(tmp)) # different assignement?
      
      df = rbind(df, tmp)
    }
    
  }
  
  df$embedding = str_replace_all(df$embedding, coll("_"), coll(" "))
  
  df.mse = df[, c("embedding","syntax_diff", "MF_semantics_diff", "BP_semantics_diff", "CC_semantics_diff")]
  df.spearman = df[, c("embedding","syntax_R2", "MF_semantics_R2", "BP_semantics_R2", "CC_semantics_R2")]
  
  colnames(df.mse) = c("embedding", "syntax", "semantics MF", "semantics BP", "semantics CC")
  colnames(df.spearman) = colnames(df.mse)
  
  # remove redundant entries
  pairs = cbind(c("syntax", "semantics MF", "semantics BP", "semantics CC"),
                c("true syntax", "true semantics MF", "true semantics BP", "true semantics CC"))
  
  for (p in 1:nrow(pairs)){
    df.mse[which(df.mse$embedding == pairs[p,2]), pairs[p, 1]] = NA
    df.spearman[which(df.spearman$embedding == pairs[p,2]), pairs[p, 1]] = NA
  }
  
  return(list(df.mse, df.spearman))
}

df_hp = merge_data(fs = fs_hp)

nrow(df_hp[[1]])/26

# df_hst = merge_data(fs = fs_hst)
# df_hm = merge_data(fs = fs_hm)


# deprecated
pick_seq2vec = function(df = "", nm = ""){
  keep = c("seq2vec", "seq2vec CCR", "seq2vec TFIDF", "seq2vec TFIDF CCR",
           "seq2vec SIF", "seq2vec SIF CCR")
  
  df = df[which(df$embedding %in% keep), ]
  
  df$embedding = paste(nm, df$embedding)
  
  return(df)
}


df_hp = pick_seq2vec(df_hp, "hp")
df_hst = pick_seq2vec(df_hst, "hst")
df_hm = pick_seq2vec(df_hm, "hm")

master = rbind(df_hp, df_hst, df_hm)


# plot it
for (c in 2:ncol(master)){
  
  p = ggplot(master, aes(factor(master$embedding), master[,c])) +
    geom_violin(scale = "width", trim = F,
                draw_quantiles = c(0.5),
                aes(fill = factor(embedding))) +
    geom_jitter(height = 0, width = 0.005) +
    stat_summary(fun=mean, geom="point", size=1, color="red")
  
  p = p +
    ggtitle("comparison of seq2vec performance on different data sets",
            subtitle = paste0(colnames(master)[c])) +
    ylab("mean squared error") +
    xlab("embedding") +
    #ylim(c(lower, upper)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90),
          legend.position = "none")
  
  
  p
  ggsave(plot = p, filename = paste0("results/seq2vec-perform_",
                                     str_replace_all(colnames(master)[c], coll(" "), coll("_")),
                                     ".png"), device = "png",
         dpi = "retina", height = 3.93*2, width = 5.56*2)
}




