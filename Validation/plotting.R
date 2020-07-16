### visualize results of validation pipeline ###

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggstatsplot)


### INPUT ###
# get all score files
# fs_hp = list.files(path = "./", pattern = "scores_hp_200715_dot", recursive = T, full.names = T)
fs_hp = list.files(path = "./", pattern = "scores_hp_200716_dot", recursive = T, full.names = T)

# fs_hst = list.files(path = "downloads", pattern = "scores_hst", recursive = T, full.names = T)
# fs_hm = list.files(path = "downloads", pattern = "scores_hm", recursive = T, full.names = T)


########## merge them in a single df ##########
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



### MAIN PART ###

# different plots for seq alinment and GO similarities
extract_scores = function(score = "", df = ""){
  return(df[, c("embedding", score)])
}




########## violin plots ##########
violin = function(score = "", df = ""){
  
  set.seed(42)
  
  title = paste0("comparison of true and predicted sequence similarity",
                 "\n (based on cosine similarity between vectors)")
  
  # mean squared error
  tbl.mse = extract_scores(score = score, as.data.frame(df[[1]])) %>%
    as.data.frame %>%
    na.omit()
  colnames(tbl.mse) = c("embedding", "metric")
  
  
  # old school ggplot
  p = ggplot(tbl.mse, aes(factor(embedding), metric)) +
    geom_violin(scale = "width", trim = F,
                draw_quantiles = c(0.5),
                aes(fill = factor(embedding))) +
    #geom_jitter(height = 0, width = 0.005) +
    stat_summary(fun=mean, geom="point", size=1, color="red")
  
  p = p +
    ggtitle(title,
            subtitle = paste0("human proteome: ", score)) +
    ylab("mean squared error") +
    xlab("embedding") +
    #ylim(c(lower, upper)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90),
          legend.position = "none")
    
  
  p
  ggsave(plot = p, filename = paste0("results/hp_", str_replace_all(score, coll(" "), coll("_")),
                                     "_mse.png"), device = "png",
         dpi = "retina", height = 3.93*2, width = 5.56*2)
  
  
  # spearman correlation
  tbl.spearman = extract_scores(score = score, df[[2]]) %>%
    as.data.frame %>%
    na.omit()
  colnames(tbl.spearman) = c("embedding", "metric")
  
  
  # old school ggplot
  p = ggplot(tbl.spearman, aes(factor(embedding), metric)) +
    geom_violin(scale = "width", trim = F,
                draw_quantiles = c(0.5),
                aes(fill = factor(embedding))) +
    #geom_jitter(height = 0, width = 0.005) +
    stat_summary(fun=mean, geom="point", size=1, color="red")
  
  p = p +
    ggtitle(title,
            subtitle = paste0("human proteome: ", score)) +
    ylab("Spearman coefficient") +
    xlab("embedding") +
    #ylim(c(lower, upper)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90),
          legend.position = "none")
  
  
  p
  ggsave(plot = p, filename = paste0("results/hp_", str_replace_all(score, coll(" "), coll("_")),
                                     "_spearman.png"), device = "png",
         dpi = "retina", height = 3.93*2, width = 5.56*2)
  {# ggstatsplot
  
  # gs = ggstatsplot::ggbetweenstats(data = tbl,
  #                              x = embedding,
  #                              y = metric)
  # gs = gs +
  #   geom_violin(scale = "width", trim = F,
  #               draw_quantiles = c(0.5)) +
  #   geom_jitter(height = 0, width = 0.01) +
  #   ggtitle("comparison of true and predicted sequence similarity (based on euclidean distance)") +
  #   ylab("mean squared error") +
  #   xlab("embedding") +
  #   #ylim(c(lower, upper)) +
  #   theme_bw() +
  #   theme(axis.text.x = element_text(angle = 90),
  #         legend.position = "none")
  # 
  # gs
  # ggsave(plot = gs, filename = paste0("results/hp_",str_replace_all(score, coll(" "), coll("_"))
  #                                     , "_stats.png"), device = "png",
  #        dpi = "retina", height = 3.93*2, width = 5.56*2)
}
}
### OUTPUT ###

scores = c("syntax", "semantics MF", "semantics BP", "semantics CC")

for (i in 1:length(scores)){
  violin(score = scores[i], df = df_hp) 
}


########## compare seq2vec for different embeddings ##########
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




