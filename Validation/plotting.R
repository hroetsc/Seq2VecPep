### visualize results of validation pipeline ###

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggstatsplot)


### INPUT ###
# get all score files
fs = list.files(path = "downloads", pattern = "scores", recursive = T, full.names = T)

# merge them in a single df
for (f in 1:length(fs)){
  
  if (f == 1){
    df = read.csv(fs[f], stringsAsFactors = F)
    
    # tmp!!!
    #df = df[c(((2*nrow(df)/3)+1):nrow(df)),]
    
  } else {
    
    tmp = read.csv(fs[f], stringsAsFactors = F)
    # tmp!!!
    #tmp = tmp[c(((2*nrow(tmp)/3)+1):nrow(tmp)),]
    
    tmp$iteration = rep(f, nrow(tmp)) # different assignement?
    
    df = rbind(df, tmp)
  }
  
}

df = df[-which(df$embedding == "true_syntax"),]
df$embedding = str_replace_all(df$embedding, coll("_"), coll(" "))


### MAIN PART ###

# different plots for seq alinment and GO similarities
extract_scores = function(score = ""){
  return(df[, c("embedding", score)])
}

# tmp
#tbl = extract_scores(score = "syntax_diff")

# axis limits
upper = max(df[,3:ncol(df)])
lower = min(df[,3:ncol(df)])

# violin plots
violin = function(score = ""){
  
  set.seed(42)
  
  tbl = extract_scores(score = score)
  colnames(tbl) = c("embedding", "metric")
  
  # old school ggplot
  p = ggplot(tbl, aes(factor(embedding), metric)) +
    geom_violin(scale = "width", trim = F,
                draw_quantiles = c(0.5),
                aes(fill = factor(embedding))) +
    geom_jitter(height = 0, width = 0.01) +
    stat_summary(fun=mean, geom="point", size=1, color="red")
  
  p = p +
    ggtitle("comparison of true and predicted sequence similarity",
            subtitle = paste0(str_replace_all(score, coll("_"), coll(" ")))) +
    ylab("mean squared error") +
    xlab("embedding") +
    #ylim(c(lower, upper)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90),
          legend.position = "none")
    
  
  p
  ggsave(plot = p, filename = paste0("results/", score, ".png"), device = "png",
         dpi = "retina", height = 3.93*2, width = 5.56*2)
  
  
  
  # ggstatsplot
  
  gs = ggstatsplot::ggbetweenstats(data = tbl,
                               x = embedding,
                               y = metric)
  gs = gs +
    geom_violin(scale = "width", trim = F,
                draw_quantiles = c(0.5)) +
    geom_jitter(height = 0, width = 0.01) +
    ggtitle("comparison of true and predicted sequence similarity") +
    ylab("mean squared error") +
    xlab("embedding") +
    #ylim(c(lower, upper)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90),
          legend.position = "none")

  gs
  ggsave(plot = gs, filename = paste0("results/", score, "_stats.png"), device = "png",
         dpi = "retina", height = 3.93*2, width = 5.56*2)

}


### OUTPUT ###

for (i in 3:ncol(df)){
 violin(score = colnames(df)[i]) 
}

