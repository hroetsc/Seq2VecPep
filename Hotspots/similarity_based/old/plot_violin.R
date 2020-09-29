### HEADER ###
# HOTSPOT REGIONS
# description: plotting
# input: mean cosine similarities of embeddings
# output: nice plots
# author: HR


library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggstatsplot)


violin_plots = function(input = "", output = "", name = ""){
  ### INPUT ###
  scores = read.csv(input,
                    stringsAsFactors = F, header = F)
  names(scores) = c("hotspot_hotspot", "non-hotspot_non-hotspot", "hotspot_non-hotspot")
  
  
  ### MAIN PART ###
  scores2 = tidyr::gather(scores)
  names(scores2) = c("region", "mean_similarity")
  means = aggregate(mean_similarity ~ region, scores2, mean)
  
  #scores2$mean_similarity = log(as.numeric(as.character(scores2$mean_similarity)))
  
  p = ggplot(scores2, aes(factor(region), mean_similarity)) +
    geom_violin(scale = "width", trim = F,
                draw_quantiles = c(0.5),
                aes(fill = factor(region))) +
    geom_jitter(height = 0, width = 0.01) +
    stat_summary(fun=mean, geom="point", size=1, color="red") +
    ylim(c(0.4, 0.6)) +
    ggtitle("comparison of hotspot and non-hotspot embedding similarity",
            subtitle = paste0("each data point: mean of N = 100 regions, ",name," (window: 3, dim: 100)")) +
    ylab("mean of normalized cosine similarity") + 
    geom_text(data = means, aes(label = mean_similarity, y = 0.6)) +
    xlab("regions") +
    theme_bw() +
    theme(legend.position = "none")
  
  p
  
  ggsave(plot = p, filename = output, device = "png",
         dpi = "retina", height = 3.93*1.5, width = 5.56*1.5)
  
  
  # p-vaues
  print(name)
  print("hsp-hsp vs. hsp-nhsp.")
  print(t.test(scores$hotspot_hotspot, scores$`hotspot_non-hotspot`)$p.value)
  
  print("nhsp-nhsp vs. hsp-nhsp.")
  print(t.test(scores$`non-hotspot_non-hotspot`, scores$`hotspot_non-hotspot`)$p.value)
  
  print("hsp-hsp vs. nhsp-nhsp.")
  print(t.test(scores$hotspot_hotspot, scores$`non-hotspot_non-hotspot`)$p.value)
}

violin_plots(input = "RegionSimilarity/sim_w3_d100_seq2vec.csv",
             output = "RegionSimilarity/sim_w3_d100_seq2vec.png",
             name = "pure seq2vec")
  
violin_plots(input = "RegionSimilarity/sim_w3_d100_seq2vec_CCR.csv",
             output = "RegionSimilarity/sim_w3_d100_seq2vec_CCR.png",
             name = "seq2vec + CCR")

violin_plots(input = "RegionSimilarity/sim_w3_d100_seq2vec-TFIDF.csv",
             output = "RegionSimilarity/sim_w3_d100_seq2vec-TFIDF.png",
             name = "seq2vec + TF-IDF")

violin_plots(input = "RegionSimilarity/sim_w3_d100_seq2vec-TFIDF_CCR.csv",
             output = "RegionSimilarity/sim_w3_d100_seq2vec-TFIDF_CCR.png",
             name = "seq2vec + TF-IDF + CCR")

violin_plots(input = "RegionSimilarity/sim_w3_d100_seq2vec-SIF.csv",
             output = "RegionSimilarity/sim_w3_d100_seq2vec-SIF.png",
             name = "seq2vec + SIF")

violin_plots(input = "RegionSimilarity/sim_w3_d100_seq2vec-SIF_CCR.csv",
             output = "RegionSimilarity/sim_w3_d100_seq2vec-SIF_CCR.png",
             name = "seq2vec + SIF + CCR")
