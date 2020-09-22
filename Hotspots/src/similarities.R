### HEADER ###
# HOTSPOT REGIONS
# description: calculate pairwise similarities between hotspot / non-hotspot embeddings
# input: embeddings for N sampled hotspot and N non-hotspot regions
# output: mean normalised similarities
# author: HR

library(foreach)
library(doParallel)
library(doMC)
library(future)

library(dplyr)
library(ggplot2)
library(tidyr)
library(stringr)

threads = availableCores()
registerDoParallel(threads)

# setwd("Documents/QuantSysBios/ProtTransEmbedding/Hotspots/")

### INPUT ###

seq2vec = read.csv("data/ext_substr_w5_d100_seq2vec.csv",
                   stringsAsFactors = F, header = T)
seq2vec.tfidf = read.csv("data/ext_substr_w5_d100_seq2vec-TFIDF.csv",
                   stringsAsFactors = F, header = T)
seq2vec.sif = read.csv("data/ext_substr_w5_d100_seq2vec-SIF.csv",
                   stringsAsFactors = F, header = T)
seq2vec.ccr = read.csv("data/ext_substr_w5_d100_seq2vec_CCR.csv",
                       stringsAsFactors = F, header = T)
seq2vec.tfidf.ccr = read.csv("data/ext_substr_w5_d100_seq2vec-TFIDF_CCR.csv",
                         stringsAsFactors = F, header = T)
seq2vec.sif.ccr = read.csv("data/ext_substr_w5_d100_seq2vec-SIF_CCR.csv",
                       stringsAsFactors = F, header = T)

# for length-corrected data set
# training = read.csv("data/classifier/training_DATA.csv", stringsAsFactors = F)
# seq2vec.tfidf.ccr$Accession = str_split_fixed(seq2vec.tfidf.ccr$Accession, coll("_"), Inf)[,1]
# training = inner_join(training, seq2vec.tfidf.ccr) %>% na.omit()
# training$start = NULL
# training$end = NULL


### MAIN PART ###
# function that returns cosine of angle between two vectors
matmult = function(v1 = "", v2 = ""){
  return(as.numeric(v1) %*% as.numeric(v2))
}

dot_product = function(v1 = "", v2 = ""){
  p = matmult(v1, v2)/(sqrt(matmult(v1, v1)) * sqrt(matmult(v2, v2)))
  return(p)
}


rm_meta = function(tbl = ""){
  {
  tbl$Accession = NULL
  tbl$region = NULL
  tbl$label = NULL
  tbl$tokens = NULL
  }
  
  tbl[, which(!is.finite(colSums(tbl)))] = NULL
  
  for (t in 1:ncol(tbl)){
    tbl[,t] = as.numeric(as.character(tbl[, t]))
  }
  
  return(tbl)
}


sim = function(tbl1 = "", tbl2 = "", res = ""){
  
  res = foreach(a = 1:nrow(tbl1), .combine = "rbind") %dopar% {
    
    res[a, ] = foreach (b = 1:nrow(tbl2), .combine = "cbind") %dopar% {
     
      res[a, b] = dot_product(v1 = tbl1[a,],
                               v2 = tbl2[b, ])
     
   }
   
  }
  
  return(res[lower.tri(res)] %>% mean())
}

# function that calculates pairwise similarities
calc = function(tbl = "", outfile = "", n_samples = 20){
  
  keep = sample(nrow(tbl), n_samples)
  tbl = tbl[keep, ]
  
  tbl = tbl[order(tbl$Accession), ]
  
  hsp = tbl[which(tbl$label == "hotspot"), ]
  hsp_prots = hsp$Accession
  hsp = rm_meta(hsp)
  
  n.hsp = tbl[which(tbl$label == "non_hotspot"), ]
  n.hsp_prots = n.hsp$Accession
  n.hsp = rm_meta(n.hsp)
  
  # 3 comparisons: hsp-hsp, n.hsp-n.hsp, hsp-n.hsp
  
  hsp_hsp = matrix(ncol = nrow(hsp), nrow = nrow(hsp))
  n.hsp_n.hsp = matrix(ncol = nrow(n.hsp), nrow = nrow(n.hsp))
  hsp_n.hsp = matrix(ncol = nrow(n.hsp), nrow = nrow(hsp))
  
  mean_hsp_hsp = sim(tbl1 = hsp, tbl2 = hsp, res = hsp_hsp)
  mean_n.hsp_n.hsp = sim(tbl1 = n.hsp, tbl2 = n.hsp, res = n.hsp_n.hsp)
  mean_hsp_n.hsp = sim(tbl1 = hsp, tbl2 = n.hsp, res = hsp_n.hsp)
  
  final = c(mean_hsp_hsp, mean_n.hsp_n.hsp, mean_hsp_n.hsp)
  
  # append mean to outfile
  if(file.exists(outfile)){
    write(final, outfile, ncolumns = 3, sep = ",", append = T)
    
  } else {
    write(final, outfile, ncolumns = 3, sep = ",", append = F)
  }
}



### OUTPUT ###
for (i in 1:40){
  calc(tbl = seq2vec, outfile = "RegionSimilarity/sim_w5_d100_seq2vec.csv")
  calc(tbl = seq2vec.tfidf, outfile = "RegionSimilarity/sim_w5_d100_seq2vec-TFIDF.csv")
  calc(tbl = seq2vec.sif, outfile = "RegionSimilarity/sim_w5_d100_seq2vec-SIF.csv")

  calc(tbl = seq2vec.ccr, outfile = "RegionSimilarity/sim_w5_d100_seq2vec_CCR.csv")
  calc(tbl = seq2vec.tfidf.ccr, outfile = "RegionSimilarity/sim_w5_d100_seq2vec-TFIDF_CCR.csv")
  calc(tbl = seq2vec.sif.ccr, outfile = "RegionSimilarity/sim_w5_d100_seq2vec-SIF_CCR.csv")

  calc(tbl = training, outfile = "RegionSimilarity/training_data.csv")
}

stopImplicitCluster()

# plotting
load_and_plot = function(infile = "", title = ""){
  
  tbl = read.csv(infile, header = F, stringsAsFactors = F) %>% na.omit()
  
  colnames(tbl) = c("hotspot-hotspot", "no hotspot - no hotspot",
                    "hotspot - no hotspot")
  
  # tmp !!
  # tbl = tbl[c(35:nrow(tbl)), ]
  
  tbl = tidyr::gather(tbl)
  
  means = aggregate(value ~ key, tbl, mean)
  means$value = round(means$value, 4)
  
  p = ggplot(tbl, aes(x = key, y = value, fill = key)) +
    geom_violin(scale = "width", trim = F,
                draw_quantiles = c(0.5)) +
    stat_summary(fun=mean, geom="point", size=1, color="red") +
    geom_text(data = means, aes(label = value, y = max(tbl$value) + 0.1)) +
    scale_fill_viridis_d(option = "viridis", direction = -1) +
    ggtitle("comparison of hotspot and non-hotspot region similarity",
            subtitle = paste0("embedded using ", title, " (window = 5, dim = 100)")) +
    ylab("mean cosine similarity") +
    xlab("regions") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 0),
          legend.position = "none")
  p
  
  out = str_split(infile, coll("."), simplify = T)[,1] %>% as.character()
  ggsave(filename = paste0(out, ".png"), plot = p, dpi = "retina")
  
}

load_and_plot(infile = "RegionSimilarity/sim_w5_d100_seq2vec.csv", title = "seq2vec")
load_and_plot(infile = "RegionSimilarity/sim_w5_d100_seq2vec-TFIDF.csv", title = "seq2vec + TFIDF")
load_and_plot(infile = "RegionSimilarity/sim_w5_d100_seq2vec-SIF.csv", title = "seq2vec + SIF")

load_and_plot(infile = "RegionSimilarity/sim_w5_d100_seq2vec_CCR.csv", title = "seq2vec + CCR")
load_and_plot(infile = "RegionSimilarity/sim_w5_d100_seq2vec-TFIDF_CCR.csv", title = "seq2vec + TFIDF + CCR")
load_and_plot(infile = "RegionSimilarity/sim_w5_d100_seq2vec-SIF_CCR.csv", title = "seq2vec + SIF + CCR")

load_and_plot(infile = "RegionSimilarity/training_data.csv", title = "seq2vec + TFIDF + CCR (length-corrected training data)")
