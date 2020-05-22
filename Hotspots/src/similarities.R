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

threads = availableCores()
cl = makeCluster(threads)
registerDoMC(threads)


### INPUT ###

seq2vec = read.csv("repres_min_regions_w3_d100_seq2vec.csv",
                   stringsAsFactors = F, header = T)
seq2vec.tfidf = read.csv("repres_min_regions_w3_d100_seq2vec-TFIDF.csv",
                   stringsAsFactors = F, header = T)
seq2vec.sif = read.csv("repres_min_regions_w3_d100_seq2vec-SIF.csv",
                   stringsAsFactors = F, header = T)
seq2vec.CCR = read.csv("repres_min_regions_w3_d100_seq2vec_CCR.csv",
                       stringsAsFactors = F, header = T)
seq2vec.tfidf.CCR = read.csv("repres_min_regions_w3_d100_seq2vec-TFIDF_CCR.csv",
                         stringsAsFactors = F, header = T)
seq2vec.sif.CCR = read.csv("repres_min_regions_w3_d100_seq2vec-SIF_CCR.csv",
                       stringsAsFactors = F, header = T)


### MAIN PART ###
# function that returns cosine of angle between two vectors
dot_product = function(v1 = "", v2 = ""){
  p = sum(v1 * v2)/(sqrt(sum(v1^2)) * sqrt(sum(v2^2)))
  return(p)
}

rm_meta = function(tbl = ""){
  {
  tbl$Accession = NULL
  tbl$region = NULL
  tbl$label = NULL
  tbl$start = NULL
  tbl$end = NULL
  tbl$tokens = NULL
  }
  
  tbl[, which(!is.finite(colSums(tbl)))] = NULL
  
  for (t in 1:ncol(tbl)){
    tbl[,t] = as.numeric(as.character(tbl[, t]))
  }
  
  return(tbl)
}


sim = function(tbl1 = "", tbl2 = "", res = ""){
  
  pb = txtProgressBar(min = 0, max = nrow(tbl1), style = 3)
  
  res = foreach(a = 1:nrow(tbl1), .combine = "rbind") %dopar% {
    
    setTxtProgressBar(pb, a)
    
    res[a, ] = foreach (b = 1:nrow(tbl2), .combine = "cbind") %dopar% {
     
      res[a, b] = dot_product(v1 = tbl1[a,],
                               v2 = tbl2[b, ])
     
   }
   
  }
  
  # scale, transform, normalize
  res = (res - min(res)) / (max(res) - min(res))
  res = (res - mean(res)) / sd(res)
  res = pnorm(res)
  
  return(mean(res))
}

# function that calculates pairwise similarities
calc = function(tbl = "", outfile = ""){
  
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
calc(tbl = seq2vec, outfile = "RegionSimilarity/sim_w3_d100_seq2vec.csv")
calc(tbl = seq2vec.tfidf, outfile = "RegionSimilarity/sim_w3_d100_seq2vec-TFIDF.csv")
calc(tbl = seq2vec.sif, outfile = "RegionSimilarity/sim_w3_d100_seq2vec-SIF.csv")

calc(tbl = seq2vec.CCR, outfile = "RegionSimilarity/sim_w3_d100_seq2vec_CCR.csv")
calc(tbl = seq2vec.tfidf.CCR, outfile = "RegionSimilarity/sim_w3_d100_seq2vec-TFIDF_CCR.csv")
calc(tbl = seq2vec.sif.CCR, outfile = "RegionSimilarity/sim_w3_d100_seq2vec-SIF_CCR.csv")

