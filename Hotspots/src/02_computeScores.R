### HEADER ###
# HOTSPOT REGIONS
# description: slide over proteins and caculate similarity to n known regions
# input: training and test data set, trainedding of training data set, TF-IDF scores, token traineddings
# output: per-residue score for hotspot / non-hotspot in training data set
# author: HR


library(dplyr)
library(stringr)
library(ggplot2)
library(tidyr)
library(foreach)
library(doParallel)
library(future)

registerDoParallel(availableCores())


### INPUT ###
# see respective scripts

### MAIN PART ###

# function that returns cosine of angle between two vectors
matmult = function(v1 = "", v2 = ""){
  return(as.numeric(v1) %*% as.numeric(v2))
}

dot_product = function(v1 = "", v2 = ""){
  p = matmult(v1, v2)/(sqrt(matmult(v1, v1)) * sqrt(matmult(v2, v2)))
  return(p)
}


# KS distance between distributions
comp_score = function(d.H = "", d.N = ""){
  
  KS = ks.test(d.H, d.N, alternative = "greater")
  
  return(KS$statistic %>% as.numeric())
  
}

Prediction = function(train = "", test = "", dimRange = "") {
  
  # hyperparameters
  n = 500 # number of comparisons per test sample
  no_samples = nrow(test) # total no. of samples
  
  
  train.hsp = train[which(train$label == "hotspot"), ]
  train.n.hsp = train[which(train$label == "non_hotspot"), ]
  
  # subsample of test data set for test of pipeline
  test = test[sample(nrow(test), no_samples), ]
  
  test.res = rep(NA, no_samples)
  test.res = foreach (i = 1:no_samples, .combine = "rbind") %dopar% {
    
    # current sequence
    sqs = test[i, dimRange]
    
    # sample known hotspots and non-hotspots and calculate dot-product similarity
    sim.hsp = rep(NA, n)
    sim.n.hsp = rep(NA, n)
    
    # set.seed(42)
    
    for (k in 1:n){
      
      sim.hsp[k] = dot_product(v1 = sqs,
                               v2 = train.hsp[sample(nrow(train.hsp), 1), dimRange])
      
      sim.n.hsp[k] = dot_product(v1 = sqs,
                                 v2 = train.n.hsp[sample(nrow(train.n.hsp), 1), dimRange])
      
    }
    
    sims = comp_score(d.H = sim.hsp,
                      d.N = sim.n.hsp)
    
    # concatenate similarity and append to df
    test.res[i] = paste(sims, mean(sim.hsp), mean(sim.n.hsp), collapse = " ")
    
  }
  
  test.res = cbind(test[c(1:length(test.res)), ],
                      test.res)
  
  test.res$score = str_split_fixed(test.res$test.res, coll(" "), Inf)[, 1]
  test.res$d_H = str_split_fixed(test.res$test.res, coll(" "), Inf)[, 2]
  test.res$d_N = str_split_fixed(test.res$test.res, coll(" "), Inf)[, 3]
  
  test.res$test.res = NULL
  
  return(test.res)
  
}


