### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  evaluate embeddings based on ability to capture similarity between sequences
# input:        sequence embeddings similarity matrices, "true" similarities (syntax and semantics)
# output:       heatmaps showing the differences, similarity scores (the smaller the better)
# author:       HR

library(stringr)
library(plyr)
library(dplyr)
library(tidyr)

library(reshape2)
library(philentropy)
library(transport)
library(distrEx)

library(foreach)
library(doParallel)
library(doMC)
library(future)

print("### COMPUTE SCORES ###")

# parallel computing
cl <- makeCluster(availableCores())
registerDoParallel(cl)
registerDoMC(availableCores())

### INPUT ###
# overall input
syntax = read.csv(snakemake@input[["true_syntax"]], stringsAsFactors = F, header = T)
semantics_MF = read.csv(snakemake@input[["true_semantics_MF"]], stringsAsFactors = F, header = T)
semantics_BP = read.csv(snakemake@input[["true_semantics_BP"]], stringsAsFactors = F, header = T)
semantics_CC = read.csv(snakemake@input[["true_semantics_CC"]], stringsAsFactors = F, header = T)

# tmp !!!
# syntax = read.csv("postprocessing/similarity_true_syntax.csv", stringsAsFactors = F, header = T)
# semantics_MF = read.csv("postprocessing/similarity_true_semantics_MF.csv", stringsAsFactors = F, header = T)
# semantics_BP = read.csv("postprocessing/similarity_true_semantics_BP.csv", stringsAsFactors = F, header = T)
# semantics_CC = read.csv("postprocessing/similarity_true_semantics_CC.csv", stringsAsFactors = F, header = T)
# pred = read.csv("postprocessing/similarity_seq2vec.csv", stringsAsFactors = F, header = T)


# functions
# plot distributions
dist_plot = function(df = "", name = "", state = ""){
  
  if(str_detect(name, "/")){
    name = str_split(name, "/", simplify = T)[,2] %>% 
      as.vector() %>%
      as.character()
  }
  
  if(! dir.exists("tmp")){
    dir.create("tmp")
  }
  
  png(filename = paste0("tmp/",name, "_", state, ".png"))
  plot(density(df),
       main = str_replace_all(name, "_", " "),
       sub = paste0(state, " normalisation"))
  dev.off()
  
}

# remove metainformation and transform into matrix
cleaning = function(df = "", col = ""){
  
  df = df[, col] %>% as.matrix() %>%
    as.character() %>%
    as.numeric() %>%
    na.omit()
  
  return(df)
}

# calculate a bunch statistics to get distance of distributions
getDist = function(v1 = "", v2 = "") {
  
  v = rbind(v1, v2)
  
  # the smaller the more similar, i.e. the better : emd, bhatt, euc
  # the higher the more similar, i.e. the better : ks, cosine, jensen-shannon
  
  # Wasserstein metric (earth mover's distance)
  emd = wasserstein1d(v1, v2)
  # Bhattacharyya (similar to Mahalanobis)
  # bhatt = distance(v, method = "bhattacharyya")
  # euclidean distance
  euc = distance(v, method = "euclidean")
  # mse
  mse = (v1-v2)^2
  
  # KS test
  ks = ks.test(v1, v2, alternative = "two.sided")$p.value
  # Pearson correlation
  pcc = cor(v1, v2, method = "pearson")
  # cosine similarity
  cos = distance(v, method = "cosine")
  # Jensen-Shannon divergence
  # js_div = distance(v, method = "jensen-shannon")
  
  
  out = c(emd, bhatt, euc, mse, ks, pcc, cos, js_div)
  names(out) = c("Wasserstein_metric", "euclidean", "mean_squared_error",
                 "KS_pvalue", "Pearson_correlation", "cosine")
  
  return(out)
}

# calculate scores
compare = function(true = "", predicted = "", n_true = "", n_pred = ""){
  
  if(! (dim(true)[1] | dim(predicted)[1]) == 0){
    
    # same proteins
    if(!dim(true)[1] == dim(predicted)[1]){
      
      if("similarity" %in% colnames(predicted)){
        colnames(predicted) = c("acc1", "acc2", "pred_similarity")
        
        tmp = inner_join(true, predicted)
        
        true = tmp[, c("acc1", "acc2", "similarity")]
        
        tmp$similarity = NULL
        predicted = tmp
        colnames(predicted) = c("acc1", "acc2", "similarity")
        
      } else {
        
        tmp = inner_join(true, predicted)
        
        true = tmp[, c("acc1", "acc2", "similarity")]
        
        tmp$similarity = NULL
        predicted = tmp
        
      }
      
      
    }
    
    true = cleaning(true, col = "similarity")
    
    if("similarity" %in% colnames(predicted)){
      col_name = "similarity"
      
    } else { col_name = "dot" }
    
    predicted = cleaning(df = predicted, col = col_name)
    
    ### plot
    dist_plot(df = true, name = n_true, state = "1_prior_to")
    dist_plot(df = predicted, name = n_pred, state = "1_prior_to")
    
    scores_pre = getDist(v1 = true, v2 = predicted)
    
    # z-transformation
    predicted = (predicted - mean(predicted)) / sd(predicted)
    true = (true - mean(true)) / sd(true)
    
    scores_post = getDist(v1 = true, v2 = predicted)
    
    ### plot again
    dist_plot(df = true, name = n_true, state = "2_after_ztransform")
    dist_plot(df = predicted, name = n_pred, state = "2_after_ztransform")
    
    out = rbind(scores_pre, scores_post)
    
    return(out)
  
  } else {
    
    return(rbind(rep(NaN, 3),
                 rep(NaN, 3)))
    
  }
  
}

# actually start computing

input = snakemake@input[["predicted"]]

foreach(i = 1:length(input)) %dopar% {

  print(snakemake@input[["predicted"]][i])

  ### INPUT ###
  pred = read.csv(snakemake@input[["predicted"]][i], stringsAsFactors = F, header = T)
  pred = as.data.frame(pred)
  
  
  ### MAIN PART ###
  
  nm = str_split(snakemake@input[["predicted"]][i], coll("."), simplify = T)[,1] %>% 
    as.character()
  # nm = str_split("postprocessing/similarity_seq2vec.csv", coll("."), simplify = T)[,1] %>%
  #   as.character()

  # calculate scores
  syn = compare(true = syntax, predicted = pred,
                n_true = "syntax", n_pred = nm)

  sem_MF = compare(true = semantics_MF, predicted = pred,
                   n_true = "semantics_MF", n_pred = nm)

  sem_BP = compare(true = semantics_BP, predicted = pred,
                   n_true = "semantics_BP", n_pred = nm)

  sem_CC = compare(true = semantics_CC, predicted = pred,
                   n_true = "semantics_CC", n_pred = nm)
  
  # concatenate scores
  scores = list(syn, sem_MF, sem_BP, sem_CC)
  names(scores) = c("syn", "sem_MF", "sem_BP", "sem_CC")
  
  ### OUTPUT ###
  save(scores,file = unlist(snakemake@output[["scores"]][i]))

  print(paste0("done with ", snakemake@input[["predicted"]][i]))
}

stopImplicitCluster()
stopCluster(cl)