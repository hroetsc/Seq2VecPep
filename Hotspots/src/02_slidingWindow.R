### HEADER ###
# HOTSPOT REGIONS
# description: slide over proteins and caculate similarity to n known regions
# input: training and testing data set, embedding of training data set, TF-IDF scores, token embeddings
# output: per-residue score for hotspot / non-hotspot in training data set
# author: HR


library(dplyr)
library(stringr)
library(WGCNA)

library(foreach)
library(doParallel)
library(future)

library(rhdf5)

registerDoParallel(availableCores())


### INPUT ###
# choose seq2vec + CCR + TFIDF
# extended substrings

TFIDF = read.csv("data/ext_substr_TFIDF.csv", stringsAsFactors = F)
TFIDF$token = toupper(TFIDF$token)

training = read.csv("data/classifier/training_DATA.csv", stringsAsFactors = F)
testing = read.csv("data/classifier/testing_DATA.csv", stringsAsFactors = F)

emb = read.csv("data/classifier/training_extSubstr_w5_d100_seq2vec-TFIDF_CCR.csv",
               stringsAsFactors = F)

dimRange = c(5:ncol(emb)) # where are weights in this table?

emb.hsp = emb[which(emb$label == "hotspot"), ]
emb.n.hsp = emb[which(emb$label == "non_hotspot"), ]


indices = read.csv("../RUNS/HumanProteome/ids_hp_w5_new.csv", stringsAsFactors = F, header = F)
weight_matrix = h5read("../RUNS/HumanProteome/word2vec_model/hp_model_w5_d100/weights.h5", "/embedding/embedding")


### MAIN PART ###

# hyperparameters
windowSize = 4
n = 500

# subsample of testing data set for testing of pipeline
set.seed(42)

testing = testing[sample(nrow(testing), 1e03), ]

########## retrieve token embeddings ########## 

{
if (ncol(indices) == 3){
  indices$V1 = NULL
}
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)

# extract weights
weight_matrix = plyr::ldply(weight_matrix)
weight_matrix = t(weight_matrix) %>% as.data.frame()

weight_matrix = weight_matrix[-1,]
colnames(weight_matrix) = seq(1, ncol(weight_matrix))

weight_matrix["word_ID"] = seq(0, (nrow(weight_matrix)-1), 1)
weight_matrix$word_ID = weight_matrix$word_ID + 1


# merge indices and weights
weights = full_join(weight_matrix, indices) %>% na.omit() %>% unique()
}


########## functions ##########
{

# functions for sequence representation
grep_weights = function(df = ""){
  return(which(grepl("^[0-9]{1,}$", colnames(df))))
}

embeddingDim = length(grep_weights(weights))


find_tokens = function(token = ""){
  if (token %in% weights$subword) {
    
    res = weights[which(token == weights$subword)[1], grep_weights(weights)][1, ]
    red = mutate_if(res, is.factor, ~ as.numeric(levels(.x))[.x])
    
    return(res %>% as.vector())
    
  } else {
    
    print(paste0("no embedding found for token: ", token))
    return(rep(NA, embeddingDim))
  }
}



# define function that finds TF-IDF score for token in correct sequence
find_TF.IDF = function(tokens = "", sequence = "") {
  scores = rep(NA, length(tokens))
  tmp = TFIDF[which(TFIDF$Accession == sequence), ]
  
  for (t in 1:length(tokens)) {
    if (tokens[t] %in% tmp$token) {
      scores[t] = tmp[which(tmp$token == tokens[t]), "tf_idf"]
      
    } else {
      scores[t] = 1
    }
  }
  return(scores)
}


# function that returns cosine of angle between two vectors
matmult = function(v1 = "", v2 = ""){
  return(as.numeric(v1) %*% as.numeric(v2))
}

dot_product = function(v1 = "", v2 = ""){
  p = matmult(v1, v2)/(sqrt(matmult(v1, v1)) * sqrt(matmult(v2, v2)))
  return(p)
}


# specific functions, here: seq2vec + CCR + TF-IDF
get_seq_repres = function(tokens = "", acc = ""){
  
  tmp = matrix(ncol = embeddingDim, nrow = length(tokens)) %>% as.data.frame()
  tmp[, "token"] = tokens
  
  # find embeddings for every token
  for (r in 1:nrow(tmp)){
    tmp[r, c(1:embeddingDim)] = find_tokens(token = tmp$token[r])
  }
  
  if (! any(is.na(tmp))){
    
    # extract TF-IDF scores for tokens and multiply them
    tmp[, "tfidf"] = find_TF.IDF(tokens = tmp$token, sequence = acc)
    tmp[, c(1:embeddingDim)] = tmp[, c(1:embeddingDim)] * tmp$tfidf
    tmp$tfidf = NULL
    tmp$token = NULL
    
    # remove mean and apply CCR
    mu = colSums(tmp)/nrow(tmp)
    
    tmp = t(apply(as.matrix(tmp), 1, function(x) x - mu)) %>%
      removePrincipalComponents(n = 1) %>% as.data.frame()
    
    # average to get sequence representation
    sqs = colSums(tmp) / nrow(tmp)
    
    return(sqs)
    
  } else {
    
    print(paste0("no sequence embedding available for ", acc))
    
    return(NA)
  }
  
}

# KS distance between distributions
comp_score = function(d.H = "", d.N = ""){
  
  KS = ks.test(d.H, d.N, alternative = "greater")
  
  return(KS$statistic %>% as.numeric())
  
}

}

########## whole sequence ##########
#pb = txtProgressBar(min = 0, max = nrow(testing), style = 3)

testing.res = rep(NA, nrow(testing))
testing.res = foreach (i = 1:nrow(testing), .combine = "rbind") %dopar% {
  
  #setTxtProgressBar(pb, i)
  
  # slide over protein sequence
  cnt_tokens = str_split(testing$tokens[i], coll(" "), simplify = T) %>%
    as.character() %>%
    as.vector()
  
  # get sequence representation of frame
  sqs = get_seq_repres(tokens = frm, acc = testing$Accession[i])
  
  # sample known hotspots and non-hotspots and calculate dot-product similarity
  sim.hsp = rep(NA, n)
  sim.n.hsp = rep(NA, n)
  
  for (k in 1:n){
    
    sim.hsp[k] = dot_product(v1 = sqs,
                             v2 = emb.hsp[sample(nrow(emb.hsp), 1), dimRange])
    
    sim.n.hsp[k] = dot_product(v1 = sqs,
                               v2 = emb.n.hsp[sample(nrow(emb.n.hsp), 1), dimRange])
    
  }
  
  sims = comp_score(d.H = sim.hsp,
                    d.N = sim.n.hsp)
  
  # concatenate similarity and append to df
  testing.res[i] = paste(sims, mean(sim.hsp), mean(sim.n.hsp), collapse = " ")
  
}

testing.res = cbind(testing[c(1:length(testing.res)), ],
                    testing.res)

testing.res$score = str_split_fixed(testing.res$testing.res, coll(" "), Inf)[, 1]
testing.res$d_H = str_split_fixed(testing.res$testing.res, coll(" "), Inf)[, 2]
testing.res$d_N = str_split_fixed(testing.res$testing.res, coll(" "), Inf)[, 3]

testing.res$testing.res = NULL

### OUTPUT ###
write.csv(testing.res, "data/classifier/testing_PREDICTION.csv", row.names = F)


########## sliding window ##########

# pb = txtProgressBar(min = 0, max = nrow(testing), style = 3)
# 
# testing$score = rep(NA, nrow(testing))
# 
# 
# # loop over proteins
# for (i in 1:nrow(testing)) {
#   
#   setTxtProgressBar(pb, i)
#   
#   # slide over protein sequence
#   cnt_tokens = str_split(testing$tokens[i], coll(" "), simplify = T) %>%
#     as.character() %>%
#     as.vector()
#   
#   sims = rep(NA, length(cnt_tokens))
#   
#   counter = 1
#   
#   for (j in 1:(length(cnt_tokens) - windowSize)) {
#     
#     # frame = tokens in sliding window
#     frm = c(counter : (counter + windowSize - 1))
#     if (length(frm) >= (counter + windowSize - 1)){
#       frm = cnt_tokens[counter : (counter + windowSize - 1)]
#       
#     } else {
#       frm = cnt_tokens[counter : length(cnt_tokens)]
#       
#     }
#     
#     # get sequence representation of frame
#     sqs = get_seq_repres(tokens = frm, acc = testing$Accession[i])
#     
#     # sample known hotspots and non-hotspots and calculate dot-product similarity
#     sim.hsp = rep(NA, n)
#     sim.n.hsp = rep(NA, n)
#     
#     for (k in 1:n){
#       
#       sim.hsp[k] = dot_product(v1 = sqs,
#                                v2 = emb.hsp[sample(nrow(emb.hsp), 1), dimRange])
#       
#       sim.n.hsp[k] = dot_product(v1 = sqs,
#                                  v2 = emb.n.hsp[sample(nrow(emb.n.hsp), 1), dimRange])
#       
#     }
#     
#     sims[j] = comp_score(d.H = mean(sim.hsp),
#                          d.N = mean(sim.n.hsp))
#     
#     counter = counter + 1
#     
#   }
#   
#   # concatenate similarity and append to df
#   testing$score[i] = paste(sims, collapse = " ")
#   
# }
# 


