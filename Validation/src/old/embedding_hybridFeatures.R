### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  generate hybrid feature embeddings
# input:        YH: biophysical property matrix for tokens, supervised Seq2Vec
#               embeddings for window sizes 3 and 5, sequences
# output:       sequence representation generated from hybrid embeddings
# author:       HR

library(plyr)
library(dplyr)
library(stringr)

print("### GENERATE EMBEDDINGS USING HYBRID FEATURES ###")


### INPUT ###
words = read.csv(file = snakemake@input[["words"]],
                     stringsAsFactors = F, header = T)
indices = read.csv(file = snakemake@input[["ids"]], stringsAsFactors = F, header = F)
w3 = read.csv(file = snakemake@input[["w3"]], stringsAsFactors = F, header = T)
w5 = read.csv(file = snakemake@input[["w5"]], stringsAsFactors = F, header = T)
sup = read.csv(file = snakemake@input[["sup"]], stringsAsFactors = F, header = T)
PropMatrix = read.csv(file = snakemake@input[["Props"]], stringsAsFactors = F, header = T)

# words = read.csv(file = "data/current_words.csv",
#                   stringsAsFactors = F, header = T)
# indices = read.csv(file = "data/ids_hp_w5.csv", stringsAsFactors = F, header = F)
# w3 = read.csv(file = "data/w3_d100_embedding.csv", stringsAsFactors = F, header = F)
# w5 = read.csv(file = "data/w5_d100_embedding.csv", stringsAsFactors = F, header = F)
# sup = read.csv(file = "data/u.sup.inv.3d.csv", stringsAsFactors = F, header = T)
# PropMatrix = read.csv(file = "data/PropMatrix.csv", stringsAsFactors = F, header = T)


### MAIN PART ###
# assign tokens to weight matrix
if (ncol(indices) == 3){
  indices$V1 = NULL
}
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)

# merge weights with
add_tokens = function(weights = ""){
  colnames(weights)[1] = "word_ID"
  w = full_join(weights, indices)
  w = na.omit(w)
  w = unique(w)
  w$word_ID = NULL
  w = w[order(w$subword), ]
  return(w)
}

w3 = add_tokens(w3)
w5 = add_tokens(w5)

# hybrid features
merge_emb = function(tbl_ls = ""){
  tbl = matrix(nrow = nrow(indices)) %>% as.data.frame()
  
  for (i in 1:length(tbl_ls)){
    tmp = tbl_ls[[i]] %>% as.data.frame()
    tmp = tmp[order(tmp$subword), ]
    tokens = tmp$subword
    
    tmp$subword = NULL
    tbl = cbind(tbl, tmp)
  }
  
  tbl = cbind(tokens, tbl)
  tbl[,2] = NULL
  
  colnames(tbl) = c("subword", seq(1, ncol(tbl)-1))
  return(tbl)
}

tokens.w3w5 = merge_emb(tbl_ls = list(w3, w5))
PropMatrix$word_ID = NULL
tokens.w3w5B = merge_emb(tbl_ls = list(w3, w5, PropMatrix))

sup$word_ID = NULL
colnames(sup) = c("subword", seq(1, ncol(sup)-1))
# sequence representation: average of token embeddings

# functions
grep_weights = function(df = ""){
  return(which(grepl("^[0-9]{1,}$", colnames(df))))
}

find_tokens = function(token = "", weights = ""){
  if (token %in% weights$subword) {
    return(weights[which(token == weights$subword)[1], grep_weights(weights)][1, ])
    
  } else {
    
    print(paste0("no embedding found for token: ", token))
    return(rep(NA, embeddingDim))
  }
}

sequence_repres = function(tok = ""){
  embeddingDim = length(grep_weights(tok))
  
  seq_rep = matrix(ncol = ncol(tok)+1, nrow = nrow(words))
  seq_rep[, 1] = words$Accession
  seq_rep[, 2] = words$tokens
  
  pb = txtProgressBar(min = 0, max = nrow(words), style = 3)
  
  for (i in 1:nrow(words)){
    setTxtProgressBar(pb, i)
    
    # build temporary table that contains all tokens and weights for the current sequences
    current_tokens = str_split(words$tokens[i], coll(" "), simplify = T) %>% as.vector()
    
    tmp = as.data.frame(matrix(ncol = embeddingDim, nrow = length(current_tokens)))
    tmp[, "token"] = current_tokens
    
    for (r in 1:nrow(tmp)) {
      tmp[r,c(1:embeddingDim)] = find_tokens(paste(tmp[r, ncol(tmp)]), weights = tok)
    }
    
    tmp$token = NULL
    
    if (!any(is.na(tmp))) {
      
      seq_rep[i, 3:ncol(seq_rep)] = colSums(tmp) / nrow(tmp)
      
    } else {
      seq_rep[i, 3:ncol(seq_rep)] = NA
    }
    
  }
  
  seq_rep = as.data.frame(seq_rep) %>% na.omit()
  colnames(seq_rep) = c("Accession", "seqs", seq(1, embeddingDim))
  
  return(seq_rep)
}

hybrid.w3w5 = sequence_repres(tokens.w3w5)
hybrid.w3w5B = sequence_repres(tokens.w3w5B)
hybrid.sup = sequence_repres(tok = sup)

### OUTPUT ###
write.csv(hybrid.w3w5, file = unlist(snakemake@output[["hybrid_w3w5"]]), row.names = F)
write.csv(hybrid.w3w5, file = unlist(snakemake@output[["hybrid_w3w5biophys"]]), row.names = F)
write.csv(hybrid.w3w5, file = unlist(snakemake@output[["hybrid_sup"]]), row.names = F)