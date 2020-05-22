### HEADER ###
# HOTSPOT REGIONS
# description:  transform embedded tokens to numerical sequence matrix
#               sample
# input:        weight matrix from seq2vec_model_training, word table (Accession, tokens), TF-IDF scores,
#               table that maps word-IDs to tokens
# output:       sequence matrices:
#               - no weighting
#               - weighting using TF-IDF
#               - weighting using SIF
# author:       HR

print("### RETRIEVE SEQUENCE REPRESENATION ###")

library(plyr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)

library(rhdf5)

library(foreach)
library(doParallel)
library(doMC)
library(future)


### INPUT ###
print("LOAD DATA")

sequences = read.csv("RegionSimilarity/data/regions_min_substr.csv", stringsAsFactors = F, header = T)
TF_IDF = read.csv("../RUNS/HumanProteome/TF_IDF.csv", stringsAsFactors = F, header = T)
words = read.csv("../RUNS/HumanProteome/words_hp.csv", stringsAsFactors = F, header = T)
indices = read.csv("../RUNS/HumanProteome/results/ids_hp_w3.csv", stringsAsFactors = F, header = F)

weight_matrix = h5read("../RUNS/HumanProteome/results/model_w3_d100/weights.h5", "/embedding/embedding")


### MAIN PART ###
# multiprocessing
#threads = as.numeric(params[which(params$parameter == "threads"), "value"])
threads = availableCores()

cl = makeCluster(threads)
registerDoMC(threads)


# assign tokens to weight matrix
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
weights = full_join(weight_matrix, indices)
weights = na.omit(weights)
weights = unique(weights)


### sampling starts here ###
N = 100
k1 = sample(which(sequences$label == "hotspot"), N)
k2 = sample(which(sequences$label == "non_hotspot"), N)
sequences.master = sequences[c(k1, k2), ]

sequences.master = unique(sequences.master)

if (any(is.na(sequences.master$tokens))){
  sequences.master = sequences.master[-which(is.na(sequences.master$tokens)), ]
}


### FUNCTIONS ###
# grep only the columns that contain actual weights in the weights data frame
grep_weights = function(df = ""){
  return(which(grepl("^[0-9]{1,}$", colnames(df))))
}

embeddingDim = length(grep_weights(weights))


# define function that searches for tokens in weight matrix
find_tokens = function(token = ""){
  if (token %in% weights$subword) {
    return(weights[which(token == weights$subword)[1], grep_weights(weights)][1, ])
    
  } else {
    
    print(paste0("no embedding found for token: ", token))
    return(rep(NA, embeddingDim))
  }
}


TF_IDF$token = toupper(TF_IDF$token)
# define function that finds TF-IDF score for token in correct sequence
find_TF.IDF = function(tokens = "", sequence = "") {
  scores = rep(NA, length(tokens))
  tmp = TF_IDF[which(TF_IDF$Accession == sequence), ]
  
  for (t in 1:length(tokens)) {
    if (tokens[t] %in% tmp$token) {
      scores[t] = tmp[which(tmp$token == tokens[t]), "tf_idf"]
    } else {
      scores[t] = 1
    }
  }
  return(scores)
}


# calculate smooth inverse frequency
a = 0.001
find_SIF = function(tokens = "", sequence = "") {
  scores = rep(NA, length(tokens))
  tmp = TF_IDF[which(TF_IDF$Accession == sequence), ]
  
  for (t in 1:length(tokens)) {
    if (tokens[t] %in% tmp$token) {
      scores[t] = tmp[which(tmp$token == tokens[t]), "idf"]
    } else {
      scores[t] = NaN
    }
  }
  # get document frequency from inverse frequency
  scores = exp(log(nrow(sequences), scores))
  # smooth inverse frequency
  scores = a/(a + scores)
  scores[which(!is.finite(scores))] = 1
  
  return(scores)
}


# write current sequence representation to file to prevent running OOM
saveSeq = function(line = "", cols = "", outfile = "", PID = ""){
  path = paste0("tmp/",PID, "_", outfile)
  
  if(! file.exists(path)){
    write(line, path, append = F, sep = ",", ncolumns = cols)
    
  } else {
    write(line, path, append = T, sep = ",", ncolumns = cols)
  }
}


# iterate sequences in master table to get their representation
print("RETRIEVING NUMERIC REPRESENTATION OF EVERY SEQUENCE")

# convert weights into numeric
for (c in grep_weights(weights)){
  weights[,c] = as.numeric(as.character(weights[,c]))
}


### outfiles ###
out = "repres_min_regions_w3_d100_seq2vec.csv"
out.tfidf = "repres_min_regions_w3_d100_seq2vec-TFIDF.csv"
out.sif = "repres_min_regions_w3_d100_seq2vec-SIF.csv"


# empty directory for tmp outfiles
if(! dir.exists("./tmp")){
  dir.create("./tmp")
} else {
  system("rm -rf tmp/")
  dir.create("./tmp")
}

# iteration
system.time(foreach (i = 1:nrow(sequences.master)) %dopar% {
  
  # build temporary table that contains all tokens and weights for the current sequences
  current_tokens = str_split(sequences.master$tokens[i], coll(" "), simplify = T) %>% as.vector()
  
  tmp = as.data.frame(matrix(ncol = embeddingDim, nrow = length(current_tokens)))
  tmp[, "token"] = current_tokens
  
  # find embeddings for every token in tmp
  for (r in 1:nrow(tmp)) {
    tmp[r,c(1:embeddingDim)] = find_tokens(paste(tmp[r, ncol(tmp)]))
  }
  
  tmp.tfidf = tmp
  tmp.sif = tmp
  
  # extract TF-IDF and SIF scores for all tokens
  tmp.tfidf[, "TF_IDF"] = find_TF.IDF(tokens = tmp.tfidf$token, sequence = sequences.master$Accession[i])
  tmp.sif[, "SIF"] = find_SIF(tokens = tmp.sif$token, sequence = sequences.master$Accession[i])
  
  # multiply token embeddings by their TF-IDF scores
  tmp.tfidf[, c(1:embeddingDim)] = tmp.tfidf[, c(1:embeddingDim)] * tmp.tfidf$TF_IDF
  tmp.tfidf$TF_IDF = NULL
  tmp.tfidf$token = NULL
  
  # multiply token embeddings by their SIF scores
  tmp.sif[, c(1:embeddingDim)] = tmp.sif[, c(1:embeddingDim)] * tmp.sif$SIF
  tmp.sif$SIF = NULL
  tmp.sif$token = NULL
  
  tmp$token = NULL
  
  # only proceed if embeddings for every token are found, otherwise discard whole sequence
  if (!any(is.na(tmp))) {
    
    # calculate mean of every token dimension to get sequence dimension
    for (c in 1:ncol(tmp)) {
      tmp[,c] = as.numeric(as.character(tmp[,c]))
      tmp.tfidf[,c] = as.numeric(as.character(tmp.tfidf[,c]))
      tmp.sif[,c] = as.numeric(as.character(tmp.sif[,c]))
    }
    
    # calculate means
    line = colSums(tmp) / nrow(tmp)
    line.tfidf = colSums(tmp.tfidf) / nrow(tmp.tfidf)
    line.sif = colSums(tmp.sif) / nrow(tmp.sif)
    
    
  } else {
    line = rep(NA, ncol(tmp))
    line.tfidf = line
    line.sif = line
  }
  
  # add current accession
  acc = sequences.master$Accession[i]
  
  # write / append to file
  saveSeq(line = c(acc, line), cols = embeddingDim + 1, outfile = out, PID = Sys.getpid())
  saveSeq(line = c(acc, line.tfidf), cols = embeddingDim + 1, outfile = out.tfidf, PID = Sys.getpid())
  saveSeq(line = c(acc, line.sif), cols = embeddingDim + 1, outfile = out.sif, PID = Sys.getpid())
  
  
})[3]

stopImplicitCluster()
stopCluster(cl)

print("DONE")


# concatenating output from all threads
# add metainformation to outfiles
mergeOut = function(out = ""){
  
  fs = list.files(path = "tmp", pattern = out, full.names = T)
  
  for (i in 1:length(fs)){
    if (i == 1){
      
      tbl = read.csv(fs[i], stringsAsFactors = F, header = F)
      
    } else {
      
      dat = read.csv(fs[i], stringsAsFactors = F, header = F)
      tbl = rbind(tbl, dat)
      
    }
  }
  
  colnames(tbl) = c("Accession", grep_weights(weights))
  
  tbl = inner_join(sequences.master, tbl)
  tbl = as.data.frame(tbl)
  
  tbl = na.omit(tbl)
  tbl = unique(tbl)
  
  return(tbl)
}


seq2vec = mergeOut(out = out)
seq2vec.tfidf = mergeOut(out = out.tfidf)
seq2vec.sif = mergeOut(out = out.sif)


### OUTPUT ###
# vector representation of sequences
write.csv(seq2vec, file = out, row.names = F)
write.csv(seq2vec.tfidf, file = out.tfidf, row.names = F)
write.csv(seq2vec.sif, file = out.sif, row.names = F)
