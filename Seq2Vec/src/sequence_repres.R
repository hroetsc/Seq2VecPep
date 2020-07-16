### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  transform embedded tokens to numerical sequence matrix
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

library(WGCNA)


### INPUT ###
print("LOAD DATA")
# sequences
params = read.csv(snakemake@input[["params"]], stringsAsFactors = F, header = T)
sequences = read.csv(file = params[which(params$parameter == "Seqinput"), "value"],
                     stringsAsFactors = F, header = T)
# TF-IDF scores
TF_IDF = read.csv(file = snakemake@input[["TF_IDF"]], stringsAsFactors = F, header = T)
# tokens
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)
# weights in .h5 format
weight_matrix = h5read(snakemake@input[["weights"]], "/embedding/embedding")
# corresponding subwords
indices = read.csv(file = snakemake@input[["ids"]], stringsAsFactors = F, header = F)

# tmp!!
sequences = read.csv("../proteome_human.csv", stringsAsFactors = F, header = T)
TF_IDF = read.csv("../TF_IDF.csv", stringsAsFactors = F, header = T)
words = read.csv("../words_hp.csv", stringsAsFactors = F, header = T)
indices = read.csv("../ids_hp_w5_new.csv", stringsAsFactors = F, header = F)
weight_matrix = h5read("hp_model_w5_d100/weights.h5", "/embedding/embedding")

# sequences = read.csv("../../../files/ProteasomeDB.csv", stringsAsFactors = F, header = T)
# TF_IDF = read.csv("../TF_IDF_ProteasomeDB.csv", stringsAsFactors = F, header = T)
# words = read.csv("../words_ProteasomeDB.csv", stringsAsFactors = F, header = T)
# words$Accession = str_replace_all(words$tokens, " ", "")
# indices = read.csv("../ids_ProteasomeDB_w5.csv", stringsAsFactors = F, header = F)
# weight_matrix = h5read("ProteasomeDB_model_w5_d100/weights.h5", "/embedding/embedding")


### MAIN PART ###
# multiprocessing
#threads = as.numeric(params[which(params$parameter == "threads"), "value"])
threads = availableCores()

cl = makeCluster(threads)
registerDoParallel(cl)
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
weights = full_join(weight_matrix, indices) %>% na.omit() %>% unique()


# combine sequences table with tokens file
sequences.master = left_join(sequences, words) %>% unique()

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
out = unlist(snakemake@output[["sequence_repres_seq2vec"]])
out.tfidf = unlist(snakemake@output[["sequence_repres_seq2vec_TFIDF"]])
out.sif = unlist(snakemake@output[["sequence_repres_seq2vec_SIF"]])

out.ccr = unlist(snakemake@output[["sequence_repres_seq2vec_CCR"]])
out.tfidf.ccr = unlist(snakemake@output[["sequence_repres_seq2vec_TFIDF_CCR"]])
out.sif.ccr = unlist(snakemake@output[["sequence_repres_seq2vec_SIF_CCR"]])

# tmp !!
# out = "hp_sequence_repres_w5_d100_seq2vec.csv"
# out.tfidf = "hp_sequence_repres_w5_d100_seq2vec-TFIDF.csv"
# out.sif = "hp_sequence_repres_w5_d100_seq2vec-SIF.csv"
# out.ccr = "hp_sequence_repres_w5_d100_seq2vec_CCR.csv"
# out.tfidf.ccr = "hp_sequence_repres_w5_d100_seq2vec-TFIDF_CCR.csv"
# out.sif.ccr = "hp_sequence_repres_w5_d100_seq2vec-SIF_CCR.csv"


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
  
  ## common component removal
  # remove mean
  mu = colSums(tmp) / nrow(tmp)
  mu.tfidf = colSums(tmp.tfidf) / nrow(tmp.tfidf)
  mu.sif = colSums(tmp.sif) / nrow(tmp.sif)
  
  tmp.ccr = tmp
  tmp.tfidf.ccr = tmp.tfidf
  tmp.sif.ccr = tmp.sif
  
  for (l in 1:nrow(tmp.ccr)){
    tmp.ccr[l,] = tmp.ccr[l, ] - mu
    tmp.tfidf.ccr[l,] = tmp.tfidf.ccr[l, ] - mu.tfidf
    tmp.sif.ccr[l,] = tmp.sif.ccr[l, ] - mu.sif
  }
  
  # only proceed if embeddings for every token are found, otherwise discard whole sequence
  if (!any(is.na(tmp))) {
    
    # calculate mean of every token dimension to get sequence dimension
    for (c in 1:ncol(tmp)) {
      tmp[,c] = as.numeric(as.character(tmp[,c]))
      tmp.tfidf[,c] = as.numeric(as.character(tmp.tfidf[,c]))
      tmp.sif[,c] = as.numeric(as.character(tmp.sif[,c]))
      
      tmp.ccr[,c] = as.numeric(as.character(tmp.ccr[,c]))
      tmp.tfidf.ccr[,c] = as.numeric(as.character(tmp.tfidf.ccr[,c]))
      tmp.sif.ccr[,c] = as.numeric(as.character(tmp.sif.ccr[,c]))
    }
    
    # calculate means
    line = colSums(tmp) / nrow(tmp)
    line.tfidf = colSums(tmp.tfidf) / nrow(tmp.tfidf)
    line.sif = colSums(tmp.sif) / nrow(tmp.sif)
    
    line.ccr = colSums(tmp.ccr) / nrow(tmp.ccr)
    line.tfidf.ccr = colSums(tmp.tfidf.ccr) / nrow(tmp.tfidf.ccr)
    line.sif.ccr = colSums(tmp.sif.ccr) / nrow(tmp.sif.ccr)
    
  } else {
    line = rep(NA, ncol(tmp))
    line.tfidf = line
    line.sif = line
    
    line.ccr = line
    line.tfidf.ccr = line
    line.sif.ccr = line
  }
  
  # add current accession
  acc = sequences.master$Accession[i]
  
  # write / append to file
  saveSeq(line = c(acc, line), cols = embeddingDim + 1, outfile = out, PID = Sys.getpid())
  saveSeq(line = c(acc, line.tfidf), cols = embeddingDim + 1, outfile = out.tfidf, PID = Sys.getpid())
  saveSeq(line = c(acc, line.sif), cols = embeddingDim + 1, outfile = out.sif, PID = Sys.getpid())
  
  saveSeq(line = c(acc, line.ccr), cols = embeddingDim + 1, outfile = out.ccr, PID = Sys.getpid())
  saveSeq(line = c(acc, line.tfidf.ccr), cols = embeddingDim + 1, outfile = out.tfidf.ccr, PID = Sys.getpid())
  saveSeq(line = c(acc, line.sif.ccr), cols = embeddingDim + 1, outfile = out.sif.ccr, PID = Sys.getpid())
  
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

seq2vec.ccr = mergeOut(out = out.ccr)
seq2vec.tfidf.ccr = mergeOut(out = out.tfidf.ccr)
seq2vec.sif.ccr = mergeOut(out = out.sif.ccr)

### OUTPUT ###
# vector representation of sequences
write.csv(seq2vec, file = out, row.names = F)
write.csv(seq2vec.tfidf, file = out.tfidf, row.names = F)
write.csv(seq2vec.sif, file = out.sif, row.names = F)

write.csv(seq2vec.ccr, file = out.ccr, row.names = F)
write.csv(seq2vec.tfidf.ccr, file = out.tfidf.ccr, row.names = F)
write.csv(seq2vec.sif.ccr, file = out.sif.ccr, row.names = F)
