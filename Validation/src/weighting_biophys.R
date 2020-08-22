### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  weight embeddings (based on token biophysical properties) by TF-IDF and SIF
# input:        sequence embeddings
# output:       embeddings weighted by TF-IDF, embeddings weighted by SIF
# author:       HR

library(plyr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)

library(seqinr)
library(protr)
library(Peptides)
library(plyr)
library(dplyr)
library(stringr)

library(WGCNA)

library(foreach)
library(doParallel)
library(doMC)
library(future)

print("### BIOPHYSICAL PROPERTIES: EMBEDDING AND WEIGHTING ###")


### INPUT ###
sequences = read.csv(file = snakemake@input[["formatted_sequence"]],
                     stringsAsFactors = F, header = T)
TF_IDF = read.csv(file = snakemake@input[["TF_IDF"]], stringsAsFactors = F, header = T)
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)
indices = read.csv(file = snakemake@input[["ids"]], stringsAsFactors = F, header = F)
PropMatrix = read.csv(file = snakemake@input[["Props"]], stringsAsFactors = F, header = T)

# sequences = read.csv("data/current_sequences.csv", stringsAsFactors = F, header = T)
# TF_IDF = read.csv("data/TF_IDF.csv", stringsAsFactors = F, header = T)
# words = read.csv("data/current_words.csv", stringsAsFactors = F, header = T)
# indices = read.csv("data/ids_hp_w5_new.csv", stringsAsFactors = F, header = F)
# PropMatrix = read.csv(file = "data/PropMatrix.csv", stringsAsFactors = F, header = T)


### MAIN PART ###
threads = 8

cl = makeCluster(threads)
registerDoParallel(cl)
registerDoMC(threads)


TF_IDF$token = toupper(TF_IDF$token) %>% as.character()

if (ncol(indices) == 3){
  indices$V1 = NULL
}
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)


# combine sequences table with tokens file
sequences.master = left_join(sequences, words) %>% na.omit()
sequences.master = sequences.master[order(sequences.master$Accession), ]


# clean sequences
# a = sapply(toupper(TF_IDF$token), protcheck)
# names(a) = NULL
# print(paste0("found ",length(which(a==F)) , " tokens that are failing the protcheck() and is removing them"))
# TF_IDF = TF_IDF[which(a == T), ]


PropMatrix[, "token"] = PropMatrix$subword

PropMatrix$word_ID = NULL
PropMatrix$subword = NULL


### FUNCTIONS ###
# grep only the columns that contain actual weights in the PropMatrix data frame
grep_weights = function(df = ""){
  c = str_count(df[2,], "\\d+\\.*\\d*")
  return(as.logical(c))
}

embeddingDim = grep_weights(PropMatrix) %>% sum()


# define function that searches for tokens in weight matrix
find_tokens = function(token = ""){
  if (token %in% PropMatrix$token) {
    return(PropMatrix[which(token == PropMatrix$token), grep_weights(PropMatrix)][1, ])
    
  } else {
    
    print(paste0("no embedding found for token: ", token))
    return(rep(NA, embeddingDim))
  }
}

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
  
  path = paste0("biophys_tmp/",PID, "_", outfile)
  
  if (str_detect(outfile, "/")){
    outfile = str_split(outfile, "/", simplify = T) %>% as.vector()
    path = paste0("biophys_tmp/", PID, "_", outfile[1], "_", outfile[2])
  }
  
  
  if(! file.exists(path)){
    write(line, path, append = F, sep = ",", ncolumns = cols)
    
  } else {
    write(line, path, append = T, sep = ",", ncolumns = cols)
  }
}


# iterate sequences in master table to get their representation
print("RETRIEVING NUMERIC REPRESENTATION OF EVERY SEQUENCE")

# convert weights into numeric
if(! mode(PropMatrix[,5]) == "numeric"){
  for (c in grep_weights(PropMatrix)){
    PropMatrix[,c] = PropMatrix[,c] %>% as.character() %>% as.numeric()
  }
  
}


### outfiles ###
out = unlist(snakemake@output[["biophys"]])
out.tfidf = unlist(snakemake@output[["biophys_TFIDF"]])
out.sif = unlist(snakemake@output[["biophys_SIF"]])

out.ccr = unlist(snakemake@output[["biophys_CCR"]])
out.tfidf.ccr = unlist(snakemake@output[["biophys_TFIDF_CCR"]])
out.sif.ccr = unlist(snakemake@output[["biophys_SIF_CCR"]])


# empty directory for tmp outfiles
if(! dir.exists("./biophys_tmp")){
  dir.create("./biophys_tmp")
} else {
  system("rm -rf biophys_tmp/")
  dir.create("./biophys_tmp")
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
  
  if (str_detect(out, "/")){
    out = str_replace(out, "/", "_")
  }
  
  fs = list.files(path = "biophys_tmp", pattern = out, full.names = T)
  
  for (i in 1:length(fs)){
    if (i == 1){
      
      tbl = read.csv(fs[i], stringsAsFactors = F, header = F)
      
    } else {
      
      dat = read.csv(fs[i], stringsAsFactors = F, header = F)
      tbl = rbind(tbl, dat)
      
    }
  }
  
  colnames(tbl) = c("Accession", seq(1,embeddingDim))
  
  tbl = inner_join(sequences.master, tbl)
  tbl = as.data.frame(tbl)
  
  tbl = na.omit(tbl)
  tbl = unique(tbl)
  
  return(tbl)
}


repres = mergeOut(out = out)
repres.tfidf = mergeOut(out = out.tfidf)
repres.sif = mergeOut(out = out.sif)

repres.ccr = mergeOut(out = out.ccr)
repres.tfidf.ccr = mergeOut(out = out.tfidf.ccr)
repres.sif.ccr = mergeOut(out = out.sif.ccr)


### OUTPUT ###
write.csv(repres, file = out, row.names = F)
write.csv(repres.tfidf, file = out.tfidf, row.names = F)
write.csv(repres.sif, file = out.sif, row.names = F)

write.csv(repres.ccr, file = out.ccr, row.names = F)
write.csv(repres.tfidf.ccr, file = out.tfidf.ccr, row.names = F)
write.csv(repres.sif.ccr, file = out.sif.ccr, row.names = F)

