### HEADER ###
# HOTSPOT REGIONS
# description: functions to generate training / testing data sets
# input: hotspot/non-hotspot regions (extended/minimal version)
# output: training data set and its embedding
# author: HR

library(dplyr)
library(stringr)
library(seqinr)

library(protr)
library(Peptides)

library(future)
library(foreach)
library(doParallel)

library(WGCNA)

library(ggplot2)
library(tidyr)

library(rhdf5)

registerDoParallel(cores=availableCores())

# nomenclature
# PURE - original regions (extended substrings)
# NTERM1 - N-terminal extension and original region embedded together (100 dim.)
# NTERM2 - N-terminal extension and original region embedded separately and appended vectors (200 dim.)
# same for CTERM1, CTERM2, RED1, RED2 (reduced hotspot/non-hotspot regions)
# PURE_PROP - original regions joint with 80-dimensional vector of biophysical properties


### INPUT ###
reg = read.csv("data/regions_ext_substr.csv", stringsAsFactors = F)


### MAIN PART ###
########## sample data sets ########## 
# majority of short sequences are non-hotspots
# select training data with same length distributions of hotspots/non-hotspots

sampleData = function(reg = "") {
  
  # length range
  rg = seq(20, 160)
  
  # sample k sequences for each range
  k = 8
  
  # how many training samples ??
  # for each length, sample 2 hotspots and 2 non_hotspots
  
  hsp.reg = reg[which(reg$label == "hotspot"), ]
  n.hsp.reg = reg[which(reg$label == "non_hotspot"), ]
  
  # empty training/testing data frames
  training.hsp = matrix(ncol = ncol(reg), nrow = length(rg)*k) %>% as.data.frame()
  training.n.hsp = matrix(ncol = ncol(reg), nrow = length(rg)*k)  %>% as.data.frame()
  
  testing.hsp = matrix(ncol = ncol(reg), nrow = length(rg)*k) %>% as.data.frame()
  testing.n.hsp = matrix(ncol = ncol(reg), nrow = length(rg)*k)  %>% as.data.frame()
  
  
  pb = txtProgressBar(min = 0, max = length(rg), style = 3)
  
  counter = 1
  
  for (i in rg){
    
    setTxtProgressBar(pb, i)
    
    tmp.hsp = hsp.reg[which(nchar(hsp.reg$region) == i), ]
    tmp.n.hsp = n.hsp.reg[which(nchar(n.hsp.reg$region) == i), ]
    
    if (nrow(tmp.hsp) > 0 & nrow(tmp.n.hsp) > 0){
      
      training.hsp[counter:(counter+k-1), ] = tmp.hsp[sample(nrow(tmp.hsp), k), ]
      training.n.hsp[counter:(counter+k-1), ] = tmp.n.hsp[sample(nrow(tmp.n.hsp), k), ]
      
      testing.hsp[counter:(counter+k-1), ] = tmp.hsp[sample(nrow(tmp.hsp), k), ]
      testing.n.hsp[counter:(counter+k-1), ] = tmp.n.hsp[sample(nrow(tmp.n.hsp), k), ]
      
    } else {
      
      training.hsp[counter:(counter+k-1), ] = NA
      training.n.hsp[counter:(counter+k-1), ] = NA
      
      testing.hsp[counter:(counter+k-1), ] = NA
      testing.hsp[counter:(counter+k-1), ] = NA
      
    }
    
    counter = counter + k
    
  }
  
  # merge to get training data
  training = rbind(training.hsp, training.n.hsp) %>% na.omit()
  names(training) = names(reg)
  # shuffle
  training = training[sample(nrow(training)), ]
  
  testing = rbind(testing.hsp, testing.n.hsp) %>% na.omit()
  names(testing) = names(reg)
  testing = testing[sample(nrow(testing)), ]
  
  
  # remove all proteins and isoforms that occur in the training data from testing data
  testing = testing[which(!str_split_fixed(testing$Accession, coll("-"), Inf)[,1] %in%
                        str_split_fixed(training$Accession, coll("-"), Inf)[,1]), ]
  
  print(paste0("size of training data set: ", nrow(training)))
  print(paste0("size of testing data set: ", nrow(testing)))
  
  out = list(training, testing)
  names(out) = c("training", "testing")
  
  return(out)
}

out = sampleData(reg)
training = out[["training"]] %>% as.data.frame()
testing = out[["testing"]] %>% as.data.frame()

### OUTPUT ###

# used in all downstream scripts!
write.csv(training, "data/classifier/training_DATA.csv", row.names = F)
write.csv(testing, "data/classifier/testing_DATA.csv", row.names = F)


########## get seq2vec + TF-IDF embeddings ########## 

### INPUT ###
indices = read.csv("../RUNS/HumanProteome/ids_hp_w5_new.csv", stringsAsFactors = F, header = F)
weight_matrix = h5read("../RUNS/HumanProteome/word2vec_model/hp_model_w5_d100/weights.h5", "/embedding/embedding")
TFIDF = read.csv("data/ext_substr_TFIDF.csv", stringsAsFactors = F)
TFIDF$token = toupper(TFIDF$token)

# retrieve token embeddings

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
  
  # convert weights into numeric
  for (c in grep_weights(weights)){
    weights[,c] = as.numeric(as.character(weights[,c]))
  }
  
}


# a lot of functions
{
  
  # functions for sequence representation
  grep_weights = function(df = ""){
    return(which(grepl("^[0-9]{1,}$", colnames(df))))
  }
  
  embeddingDim = length(grep_weights(weights))
  
  
  find_tokens = function(token = ""){
    if (token %in% weights$subword) {
      
      res = weights[which(token == weights$subword)[1], grep_weights(weights)][1, ]
      res = mutate_if(res, is.factor, ~ as.numeric(levels(.x))[.x])
      
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
  
  # write current sequence representation to file to prevent running OOM
  saveSeq = function(line = "", cols = "", outfile = "", PID = ""){
    path = paste0("tmp/",PID, "_", outfile)
    
    if(! file.exists(path)){
      write(line, path, append = F, sep = ",", ncolumns = cols)
      
    } else {
      write(line, path, append = T, sep = ",", ncolumns = cols)
    }
  }
  
  
  
  # concatenating output from all threads
  # add metainformation to outfiles
  mergeOut = function(out = "", sequences.master = ""){
    
    fs = list.files(path = "tmp", pattern = out, full.names = T)
    
    if (length(fs) > 0) {
      
      for (i in 1:length(fs)){
        if (i == 1){
          
          tbl = read.csv(fs[i], stringsAsFactors = F, header = F)
          
          
        } else {
          
          dat = read.csv(fs[i], stringsAsFactors = F, header = F)
          tbl = rbind(tbl, dat)
          
        }
      }
      
      colnames(tbl) = c("Accession", grep_weights(weights))
      
      tbl = inner_join(sequences.master, tbl) %>%
        as.data.frame() %>%
        na.omit() %>%
        unique()
      
      # tmp !!! combine with existing file
      
      if (file.exists(out)) {
        ex = read.csv(out, stringsAsFactors = F, header = T)
        tbl = rbind(ex, tbl) %>% unique()
      }
      
      
      return(tbl)
      
    } else {
      
      return(NA)
      
    }
    
  }
  
  
  # here: seq2vec + TF-IDF
  get_seq_repres = function(sequences.master = "", out = ""){
    
    # make sure that accessions are unique
    sequences.master$Accession = paste0(sequences.master$Accession, "_",
                                        seq(1, nrow(sequences.master)))
    
    # empty directory for tmp outfiles
    if(! dir.exists("./tmp")){
      dir.create("./tmp")
    } else {
      system("rm -rf tmp/")
      dir.create("./tmp")
    }
    
    
    system.time(foreach (i = 1:nrow(sequences.master)) %dopar% {
      
      # build temporary table that contains all tokens and weights for the current sequences
      current_tokens = str_split(sequences.master$tokens[i], coll(" "), simplify = T) %>% as.vector()
      
      tmp = matrix(ncol = embeddingDim, nrow = length(current_tokens)) %>% as.data.frame()
      tmp[, "token"] = current_tokens
      
      # find embeddings for every token in tmp
      for (r in 1:nrow(tmp)) {
        tmp[r, c(1:embeddingDim)] = find_tokens(tmp$token[r])
      }
      
      # only proceed if embeddings for every token are found, otherwise discard whole sequence
      if (!any(is.na(tmp))) {
        
        tmp.tfidf = tmp
        
        # extract TF-IDF scores for all tokens
        tmp.tfidf[, "TF_IDF"] = find_TF.IDF(tokens = tmp.tfidf$token, sequence = sequences.master$Accession[i])
        
        # multiply token embeddings by their TF-IDF scores
        tmp.tfidf[, c(1:embeddingDim)] = tmp.tfidf[, c(1:embeddingDim)] * tmp.tfidf$TF_IDF
        tmp.tfidf$TF_IDF = NULL
        tmp.tfidf$token = NULL
        
        # get means
        mu.tfidf = colSums(tmp.tfidf) / nrow(tmp.tfidf)
        
        #!!! z-transform to normalise vector
        mu.tfidf = (mu.tfidf - mean(mu.tfidf)) / sd(mu.tfidf)
        
        # add current accession
        acc = sequences.master$Accession[i]
        
        # write / append to file
        saveSeq(line = c(acc, mu.tfidf), cols = embeddingDim + 1, outfile = out, PID = Sys.getpid())
        
      }
      
      # no else to save time (sequences are discarded anyways)
      
    })[3]
    
    seq2vec.tfidf = mergeOut(out = out, sequences.master = sequences.master)
    return(seq2vec.tfidf)
    
  }
  
}

########## extensions ########## 
words = read.csv("../RUNS/HumanProteome/words_hp.csv", stringsAsFactors = F)

cleanup = function(tbl){
  tbl$seqs = NULL
  tbl$tokens = NULL
  
  if("shifted" %in% names(tbl)) {
    colnames(tbl) = c("Accession", "region", "label", "start", "end", "shifted", "tokens")
  } else {
    colnames(tbl) = c("Accession", "region", "label", "start", "end", "tokens")
  }
  
  
  return(tbl)
}


getTokens = function(regions = "", pos1 = "", pos2 = "") {
  
  pb = txtProgressBar(min = 0, max = nrow(regions), style = 3)
  
  for (m in 1:nrow(regions)){
    
    setTxtProgressBar(pb, m)
    
    tokens = str_split(regions$tokens[m], coll(" "), simplify = T) %>% t() %>% as.data.frame()
    tokens$V1 = as.character(tokens$V1)
    tokens[, "start"] = rep(NA, nrow(tokens))
    tokens[, "end"] = tokens$start
    
    # token positions
    end = 0
    
    for (t in 1:nrow(tokens)){
      tokens[t, pos1] = end + 1
      end = end + nchar(tokens$V1[t])
      tokens[t, pos2] = end
      
    }
    
    r = seq(regions[m, pos1], regions[m, pos2])
    
    ext.regions = tokens[which(tokens[, pos1] %in% r | tokens[, pos2] %in% r), "V1"]
    regions[m, "extr_token"] = paste(ext.regions, collapse = " ")
    
  }
  
  regions = cleanup(regions) %>%
    unique()
  
  return(regions)
  
}


extensions = function(tbl = "", direction = "", by = "") {
  
  tbl$tokens = NULL
  
  # add protein sequence
  tbl = left_join(tbl, words) %>% na.omit()
  
  # change positions
  if (direction == "N") {
    
    tbl$shifted = tbl$start - by
    tbl$shifted[tbl$shifted < 1] = 1
    
    joint = getTokens(regions = tbl, pos1 = "shifted", pos2 = "end")
    only_ext = getTokens(tbl, "shifted", "start")
    
    
    
  } else if (direction == "C") {
    
    tbl$shifted = tbl$end + by
    tbl$shifted[tbl$shifted > nchar(tbl$seqs)] = nchar(tbl$seqs)
    
    
    joint = getTokens(tbl, "start", "shifted")
    only_ext = getTokens(tbl, "end", "shifted")
    
    
  } else if (direction == "R") {
    
    joint = getTokens(tbl, "start", "end")
    
    tbl$start = tbl$start + by
    tbl$start[tbl$start > nchar(tbl$seqs)] = nchar(tbl$seqs)
    
    tbl$end = tbl$end - by
    tbl$end[tbl$end < 1] = 1
    
    only_ext = getTokens(tbl, "start", "end")
  }
  
  out = list(joint, only_ext)
  names(out) = c("joint", "only_ext")
  
  return(out)
}


########## biophysical properties ########## 

# don't forget z-transformation!!!

PepSummary <- function(Peptides.input) {
  progressBar = txtProgressBar(min = 0, max = length(Peptides.input), style = 3)
  # Equivalent function for chatacter string input
  
  # Compute the amino acid composition of a protein sequence
  AACOMP <- data.frame()
  print("AACOMP")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(aaComp(seq = Peptides.input[i])))
    
    AACOMP.numbers <- a[1,]
    AACOMP.Mole.percent <- a[2,]
    names(AACOMP.numbers) <- c("Tiny.number","Small.number","Aliphatic.number","Aromatic.number","NonPolar.number","Polar.number","Charged.number","Basic.number","Acidic.number")
    names(AACOMP.Mole.percent) <- c("Tiny.Mole.percent","Small.Mole.percent","Aliphatic.Mole.percent","Aromatic.Mole.percent","NonPolar.Mole.percent","Polar.Mole.percent","Charged.Mole.percent","Basic.Mole.percent","Acidic.Mole.percent")
    a <- t(data.frame(c(AACOMP.numbers,AACOMP.Mole.percent)))
    rownames(a) <- Peptides.input[i]
    
    AACOMP <- rbind(AACOMP,a)
  }
  
  # Compute the aliphatic index of a protein sequence
  AINDEX <- data.frame()
  print("AAINDEX")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- as.data.frame(aIndex(seq = Peptides.input[i]))
    rownames(a) <- Peptides.input[i]
    colnames(a) <- c("aliphatic.index")
    AINDEX <- rbind(AINDEX,a)
  }
  
  # Compute the BLOSUM62 derived indices of a protein sequence
  BLOSUM62 <- data.frame()
  print("BLOSUM62")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(blosumIndices(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    BLOSUM62 <- rbind(BLOSUM62,a)
  }
  
  # Compute the Boman (Potential Protein Interaction) index
  # Important if higher then 2.48
  BOMANINDEX <- data.frame()
  print("BOMANINDEX")
  for (i in 1:length(Peptides.input)) {
    
    setTxtProgressBar(progressBar, i)
    a <- as.data.frame(boman(seq = Peptides.input[i]))
    rownames(a) <- Peptides.input[i]
    colnames(a) <- c("boman.index")
    BOMANINDEX <- rbind(BOMANINDEX,a)
  }
  
  # Compute the theoretical net charge of a protein sequence
  PROTCHARGE <- data.frame()
  print("PROTCHARGE")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- as.data.frame(charge(seq = Peptides.input[i], pH = 7, pKscale = "Lehninger"))
    rownames(a) <- Peptides.input[i]
    colnames(a) <- c("charge")
    PROTCHARGE <- rbind(PROTCHARGE,a)
  }
  
  # Compute the Cruciani properties of a protein sequence
  CRUCIANI <- data.frame()
  print("CRUCIANI")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(crucianiProperties(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    colnames(a) <- c("Polarity","Hydrophobicity","H-bonding")
    CRUCIANI <- rbind(CRUCIANI,a)
  }
  
  # Compute the FASGAI vectors of a protein sequence
  FASGAI <- data.frame()
  print("FASGAI")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(fasgaiVectors(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    FASGAI <- rbind(FASGAI,a)
  }
  
  # Compute the instability index of a protein sequence
  # INSTAINDEX <- data.frame()
  # print("INSTAINDEX")
  # for (i in 1:length(Peptides.input)) {
  #   setTxtProgressBar(progressBar, i)
  #   a <- t(as.data.frame(instaIndex(seq = Peptides.input[i])))
  #   rownames(a) <- Peptides.input[i]
  #   INSTAINDEX <- rbind(INSTAINDEX,a)
  # }
  # colnames(INSTAINDEX) <- c("instability.index")
  #
  
  # Compute the Kidera factors of a protein sequence
  KIDERA <- data.frame()
  print("KIDERA")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(kideraFactors(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    KIDERA <- rbind(KIDERA,a)
  }
  
  # Compute the amino acid length of a protein sequence
  print("LENGTHP")
  LENGTHP <- data.frame()
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(lengthpep(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    LENGTHP <- rbind(LENGTHP,a)
  }
  colnames(LENGTHP) <- c("protein.length")
  
  # Compute the MS-WHIM scores of a protein sequence
  MSWHIM <- data.frame()
  print("MSWHIM")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(mswhimScores(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    MSWHIM <- rbind(MSWHIM,a)
  }
  
  # Compute the molecular weight of a protein sequence
  MOLWEIGHT <- data.frame()
  print("MOLWEIGHT")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(mw(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    MOLWEIGHT <- rbind(MOLWEIGHT,a)
  }
  colnames(MOLWEIGHT) <- c("mol.weight")
  
  # Compute the isoelectic point (pI) of a protein sequence
  ISOELECTRIC <- data.frame()
  print("ISOELECTRIC")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(pI(seq = Peptides.input[i],pKscale = "Lehninger")))
    rownames(a) <- Peptides.input[i]
    ISOELECTRIC <- rbind(ISOELECTRIC,a)
  }
  colnames(ISOELECTRIC) <- c("pI")
  
  # Compute the protFP descriptors of a protein sequence
  PROTFP <- data.frame()
  print("PROTFP")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(protFP(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    PROTFP <- rbind(PROTFP,a)
  }
  
  # Compute the ST-scales of a protein sequence
  STSC <- data.frame()
  print("STSC")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(stScales(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    STSC <- rbind(STSC,a)
  }
  
  # Compute the T-scales of a protein sequence
  TSC <- data.frame()
  print("TSC")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(tScales(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    TSC <- rbind(TSC,a)
  }
  
  # Compute the VHSE-scales of a protein sequence
  VHSE <- data.frame()
  print("VHSE")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(vhseScales(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    VHSE <- rbind(VHSE,a)
  }
  
  # Compute the Z-scales of a protein sequence
  ZSC <- data.frame()
  print("ZSC")
  for (i in 1:length(Peptides.input)) {
    setTxtProgressBar(progressBar, i)
    a <- t(as.data.frame(zScales(seq = Peptides.input[i])))
    rownames(a) <- Peptides.input[i]
    ZSC <- rbind(ZSC,a)
  }
  # Bind a summary
  ProtProp <- cbind(AACOMP,AINDEX,BLOSUM62,BOMANINDEX,CRUCIANI,FASGAI,ISOELECTRIC,KIDERA,LENGTHP,MOLWEIGHT,MSWHIM,PROTCHARGE,PROTFP,STSC,TSC,VHSE,ZSC)
  return(ProtProp)
}

# clean proteome
ProtCheck = function(tbl = ""){
  tbl$region = as.character(tbl$region)
  a <- sapply(toupper(tbl$region), protcheck)
  names(a) <- NULL
  
  print(paste0("found ",length(which(a==F)) , " proteins that are failing the protcheck() and is removing them"))
  
  tbl = tbl[which(a == T), ]
  return(tbl)
}

# z-transformation
Z_transform = function(tbl = "") {
  
  tbl = as.matrix(tbl)
  
  for ( r in 1:nrow(tbl)) {
    
    tbl[r,] = (tbl[r, ] - mean(tbl[r, ])) / sd(tbl[r, ])
    
  }
  
  tbl = as.data.frame(tbl)
  
  return(tbl)
}


getPropMatrix = function(inp = "") {
  
  inp = ProtCheck(inp)
  
  PropMatrix = PepSummary(inp$region)
  PropMatrix = Z_transform(PropMatrix)
  PropMatrix$region = rownames(PropMatrix)
  
  # remove properties that correlate with sequence length
  rm = c("protein.length", "mol.weight", "Tiny.number", "Small.number", "Aliphatic.number",
         "NonPolar.number", "Charged.number", "Basic.number", "Acidic.number", "Polar.number",
         "Aromatic.number")
  
  PropMatrix = PropMatrix[ , -which(names(PropMatrix) %in% rm)]
  
  return(PropMatrix)
}

