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


print("### WEIIGHTING OF BIOPHYSICAL PROPERTIES ###")

### INPUT ###
sequences = read.csv(file = snakemake@input[["formatted_sequence"]],
                     stringsAsFactors = F, header = T)
TF_IDF = read.csv(file = snakemake@input[["TF_IDF"]], stringsAsFactors = F, header = T)
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)
weight_matrix = read.csv(file = snakemake@input[["weights"]], stringsAsFactors = F, header = F)
indices = read.csv(file = snakemake@input[["ids"]], stringsAsFactors = F, header = F)

# sequences = read.csv("data/current_sequences.csv", stringsAsFactors = F, header = T)
# TF_IDF = read.csv("data/TF_IDF.csv", stringsAsFactors = F, header = T)
# words = read.csv("data/current_words.csv", stringsAsFactors = F, header = T)
# weight_matrix = read.csv("data/weights_w5_d100.csv", stringsAsFactors = F, header = F)
# indices = read.csv("data/ids_hp_w5.csv", stringsAsFactors = F, header = F)


### MAIN PART ###
TF_IDF$token = toupper(TF_IDF$token)

if (ncol(indices) == 3){
  indices$V1 = NULL
}
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)


# merge indices and weights
colnames(weight_matrix)[1] = "word_ID"
weight_matrix$word_ID = weight_matrix$word_ID + 1

weights = full_join(weight_matrix, indices)
weights = na.omit(weights)
weights = unique(weights)

# combine sequences table with tokens file
sequences.master = left_join(sequences, words)
sequences.master = sequences.master[order(sequences.master$Accession), ]


# clean sequences
TF_IDF$token = as.character(TF_IDF$token)
a = sapply(toupper(TF_IDF$token), protcheck)
names(a) = NULL
print(paste0("found ",length(which(a==F)) , " tokens that are failing the protcheck() and is removing them"))
TF_IDF = TF_IDF[which(a == T), ]

# function that calculates bunch of properties
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

# actually calculate
PropMatrix = PepSummary(unique(TF_IDF$token))

# remove properties that correlate with sequence length
rm = c("protein.length", "mol.weight", "Tiny.number", "Small.number", "Aliphatic.number",
       "NonPolar.number", "Charged.number", "Basic.number", "Acidic.number", "Polar.number",
       "Aromatic.number")
for (r in 1:length(rm)){
  PropMatrix[, rm[r]] = NULL
}

PropMatrix[, "token"] = rownames(PropMatrix)
TF_IDF = left_join(TF_IDF, PropMatrix)

# define function searching for tokens
find_tokens = function(token = ""){
  if (token %in% TF_IDF$token) {
    return(TF_IDF[which(token == TF_IDF$token)[1], c(8:ncol(TF_IDF))])[1, ]
  } else {
    print(paste0("no embedding found for token: ", token))
    return(rep(NA, length(c(1:ncol(PropMatrix)))))
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

# matrix for TF-IDF weighting
repres.TFIDF = as.data.frame(matrix(ncol = ncol(sequences.master)+ncol(PropMatrix),
                                    nrow = nrow(sequences.master))) # contains vector representation for every sequence
colnames(repres.TFIDF) = c(colnames(sequences.master), seq(1,ncol(PropMatrix),1))
dim_range = c(ncol(sequences.master)+1, ncol(repres.TFIDF))
repres.TFIDF[, c(1:(dim_range[1]-1))] = sequences.master

# matrix for SIF weighting
repres.SIF = as.data.frame(matrix(ncol = ncol(sequences.master)+ncol(PropMatrix),
                                  nrow = nrow(sequences.master))) # contains vector representation for every sequence
colnames(repres.SIF) = c(colnames(sequences.master), seq(1,ncol(PropMatrix),1))
dim_range = c(ncol(sequences.master)+1, ncol(repres.SIF))
repres.SIF[, c(1:(dim_range[1]-1))] = sequences.master


# iterate sequences in master table to get their representation
progressBar = txtProgressBar(min = 0, max = nrow(sequences.master), style = 3)
for (i in 1:nrow(sequences.master)) {
  setTxtProgressBar(progressBar, i)
  
  # build temporary table that contains all tokens and weights for the current sequences
  current_tokens = t(str_split(sequences.master$tokens[i], pattern = " ", simplify = T))
  
  tfidf = as.data.frame(matrix(ncol = ncol(PropMatrix), nrow = nrow(current_tokens)))
  tfidf[, "token"] = current_tokens
  
  # find embeddings for every token in tmp
  for (r in 1:nrow(tfidf)) {
    tfidf[r,c(1:ncol(PropMatrix))] = find_tokens(paste(tfidf[r, ncol(tfidf)]))
  }
  
  sif = tfidf
  
  # extract TF-IDF scores for all tokens
  tfidf[, "TF_IDF"] = find_TF.IDF(tokens = tfidf$token, sequence = sequences.master$Accession[i])
  sif[, "SIF"] = find_SIF(tokens = sif$token, sequence = sequences.master$Accession[i])
  
  # multiply token embeddings by their TF-IDF scores
  tfidf[, c(1:(ncol(tfidf)-2))] = tfidf[, c(1:(ncol(tfidf)-2))] * tfidf$TF_IDF
  tfidf$TF_IDF = NULL
  tfidf$token = NULL
  
  # multiply token embeddings by their SIF scores
  sif[, c(1:(ncol(sif)-2))] = sif[, c(1:(ncol(sif)-2))] * sif$SIF
  sif$SIF = NULL
  sif$token = NULL
  
  # only proceed if embeddings for every token are found, otherwise discard sequence
  if (!any(is.na(tfidf) & is.na(sif))) {
    # calculate mean of every token dimension to get sequence dimension
    for (c in 1:ncol(tfidf)) {
      tfidf[,c] = as.numeric(as.character(tfidf[,c]))
      sif[,c] = as.numeric(as.character(sif[,c]))
    }
    repres.TFIDF[i, c(dim_range[1]:dim_range[2])] = colSums(tfidf[,c(1:ncol(PropMatrix))]) / nrow(tfidf)
    repres.SIF[i, c(dim_range[1]:dim_range[2])] = colSums(sif[,c(1:ncol(PropMatrix))]) / nrow(sif)
    
  } else {
    repres.TFIDF[i, c(dim_range[1]:dim_range[2])] = rep(NA, length(c(dim_range[1]:dim_range[2])))
    repres.SIF[i, c(dim_range[1]:dim_range[2])] = rep(NA, length(c(dim_range[1]:dim_range[2])))
  }
}

repres.TFIDF = na.omit(repres.TFIDF)
repres.TFIDF = unique(repres.TFIDF)
repres.TFIDF$tokens = NULL

repres.SIF = na.omit(repres.SIF)
repres.SIF = unique(repres.SIF)
repres.SIF$tokens = NULL


### OUTPUT ###
write.csv(repres.TFIDF, file = unlist(snakemake@output[["biophys_TFIDF"]]), row.names = F)
write.csv(repres.SIF, file = unlist(snakemake@output[["biophys_SIF"]]), row.names = F)
