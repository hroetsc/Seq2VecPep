### HEADER ###
# HOTSPOT REGIONS
# description: try out different classifiers
# input: sequence repres after DAN
# output: -
# author: HR

{
library(dplyr)
library(caret)
library(dbscan)
library(MLeval)
library(plotROC)
library(mlbench)
library(pROC)

library(scatterplot3d)
library(car)
library(rgl)

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(uwot)
library(ggraptR)
library(ggthemes)

library(foreach)
library(doParallel)

registerDoParallel(cores=availableCores())

}

### INPUT ###

dt = read.csv("ext_substr_w5_d100_seq2vec-TFIDF.csv", stringsAsFactors = F)


### MAIN PART ###

########## UMAP ##########
grep_weights = function(df = ""){
  c = str_count(df[2,], "\\d+\\.*\\d*")
  return(as.logical(c))
}


UMAP = function(tbl = ""){
  set.seed(42)
  
  for (c in 1:ncol(tbl)){
    tbl[, c] = tbl[, c] %>% as.character() %>% as.numeric()
  }
  
  tbl$Accession = NULL
  
  coord = umap(tbl,
               n_neighbors = 5,
               min_dist = 0.8,
               n_epochs = 300,
               n_trees = 15,
               metric = "cosine",
               verbose = T,
               approx_pow = T,
               ret_model = T,
               init = "spca",
               n_threads = availableCores(),
               n_components = 3)
  
  um = data.frame(UMAP1 = coord$embedding[,1],
                  UMAP2 = coord$embedding[,2],
                  UMAP3 = coord$embedding[,3])
  
  return(um)
}

um = UMAP(dt[, grep_weights(dt)])



# plotting
{
  col_by = dt$label
  nColor = length(levels(as.factor(col_by)))
  colors = paletteer_c("viridis::viridis", n = nColor)
  colors = c("darkturquoise", "firebrick1")
  
  rank = rep(NA, length(col_by))
  for (i in 1:length(col_by)){
    if (col_by[i] == "non_hotspot") { rank[i] = colors[1] } else { rank[i] = colors[2] }
  }
}

png(filename = "seq2vec_TFIDF.png",
    width = 2000, height = 2000, res = 300)
plot(um$UMAP1, um$UMAP2,
     col = rank,
     cex = 0.3,
     pch = 1,
     xlab = "UMAP 1", ylab = "UMAP 2",
     main = "human proteins: seq2vec + TFIDF",
     sub = "colored by hotspot/non-hotspot")
dev.off()

scatter3d(x = um$UMAP1,
          y = um$UMAP2,
          z = um$UMAP3,
          surface = F,
          point.col = rank)

########## HDBSCAN ##########
# emb = dt[, grep_weights(dt)]
# emb$Accession = NULL

cl = hdbscan(um, minPts = 100)
cl
plot(um$UMAP1, um$UMAP2,
     col = cl$cluster + 1,
     cex = 0.3,
     pch = 1,
     xlab = "UMAP 1", ylab = "UMAP 2",
     main = "human proteins: seq2vec + TFIDF",
     sub = "colored by hotspot/non-hotspot")


########## characterise cluster ##########
cl.seqs = cbind(cl$cluster, dt[, !grep_weights(dt)], um)

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

# biophys properties
PropMatrix0 = foreach(i=1:length(cl.seqs$region), 
                      .packages = c("Peptides")) %dopar% PepSummary(cl.seqs$region[i])
PropMatrix0 <- bind_rows(PropMatrix0)
PropMatrix0$region = rownames(PropMatrix0)

cl.seqs = left_join(cl.seqs, PropMatrix0) %>% na.omit()


for ( i in 8 : ncol(cl.seqs)){
  
  if(!dir.exists("cluster_characterisation")){
    dir.create("cluster_characterisation")
  }
  
  col_by = cl.seqs[,i]
  nColor = length(levels(as.factor(col_by)))
  colors = paletteer_c("viridis::inferno", n = nColor)
  
  col_by = as.numeric(as.character(col_by))
  rank = as.factor( as.numeric(col_by))
  
  prop = colnames(PropMatrix0)[i]
  prop = str_replace_all(prop, coll("."), coll("_"))
  
  png(filename = paste0("cluster_characterisation/seq2vec_TFIDF_",
                        prop, ".png"),
      width = 2000, height = 2000, res = 300)
  
  plot(cl.seqs$UMAP1,
       cl.seqs$UMAP2,
       col = colors[ rank ],
       cex = 0.5,
       pch = cl.seqs$`cl$cluster` + 1,
       xlab = "UMAP 1", ylab = "UMAP 2",
       main = "hotspot regions: seq2vec + TFIDF",
       sub = paste0("colored by ", str_replace_all(prop, coll("_"), coll(" "))))
  
  dev.off()
  
  
}


########## combine with protein embeddings #############
prot.emb = read.csv("../RUNS/HumanProteome/word2vec_model/hp_sequence_repres_w5_d100_seq2vec-TFIDF.csv",
                    stringsAsFactors = F)

colnames(prot.emb)[4:103] = paste0("Y", seq(1, 100))
prot.emb$tokens = NULL

# concatenate with regions
dt$Accession = str_split_fixed(dt$Accession, "_", Inf)[,1]
master = left_join(dt, prot.emb) %>% na.omit()
master$seqs = NULL

master = cbind(cl$cluster, master)

UMAP.both = function(tbl = ""){
  set.seed(42)
  
  for (c in 1:ncol(tbl)){
    tbl[, c] = tbl[, c] %>% as.character() %>% as.numeric()
  }
  
  tbl$Accession = NULL
  
  coord = umap(tbl,
               n_neighbors = 5,
               min_dist = 0.8,
               n_epochs = 300,
               n_trees = 15,
               metric = "cosine",
               verbose = T,
               approx_pow = T,
               ret_model = T,
               init = "spca",
               n_threads = availableCores(),
               n_components = 3)
  
  um = data.frame(UMAP1 = coord$embedding[,1],
                  UMAP2 = coord$embedding[,2],
                  UMAP3 = coord$embedding[,3])
  
  return(um)
}



um = UMAP.both(master[, c(6:ncol(master))])

# plotting
{
  col_by = master$label
  nColor = length(levels(as.factor(col_by)))
  colors = paletteer_c("viridis::viridis", n = nColor)
  colors = c("darkturquoise", "firebrick1")
  
  rank = rep(NA, length(col_by))
  for (i in 1:length(col_by)){
    if (col_by[i] == "non_hotspot") { rank[i] = colors[1] } else { rank[i] = colors[2] }
  }
}


png(filename = "protAndRegion_seq2vec_TFIDF.png",
    width = 2000, height = 2000, res = 300)
plot(um$UMAP1, um$UMAP2,
     col = rank,
     cex = 0.3,
     pch = 1,
     xlab = "UMAP 1", ylab = "UMAP 2",
     main = "protein + region embedding: seq2vec + TFIDF",
     sub = "colored by hotspot/non-hotspot")
dev.off()

cl.seqs = cbind(cl$cluster, master[, !grep_weights(master)], um)

# biophys properties
PropMatrix0 = foreach(i=1:length(cl.seqs$region), 
                      .packages = c("Peptides")) %dopar% PepSummary(cl.seqs$region[i])
PropMatrix0 <- bind_rows(PropMatrix0)
PropMatrix0$region = rownames(PropMatrix0)

cl.seqs = left_join(cl.seqs, PropMatrix0) %>% na.omit()


for ( i in 8 : ncol(cl.seqs)){
  
  if(!dir.exists("cluster_characterisation")){
    dir.create("cluster_characterisation")
  }
  
  col_by = cl.seqs[,i]
  nColor = length(levels(as.factor(col_by)))
  colors = paletteer_c("viridis::inferno", n = nColor)
  
  col_by = as.numeric(as.character(col_by))
  rank = as.factor( as.numeric(col_by))
  
  prop = colnames(PropMatrix0)[i]
  prop = str_replace_all(prop, coll("."), coll("_"))
  
  png(filename = paste0("cluster_characterisation/protAndRegion_seq2vec_TFIDF_",
                        prop, ".png"),
      width = 2000, height = 2000, res = 300)
  
  plot(cl.seqs$UMAP1,
       cl.seqs$UMAP2,
       col = colors[ rank ],
       cex = 0.5,
       pch = cl.seqs$`cl$cluster` + 1,
       xlab = "UMAP 1", ylab = "UMAP 2",
       main = "protein + region embedding: seq2vec + TFIDF",
       sub = paste0("colored by ", str_replace_all(prop, coll("_"), coll(" "))))
  
  dev.off()
  
  
}

