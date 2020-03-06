### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  concatenate embedded tokens to numerical antigen matrix
# input:        weight matrix from the 3_word2vec.py script, encoded antigens table
# output:       antigen matrices
# author:       HR, parts from YH
setwd("/home/hroetsc/Documents/ProtTransEmbedding/Snakemake/")

library(seqinr)
library(protr)
library(Peptides)
library(dplyr)
library(stringr)
library(tidyr)
library(plyr)
library(dplyr)
library(readr)
library(ggplot2)
library(ggthemes)
library(ggpubr)
library(uwot)
library(reshape2)

### INPUT ###
# tmp!!!!!!!!!!!
antigens.Encoded = read.csv(snakemake@input[["encoded_antigens"]], stringsAsFactors = F, header = T)
antigens.Encoded = na.omit(antigens.Encoded)
# weight matrix
weights = read.csv(snakemake@input[["weights"]], stringsAsFactors = F, header = F)
weights$V1 = NULL
# corresponding subwords
indices = read.csv(snakemake@input[["ids"]], stringsAsFactors = F, header = F)
indices$V1 = NULL
indices[nrow(indices)+1, 1] = ""
# merge indices and weights
weights = cbind(indices, weights)
colnames(weights)[1] = "subword"
weights = na.omit(weights)
weights = unique(weights)

### MAIN PART ###
# add rPCP
antigens.Encoded[,"subword"] = antigens.Encoded$SegmentedSeq
antigens.Encoded$SegmentedSeq = NULL
weights$subword = toupper(weights$subword)
weights = inner_join(antigens.Encoded[,c("subword", "rPCP")], weights)

# convert weights into numeric
for (p in 2:ncol(weights)){
  weights[,p] = as.numeric(as.character(weights[,p]))
}
weights = na.omit(weights)
head(weights)

# add seq vectors to antigens.Encoded
# memory saving (but slower) way to merge the dataframes
antigens.Encoded.list = split.data.frame(antigens.Encoded, antigens.Encoded$UniProtID)
antigens.Encoded.weights = list()

for (i in 1:length(antigens.Encoded.list)) {
  tmp = inner_join(antigens.Encoded.list[[i]], weights)
  antigens.Encoded.weights[[i]] = tmp
}
antigens.Encoded.weights = ldply(antigens.Encoded.weights, rbind)

# find tokens for every unique antigen (UniProtID) and sum up all dimensions of the tokens
# to get the respective embedding dimension
UniProtIDs = unique(as.character(antigens.Encoded.weights$UniProtID))
progressBar = txtProgressBar(min = 0, max = length(UniProtIDs), style = 3)

antigen.repres = matrix(ncol = ncol(antigens.Encoded.weights), nrow = length(UniProtIDs)) # contains vector representation for every antigen
colnames(antigen.repres) = colnames(antigens.Encoded.weights)
dim_range = c(ncol(antigens.Encoded)+1, ncol(antigen.repres))

for (u in 1:length(UniProtIDs)) {
  setTxtProgressBar(progressBar, u)
  # get tokens for current antigen
  tmp = antigens.Encoded.weights[which(antigens.Encoded.weights$UniProtID == UniProtIDs[u]),]
  # only take the 1st accession (some antigens occur within different sets of accessions)
  if (length(levels(as.factor(tmp$Accession))) > 1) {
    tmp = tmp %>% group_split(Accession)
    tmp = as.data.frame(tmp[[1]])
  }
  for (c in seq(dim_range[1], dim_range[2])) {
    tmp[,c] = as.numeric(tmp[,c])
  }
  # calculate colsums
  c_sum = colSums(tmp[,c(dim_range[1]:dim_range[2])])
  c_sum = c_sum / nrow(tmp)
  # add to df
  ln = antigens.Encoded.weights[which(antigens.Encoded.weights$UniProtID == UniProtIDs[u])[1],c(1:dim_range[1]-1)]
  antigen.repres[u, seq(1,dim_range[1]-1)] = as.character(ln[1,])
  antigen.repres[u, c(dim_range[1]:dim_range[2])] = c_sum
}

### EVALUATION ###
# get statistics summary
for (c in 1:nrow(antigen.repres)) {
  ln = as.numeric(antigen.repres[c, c(dim_range[1]:ncol(antigen.repres))])
  print(summary(ln))
  dens = density(ln)
  plot(dens, main = paste0("antigen: ", antigen.repres[c,"UniProtID"]))
}
antigen.repres = as.data.frame(antigen.repres)
antigen.repres = na.omit(antigen.repres)
antigen.repres = unique(antigen.repres)
for (p in dim_range[1]:ncol(antigen.repres)){
  antigen.repres[,p] = as.numeric(as.character(antigen.repres[,p]))
}

### VISUALIZATION ###
set.seed(42)
dims_UMAP = umap(antigen.repres,
                 n_neighbors = 4,
                 min_dist = 0.01,
                 #spread = 2,
                 #n_trees = 50,
                 verbose = T,
                 approx_pow = T,
                 ret_model = T,
                 metric = list("categorical" = colnames(antigen.repres[,c(1:dim_range[1]-1)]),
                               "cosine" = colnames(antigen.repres[,c(dim_range[1]:ncol(antigen.repres))])),
                 scale = "none",
                 n_epochs = 500,
                 n_threads = 11)

antigen.repres$subword = NULL # to avoid confusion
umap_coords <- data.frame("X1"=dims_UMAP$embedding[,1],"X2"=dims_UMAP$embedding[,2])
antigensUMAP <- cbind(umap_coords, antigen.repres)

antigensUMAP$rPCP = as.numeric(as.character(antigensUMAP$rPCP))
antigensUMAP$rPCP = 10^(-antigensUMAP$rPCP) # convert rPCP back
UMAP_rPCP <- ggplot(antigensUMAP, aes(x = X1, y = X2)) + 
  geom_point(aes(fill = rPCP, size = rPCP), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_color_gradient2(aesthetics = "fill", midpoint=median(antigensUMAP$rPCP), low="red", mid="grey80",high="green") +
  scale_radius(range = c(1,3))+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("embedded source antigens")
UMAP_rPCP

### BIOPHYSICAL PROPERTIES ###
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
# calculate biophysical properties of antigens
antigensUMAP$antigenSeq = as.character(antigensUMAP$antigenSeq)
a <- sapply(toupper(antigensUMAP$antigenSeq), protcheck)
names(a) <- NULL
antigensUMAP = antigensUMAP[which(a==T),]
PropMatrix = PepSummary(antigensUMAP$antigenSeq)
antigensUMAP.Props = as.data.frame(cbind(antigensUMAP, PropMatrix))
antigensUMAP.Props = na.omit(antigensUMAP.Props)

# does not work at the moment
prop_plots = function(df = "", props = ""){
  plots = list()
  for (i in 1:10) {
    p = ggplot(df, aes(x = df[,"X1"], y = df[,"X2"])) +
      geom_point(aes(fill = df[,props[i]], size = df[,"rPCP"]), alpha = 0.85, pch=21, stroke = 1) +
      theme_few() + 
      theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
      scale_fill_gradient2(aesthetics = "fill",
                            low="blue", high="red", mid = "yellow",
                            name = as.character(props[i])) +
      scale_radius(range = c(1,4), name = "rPCP")+
      xlab("UMAP 1") +
      ylab("UMAP 2") +
      ggtitle(paste0("embedded source antigens, by ", as.character(props[i])))
    p
    png(paste0("./results/plots/", as.character(gsub("[.]", "_", props[i])), ".png"))
    print(p)
    dev.off()
    
    plots[[i]] = p
  }
  return(plots)
}
prop_plots(df = antigensUMAP.Props, props = colnames(PropMatrix))

# plot some particular properties
df = antigensUMAP.Props
F6 = ggplot(df, aes(x = df[,"X1"], y = df[,"X2"])) +
  geom_point(aes(fill = df[,"F6"], size = df[,"rPCP"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "fill",
                       low="blue", high="red", mid = "yellow",
                       name = "F6") +
  scale_radius(range = c(1,5), name = "rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded source antigens, by F6"))

Z3 = ggplot(df, aes(x = df[,"X1"], y = df[,"X2"])) +
  geom_point(aes(fill = df[,"Z3"], size = df[,"rPCP"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "fill",
                       low="blue", high="red", mid = "yellow",
                       name = "Z3") +
  scale_radius(range = c(1,5), name = "rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded source antigens, by Z3"))

BLOSUM1 = ggplot(df, aes(x = df[,"X1"], y = df[,"X2"])) +
  geom_point(aes(fill = df[,"BLOSUM1"], size = df[,"rPCP"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "fill",
                       low="blue", high="red", mid = "yellow",
                       name = "BLOSUM1") +
  scale_radius(range = c(1,5), name = "rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded source antigens, by BLOSUM1"))

charge = ggplot(df, aes(x = df[,"X1"], y = df[,"X2"])) +
  geom_point(aes(fill = df[,"charge"], size = df[,"rPCP"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "fill",
                       low="blue", high="red", mid = "yellow",
                       name = "charge") +
  scale_radius(range = c(1,5), name = "rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded source antigens, by charge"))

pI = ggplot(df, aes(x = df[,"X1"], y = df[,"X2"])) +
  geom_point(aes(fill = df[,"pI"], size = df[,"rPCP"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "fill",
                       low="blue", high="red", mid = "yellow",
                       name = "pI") +
  scale_radius(range = c(1,5), name = "rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded source antigens, by pI"))

Hydrophobicity = ggplot(df, aes(x = df[,"X1"], y = df[,"X2"])) +
  geom_point(aes(fill = df[,"Hydrophobicity"], size = df[,"rPCP"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "fill",
                       low="blue", high="red", mid = "yellow",
                       name = "Hydrophobicity") +
  scale_radius(range = c(1,5), name = "rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded source antigens, by Hydrophobicity"))

H_bonding = ggplot(df, aes(x = df[,"X1"], y = df[,"X2"])) +
  geom_point(aes(fill = df[,"H-bonding"], size = df[,"rPCP"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "fill",
                       low="blue", high="red", mid = "yellow",
                       name = "H-bonding") +
  scale_radius(range = c(1,5), name = "rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded source antigens, by H-bonding"))

Polarity = ggplot(df, aes(x = df[,"X1"], y = df[,"X2"])) +
  geom_point(aes(fill = df[,"Polarity"], size = df[,"rPCP"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "fill",
                       low="blue", high="red", mid = "yellow",
                       name = "Polarity") +
  scale_radius(range = c(1,5), name = "rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded source antigens, by Polarity"))

p1 = ggarrange(UMAP_rPCP, F6, Z3, ncol = 2, nrow = 2)
p2 = ggarrange(UMAP_rPCP, BLOSUM1, charge, pI, ncol = 2, nrow = 2)
p3 = ggarrange(UMAP_rPCP, Hydrophobicity, H_bonding, Polarity, ncol = 2, nrow = 2)

### OUTPUT ###
# merged antigen table and token vectors
write.csv(antigens.Encoded.weights, snakemake@output[["antigen_weights"]])
# vector representation of antigens
write.csv(antigen.repres, snakemake@output[["antigen_repres"]])
# antigens with biophysical properties
write.csv(antigensUMAP.Props, snakemake@output[["antigen_props"]])
# plots
ggsave(filename = snakemake@output[["p_rPCP"]], plot = UMAP_rPCP, device = "png", dpi = "retina")
ggsave(filename = snakemake@output[["p_F6"]], plot = F6, device = "png", dpi = "retina")
ggsave(filename = snakemake@output[["p_Z3"]], plot = Z3, device = "png", dpi = "retina")
ggsave(filename = snakemake@output[["p_BLOSUM1"]], plot = BLOSUM1, device = "png", dpi = "retina")
ggsave(filename = snakemake@output[["p_charge"]], plot = charge, device = "png", dpi = "retina")
ggsave(filename = snakemake@output[["p_pI"]], plot = pI, device = "png", dpi = "retina")
ggsave(filename = snakemake@output[["p_hydrophobicity"]], plot = Hydrophobicity, device = "png", dpi = "retina")
ggsave(filename = snakemake@output[["p_H_bonding"]], plot = H_bonding, device = "png", dpi = "retina")
ggsave(filename = snakemake@output[["p_Polarity"]], plot = Polarity, device = "png", dpi = "retina")