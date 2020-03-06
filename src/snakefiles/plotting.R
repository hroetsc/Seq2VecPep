### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  dimension reduction and plotting
# input:        protein matrices
# output:       some nice plots
# author:       HR, PepSummary function from YH

# tmp!!!
protein.repres = read.csv(file = "results/embedded_proteome/proteome_repres.csv", stringsAsFactors = F, header = T)

print("### DIMENSION REDUCTION / PLOTTING ###")

library(seqinr)
library(protr)
library(Peptides)
library(plyr)
library(dplyr)
library(stringr)
#library(tidyr)
library(readr)
library(ggplot2)
library(ggthemes)
library(ggpubr)
library(uwot)
library(reshape2)

### INPUT ###
protein.repres = read.csv(file = snakemake@input[["proteome_repres"]], stringsAsFactors = F, header = T)
dim_range = c(which(colnames(protein.repres)=="TF_IDF_score")+1, ncol(protein.repres))
print(dim_range)
colnames(protein.repres)[c(dim_range[1]:dim_range[2])] = seq(1, (dim_range[2]-dim_range[1]+1))

### MAIN PART ###
print("APPLY UMAP TO EMBEDDINGS")
set.seed(42)
dims_UMAP = umap(protein.repres[,c(dim_range[1]:ncol(protein.repres))],
                 n_neighbors = 10,
                 min_dist = 0.01,
                 #spread = 2,
                 #n_trees = 50,
                 verbose = T,
                 approx_pow = T,
                 ret_model = T,
                 metric = "cosine",
                 scale = "none",
                 n_epochs = 500,
                 n_threads = 11)

umap_coords <- data.frame("X1"=dims_UMAP$embedding[,1],"X2"=dims_UMAP$embedding[,2])
proteinsUMAP <- cbind(umap_coords, protein.repres)

proteinsUMAP$shared_rPCP = as.numeric(as.character(proteinsUMAP$shared_rPCP))
UMAP_rPCP <- ggplot(proteinsUMAP, aes(x = X1, y = X2)) +
  geom_point(aes(fill = class, size = shared_rPCP, color = shared_rPCP),
             alpha = 0.85, pch=21, stroke = 1) +
  theme_few() +
  theme(text=element_text(family="sans", face="plain", color="#000000",
                          size=15, hjust=0.5, vjust=0.5))+
  scale_color_gradient2(aesthetics = "color",
                        low="blue", high="red", mid = "yellow",
                        name = "shared_rPCP") +
  scale_fill_manual(aesthetics = "fill", values = c("black", "white"), name = "class")+
  scale_radius(range = c(1,10), name = "shared_rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("embedded proteins")
UMAP_rPCP

# density plot
UMAP_rPCP_dens <- ggplot(proteinsUMAP, aes(x = X1, y = X2)) +
  geom_bin2d(bins=200) +
  scale_fill_continuous(type = "viridis") +
  theme_bw() +
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("embedded proteins - density plot")

### BIOPHYSICAL PROPERTIES ###
print("CALCULATE BIOCHEMICAL PROPERTIES OF PROTEOME")
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
# calculate biophysical properties of proteins
proteinsUMAP$seqs = as.character(proteinsUMAP$seqs)
a <- sapply(toupper(proteinsUMAP$seqs), protcheck)
names(a) <- NULL
proteinsUMAP = proteinsUMAP[which(a==T),]
PropMatrix = PepSummary(proteinsUMAP$seqs)
proteinsUMAP.Props = as.data.frame(cbind(proteinsUMAP, PropMatrix))
proteinsUMAP.Props = na.omit(proteinsUMAP.Props)

# does not work at the moment
# prop_plots = function(proteinsUMAP.Props = "", props = ""){
#   plots = list()
#   for (i in 1:10) {
#     p = ggplot(proteinsUMAP.Props, aes(x = proteinsUMAP.Props[,"X1"], y = proteinsUMAP.Props[,"X2"])) +
#       geom_point(aes(fill = proteinsUMAP.Props[,props[i]], size = proteinsUMAP.Props[,"shared_rPCP"], color = class), alpha = 0.85, pch=21, stroke = 1) +
#       theme_few() +
#       theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
#       scale_fill_gradient2(aesthetics = "fill",
#                            low="blue", high="red", mid = "yellow",
#                            name = as.character(props[i])) +
#       scale_radius(range = c(1,4), name = "shared_rPCP")+
#       xlab("UMAP 1") +
#       ylab("UMAP 2") +
#       ggtitle(paste0("embedded proteins, by ", as.character(props[i])))
#     p
#     png(paste0("./results/plots/", as.character(gsub("[.]", "_", props[i])), ".png"))
#     print(p)
#     dev.off()
#     
#     plots[[i]] = p
#   }
#   return(plots)
# }

print("GENERATE PLOTS")
# plot some particular properties
F6 = ggplot(proteinsUMAP.Props, aes(x = proteinsUMAP.Props[,"X1"], y = proteinsUMAP.Props[,"X2"])) +
  geom_point(aes(color = proteinsUMAP.Props[,"F6"], size = proteinsUMAP.Props[,"shared_rPCP"],
                 fill = class), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() +
  theme(text=element_text(family="sans", face="plain", color="#000000",
                          size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "color",
                       low="blue", high="red", mid = "yellow",
                       name = "F6") +
  scale_fill_manual(aesthetics = "fill", values = c("black", "white"), name = "class")+
  scale_radius(range = c(1,5), name = "shared_rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded proteins, by F6"))

Z3 = ggplot(proteinsUMAP.Props, aes(x = proteinsUMAP.Props[,"X1"], y = proteinsUMAP.Props[,"X2"])) +
  geom_point(aes(fill = class, size = proteinsUMAP.Props[,"shared_rPCP"],
                 color = proteinsUMAP.Props[,"Z3"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() +
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "color",
                       low="blue", high="red", mid = "yellow",
                       name = "Z3") +
  scale_fill_manual(aesthetics = "fill", values = c("black", "white"), name = "class")+
  scale_radius(range = c(1,10), name = "shared_rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded proteins, by Z3"))

BLOSUM1 = ggplot(proteinsUMAP.Props, aes(x = proteinsUMAP.Props[,"X1"], y = proteinsUMAP.Props[,"X2"])) +
  geom_point(aes(fill = class, size = proteinsUMAP.Props[,"shared_rPCP"],
                 color = proteinsUMAP.Props[,"BLOSUM1"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() +
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "color",
                       low="blue", high="red", mid = "yellow",
                       name = "BLOSUM1") +
  scale_fill_manual(aesthetics = "fill", values = c("black", "white"), name = "class")+
  scale_radius(range = c(1,10), name = "shared_rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded proteins, by BLOSUM1"))

charge = ggplot(proteinsUMAP.Props, aes(x = proteinsUMAP.Props[,"X1"], y = proteinsUMAP.Props[,"X2"])) +
  geom_point(aes(fill = class, size = proteinsUMAP.Props[,"shared_rPCP"],
                 color = proteinsUMAP.Props[,"charge"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() +
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "color",
                       low="blue", high="red", mid = "yellow",
                       name = "charge") +
  scale_fill_manual(aesthetics = "fill", values = c("black", "white"), name = "class")+
  scale_radius(range = c(1,10), name = "shared_rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded proteins, by charge"))

pI = ggplot(proteinsUMAP.Props, aes(x = proteinsUMAP.Props[,"X1"], y = proteinsUMAP.Props[,"X2"])) +
  geom_point(aes(fill = class, size = proteinsUMAP.Props[,"shared_rPCP"],
                 color = proteinsUMAP.Props[,"pI"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() +
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "color",
                       low="blue", high="red", mid = "yellow",
                       name = "pI") +
  scale_fill_manual(aesthetics = "fill", values = c("black", "white"), name = "class")+
  scale_radius(range = c(1,10), name = "shared_rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded proteins, by pI"))

Hydrophobicity = ggplot(proteinsUMAP.Props, aes(x = proteinsUMAP.Props[,"X1"], y = proteinsUMAP.Props[,"X2"])) +
  geom_point(aes(fill = class, size = proteinsUMAP.Props[,"shared_rPCP"],
                 color = proteinsUMAP.Props[,"Hydrophobicity"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() +
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "color",
                       low="blue", high="red", mid = "yellow",
                       name = "Hydrophobicity") +
  scale_fill_manual(aesthetics = "fill", values = c("black", "white"), name = "class")+
  scale_radius(range = c(1,10), name = "shared_rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded proteins, by Hydrophobicity"))

H_bonding = ggplot(proteinsUMAP.Props, aes(x = proteinsUMAP.Props[,"X1"], y = proteinsUMAP.Props[,"X2"])) +
  geom_point(aes(fill = class, size = proteinsUMAP.Props[,"shared_rPCP"],
                 color = proteinsUMAP.Props[,"H-bonding"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() +
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "color",
                       low="blue", high="red", mid = "yellow",
                       name = "H-bonding") +
  scale_fill_manual(aesthetics = "fill", values = c("black", "white"), name = "class")+
  scale_radius(range = c(1,10), name = "shared_rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded proteins, by H-bonding"))

Polarity = ggplot(proteinsUMAP.Props, aes(x = proteinsUMAP.Props[,"X1"], y = proteinsUMAP.Props[,"X2"])) +
  geom_point(aes(fill = class, size = proteinsUMAP.Props[,"shared_rPCP"],
                 color = proteinsUMAP.Props[,"Polarity"]), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() +
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_fill_gradient2(aesthetics = "color",
                       low="blue", high="red", mid = "yellow",
                       name = "Polarity") +
  scale_fill_manual(aesthetics = "fill", values = c("black", "white"), name = "class")+
  scale_radius(range = c(1,10), name = "shared_rPCP")+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle(paste0("embedded proteins, by Polarity"))

### OUTPUT ###
# proteins with biophysical properties
write.csv(proteinsUMAP.Props, file = unlist(snakemake@output[["proteome_props"]]))
# plots
ggsave(filename = unlist(snakemake@output[["p_rPCP"]]), plot = UMAP_rPCP, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["p_rPCP_dens"]]), plot = UMAP_rPCP_dens, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["p_F6"]]), plot = F6, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["p_Z3"]]), plot = Z3, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["p_BLOSUM1"]]), plot = BLOSUM1, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["p_charge"]]), plot = charge, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["p_pI"]]), plot = pI, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["p_hydrophobicity"]]), plot = Hydrophobicity, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["p_H_bonding"]]), plot = H_bonding, device = "png", dpi = "retina")
ggsave(filename = unlist(snakemake@output[["p_Polarity"]]), plot = Polarity, device = "png", dpi = "retina")
