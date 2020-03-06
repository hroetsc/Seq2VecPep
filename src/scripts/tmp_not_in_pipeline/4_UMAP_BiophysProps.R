### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  apply UMAP to embedded tokens and visualize them by biophysical properties
# input:        weight matrix from the 3_word2vec.py script and their corresponding IDs (subwords)
# output:       some nice figures
# author:       HR, YH
setwd("/home/hroetsc/Documents/ProtTransEmbedding")

library(seqinr)
library(protr)
library(Peptides)
#library(rattle)
library(dplyr)
library(stringr)
library(tidyr)
library(plyr)
library(dplyr)
#library(tximport)
library(readr)
#library(biomaRt)
library(ggplot2)
library(ggthemes)
library(gridExtra)
library(uwot)

### INPUT ###
# encoded antigens
load("./seq2vec/source_antigens_encoded.RData")
# weight matrix
weights = read.csv("./seq2vec/word2vec_weights_batch_rdn.csv", stringsAsFactors = F, header = F)
weights$V1 = NULL
# corresponding subwords
indices = read.csv("./seq2vec/word2vec_subwords_batch_rdn.csv", stringsAsFactors = F, header = F)
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

# UMAP
rPCP = weights$rPCP
weights$rPCP = NULL
set.seed(42)
seq2vec_UMAP = umap(weights,
                    n_neighbors = 100,
                    min_dist = 0.00,
                    spread = 2,
                    n_trees = 100,
                    a = 10,
                    b = 0.9,
                    verbose = T,
                    approx_pow = T,
                    ret_model = T,
                    metric = list("categorical" = c("subword"),
                                  "cosine" = colnames(weights)[2:ncol(weights)]),
                    scale = "none",
                    n_epochs = 300,
                    n_threads = 11)

# plotting
umap_coords <- data.frame("X1"=seq2vec_UMAP$embedding[,1],"X2"=seq2vec_UMAP$embedding[,2])
weightsUMAP <- cbind(umap_coords, weights, rPCP)

UMAP_rPCP <- ggplot(weightsUMAP, aes(x = X1, y = X2)) + 
  geom_point(aes(fill = rPCP, size = rPCP), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_color_gradient2(aesthetics = "fill", midpoint=median(weightsUMAP$rPCP), low="red", mid="grey80",high="green") +
  scale_radius(range = c(1,4))+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("segmented embedded source antigens")
UMAP_rPCP

# add biophysical properties
# function that calculates biophysical properties of given protein sequence
PepSummary2 <- function(Peptides.input) {
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
# calculate them
weightsUMAP$subword <- as.character(weightsUMAP$subword)
a <- sapply(toupper(weightsUMAP$subword), protcheck)
names(a) <- NULL
weightsUMAP <- weightsUMAP[which(a == T),]
PropMatrix = PepSummary2(weightsUMAP$subword)
# concatenate weights and properties
Weights.and.Props = as.data.frame(cbind(weightsUMAP, PropMatrix))

# plotting
umap_plot_Hbond <- ggplot(Weights.and.Props, aes(x = X1, y = X2)) + 
  geom_point(aes(fill = `H-bonding`, size = rPCP), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_color_gradient2(aesthetics = "fill", midpoint=median(Weights.and.Props$`H-bonding`), low="red", mid="grey80",high="green") +
  scale_radius(range = c(1,4))+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("IP in low dimensions, by hydrogen bonding")

umap_plot_mol.weight <- ggplot(Weights.and.Props, aes(x = X1, y = X2)) + 
  geom_point(aes(fill = mol.weight, size = rPCP), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_color_gradient2(aesthetics = "fill", midpoint=median(Weights.and.Props$mol.weight), low="red", mid="grey80",high="green") +
  scale_radius(range = c(1,4))+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("IP in low dimensions, by molecular weight")

umap_plot_hydrophobicity <- ggplot(Weights.and.Props, aes(x = X1, y = X2)) + 
  geom_point(aes(fill = Hydrophobicity, size = rPCP), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_color_gradient2(aesthetics = "fill", midpoint=median(Weights.and.Props$Hydrophobicity), low="red", mid="grey80",high="green") +
  scale_radius(range = c(1,4))+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("IP in low dimensions, by hydrophobicity")

umap_plot_polarity <- ggplot(Weights.and.Props, aes(x = X1, y = X2)) + 
  geom_point(aes(fill = Polarity, size = rPCP), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_color_gradient2(aesthetics = "fill", midpoint=median(Weights.and.Props$Polarity), low="red", mid="grey80",high="green") +
  scale_radius(range = c(1,4))+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("IP in low dimensions, by polarity")

umap_plot_Z3 <- ggplot(Weights.and.Props, aes(x = X1, y = X2)) + 
  geom_point(aes(fill = Z3, size = rPCP), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_color_gradient2(aesthetics = "fill", midpoint=median(Weights.and.Props$Z3), low="red", mid="grey80",high="green") +
  scale_radius(range = c(1,4))+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("IP in low dimensions, by Z3")

umap_plot_length <- ggplot(Weights.and.Props, aes(x = X1, y = X2)) + 
  geom_point(aes(fill = protein.length, size = rPCP), alpha = 0.85, pch=21, stroke = 1) +
  theme_few() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5))+
  scale_color_gradient2(aesthetics = "fill", midpoint=median(Weights.and.Props$protein.length), low="red", mid="grey80",high="green") +
  scale_radius(range = c(1,4))+
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  ggtitle("IP in low dimensions, by protein length")

umap_plot_Hbond
umap_plot_mol.weight
umap_plot_hydrophobicity
umap_plot_polarity
umap_plot_Z3
umap_plot_length

UMAP_props = grid.arrange(umap_plot_Hbond, umap_plot_mol.weight, umap_plot_hydrophobicity,
                          umap_plot_polarity, umap_plot_Z3, umap_plot_length,
                          ncol=3, nrow=2)

### OUTPUT ###
# weights with rPCP
save(file = "./seq2vec/weights_subwords_rPCP.RData", weights)
# weights with rPCP and UMAP
save(file = "./seq2vec/weights_UMAP.RData", weightsUMAP)
# weights with numerical biophysical properties
save(file = "./seq2vec/weights_and_properties.RData", Weights.and.Props)
write.csv(Weights.and.Props, "./seq2vec/weights_and_properties.csv")

# plots
ggsave(filename = "./seq2vec/embedding_batch.png", plot = UMAP_rPCP, device = "png", dpi = "retina")
ggsave(filename = "./seq2vec/embedding_Hbond.png", plot = umap_plot_Hbond, device = "png", dpi = "retina")
ggsave(filename = "./seq2vec/embedding_molWeight.png", plot = umap_plot_mol.weight, device = "png", dpi = "retina")
ggsave(filename = "./seq2vec/embedding_hydrophobicity.png", plot = umap_plot_hydrophobicity, device = "png", dpi = "retina")
ggsave(filename = "./seq2vec/embedding_polarity.png", plot = umap_plot_polarity, device = "png", dpi = "retina")
ggsave(filename = "./seq2vec/embedding_Z3.png", plot = umap_plot_Z3, device = "png", dpi = "retina")
ggsave(filename = "./seq2vec/embedding_length.png", plot = umap_plot_length, device = "png", dpi = "retina")
ggsave(filename = "./seq2vec/embedding_props.png", plot = UMAP_props, device = "png", dpi = "retina")
