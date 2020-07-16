### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  dimension of token embeddings
# input:        token weights
# output:       some nice plots
# author:       YH, HR

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(uwot)
library(ggraptR)
library(ggthemes)
#library(mixOmics)
library(Peptides)
library(caret)
library(GGally)
library(rhdf5)
library(doParallel)
library(future)

registerDoParallel(cores=availableCores())


### INPUT ###
indices = read.csv("../ids_hp_w5_new.csv", stringsAsFactors = F, header = F)
weight_matrix = h5read("hp_model_w5_d100/weights.h5", "/embedding/embedding")

if(! dir.exists("token_embeddings")){
  dir.create("token_embeddings")
}

### MAIN PART ###
########### preprocessing and biophys properties ###########
# assign tokens to weight matrix
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
}

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

# substitute selenocysteine with cysteine
U.idx = which(str_detect(weights$subword, "U"))
weights[U.idx, "subword"] = str_replace(weights[U.idx, "subword"],"U", "C")
# substitute X with glycine
X.idx = which(str_detect(weights$subword, "X"))
weights[X.idx, "subword"] = str_replace(weights[X.idx, "subword"],"X", "G")

# biophys properties
PropMatrix0 = foreach(i=1:length(weights$subword), 
                      .packages = c("Peptides")) %dopar% PepSummary(weights$subword[i])
PropMatrix0 <- bind_rows(PropMatrix0)
PropMatrix0$subword = rownames(PropMatrix0)

# return tokens
weights[U.idx, "subword"] = str_replace(weights[U.idx, "subword"],"C", "U")
weights[X.idx, "subword"] = str_replace(weights[X.idx, "subword"],"G", "X")


########### single amino acid tokens ###########

grep_weights = function(df = ""){
  return(which(grepl("^[0-9]{1,}$", colnames(df))))
}

single.aa = weights[which(nchar(weights$subword) == 1), ]

UMAP_single.aa = function(tbl = ""){
  
  for (c in 1:ncol(tbl)){
    tbl[, c] = tbl[, c] %>% as.character() %>% as.numeric()
  }
  
  set.seed(42)
  
  coord = uwot::umap(tbl,
                     init = "normlaplacian",
                     metric = "cosine", 
                     n_trees = 200,
                     n_epochs = 200,
                     n_neighbors = 2,
                     min_dist = 0.1, 
                     n_threads = 3, n_sgd_threads = 3,
                     approx_pow = T, 
                     ret_model = T, 
                     verbose = T,
                     ret_nn = T, 
                     n_components = 2)
  
  um = data.frame(UMAP1 = coord$embedding[, 1],
                  UMAP2 = coord$embedding[, 2])
  
  return(um)
  
}

um_single.aa = UMAP_single.aa(single.aa[, grep_weights(single.aa)])

# add metainformation
um_single.aa = cbind(single.aa$subword, um_single.aa)
colnames(um_single.aa)[1] = "subword"
um_single.aa = left_join(um_single.aa, PropMatrix0)
um_single.aa[is.na(um_single.aa)] = 0

# plot

{
# BLOSUM 10
s.aa = ggplot(um_single.aa, aes(UMAP1, UMAP2, color = BLOSUM9)) +
  geom_text(aes(label = subword, color = BLOSUM9), size=4,
            show.legend = T) +
  scale_color_viridis_c(option = "inferno") + 
  theme_bw()
s.aa
ggsave(filename = paste0("token_embeddings/singleAA_",
                         "BLOSUM9", ".png"),
       plot = s.aa, device = "png", dpi = "retina")

# Hydrophobicity
s.aa = ggplot(um_single.aa, aes(UMAP1, UMAP2, color = Hydrophobicity)) +
  geom_text(aes(label = subword, color = Hydrophobicity), size=4,
            show.legend = T) +
  scale_color_viridis_c(option = "inferno") + 
  theme_bw()
s.aa
ggsave(filename = paste0("token_embeddings/singleAA_",
                         "Hydrophobicity", ".png"),
       plot = s.aa, device = "png", dpi = "retina")

# Polarity
s.aa = ggplot(um_single.aa, aes(UMAP1, UMAP2, color = Polarity)) +
  geom_text(aes(label = subword, color = Polarity), size=4,
            show.legend = T) +
  scale_color_viridis_c(option = "inferno") + 
  theme_bw()
s.aa
ggsave(filename = paste0("token_embeddings/singleAA_",
                         "Polarity", ".png"),
       plot = s.aa, device = "png", dpi = "retina")

# H-bonding
s.aa = ggplot(um_single.aa, aes(UMAP1, UMAP2, color = `H-bonding`)) +
  geom_text(aes(label = subword, color = `H-bonding`), size=4,
            show.legend = T) +
  scale_color_viridis_c(option = "inferno") + 
  theme_bw()
s.aa
ggsave(filename = paste0("token_embeddings/singleAA_",
                         "H-bonding", ".png"),
       plot = s.aa, device = "png", dpi = "retina")

# mol.weight
s.aa = ggplot(um_single.aa, aes(UMAP1, UMAP2, color = mol.weight)) +
  geom_text(aes(label = subword, color = mol.weight), size=4,
            show.legend = T) +
  scale_color_viridis_c(option = "inferno") + 
  theme_bw()
s.aa
ggsave(filename = paste0("token_embeddings/singleAA_",
                         "mol_weight", ".png"),
       plot = s.aa, device = "png", dpi = "retina")

# Aliphatic.number
s.aa = ggplot(um_single.aa, aes(UMAP1, UMAP2, color = Aliphatic.number)) +
  geom_text(aes(label = subword, color = Aliphatic.number), size=4,
            show.legend = T) +
  scale_color_viridis_c(option = "inferno") + 
  theme_bw()
s.aa
ggsave(filename = paste0("token_embeddings/singleAA_",
                         "Aliphatic_number", ".png"),
       plot = s.aa, device = "png", dpi = "retina")

# Aromatic.number
s.aa = ggplot(um_single.aa, aes(UMAP1, UMAP2, color = Aromatic.number)) +
  geom_text(aes(label = subword, color = Aromatic.number), size=4,
            show.legend = T) +
  scale_color_viridis_c(option = "inferno") + 
  theme_bw()
s.aa
ggsave(filename = paste0("token_embeddings/singleAA_",
                         "Aromatic_number", ".png"),
       plot = s.aa, device = "png", dpi = "retina")

# Acidic.number
s.aa = ggplot(um_single.aa, aes(UMAP1, UMAP2, color = Acidic.number)) +
  geom_text(aes(label = subword, color = Acidic.number), size=4,
            show.legend = T) +
  scale_color_viridis_c(option = "inferno") + 
  theme_bw()
s.aa
ggsave(filename = paste0("token_embeddings/singleAA_",
                         "Acidic_number", ".png"),
       plot = s.aa, device = "png", dpi = "retina")
}

########### all tokens ###########

UMAP_all.tokens = function(tbl = ""){
  
  for (c in 1:ncol(tbl)){
    tbl[, c] = tbl[, c] %>% as.character() %>% as.numeric()
  }
  
  set.seed(42)
  
  coord = uwot::umap(tbl,
                     init = "normlaplacian",
                     metric = "euclidean", 
                     n_trees = 100, n_epochs = 500,
                     n_neighbors = 3,
                     min_dist = 0.1, 
                     n_threads = 3, n_sgd_threads = 3,
                     approx_pow = T, 
                     ret_model = T, 
                     verbose = T,
                     ret_nn = T, 
                     n_components = 2)
  
  um = data.frame(UMAP1 = coord$embedding[, 1],
                  UMAP2 = coord$embedding[, 2])
  
  return(um)
  
}

um_all.tokens = UMAP_all.tokens(weights[, grep_weights(weights)])

um_all.tokens = cbind(weights$subword, um_all.tokens)
colnames(um_all.tokens)[1] = "subword"
um_all.tokens = left_join(um_all.tokens, PropMatrix0)
um_all.tokens[is.na(um_all.tokens)] = 0

# plot

{
  # protein.length
  s.aa = ggplot(um_all.tokens, aes(UMAP1, UMAP2, color = protein.length)) +
    geom_text(aes(label = subword, color = protein.length), size=1,
              show.legend = T) +
    scale_color_viridis_c(option = "inferno") + 
    theme_bw()
  s.aa
  ggsave(filename = paste0("token_embeddings/allTokens_",
                           "protein_length", ".png"),
         plot = s.aa, device = "png", dpi = 600,
         height = 8.2, width = 12.84)
  
  # BLOSUM2
  s.aa = ggplot(um_all.tokens, aes(UMAP1, UMAP2, color = BLOSUM2)) +
    geom_text(aes(label = subword, color = BLOSUM2), size=1,
              show.legend = T) +
    scale_color_viridis_c(option = "inferno") + 
    theme_bw()
  s.aa
  ggsave(filename = paste0("token_embeddings/allTokens_",
                           "BLOSUM2", ".png"),
         plot = s.aa, device = "png", dpi = 600,
         height = 8.2, width = 12.84)
  
  # Hydrophobicity
  s.aa = ggplot(um_all.tokens, aes(UMAP1, UMAP2, color = Hydrophobicity)) +
    geom_text(aes(label = subword, color = Hydrophobicity), size = 1,
              show.legend = T) +
    scale_color_viridis_c(option = "inferno") + 
    theme_bw()
  s.aa
  ggsave(filename = paste0("token_embeddings/allTokens_",
                           "Hydrophobicity", ".png"),
         plot = s.aa, device = "png", dpi = 600,
         height = 8.2, width = 12.84)
  
  # Polarity
  s.aa = ggplot(um_all.tokens, aes(UMAP1, UMAP2, color = Polarity)) +
    geom_text(aes(label = subword, color = Polarity), size = 1,
              show.legend = T) +
    scale_color_viridis_c(option = "inferno") + 
    theme_bw()
  s.aa
  ggsave(filename = paste0("token_embeddings/allTokens_",
                           "Polarity", ".png"),
         plot = s.aa, device = "png", dpi = 600,
         height = 8.2, width = 12.84)
  
  # H-bonding
  s.aa = ggplot(um_all.tokens, aes(UMAP1, UMAP2, color = `H-bonding`)) +
    geom_text(aes(label = subword, color = `H-bonding`), size = 1,
              show.legend = T) +
    scale_color_viridis_c(option = "inferno") + 
    theme_bw()
  s.aa
  ggsave(filename = paste0("token_embeddings/allTokens_",
                           "H-bonding", ".png"),
         plot = s.aa, device = "png", dpi = 600,
         height = 8.2, width = 12.84)
  
  # mol.weight
  s.aa = ggplot(um_all.tokens, aes(UMAP1, UMAP2, color = mol.weight)) +
    geom_text(aes(label = subword, color = mol.weight), size = 1,
              show.legend = T) +
    scale_color_viridis_c(option = "inferno") + 
    theme_bw()
  s.aa
  ggsave(filename = paste0("token_embeddings/allTokens_",
                           "mol_weight", ".png"),
         plot = s.aa, device = "png", dpi = 600,
         height = 8.2, width = 12.84)
  
  # Aliphatic.number
  s.aa = ggplot(um_all.tokens, aes(UMAP1, UMAP2, color = Aliphatic.number)) +
    geom_text(aes(label = subword, color = Aliphatic.number), size = 1,
              show.legend = T) +
    scale_color_viridis_c(option = "inferno") + 
    theme_bw()
  s.aa
  ggsave(filename = paste0("token_embeddings/allTokens_",
                           "Aliphatic_number", ".png"),
         plot = s.aa, device = "png", dpi = 600,
         height = 8.2, width = 12.84)
  
  # Aromatic.number
  s.aa = ggplot(um_all.tokens, aes(UMAP1, UMAP2, color = Aromatic.number)) +
    geom_text(aes(label = subword, color = Aromatic.number), size = 1,
              show.legend = T) +
    scale_color_viridis_c(option = "inferno") + 
    theme_bw()
  s.aa
  ggsave(filename = paste0("token_embeddings/allTokens_",
                           "Aromatic_number", ".png"),
         plot = s.aa, device = "png", dpi = 600,
         height = 8.2, width = 12.84)
  
  # Acidic.number
  s.aa = ggplot(um_all.tokens, aes(UMAP1, UMAP2, color = Acidic.number)) +
    geom_text(aes(label = subword, color = Acidic.number), size = 1,
              show.legend = T) +
    scale_color_viridis_c(option = "inferno") + 
    theme_bw()
  s.aa
  ggsave(filename = paste0("token_embeddings/allTokens_",
                           "Acidic_number", ".png"),
         plot = s.aa, device = "png", dpi = 600,
         height = 8.2, width = 12.84)
}


# 3D scatterplot
UMAP_3d <- function(tbl = "") {
  
  for (c in 1:ncol(tbl)){
    tbl[, c] = tbl[, c] %>% as.character() %>% as.numeric()
  }
  
  set.seed(42)
  
  coord = umap(tbl,
             n_components = 3,
            init = "spca",
            metric = "cosine",
            n_neighbors = 3,
            n_trees = 500,
            n_epochs = 500,
            approx_pow = T,
            ret_model = T,
            verbose = T,
            min_dist = 1,
            ret_nn = T)
  
  um = data.frame(UMAP1 = coord$embedding[, 1],
                  UMAP2 = coord$embedding[, 2],
                  UMAP3 = coord$embedding[, 3])
  
  return(um)
}

um_3d = UMAP_3d(weights[, grep_weights(weights)])

um_3d = cbind(weights$subword, um_3d)
colnames(um_3d)[1] = "subword"
um_3d = left_join(um_3d, PropMatrix0)
um_3d[is.na(um_3d)] = 0

library(scatterplot3d)
library(car)
library(rgl)
library(paletteer)


fig = scatterplot3d(um_3d[, c("UMAP1", "UMAP2", "UMAP3")],
                    color = um_3d$protein.length)
{
# protein.length
col_by = um_3d$protein.length
nColor = length(levels(as.factor(col_by)))
colors = paletteer_c("viridis::inferno", n = nColor)
col_by = as.numeric(as.character(col_by))
rank = as.factor( as.numeric( cut(col_by, nColor) ))

scatter3d(x = um_3d$UMAP1,
          y = um_3d$UMAP2,
          z = um_3d$UMAP3,
          surface = F,
          point.col = colors [ rank ])
rgl.postscript("token_embeddings/3d_protein_length.ps",fmt="ps")


# BLOSUM 2
col_by = um_3d$BLOSUM2
nColor = length(levels(as.factor(col_by)))
colors = paletteer_c("viridis::inferno", n = nColor)
col_by = as.numeric(as.character(col_by))
rank = as.factor( as.numeric( cut(col_by, nColor) ))

scatter3d(x = um_3d$UMAP1,
          y = um_3d$UMAP2,
          z = um_3d$UMAP3,
          surface = F,
          point.col = colors [ rank ])
rgl.postscript("token_embeddings/3d_BLOSUM2.ps",fmt="ps")


# Hydrophobicity
col_by = um_3d$Hydrophobicity
nColor = length(levels(as.factor(col_by)))
colors = paletteer_c("viridis::inferno", n = nColor)
col_by = as.numeric(as.character(col_by))
rank = as.factor( as.numeric( cut(col_by, nColor) ))

scatter3d(x = um_3d$UMAP1,
          y = um_3d$UMAP2,
          z = um_3d$UMAP3,
          surface = F,
          point.col = colors [ rank ])
rgl.postscript("token_embeddings/3d_Hydrophobicity.ps",fmt="ps")
}