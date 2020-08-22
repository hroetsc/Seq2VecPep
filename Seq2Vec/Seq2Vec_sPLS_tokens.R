# Create sPLS representation of seq2vec data
library(readr)
library(stringr)
library(tidyr)
library(uwot)
library(ggraptR)
library(ggthemes)
library(mixOmics)
library(Peptides)
library(caret)
library(GGally)
library(doParallel)
registerDoParallel(cores=3)

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

# Load seq2vec tokens
sv.5 <- read_csv("R_data/token_weights_w5_d100.csv")
sv.3 <- read_csv("R_data/token_weights_w3_d100.csv")

tokens <- c("word_ID","subword")
X.5 <- sv.5[,!colnames(sv.5) %in% tokens]
X.3 <- sv.3[,!colnames(sv.3) %in% tokens]

tokens <- sv.5[,colnames(sv.5) %in% tokens]
tokens.special <- tokens$subword %>%
  str_replace_all(pattern = "[\\ACDEFGHIKLMNPQRSTVWY]", replacement = "") %>%
  nchar()
tokens.special <- tokens[which(tokens.special > 0),]
print(tokens.special)

# Substitute selenocysteine with cysteine
tokens$subword[c(4985)] <- c("C")

# Substitute gap token with glycine
tokens$subword[c(4996)] <- c("G")

# biophys properties
PropMatrix0 = foreach(i=1:length(tokens$subword), 
                      .packages = c("Peptides")) %dopar% PepSummary(tokens$subword[i])
PropMatrix0 <- bind_rows(PropMatrix0)

# PropMatrix0$log10.mol.weight <- log10(PropMatrix0$mol.weight)
# PropMatrix0$log10.charge <- c(log10(abs(PropMatrix0$charge)) * sign(PropMatrix0$charge))
# PropMatrix0 <- na.omit(PropMatrix0)

keep = colnames(PropMatrix0)[19:ncol(PropMatrix0)]
keep <- keep[!keep %in% c( "mol.weight", "charge")]
PropMatrix <- PropMatrix0[,keep] 
PropMatrix <- PropMatrix0[,keep]  %>%
  na.omit() %>% 
  data.matrix()

dim(PropMatrix)

# Return selenocysteine and gap token 
tokens$subword[c(4985)] <- tokens.special$subword[c(1)]
tokens$subword[c(4996)] <- tokens.special$subword[c(2)]

# nzv <- nearZeroVar(PropMatrix, saveMetrics = T)
# nzv
# nzv <- nearZeroVar(PropMatrix, saveMetrics = F)
# filteredDescr <- PropMatrix[, -nzv]

comboInfo <- caret::findLinearCombos(PropMatrix)
colnames(PropMatrix)[comboInfo$remove]
PropMatrix <- PropMatrix[, -comboInfo$remove]

descrCor <- cor(PropMatrix, method = "spearman")
summary(descrCor[upper.tri(descrCor)])

ggcorr(PropMatrix,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
ggcorr(X.5,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
ggcorr(X.3,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
ggcorr(c(X.3, X.5, PropMatrix0),  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
ggcorr(c(X.3, X.5, as.data.frame(PropMatrix)),  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")

cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_token_human_05_2020\\corr_bioph.pdf", width = 10, height = 5)
ggcorr(PropMatrix,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_token_human_05_2020\\corr_sv_X5.pdf", width = 10, height = 5)
ggcorr(X.5,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_token_human_05_2020\\corr_sv_X3.pdf", width = 10, height = 5)
ggcorr(X.3,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_token_human_05_2020\\corr_svX3X5bioph.pdf", width = 10, height = 5)
ggcorr(c(X.3, X.5, PropMatrix0),  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_token_human_05_2020\\corr_svX3X5bioph_all.pdf", width = 10, height = 5)
ggcorr(c(X.3, X.5, as.data.frame(PropMatrix)),  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()

#-------------------- Plot seq2vec --------------------
summary(X.5[upper.tri(X.5)])
summary(X.3[upper.tri(X.3)])

X <- cbind.data.frame(X.3, X.5)
colnames(X) <- paste0(rep(c("X.3","X.5"),each=100), c(colnames(X.3),colnames(X.5)))

# Plot umap embeddings of raw seq2vec
u.pca <- X %>% as_tibble() %>% spca(ncomp = 2)
u.pca$variates$X %>% head()
u.pca.100d <- X %>% as_tibble() %>% ipca(ncomp = 170)
u.pca.100d$explained_variance %>% 
  cumsum() %>%
  plot(type = "s", 
       main = "Cumulative sum of variance explained by PCA", 
       sub = "sparse PCA to perform variable selection by using Singular Value Decomposition",
       ylab = "Variance explained", 
       xlab = "N Principal Components", ylim = c(0,1))
abline(h = 0.9, col = "red")
text(1,0.9, "90% variance", col = "red", adj = c(-.1, -.12))

{
  cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_token_human_05_2020\\sPCA_var_explained.pdf", width = 10, height = 10)
  u.pca.100d$explained_variance %>% 
    cumsum() %>%
    plot(type = "s", 
         main = "Cumulative sum of variance explained by PCA", 
         sub = "sparse PCA to perform variable selection by using Singular Value Decomposition",
         ylab = "Variance explained", 
         xlab = "N Principal Components", ylim = c(0,1))
  abline(h = 0.9, col = "red")
  text(1,0.9, "90% variance", col = "red", adj = c(-.1, -.12))
  dev.off()
}

plotIndiv(u.pca)
plot(u.pca$variates$X)
plotIndiv(u.pca.100d)
plot(u.pca.100d$variates$X)

u.pca.umap <- X %>%
  umap(init = "spca",
       metric = "euclidean", 
       n_trees = 300, n_epochs = 200,
       n_neighbors = 3,
       min_dist = 1, 
       n_threads = 3, n_sgd_threads = 3,
       approx_pow = T, 
       ret_model = T, 
       verbose = T,
       ret_nn = T, 
       n_components = 2)
plot(u.pca.umap$embedding[,1], u.pca.umap$embedding[,2], main =  "umap from 2D sPCA")

u.pca.umap <- X %>%
  # cbind(PropMatrix) %>%
  umap(init = "spca", pca = 150,
       metric = "euclidean", 
       n_trees = 300, n_epochs = 500,
       n_neighbors = 2,
       min_dist = 1, 
       n_threads = 3, n_sgd_threads = 3,
       approx_pow = T, 
       ret_model = T, 
       verbose = T,
       ret_nn = T, 
       n_components = 2)
plot(u.pca.umap$embedding[,1], u.pca.umap$embedding[,2], main =  "umap from 2D sPCA")


# u.nn <- X %>%
#   umap(metric = "euclidean", 
#        n_trees = 200, n_threads = 3, n_sgd_threads = 3,
#        n_neighbors = 200, 
#        min_dist = 1,
#        approx_pow = T, 
#        ret_model = T, 
#        verbose = T,
#        ret_nn = T)
# plot(u.nn$embedding[,1], u.nn$embedding[,2], main ="UMAP many nn (no smooth breaks)")

u.ags <- X %>%
  # cbind(as.data.frame(PropMatrix)) %>%
  umap(init = "agspectral",
       # metric = list(euclidean = 1:ncol(X),
                      # cosine = (ncol(X)+1) : (ncol(X)+ncol(PropMatrix)) ),
       metric = "euclidean",
       min_dist = 1,
       n_trees = 300, n_epochs = 500, 
       repulsion_strength = 5, negative_sample_rate = 15,
       n_threads = 3, n_sgd_threads = 3,
       n_neighbors = 2, local_connectivity = 2,
       approx_pow = T, 
       ret_model = T, 
       verbose = T,
       ret_nn = T)
plot(u.ags$embedding[,1], u.ags$embedding[,2], main = "UMAP euclidean agspectral")

#-------------------- mixOmics --------------------

ndim = ncol(X)
ndim
{
  ipca.res <- ipca(X, ncomp = ndim * 0.8, mode="deflation", max.iter = 500)
  ipca.res$explained_variance %>% sum() %>% print()
  
  sipca.res <- sipca(X, ncomp = ndim * 0.8, mode="deflation", max.iter = 500)
  sipca.res$explained_variance %>% sum() %>% print()
  
  ipca.all.res <- ipca(cbind(X, as.data.frame(PropMatrix)), ncomp = 5, mode="deflation", max.iter = 500)
  ipca.all.res$explained_variance %>% print()
  ipca.all.res$explained_variance %>% sum() %>% print()
  
  sipca.all.res  <- sipca(cbind(X, as.data.frame(PropMatrix)), ncomp = 5, mode="deflation", max.iter = 500)
  sipca.all.res$explained_variance %>% print()
  sipca.all.res$explained_variance %>% sum() %>% print()
}


ndim2 = ncol(X)
ndim3 = ncol(PropMatrix)
{
  # spls.reg <- spls(X, PropMatrix, ncomp = ndim2 * 0.9, mode = "regression", max.iter = 500)
  # spls.can <- spls(X, PropMatrix, ncomp = ndim3 * 0.9,  mode = "canonical", max.iter = 500)
  # spls.inv <- spls(X, PropMatrix, ncomp = ndim2 * 0.9,  mode = "invariant", max.iter = 500)
  
  spls.reg <- spls(X, PropMatrix, ncomp = 5, mode = "regression", max.iter = 500)
  spls.can <- spls(X, PropMatrix, ncomp = ndim3 * 0.9,  mode = "canonical", max.iter = 500)
  spls.inv <- spls(X, PropMatrix, ncomp = 5,  mode = "invariant", max.iter = 500)
  spls.class <- spls(X, PropMatrix, ncomp = 15, mode = "classic", max.iter = 500)
  
  inv.spls.reg <- spls(PropMatrix, X, ncomp = ndim3 * 0.5, mode = "regression", max.iter = 500)
  inv.spls.can <- spls(PropMatrix, X, ncomp = ndim3 * 0.9,  mode = "canonical", max.iter = 500)
  inv.spls.inv <- spls(PropMatrix, X, ncomp = ndim3 * 0.5,  mode = "invariant", max.iter = 500)
  
  shrink.res <- rcc(X, PropMatrix, ncomp = ndim3, method = 'shrinkage')
  inv.shrink.res <- rcc(PropMatrix, X, ncomp = ndim3, method = 'shrinkage')
  
  lapply(spls.reg$explained_variance, sum) %>% print()
  lapply(spls.can$explained_variance, sum) %>% print()
  lapply(spls.inv$explained_variance, sum) %>% print()
  lapply(spls.class$explained_variance, sum) %>% print()
  
  lapply(inv.spls.reg$explained_variance, sum) %>% print()
  lapply(inv.spls.can$explained_variance, sum) %>% print()
  lapply(inv.spls.inv$explained_variance, sum) %>% print()
  
  lapply(shrink.res$explained_variance, sum) %>% print()
  lapply(inv.shrink.res$explained_variance, sum) %>% print()
  
  p <- spls.reg
  p <- perf(p, validation = "Mfold", folds = 5, progressBar = T)
  {
    # Q2.total
    plot(p$Q2.total, type = 'l', col = 'red', 
         xlab = 'PLS components', ylab = 'Q2 total')
    abline(h = 0.0975, col = 'darkgreen')
    legend('topright', col = c('red', 'darkgreen'),
           legend = c('Q2 total', 'threshold 0.0975'), lty = 1)
    title('PLS 5-fold, MArginal contributions of the latent variables; good variables Q2.total^2 >= 0.0975')
    
    # R2
    p$R2
    matplot(t(p$R2), type = 'l', xlab = 'PLS components', ylab = 'R2 for each variable')
    title('PLS 5-fold, R2 values')
    
    # MSEP
    p$MSEP
    matplot(t(p$MSEP), type = 'l', xlab = 'PLS components', ylab = 'MSEP for each variable')
    title('5-fold, mean squared error of prediction (MSEP)')
    
    # Plot in 2 first dimensions
    plotIndiv(p)
  }

  
}

#--------------------  3-dataset mixOmics --------------------
{
  block.X <- list(X.3, X.5)
  names(block.X) <- c("X.3", "X.5")
  rownames(block.X$X.3) <- rownames(PropMatrix)
  rownames(block.X$X.5) <- rownames(PropMatrix)

  block.spls.reg <- block.spls(block.X, PropMatrix, ncomp = 95, mode = "regression", max.iter = 500)
  block.spls.can <- block.spls(block.X, PropMatrix, ncomp = 95,  mode = "canonical", max.iter = 500)
  
  pls.reg.x3x5 <- pls(block.X$X.3, block.X$X.5, ncomp = 95, mode = "regression", max.iter = 500)
  pls.can.x3x5 <- pls(block.X$X.3, block.X$X.5, ncomp = 95,  mode = "canonical", max.iter = 500)
  pls.inv.x3x5 <- pls(block.X$X.5, block.X$X.3, ncomp = 60,  mode = "invariant", max.iter = 500)
  
  # pls.reg.sif <- pls(block.X$sv.sif, PropMatrix, ncomp = 4, mode = "regression", max.iter = 500)
  # pls.can.sif <- pls(block.X$sv.sif, PropMatrix, ncomp = 8,  mode = "canonical", max.iter = 500)
  # pls.inv.sif <- pls(block.X$sv.sif, PropMatrix, ncomp = 4,  mode = "invariant", max.iter = 500)
  # pls.class.sif <- pls(block.X$sv.sif, PropMatrix, ncomp = 8,  mode = "classic", max.iter = 500)
  # 
  # pls.reg.sv <- pls(sv, PropMatrix, ncomp = 4, mode = "regression", max.iter = 500)
  # pls.can.sv <- pls(sv, PropMatrix, ncomp = 8,  mode = "canonical", max.iter = 500)
  # pls.inv.sv <- pls(sv, PropMatrix, ncomp = 4,  mode = "invariant", max.iter = 500)
  # pls.class.sv <- pls(sv, PropMatrix, ncomp = 5,  mode = "classic", max.iter = 500)
  
  lapply(block.spls.reg$explained_variance, sum) %>% print()
  lapply(block.spls.can$explained_variance, sum) %>% print()
  
  lapply(pls.reg.x3x5$explained_variance, sum) %>% print()
  lapply(pls.can.x3x5$explained_variance, sum) %>% print()
  lapply(pls.inv.x3x5$explained_variance, sum) %>% print()
  
  p <- pls.reg.x3x5
  lapply(p$explained_variance, sum) %>% print()
  p$explained_variance
  p <- perf(p, validation = "Mfold", folds = 5, progressBar = T)
  {
    # Q2.total
    plot(p$Q2.total, type = 'l', col = 'red', 
         xlab = 'PLS components', ylab = 'Q2 total')
    abline(h = 0.0975, col = 'darkgreen')
    legend('topright', col = c('red', 'darkgreen'),
           legend = c('Q2 total', 'threshold 0.0975'), lty = 1)
    title('PLS 5-fold, MArginal contributions of the latent variables; good variables Q2.total^2 >= 0.0975')
    
    # R2
    p$R2
    matplot(t(p$R2), type = 'l', xlab = 'PLS components', ylab = 'R2 for each variable')
    title('PLS 5-fold, R2 values')
    
    # MSEP
    p$MSEP
    matplot(t(p$MSEP), type = 'l', xlab = 'PLS components', ylab = 'MSEP for each variable')
    title('5-fold, mean squared error of prediction (MSEP)')
    t(p$MSEP)
  }
  
  # Plot in 2 first dimensions
  plotIndiv(pls.inv.sif)

}
  
{  
  u.block.spls.reg <- cbind(block.spls.reg$variates$X.3, 
                            block.spls.reg$variates$X.5) %>%
    umap(metric = "euclidean", init = "agspectral",
         min_dist = 1, 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = 2, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.block.spls.reg.sup <- cbind(block.spls.reg$variates$X.3, 
                                block.spls.reg$variates$X.5) %>%
    umap(y = PropMatrix,
         metric = "euclidean",  target_metric = "cosine", init = "agspectral",
         min_dist = 1, 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = 2, target_n_neighbors = 30,
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.block.spls.reg.cat <- cbind(block.spls.reg$variates$X.3, 
                                block.spls.reg$variates$X.5,
                                PropMatrix) %>%
    umap(metric = "euclidean",  init = "agspectral",
         min_dist = 1, 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = 15, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.block.spls.can <- cbind(block.spls.can$variates$X.3, 
                            block.spls.can$variates$X.5) %>%
    umap(init = "agspectral", min_dist = 1, 
         n_threads = 3, n_sgd_threads = 3,
         metric = "euclidean", 
         n_trees = nt,
         n_neighbors = 2, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.block.spls.can.sup <- cbind(block.spls.can$variates$X.3, 
                                block.spls.can$variates$X.5) %>%
    umap(y = PropMatrix, 
         init = "agspectral", min_dist = 1, 
         n_threads = 3, n_sgd_threads = 3,
         metric = "euclidean",  target_metric = "cosine", 
         n_trees = nt,
         n_neighbors = 2, target_n_neighbors = 20,
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.reg.x3x5 <- pls.reg.x3x5$variates$X %>%
    umap(y = pls.reg.x3x5$variates$Y, 
         init = "agspectral",
         n_threads = 3, n_sgd_threads = 3,
         min_dist = 1,
         metric = "euclidean", 
         n_trees = nt,
         n_neighbors = 2, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.reg.x3x5.cat <- cbind(pls.reg.x3x5$variates$X, 
                              pls.reg.x3x5$variates$Y) %>%
    umap(init = "agspectral",
         min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = 2, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.can.x3x5.sup <- cbind(pls.can.x3x5$variates$X, 
                              pls.can.x3x5$variates$Y) %>%
    umap(y = PropMatrix,   
         init = "agspectral",
         n_threads = 3, n_sgd_threads = 3,
         min_dist = 1,
         metric = "euclidean", target_metric = "cosine",
         n_trees = nt,
         n_neighbors = 2, target_n_neighbors = 20,
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.can.x3x5.cat <- cbind(pls.can.x3x5$variates$X,
                              pls.can.x3x5$variates$Y) %>%
    umap(init = "agspectral",
         min_dist = 1,  
         n_threads = 3, n_sgd_threads = 3,
         metric = "euclidean", 
         n_trees = nt,
         n_neighbors = 2, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  
  u.pls.inv.x3x5 <- pls.inv.x3x5$variates$X %>%
    umap(y = pls.inv.x3x5$variates$Y,
         min_dist = 1,  
         n_threads = 3, n_sgd_threads = 3,
         metric = "euclidean", 
         n_trees = nt,
         n_neighbors = 2, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.inv.x3x5.cat <- cbind(pls.inv.x3x5$variates$X, 
                              pls.inv.x3x5$variates$Y) %>%
    umap(min_dist = 1,  
         n_threads = 3, n_sgd_threads = 3,
         metric = "euclidean", 
         n_trees = nt,
         n_neighbors = 2, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.inv.x3x5.cat.sup <- cbind(pls.inv.x3x5$variates$X, 
                              pls.inv.x3x5$variates$Y) %>%
    umap(y=PropMatrix,  
         n_threads = 3, n_sgd_threads = 3,
         min_dist = 1,
         metric = "euclidean", target_metric = "cosine", 
         n_trees = nt,
         n_neighbors = 2, target_n_neighbors = 20,
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
}


# 2 sPCS dim + 40nn + cosine
nn = 15
nt = 300
{
  u.biophys <- PropMatrix %>%
    umap(init = "agspectral", metric = "cosine", 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  
  u.ipca <- ipca.res$x %>%
    umap(metric = "euclidean", min_dist = 1, 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.sipca <- sipca.res$x %>%
    umap(metric = "euclidean", min_dist = 1, 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  # -------------------- << Bookmark >> ----------------------------
  
  u.ipca.all <- ipca.all.res$variates$X %>%
    umap(metric = "euclidean", min_dist = 1, 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt, n_epochs = 500,
         n_neighbors = 45, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.sipca.all <- sipca.all.res$variates$X %>%
    umap(metric = "euclidean", min_dist = 1, 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt, n_epochs = 500,
         n_neighbors = 45, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.reg <- spls.reg$variates$X %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.reg.all <- spls.reg$variates$X %>%
    umap(y = spls.reg$variates$Y, min_dist = 1,
         n_threads = 3, n_sgd_threads = 3, 
         metric = "cosine", 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.reg.allcat <- cbind(spls.reg$variates$X, spls.reg$variates$Y) %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.can <- spls.can$variates$X %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.can.all <- spls.can$variates$X %>%
    umap(y = spls.can$variates$Y, min_dist = 1,
         n_threads = 3, n_sgd_threads = 3, 
         metric = "cosine", 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.inv <- spls.inv$variates$X %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.inv.all <- spls.inv$variates$X %>%
    umap(y = spls.reg$variates$Y,
         n_threads = 3, n_sgd_threads = 3, 
         metric = "cosine", 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.inv.allcat <- cbind(spls.inv$variates$X, spls.inv$variates$Y) %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.inv.spls.reg <- inv.spls.reg$variates$Y %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.inv.spls.can <- inv.spls.can$variates$Y %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.inv.spls.inv <- inv.spls.inv$variates$Y %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.shrink <-  shrink.res$variates$X %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  u.inv.shrink <- inv.shrink.res$variates$Y %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3, 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
}

plot(u.pca$embedding[,1], u.pca$embedding[,2], main = "UMAP 5nn from 2D PCA")
plot(u.nn$embedding[,1], u.nn$embedding[,2], main = "UMAP 50 nn smooth")
# plot(ipca.res$x[,1], ipca.res$x[,2], main = "Independent Principal Component Analysis (IPCA)")
# plot(sipca.res$x[,1], sipca.res$x[,2], main = "Sparse independent principal component analysis (SIPCA)")
plot(u.ipca$embedding[,1], u.ipca$embedding[,2], main = "UMAP from IPCA")
plot(u.sipca$embedding[,1], u.sipca$embedding[,2], main = "UMAP from SIPCA")

plot(u.ipca$embedding[,1], u.ipca$embedding[,2], main = "UMAP from IPCA incl.biophys")
plot(u.sipca$embedding[,1], u.sipca$embedding[,2], main = "UMAP from SIPCA incl.biophys")


plot(u.pls$embedding[,1], u.pls$embedding[,2], main = "UMAP from pls")
plot(u.spls$embedding[,1], u.spls$embedding[,2], main = "UMAP from spls")
plot(u.shrink$embedding[,1], u.shrink$embedding[,2], main = "UMAP from shrink")

plot(u.inv.pls$embedding[,1], u.inv.pls$embedding[,2], main = "UMAP from inv.pls")
plot(u.inv.spls$embedding[,1], u.inv.spls$embedding[,2], main = "UMAP from inv.spls")
plot(u.inv.shrink$embedding[,1], u.inv.shrink$embedding[,2], main = "UMAP from inv.shrink")


#-------------------- Plot token properties --------------------
# u.nn     u.100d
# u.sipca       u.ipca
# u.sipca.all   u.ipca.all
# u.pls        u.spls       u.shrink    
# u.inv.pls    u.inv.spls   u.inv.shrink
# a <- spls.reg$variates$Y %>%
u.extreme.nn <- X %>%
  umap(metric = "cosine", 
       n_trees = 100,
       n_threads = 3, n_sgd_threads = 3, 
       n_neighbors = nrow(X) * 0.01, n_threads = 3,
       approx_pow = T, 
       ret_model = T, 
       verbose = T, 
       min_dist = 1,
       ret_nn = T)

# u.extreme.nn.x3x5 <- X.3 %>%
#   umap(y = X.5,
# n_threads = 3, n_sgd_threads = 3, 
#        metric = "cosine", 
#        n_trees = 100,
#        n_neighbors = nrow(X) * 1, n_threads = 3,
#        approx_pow = T, 
#        ret_model = T, 
#        verbose = T, 
#        min_dist = 1,
#        ret_nn = T)
# 
# u.many.nn.x3x5 <- X.3 %>%
#   umap(y = X.5,
# n_threads = 3, n_sgd_threads = 3, 
#        metric = "cosine", 
#        n_trees = 300,
#        n_neighbors = 100, n_threads = 3,
#        approx_pow = T, 
#        ret_model = T, 
#        verbose = T, 
#        min_dist = 1,
#        ret_nn = T)
dat <- cbind(X.3, X.5)
# This can be slow with high-dimensional input

ggcorr(dat, method = c("pairwise", "spearman"))
ggcorr(PropMatrix, method = c("pairwise", "spearman"))
ggcorr(cbind(dat, PropMatrix), method = c("pairwise", "spearman"))
ggcorr(cbind(dat, PropMatrix0), method = c("pairwise", "spearman"), 
       name = "Pairwise spearman correlation of: s2v X.3 and X.5 tokens, biochem props")

u.sup <- dat %>%
  umap(init = "agspectral", 
       y = PropMatrix, 
       metric = "euclidean", target_metric = "cosine", 
       n_neighbors = 3, target_n_neighbors = 30,
       n_threads = 1, n_sgd_threads = 3,
       n_trees = 500, 
       n_epochs = 200,
       approx_pow = T,
       ret_model = T,
       verbose = T,
       min_dist = 1,
       ret_nn = T)


u.sup.inv <- PropMatrix %>%
  cbind(PropMatrix0$protein.length) %>%
  umap(y = dat, target_weight = 0.5, 
       init = "agspectral",
       metric = "cosine", target_metric = "euclidean", 
       n_neighbors = 60, target_n_neighbors = 3,
       n_trees = 500, 
       n_epochs = 200,
       approx_pow = T,
       ret_model = T,
       verbose = T,
       min_dist = 1,
       ret_nn = T)

u.unsup <- dat %>% 
  cbind(PropMatrix) %>%
  umap(init = "agspectral",
       metric = list(euclidean = 1:ncol(dat), 
                     cosine = (ncol(dat)+1) : (ncol(dat)+ncol(PropMatrix)) ),
       n_trees = 300, 
       n_epochs = 100,
       n_neighbors = 15, 
       n_threads = 3, n_sgd_threads = 3,
       approx_pow = T,
       ret_model = T,
       verbose = T,
       min_dist = 1,
       ret_nn = T)

u.X3X5 <- dat %>%
  umap(init = "agspectral", 
       metric = "euclidean", 
       n_trees = 300,
       n_threads = 3, n_sgd_threads = 3,
       n_neighbors = 3,  n_epochs = 1000,
       approx_pow = T, 
       ret_model = T, 
       verbose = T,
       ret_nn = T)

u.X3X5.s <- X.3 %>%
  umap(y = X.5,
       metric = "euclidean", 
       n_trees = 300, repulsion_strength = 5, negative_sample_rate = 9,
       n_threads = 3, n_sgd_threads = 3,
       n_neighbors = 3, target_n_neighbors = 5,
       approx_pow = T, 
       ret_model = T, 
       verbose = T,
       ret_nn = T)
#-------------------- Plot  --------------------
# PM <- as.data.frame(PropMatrix)
# PM$pI <- NULL
# PM$log10.mol.weight <- NULL
# PM$log10.charge <- NULL
# u.biophys <- PM %>%
#   umap(metric = "cosine", init = "agspectral",
#        n_trees = 200,  n_threads = 3, n_sgd_threads = 3,
#        n_epochs = 20,
#        n_neighbors = 50, min_dist = 1,
#        approx_pow = T, 
#        ret_model = T,
#        verbose = T,
#        ret_nn = T)

a <- u.sipca.all
{
  # a$nn %>% lapply(head) 
  df <- data.frame(X1 = a$embedding[,1],
                   X2 = a$embedding[,2]) %>% na.omit()
  df <- cbind(df,PropMatrix0) %>%
    na.omit() %>%
    as_tibble()
}
{
  plot(cumsum(a$nn$euclidean$dist[1,]), type = "s", main = "Cumulative distance sum of umap nn")
  abline(v = 5)
}
{
  plot(cumsum(a$nn$cosine$dist[1,]), type = "s", main = "Cumulative distance sum of umap nn")
  abline(v = 5)
}

ggplot(df, aes(x=X1, y=X2, color=pI)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c() + 
  ggtitle("pI")
ggplot(df, aes(x=X1, y=X2) ) +
  geom_point(size=0.1) +
  geom_hex(bins = 15, alpha = 0.33) +
  theme_bw() +
  scale_fill_continuous(low = "grey80", high = "darkred")
# Single AA
{
  df_aa <- df[df$protein.length < 2,]
  ggplot(df_aa, aes(x=X1, y=X2, color=`Hydrophobicity`)) +
    geom_text(label = tokens$subword[df$protein.length < 2], size = 4, alpha = 1) +
    theme_base() +
    scale_color_continuous(low = "red" , high = "black") +
    ggtitle("Single AAs only plotted")
}

ggplot(df, aes(x=X1, y=X2, color=pI)) +
  geom_text(label = tokens$subword, size = 2, alpha = 1) +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=as.factor(protein.length))) +
  geom_text(label = nchar(tokens$subword), size = 3, alpha = 1) +
  theme_gray()
ggplot(df, aes(x=X1, y=X2, color=as.factor(protein.length))) +
  geom_text(label = tokens$subword, size = 3, alpha = 1) +
  theme_gray()

ggplot(df, aes(x=X1, y=X2, color=protein.length)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=Hydrophobicity)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=aliphatic.index)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()

ggplot(df, aes(x=X1, y=X2, color=Aliphatic.Mole.percent)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=Aromatic.Mole.percent)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=NonPolar.Mole.percent)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=Charged.Mole.percent)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=Basic.Mole.percent)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=Acidic.Mole.percent)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()

colnames(df)

# ggplot(df, aes(x=X1, y=X2, color=pI)) +
#   geom_text(label = tokens$subword, size = 2, alpha = 1) +
#   theme_gray() +
#   scale_color_viridis_c()
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM1)) +
#   geom_text(label = tokens$subword, size = 2, alpha = 1) +
#   theme_gray() +
#   scale_color_viridis_c() 
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM1)) +
#   geom_text(label = tokens$subword, size = 2, alpha = 1) +
#   theme_gray() +
#   scale_color_viridis_c() +
#   xlim(-15,2) +
#   ylim(-6,2)
# ggplot(df, aes(x=X1, y=X2, color=as.factor(protein.length))) +
#   geom_text(label = tokens$subword, size = 2, alpha = 1) +
#   theme_gray() 
# 
# 
# 




ggplot(df, aes(x=X1, y=X2, color=F1)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c() +
  ggtitle("F1: Hydrophobicity index")
ggplot(df, aes(x=X1, y=X2, color=F2)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()+ 
  ggtitle("F2: Alpha and turn propensities")
ggplot(df, aes(x=X1, y=X2, color=F3)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()+ 
  ggtitle("F3: Bulky properties")
ggplot(df, aes(x=X1, y=X2, color=F4)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()+ 
  ggtitle("F4: Compositional characteristic index")
ggplot(df, aes(x=X1, y=X2, color=F5)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()+ 
  ggtitle("F5: Local flexibility")
ggplot(df, aes(x=X1, y=X2, color=F6)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c() + 
  ggtitle("F6: Electronic properties")


ggplot(df, aes(x=X1, y=X2, color=BLOSUM1)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c() + 
  ggtitle("BLOSUM1")
ggplot(df, aes(x=X1, y=X2, color=BLOSUM2)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c() +
  ggtitle("BLOSUM2")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM3)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM3")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM4)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM4")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM5)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM5")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM4)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM4")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM5)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM5")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM6)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM6")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM7)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM7")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM8)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM8")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM9)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM9")
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM10)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM10")

# ---------------- 3D plots ---------------
# Graphing your 3d scatterplot using plotly's scatter3d type:
u.sup.inv.3d <- PropMatrix %>%
  umap(y = dat, n_components = 3, init = "agspectral",
       metric = "cosine", target_metric = "euclidean", target_weight = 0.90,
       n_neighbors = 30, target_n_neighbors = 3,
       n_trees = 500, 
       n_epochs = 200,
       approx_pow = T,
       ret_model = T,
       verbose = T,
       min_dist = 1,
       ret_nn = T)
# ---------------- export selected results ---------------

tokens %>%
  as.data.frame() %>%
  cbind(u.sup.inv.3d$embedding) %>%
  write_csv(path = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_token_human_05_2020\\u.sup.inv.3d.csv")

tokens %>% 
  as.data.frame() %>%
  cbind(sipca.all.res$variates$X)  %>%
  write_csv(path = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_token_human_05_2020\\sipca_X3X5bioph.csv")

tokens %>%
  cbind(as.data.frame(PropMatrix)) %>%
write_csv(path = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_token_human_05_2020\\PropMatrix.csv")


u.X3X5.s3D <- X.3 %>%
  umap(y = X.5, n_components = 3, 
       metric = "euclidean", 
       n_trees = 300, repulsion_strength = 3, negative_sample_rate = 9,
       n_threads = 3, n_sgd_threads = 3,
       n_neighbors = 2, target_n_neighbors = 5,
       approx_pow = T, 
       ret_model = T, 
       verbose = T,
       ret_nn = T)


a  <- sipca(cbind(X, as.data.frame(PropMatrix)), ncomp = 3, mode="deflation", max.iter = 500)
a$explained_variance %>% print()
a$explained_variance %>% sum() %>% print()

a <- u.sup.inv.3d
{
  # a$nn %>% lapply(head) 
  df <- data.frame(X1 = a$embedding[,1],
                   X2 = a$embedding[,2],
                   X3 = a$embedding[,3]) %>% na.omit()
  
  df <- data.frame(X1 = a$variates$X[,1],
                   X2 = a$variates$X[,2],
                   X3 = a$variates$X[,3]) %>% na.omit()
  
  df <- cbind(df,PropMatrix0) %>%
    na.omit() %>%
    as_tibble()
}

library(plotly)
library(rgl)
colnames(df)
plot_ly(data = df, 
        x=df$X1, 
        y=df$X2, 
        z=df$X3, 
        type="scatter3d", mode="markers", size = 5,
        color=df$pI)
