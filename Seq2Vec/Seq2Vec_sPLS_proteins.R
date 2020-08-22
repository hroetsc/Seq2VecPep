# Create sPLS representation of seq2vec data
library(stringr)
library(readr)
library(tidyr)
library(uwot)
library(ggraptR)
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

# Load SCOP classification
scop <- readr::read_delim("R_data/scop_cla_latest.txt", skip = 5, delim = " ")
scop_class <- scop$`SF-UNIREG` %>%
  str_remove_all(pattern = "TP=") %>%
  str_remove_all(pattern = "CL=") %>%
  str_remove_all(pattern = "CF=") %>%
  str_remove_all(pattern = "SF=") %>%
  str_remove_all(pattern = "FA=") %>% 
  str_split_fixed(pattern = ",", Inf)
scop_class <- as_tibble(scop_class)
colnames(scop_class) <- c("TP", "CL", "CF", "SF", "FA")


# Load seq2vec protein embeddings
sv <- read_csv("R_data/sequence_repres_w3_d100_seq2vec.csv")
sv.sif <- read_csv("R_data/sequence_repres_w3_d100_seq2vec-SIF.csv")
sv.tfidf <- read_csv("R_data/sequence_repres_w3_d100_seq2vec-TFIDF.csv")
sv.CCR <- read_csv("R_data/sequence_repres_seq2vec_CCR.csv")

seqs <- sv[,colnames(sv) %in% c("Accession","seqs", "tokens")]
sv <- sv[,!colnames(sv) %in% c("Accession","seqs", "tokens")]
sv.sif <- sv.sif[,!colnames(sv.sif) %in% c("Accession","seqs", "tokens")]
sv.tfidf <- sv.tfidf[,!colnames(sv.tfidf) %in% c("Accession","seqs", "tokens")]
sv.CCR <- sv.CCR[,!colnames(sv.CCR) %in% c("Accession","seqs", "tokens")]
sv.CCR$X1 <- NULL

# non-standard AAs
seqs.special <- seqs$seqs %>%
  str_replace_all(pattern = "[\\ACDEFGHIKLMNPQRSTVWY]", replacement = "") %>%
  nchar()
seqs.special <- seqs[which(seqs.special > 0),]
print(seqs.special)

seqs.prep <- seqs$seqs %>%
  str_replace_all(pattern = "X", replacement = "") %>%
  str_replace_all(pattern = "U", replacement = "C")
max(nchar(seqs.prep) - nchar(seqs$seqs))
  
# biophys properties
PropMatrix0 = foreach(i=1:length(seqs.prep), 
                      .packages = c("Peptides")) %dopar% PepSummary(seqs.prep[i])
PropMatrix0 <- bind_rows(PropMatrix0)

PropMatrix0$log10.mol.weight <- log10(PropMatrix0$mol.weight)
PropMatrix0$log10.charge <- sign(PropMatrix0$charge) * log10(abs(PropMatrix0$charge))
dim(na.omit(PropMatrix0))
PropMatrix0 <- na.omit(PropMatrix0)

keep = colnames(PropMatrix0)[19:ncol(PropMatrix0)]
keep <- keep[!keep %in% c("protein.length", "mol.weight", "charge")]
PropMatrix <- PropMatrix0[,keep] 
PropMatrix <- PropMatrix0[,keep]  %>%
  na.omit() %>% 
  data.matrix() 

# nzv <- nearZeroVar(PropMatrix, saveMetrics = T)
# nzv
# nzv <- nearZeroVar(PropMatrix, saveMetrics = F)
# filteredDescr <- PropMatrix[, -nzv]

comboInfo <- caret::findLinearCombos(PropMatrix)
colnames(PropMatrix)[comboInfo$remove]
PropMatrix <- PropMatrix[, -comboInfo$remove]

descrCor <- cor(PropMatrix)
summary(descrCor[upper.tri(descrCor)])

ggcorr(PropMatrix,  method = c("pairwise", "spearman"))
ggcorr(sv,  method = c("pairwise", "spearman"))
ggcorr(sv.sif,  method = c("pairwise", "spearman"))
ggcorr(sv.tfidf,  method = c("pairwise", "spearman"))
ggcorr(sv.CCR,  method = c("pairwise", "spearman"))

cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_prot_human_05_2020\\corr_bioph.pdf", width = 20, height = 10)
ggcorr(PropMatrix0,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_prot_human_010_2020\\corr_bioph.final.pdf", width = 20, height = 10)
ggcorr(PropMatrix,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_prot_human_05_2020\\corr_sv.pdf", width = 20, height = 10)
ggcorr(sv,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_prot_human_05_2020\\corr_sv.sif.pdf", width = 20, height = 10)
ggcorr(sv.sif,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_prot_human_05_2020\\corr_sv.tfidf.pdf", width = 20, height = 10)
ggcorr(sv.tfidf,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_prot_human_05_2020\\corr_sv.CCR.pdf", width = 20, height = 10)
ggcorr(sv.CCR,  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_prot_human_05_2020\\corr_sv_sif_bioph.pdf", width = 20, height = 10)
ggcorr(c(sv, sv.sif, as.data.frame(PropMatrix)),  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()
cairo_pdf(filename = "C:\\Users\\PC\\Desktop\\R_results\\seq2vec_prot_human_05_2020\\corr_sv_sif_tfidf_CCR_bioph_all.pdf", width = 20, height = 10)
ggcorr(c(sv, sv.sif, sv.tfidf, sv.CCR, as.data.frame(PropMatrix0)),  method = c("pairwise", "spearman"), name = "pairwise Spearman correlation")
dev.off()

#-------------------- Plot seq2vec --------------------
summary(sv[upper.tri(sv)])
summary(sv.sif[upper.tri(sv.sif)])
summary(sv.tfidf[upper.tri(sv.tfidf)])
summary(sv.CCR[upper.tri(sv.CCR)])

X <- cbind(sv, sv.sif, sv.tfidf, sv.CCR)
# X <- cbind.data.frame(sv)
colnames(X) <- paste0(rep(c("sv","sv.sif","sv.tfidf", "sv.CCR"), each=nrow(sv)), 
                      c(colnames(sv),colnames(sv.sif), colnames(sv.tfidf), colnames(sv.CCR)))

# Plot umap embeddings of raw seq2vec
u.pca <- X %>% as_tibble() %>% ipca(ncomp = 2)
u.pca$variates$X %>% head()
u.pca.100d <- X %>% as_tibble() %>% ipca(ncomp = 100)
plotIndiv(u.pca)
plot(u.pca$variates$X)
plotIndiv(u.pca.100d)
plot(u.pca.100d$variates$X)

u.pca.umap <- X %>%
  umap(metric = "euclidean", 
       n_threads = 3, n_sgd_threads = 3,
       n_trees = 300, 
       n_neighbors = 15,  
       min_dist = 1, 
       approx_pow = T, 
       ret_model = T, 
       verbose = T,
       ret_nn = T,  
       init = "spca", 
       n_components = 15)
plot(u.pca.umap$embedding[,1], u.pca.umap$embedding[,2], main =  "umap from 2D PCA")

u.nn <- X %>%
  umap(metric = "euclidean", 
       n_trees = 200, 
       n_threads = 3, n_sgd_threads = 3,
       n_neighbors = 200, 
       min_dist = 1,
       approx_pow = T, 
       ret_model = T, 
       verbose = T,
       ret_nn = T)
plot(u.nn$embedding[,1], u.nn$embedding[,2], main ="UMAP many nn (no smooth breaks)")


u.ags <- X %>%
  umap(metric = "euclidean", 
       n_trees = 300, 
       n_threads = 3, n_sgd_threads = 3,
       n_neighbors = 5,  
       min_dist = 1,
       approx_pow = T, 
       ret_model = T, 
       verbose = T,
       ret_nn = T, 
       init = "agspectral")
plot(u.ags$embedding[,1], u.ags$embedding[,2], main = "UMAP euclidean agspectral")

#-------------------- mixOmics --------------------

ndim = ncol(X)
ndim
{
  ipca.res <- ipca(X, ncomp = ndim * 0.3, mode="deflation", max.iter = 500)
  sipca.res <- sipca(X, ncomp = ndim * 0.3, mode="deflation", max.iter = 500)
  tmp <- sv
  colnames(tmp) = 1:ncol(tmp)
  ipca.sv <- ipca(tmp, ncomp = ndim * 0.3, mode="deflation", max.iter = 500)
  sipca.sv  <- sipca(tmp, ncomp = ndim * 0.3, mode="deflation", max.iter = 500)
  
  tmp <- cbind(X, PropMatrix)
  # colnames(tmp) = 1:ncol(tmp)
  ipca.all <- ipca(tmp, ncomp = 2, mode="deflation", max.iter = 500)
  sipca.all  <- sipca(tmp, ncomp = 2, mode="deflation", max.iter = 500)

  ipca.res$explained_variance %>% sum() %>% print()
  sipca.res$explained_variance %>% sum() %>% print()
  
  ipca.sv$explained_variance %>% sum() %>% print()
  sipca.sv$explained_variance %>% sum() %>% print()
  
  ipca.all$explained_variance %>% sum() %>% print()
  sipca.all$explained_variance %>% sum() %>% print()
}


ndim2 = ncol(X)
ndim3 = ncol(PropMatrix)
ndim3
{
  # spls.reg <- spls(X, PropMatrix, ncomp = ndim2 * 0.9, mode = "regression", max.iter = 500)
  # spls.can <- spls(X, PropMatrix, ncomp = ndim3 * 0.9,  mode = "canonical", max.iter = 500)
  # spls.inv <- spls(X, PropMatrix, ncomp = ndim2 * 0.9,  mode = "invariant", max.iter = 500)
  
  spls.reg <- spls(X, PropMatrix, ncomp = 5, mode = "regression", max.iter = 500)
  spls.can <- spls(X, PropMatrix, ncomp = 15,  mode = "canonical", max.iter = 500)
  spls.inv <- spls(X, PropMatrix, ncomp = 3,  mode = "invariant", max.iter = 500)
  spls.class <- spls(X, PropMatrix, ncomp = 25, mode = "classic", max.iter = 500)
  
  
  inv.spls.reg <- spls(PropMatrix, X, ncomp = 6 , mode = "regression", max.iter = 500)
  inv.spls.can <- spls(PropMatrix, X, ncomp = ndim3 * 0.9,  mode = "canonical", max.iter = 500)
  inv.spls.inv <- spls(PropMatrix, X, ncomp = 5,  mode = "invariant", max.iter = 500)
  
  # shrink.res <- rcc(X, PropMatrix, ncomp = 2, method = 'shrinkage')
  # inv.shrink.res <- rcc(PropMatrix, X, ncomp = 2, method = 'shrinkage')
  
  lapply(spls.reg$explained_variance, sum) %>% print()
  lapply(spls.can$explained_variance, sum) %>% print()
  lapply(spls.inv$explained_variance, sum) %>% print()
  lapply(spls.class$explained_variance, sum) %>% print()
  
  lapply(inv.spls.reg$explained_variance, sum) %>% print()
  lapply(inv.spls.can$explained_variance, sum) %>% print()
  lapply(inv.spls.inv$explained_variance, sum) %>% print()
  
  # lapply(shrink.res$explained_variance, sum) %>% print()
  # lapply(inv.shrink.res$explained_variance, sum) %>% print()
  
  p <- inv.spls.inv
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
  # block.X <- list(X.3, X.5)
  # names(block.X) <- c("X.3", "X.5")
  # rownames(block.X$X.3) <- rownames(PropMatrix)
  # rownames(block.X$X.5) <- rownames(PropMatrix)
  
  block.X <- list(sv, sv.sif, sv.tfidf)
  names(block.X) <- c("sv", "sv.sif", "sv.tfidf")
  rownames(block.X$sv.sif) <- rownames(PropMatrix)
  rownames(block.X$sv) <- rownames(PropMatrix)
  rownames(block.X$sv.tfidf) <- rownames(PropMatrix)

  block.spls.reg <- block.spls(block.X, PropMatrix, ncomp = 5, mode = "regression", max.iter = 500)
  block.spls.can <- block.spls(block.X, PropMatrix, ncomp = 10,  mode = "canonical", max.iter = 500)
  
  pls.reg.sif <- pls(block.X$sv.sif, PropMatrix, ncomp = 5, mode = "regression", max.iter = 500)
  pls.can.sif <- pls(block.X$sv.sif, PropMatrix, ncomp = 9,  mode = "canonical", max.iter = 500)
  pls.inv.sif <- pls(block.X$sv.sif, PropMatrix, ncomp = 4,  mode = "invariant", max.iter = 500)
  pls.class.sif <- pls(block.X$sv.sif, PropMatrix, ncomp = 20,  mode = "classic", max.iter = 500)
  
  pls.reg.sv <- pls(sv, PropMatrix, ncomp = 5, mode = "regression", max.iter = 500)
  pls.can.sv <- pls(sv, PropMatrix, ncomp = 9,  mode = "canonical", max.iter = 500)
  pls.inv.sv <- pls(sv, PropMatrix, ncomp = 4,  mode = "invariant", max.iter = 500)
  pls.class.sv <- pls(sv, PropMatrix, ncomp = 12,  mode = "classic", max.iter = 500)
  
  lapply(block.spls.reg$explained_variance, sum) %>% print()
  lapply(block.spls.can$explained_variance, sum) %>% print()
  
  lapply(pls.reg.sif$explained_variance, sum) %>% print()
  lapply(pls.can.sif$explained_variance, sum) %>% print()
  lapply(pls.inv.sif$explained_variance, sum) %>% print()
  lapply(pls.class.sif$explained_variance, sum) %>% print()
  
  lapply(pls.reg.sv$explained_variance, sum) %>% print()
  lapply(pls.can.sv$explained_variance, sum) %>% print()
  lapply(pls.inv.sv$explained_variance, sum) %>% print()
  lapply(pls.class.sv$explained_variance, sum) %>% print()
  
  
  p <- pls.class.sif
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
  nt = 300
  nn = 7
  
  u.block.spls.reg <- cbind(block.spls.reg$variates$sv, 
                             block.spls.reg$variates$sv.sif, 
                             block.spls.reg$variates$sv.tfidf) %>%
    umap(init = "agspectral",
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         min_dist = 1,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.block.spls.reg.cat <- cbind(block.spls.reg$variates$sv, 
                                block.spls.reg$variates$sv.sif, 
                                block.spls.reg$variates$sv.tfidf,
                                block.spls.reg$variates$Y) %>%
    umap(init = "agspectral", min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.block.spls.reg.sup <- cbind(block.spls.reg$variates$sv, 
                                block.spls.reg$variates$sv.sif, 
                                block.spls.reg$variates$sv.tfidf) %>%
    umap(y = block.spls.reg$variates$Y,
         min_dist = 1,
         metric = "euclidean", target_metric = "cosine",
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = 5, target_n_neighbors = 30,
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  
  u.block.spls.can <- cbind(block.spls.reg$variates$sv, 
                            block.spls.reg$variates$sv.sif, 
                            block.spls.reg$variates$sv.tfidf) %>%
    umap(min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.block.spls.can.cat <- cbind(block.spls.reg$variates$sv, 
                                block.spls.reg$variates$sv.sif, 
                                block.spls.reg$variates$sv.tfidf,
                                block.spls.can$variates$Y) %>%
    umap(min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.block.spls.can.sup <- cbind(block.spls.reg$variates$sv, 
                                block.spls.reg$variates$sv.sif, 
                                block.spls.reg$variates$sv.tfidf) %>%
    umap(y = block.spls.can$variates$Y,
         min_dist = 1,
         metric = "euclidean", target_metric = "cosine",
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = 5, target_n_neighbors = 30, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
    
  u.pls.reg.sv <- pls.reg.sv$variates$X %>%
    umap(y = pls.reg.sv$variates$Y,
         min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.reg.sv.cat <- cbind(pls.reg.sv$variates$X, 
                              pls.reg.sv$variates$Y) %>%
    umap(min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.can.sv <- pls.can.sv$variates$X %>%
    umap(y = pls.can.sv$variates$Y,
         min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.can.sv.cat <- cbind(pls.can.sv$variates$X,
                              pls.can.sv$variates$Y) %>%
    umap(min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  
  u.pls.inv.sv <- pls.inv.sv$variates$X %>%
    umap(y = pls.inv.sv$variates$Y,
         min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.pls.inv.sv.cat <- cbind(pls.inv.sv$variates$X, 
                              pls.inv.sv$variates$Y) %>%
    umap(min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
}


# 2 sPCS dim + 30nn + cosine
nn = 15
nt = 300
{
  u.biophys <- PropMatrix %>%
    umap(metric = "cosine", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = 30, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.ipca <- ipca.res$x %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.sipca <- sipca.res$x %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.ipca.all <- ipca.all$variates$X %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.sipca.all <- sipca.all.res$variates$X %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.reg <- spls.reg$variates$X %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.reg.all <- spls.reg$variates$X %>%
    umap(y = spls.reg$variates$Y, min_dist = 1,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.reg.allcat <- cbind(spls.reg$variates$X, spls.reg$variates$Y) %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.can <- spls.can$variates$X %>%
    umap(metric = "euclidean", 
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
         metric = "euclidean", 
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.inv <- spls.inv$variates$X %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.inv.all <- spls.inv$variates$X %>%
    umap(y = spls.reg$variates$Y,
         metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.spls.inv.allcat <- cbind(spls.inv$variates$X, spls.inv$variates$Y) %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.inv.spls.reg <- inv.spls.reg$variates$Y %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.inv.spls.can <- inv.spls.can$variates$Y %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.inv.spls.inv <- inv.spls.inv$variates$Y %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  
  u.shrink <-  shrink.res$variates$X %>%
    umap(metric = "euclidean", 
         n_threads = 3, n_sgd_threads = 3,
         n_trees = nt,
         n_neighbors = nn, 
         approx_pow = T, 
         ret_model = T, 
         verbose = T,
         ret_nn = T)
  u.inv.shrink <- inv.shrink.res$variates$Y %>%
    umap(metric = "euclidean", 
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


#-------------------- Plot seq2vec embeddings  --------------------
dat <- cbind(sv, sv.sif, sv.tfidf, sv.CCR)

# u.extreme.nn <- dat %>%
#   umap(metric = "euclidean",
#        n_threads = 3, n_sgd_threads = 3,
#        n_trees = 300,
#        n_neighbors = 500,
#        approx_pow = T,
#        ret_model = T,
#        verbose = T,
#        min_dist = 1,
#        ret_nn = T)

u.CCR <- sv.CCR %>%
  umap(init = "agspectral", negative_sample_rate = 9, repulsion_strength = 50,
       metric = "euclidean",
       n_threads = 3, n_sgd_threads = 3,
       n_trees = 300, n_epochs = 500,
       n_neighbors = 7, 
       approx_pow = T,
       ret_model = T, 
       verbose = T,
       min_dist = 1,
       ret_nn = T)

u.sv <- sv %>%
  umap(init = "agspectral", negative_sample_rate = 9, repulsion_strength = 50,
       metric = "euclidean",
       n_threads = 3, n_sgd_threads = 3,
       n_trees = 300, n_epochs = 500,
       n_neighbors = 7, 
       approx_pow = T,
       ret_model = T, 
       verbose = T,
       min_dist = 1,
       ret_nn = T)

u.sv2CCR <- sv %>%
  umap(y = sv.CCR, scale = "colrange",
       init = "agspectral", negative_sample_rate = 9, repulsion_strength = 50,
       metric = "euclidean",
       n_threads = 3, n_sgd_threads = 3,
       n_trees = 300, n_epochs = 500,
       n_neighbors = 5, target_n_neighbors = 5,
       approx_pow = T,
       ret_model = T, 
       verbose = T,
       min_dist = 1,
       ret_nn = T)
# -------------------- << Bookmark >> ----------------------------

u.dat <- dat %>%
  umap(init = "agspectral",
       metric = "euclidean", 
       negative_sample_rate = 9, repulsion_strength = 50,
       n_threads = 3, n_sgd_threads = 3,
       n_trees = 300, 
       n_neighbors = 2, 
       approx_pow = T,
       ret_model = T, 
       verbose = T,
       min_dist = 1,
       ret_nn = T)

u.dat.2 <- cbind(sv.tfidf, sv.sif) %>%
  umap(y = sv.CCR, 
       init = "agspectral", scale = "colrange",
       metric = "euclidean", target_metric = "euclidean", 
       negative_sample_rate = 9, repulsion_strength = 50,
       n_threads = 3, n_sgd_threads = 3,
       n_trees = 300, 
       n_neighbors = 2, target_n_neighbors = 2,
       approx_pow = T,
       ret_model = T, 
       verbose = T,
       min_dist = 1,
       ret_nn = T)

u.sup <- dat %>%
  umap(y = PropMatrix, scale = "colrange",
       init = "agspectral", 
       metric = "euclidean", target_metric = "cosine",
       n_threads = 3, n_sgd_threads = 3,
       n_trees = 500, 
       n_epochs = 500,
       n_neighbors = 3, target_n_neighbors = 100,
       approx_pow = T,
       ret_model = T,
       verbose = T,
       min_dist = 1,
       ret_nn = T)

u.sup.inv <- PropMatrix %>%
  umap(y = dat, scale = "colrange",
       metric = "cosine", target_metric = "euclidean",
       n_threads = 3, n_sgd_threads = 3,
       n_trees = 500, 
       n_epochs = 500,
       n_neighbors = 50, target_n_neighbors = 3,
       approx_pow = T,
       ret_model = T,
       verbose = T,
       min_dist = 1,
       ret_nn = T)

u.unsup <- dat %>% 
  cbind(PropMatrix) %>%
  umap(metric = list(euclidean = 1:ncol(dat), 
                     cosine = (ncol(dat)+1) : (ncol(dat)+ncol(PropMatrix)) ),
       scale = "colrange",
       n_trees = 300, 
       n_epochs = 500,
       n_neighbors = 10, 
       n_threads = 3, n_sgd_threads = 3,
       approx_pow = T,
       ret_model = T,
       verbose = T,
       min_dist = 1,
       ret_nn = T)

#-------------------- Plot  biophys --------------------
# u.dat
a <- u.scop


a$nn$cosine %>% lapply(head)
a$nn$euclidean %>% lapply(head)
{
  plot(cumsum(a$nn$cosine$dist[1,]), type = "s", main = "Cumulative cosine distance sum of umap nn")
  abline(v = 5)
  
}
{
  plot(cumsum(a$nn$euclidean$dist[1,]), type = "s", main = "Cumulative euclidean distance sum of umap nn")
  abline(v = 5)
  
}
df <- data.frame(X1 = a$embedding[,1],
                 X2 = a$embedding[,2]) %>% na.omit()
df <- cbind(df,PropMatrix0) %>%
  na.omit() %>%
  as_tibble()


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


ggplot(df, aes(x=X1, y=X2, color=log10(protein.length))) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=Hydrophobicity)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
# ggplot(df, aes(x=X1, y=X2, color=aliphatic.index)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c()


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
# ggplot(df, aes(x=X1, y=X2, color=Charged.Mole.percent)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=Basic.Mole.percent)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()
ggplot(df, aes(x=X1, y=X2, color=Acidic.Mole.percent)) +
  geom_point() +
  theme_gray() +
  scale_color_viridis_c()

colnames(df)

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
# ggplot(df, aes(x=X1, y=X2, color=BLOSUM2)) +
#   geom_point() +
#   theme_gray() +
#   scale_color_viridis_c() + 
#   ggtitle("BLOSUM2")
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
