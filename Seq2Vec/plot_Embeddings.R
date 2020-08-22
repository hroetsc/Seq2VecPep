### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  dimension reduction and plotting
# input:        protein matrices, biophysical properties, enzyme commission no., SCOP, ...
# output:       some nice plots
# author:       HR


library(seqinr)
library(protr)
library(UniProt.ws)
library(Peptides)
library(plyr)
library(dplyr)
library(stringr)
library(tibble)
library(readr)
library(uwot)
library(future)
library(paletteer)
library(foreach)
library(doParallel)
registerDoParallel(cores=availableCores())

### INPUT ###
fs = list.files(pattern = "hp_v50k_sequence_repres", full.names = T)

f1 = read.csv(fs[1], stringsAsFactors = F, header = T)
Prots = f1$Accession

########## biophysical properties ########## 

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
PropMatrix0 = foreach(i=1:length(f1$seqs), 
                      .packages = c("Peptides")) %dopar% PepSummary(f1$seqs[i])
PropMatrix0 <- bind_rows(PropMatrix0)
PropMatrix0$seqs = rownames(PropMatrix0)


########## enzyme commission numbers ##########

# enzyme commission number
# data from UniProt web page
ref = read_file("../../../files/SwissProt_Human_canonicalAndIsoforms_meta.tab")
ref = str_split(ref, coll("\n"), simplify = T) %>% as.data.frame() %>% t()
ref = str_split_fixed(ref, coll("\t"), Inf)

colnames(ref) = ref[1,]
ref = ref[-1,]

EC = ref[, c("Entry", "EC number")] %>% as.data.frame()
colnames(EC) = c("Accession", "EC_number")
EC$EC_number = as.character(EC$EC_number)


EC = EC[-which(EC$EC_number == ""),]

master = str_split_fixed(Prots, coll("-"), Inf) %>%  
  as.character() %>% 
  unique() %>%
  as.data.frame()
colnames(master) = "Accession"

master = left_join(master, EC)
master$EC_number = str_split_fixed(master$EC_number, coll("."), Inf)[,1]
master[which(master$EC_number == ""), "EC_number"] = "0"
master$Accession = as.character(master$Accession)

########## SCOP: structural clafssification of proteins ##########
# Load SCOP classification
scop <- readr::read_delim("../../../files/scop_cla_latest.txt",skip = 5, delim = " ")
header = colnames(scop)
header = header[-1]

scop$SCOPCLA = NULL
colnames(scop) = header

scop_class <- scop$SCOPCLA %>%
  str_remove_all(pattern = "TP=") %>%
  str_remove_all(pattern = "CL=") %>%
  str_remove_all(pattern = "CF=") %>%
  str_remove_all(pattern = "SF=") %>%
  str_remove_all(pattern = "FA=") %>% 
  str_split_fixed(pattern = ",", Inf)
scop_class <- as_tibble(scop_class) %>% as.data.frame()
scop_class = cbind(scop$`FA-UNIID`, scop_class)
colnames(scop_class) <- c("Accession","TP", "CL", "CF", "SF", "FA")
scop_class$Accession = as.character(scop_class$Accession)
# TP=protein type, CL=protein class, CF=fold, SF=superfamily, FA=family

# SCOP node descriptions
scop_des <- readr::read_delim("../../files/scop_des_latest.txt", delim = " ")

# convert headers into UniProt identifiers
Prots = str_split_fixed(Prots, coll("-"), Inf)[,1]
intersect(scop_class$Accession, Prots) %>% length()


########## plotting ##########

spectral = RColorBrewer::brewer.pal(10, "Spectral")

color.gradient = function(x, colsteps=1000) {
  return( colorRampPalette(spectral) (colsteps) [ findInterval(x, seq(min(x),max(x), length.out=colsteps)) ] )
}


discrete.cols = function(data = ""){
  co = c(rep(NA, length(data)))
  
  co[which(data == "0")] = "cornsilk" # no enzyme
  co[which(data == "1")] = "blue4" # oxidoreductases
  co[which(data == "2")] = "cyan" # transferases
  co[which(data == "3")] = "darkseagreen" # hydrolases
  co[which(data == "4")] = "gold" #lyases
  co[which(data == "5")] = "firebrick" # isomerases
  co[which(data == "6")] = "deeppink" #ligases
  co[which(data == "7")] = "darkgreen" #translocases
  
  return(co)
}

UMAP = function(tbl = ""){
  set.seed(42)
  
  return(
    umap(tbl,
         n_neighbors = 10,
         min_dist = 1,
         n_trees = 1000,
         metric = "euclidean",
         verbose = T,
         approx_pow = T,
         ret_model = T,
         init = "spca",
         n_threads = availableCores())$embedding
  )
}



plotting_EC = function(tbl = "", prop = "", nm = "", col_by = ""){
  
  png(filename = paste0("plots/", nm, "_", prop, ".png"),
      width = 2000, height = 2000, res = 300)
  
  plot(tbl,
       col = discrete.cols(col_by),
       cex = 0.1,
       pch = 1,
       xlab = "UMAP 1", ylab = "UMAP 2",
       main = paste0("human proteins: ", nm),
       sub = paste0("colored by ", prop))
  
  dev.off()
}


plotting_SCOP = function(tbl = "", prop = "", nm = "", col_by = ""){
  
  if (str_detect(prop, ".")){
    prop = str_replace_all(prop, coll("."), coll("_"))
  }
  
  nColor = length(levels(as.factor(col_by)))
  
  colors = paletteer_c("viridis::inferno", n = nColor)
  
  col_by = as.numeric(as.character(col_by))
  rank = as.factor( as.numeric( col_by ))
  
  png(filename = paste0("plots/", nm, "_", prop, ".png"),
      width = 2000, height = 2000, res = 300)
  
  plot(tbl,
       col = colors[ rank ],
       cex = 0.1,
       pch = 1,
       xlab = "UMAP 1", ylab = "UMAP 2",
       main = paste0("human proteins: ", nm),
       sub = paste0("colored by ", prop))
  
  dev.off()
}

fs
fs = fs[c(3,5,6)]

for (i in 1:length(fs)){
  f = read.csv(fs[i], stringsAsFactors = F, header = T)
  
  
  f = f[order(f$Accession), ]
  f$Accession = str_split_fixed(f$Accession, coll("-"), Inf)[,1]
  
  # join with EC number
  f = left_join(f, master)
  f$EC_number[which(f$EC_number == "")] = 0

  table(f$EC_number)

  # join with SCOP
  f = left_join(f, scop_class)
  f[is.na(f)] = 0
  f = left_join(f, PropMatrix0)
  
  
  um = UMAP(tbl = f[, c(4:131)])
  
  x = str_locate(fs[i], "seq2vec")
  nm = str_sub(fs[i], start = x[1])
  nm = str_split(nm, coll("."), simplify = T)[1]
  
  
  if(!dir.exists("plots")){
    dir.create("plots")
  }
  
  plotting_EC(tbl = um, prop = "EC_number", nm = paste0(nm, "_EC"),
            col_by = f$EC_number)
  
  col_tbl = f[, c(132:ncol(f))]
  
  for (j in 1:ncol(col_tbl)){
    
    plotting_SCOP(tbl = um, prop = colnames(col_tbl)[j],
                  nm = nm,
                  col_by = col_tbl[,j])
  }
  
  
  # for (p in 1:length(keep)){
  #   plotting(tbl = um, prop = colnames(Props)[p], nm = nm)
  #   
  # }
  
}

### OUTPUT ### 
write.csv(PropMatrix0, "PropMatrix_seqs.csv", row.names = T)

