### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  "true" similarity between protein sequences based on GO-term similarity
# input:        sequences
# output:       semantic similarity matrices (MF, BP, CC)
# author:       HR

library(protr)
library(plyr)
library(dplyr)


### INPUT ###
# formatted sequences
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)
GOterms = read.csv(snakemake@input[["GO_terms"]], stringsAsFactors = F, header = T)

### MAIN PART ###
# order accessions alphabetically
sequences = sequences[order(sequences$Accession), ]

# SwissProt matches
GO_terms_sp = GOterms[,which(colnames(GOterms) %in% c("go_id", "uniprotswissprot"))]
colnames(GO_terms_sp) = c("GOterm", "Accession")
sequences_sp = left_join(sequences, GO_terms_sp)
sequences_sp = na.omit(sequences_sp)

# Trembl matches
GO_terms_tr = GOterms[,which(colnames(GOterms) %in% c("go_id", "uniprotsptrembl"))]
colnames(GO_terms_tr) = c("GOterm", "Accession")
sequences_tr = left_join(sequences, GO_terms_tr)
sequences_tr = na.omit(sequences_tr)

sequences = full_join(sequences_sp, sequences_tr)
print(paste0("found GO terms for ", length(unique(sequences$Accession)), " proteins"))


# create master table for GO term similarity for each protein
sequences = sequences[order(sequences$Accession), ]
proteins = unique(sequences$Accession)

GOterms = split.data.frame(sequences, sequences$Accession)
terms = list()
progressBar = txtProgressBar(min = 0, max = length(proteins), style = 3)

for (p in 1:length(proteins)){
  setTxtProgressBar(progressBar, p)
  terms[[p]] = GOterms[[p]][, "GOterm"]
}

### molecular function ###
alig_MF = parGOSim(golist = GOterms,
                   type = "go", organism = "mouse",
                   measure = "Wang", ont = "MF")

res_MF = matrix(ncol = ncol(alig_MF)+1, nrow = nrow(alig_MF))
res_MF[, 1] = proteins
res_MF[, c(2:ncol(res_MF))] = alig_MF
colnames(res_MF) = c("Accession", seq(1, ncol(alig_MF)))

### biological process ###
alig_BP = parGOSim(golist = GOterms,
                   type = "go", organism = "mouse",
                   measure = "Wang", ont = "BP")

res_BP = matrix(ncol = ncol(alig_BP)+1, nrow = nrow(alig_BP))
res_BP[, 1] = proteins
res_BP[, c(2:ncol(res_BP))] = alig_BP
colnames(res_BP) = c("Accession", seq(1, ncol(alig_BP)))

### chemical component ###
alig_CC = parGOSim(golist = GOterms,
                   type = "go", organism = "mouse",
                   measure = "Wang", ont = "CC")

res_CC = matrix(ncol = ncol(alig_CC)+1, nrow = nrow(alig_CC))
res_CC[, 1] = proteins
res_CC[, c(2:ncol(res_CC))] = alig_CC
colnames(res_CC) = c("Accession", seq(1, ncol(alig_CC)))

### OUTPUT ###
write.csv(res_MF, file = unlist(snakemake@output[["semantics_MF"]]), row.names = F)
write.csv(res_BP, file = unlist(snakemake@output[["semantics_BP"]]), row.names = F)
write.csv(res_CC, file = unlist(snakemake@output[["semantics_CC"]]), row.names = F)