### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  calculate rate of antigen representation (rPCP)
# input:        PEAKS search results, proteome, info about transcript-protein
# output:       table with immunopeptides
# author:       HR

setwd("/home/hroetsc/Documents/ProtTransEmbedding")

library(seqinr)
library(dplyr)

### INPUT ###
peptides = read.csv("./mouse_IP/Mouse_lymphioma_IP_01.2020_PEAKS_86/proteins.csv", stringsAsFactors = F, header = T)
proteins = read.fasta("./mouse_IP/expr_prot_incl_deep_nodup.fasta",seqtype = "AA", whole.header = T)
info = read.csv("./mouse_IP/gene_tr_prot.csv", stringsAsFactors = F, header = T)

#### MAIN PART ###
seqs = c()
origin = c()
for (e in 1:length(proteins)) {
  seqs = c(seqs, paste(proteins[[e]], sep = "", collapse = ""))
  origin = c(origin, getAnnot(proteins[[e]]))
}
proteins = as.data.frame(cbind(origin, seqs))
for (i in 1:nrow(proteins)) {
  proteins[i,"length"] = nchar(as.character(proteins$seqs[i]))
}

# add protein length to peptides table
for (j in 1:nrow(peptides)) {
  peptides[j,"prot_len"] = proteins[which(peptides$Accession[j] %in% proteins$origin),"length"]
}
