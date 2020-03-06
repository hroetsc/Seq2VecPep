### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  calculate rate of antigen representation (rPCP)
# input:        PEAKS search results, proteome, info about transcript-protein
# output:       table with rPCP values for immunopeptides
# author:       HR
print("### rPCP CALCULATION ###")

library(dplyr)
library(stringr)
library(seqinr)
#library(BiocManager)

# tmp !!!
# setwd("/home/hanna/Desktop/Snakemake/")
# PEAKS.proteins = read.csv(file = "data/peptidome/PEAKS_proteins_01_2020.csv", stringsAsFactors = F, header = T)
# ref.proteins = read.fasta(file = "data/peptidome/expr_prot_incl_deep_nodup.fasta", seqtype = "AA", whole.header = F,
#                           strip.desc = T)
# gene.tr.prot_info = read.csv(file = "data/peptidome/gene_tr_prot.csv", stringsAsFactors = F, header = T)
# gencode_annot = read.table(file = "data/peptidome/gencode_vM24_TrEMBL_annot", stringsAsFactors = F)
# biomart_annot = read.csv(file = "data/peptidome/biomaRt_annot.csv", stringsAsFactors = F, header = T)

### INPUT ###
PEAKS.proteins = read.csv(file = snakemake@input[["PEAKS_results"]], stringsAsFactors = F, header = T)
ref.proteins = read.fasta(file = snakemake@input[["ref_proteins"]], seqtype = "AA", whole.header = F,
                          strip.desc = T)
gene.tr.prot_info = read.csv(file = snakemake@input[["gene_tr_prot"]], stringsAsFactors = F, header = T)
gencode_annot = read.table(file = snakemake@input[["gencode_annot"]], stringsAsFactors = F)
if (nrow(gencode_annot[which(!gencode_annot$V2==gencode_annot$V3),]) == 0){
  gencode_annot$V3 = NULL
  colnames(gencode_annot) = c("ensembl_transcript", "UniProtID")
}
biomart_annot = read.csv(file = snakemake@input[["biomart_annot"]], stringsAsFactors = F, header = T)

### MAIN PART ###
# rPCP = number of unique matches in PEAKS.proteins divided by protein length (ref.proteins)
# extract sequences and annotations from ref.proteins
print("EXTRACT SEQUENCES AND ANNOTATION FROM FASTA FILE")
seqs = c()
origin = c()
for (e in 1:length(ref.proteins)) {
  seqs = c(seqs, paste(ref.proteins[[e]], sep = "", collapse = ""))
  origin = c(origin, getAnnot(ref.proteins[[e]]))
}
prot.ref = as.data.frame(cbind(seqs, origin))

# split annotation into UniProtID and description
print("FORMAT REFERENCE TABLE AND RETRIEVE PROTEIN LENGTH")
for (r in 1:nrow(prot.ref)) {
  tmp = str_split(prot.ref$origin[r], coll(" "), simplify = T)
  if ((tmp[,1] == t(str_split(tmp, coll("|"), simplify = T)))[1] == TRUE ){
    prot.ref[r, "protein"] = tmp[,1]
  } 
  # else if (ncol(tmp) > 5) {
  #   prot.ref[r, "gene"] = str_split(tmp[,4], coll(":"), simplify = T)[,2]
  #   prot.ref[r, "transcript"] = str_split(tmp[,5], coll(":"), simplify = T)[,2]
  # }
  else {
    tmp = str_split(tmp, coll("|"), simplify = T)
    prot.ref[r, "transcript"] = tmp[,1]
    prot.ref[r, "gene"] = tmp[,2]
  }
  prot.ref[r, "protein_length"] = nchar(as.character(prot.ref$seqs[r]))
}
prot.ref$origin = NULL

# get number of unique matches
print("CALCULATE rPCP")
progressBar = txtProgressBar(min = 0, max = nrow(PEAKS.proteins), style = 3)

for (i in 1: nrow(PEAKS.proteins)) {
  setTxtProgressBar(progressBar, i)
  
  currentProt = str_split(PEAKS.proteins$Accession[i], coll("|"), simplify = T)
  # sometimes more than one accession
  # if only one accession take the length of the corresponding protein in prot.ref to calculate rPCP
  if (ncol(currentProt) == 1){
  len = prot.ref[which(currentProt[,1] == prot.ref$protein), "protein_length"]
  len = as.numeric(as.character(len))
  PEAKS.proteins[i,"protein_length"] = len
  
    if (len > 0) {
     PEAKS.proteins[i, "rPCP"] = as.numeric(as.character(PEAKS.proteins$X.Unique[i])) / len
    } else {
      PEAKS.proteins[i, "rPCP"] = NA
   }
  # if more than one accession (no peptide match but gene/transcript) search for match
  # in gene/transcript column of prot.ref
  } else {
    len = prot.ref[which(currentProt[,1] == prot.ref$transcript), "protein_length"]
    len = as.numeric(as.character(len))
    PEAKS.proteins[i,"protein_length"] = len
    if (len > 0) {
      PEAKS.proteins[i, "rPCP"] = as.numeric(as.character(PEAKS.proteins$X.Unique[i])) / len
    } else {
      PEAKS.proteins[i, "rPCP"] = NA
    }
  }
}
antigens = PEAKS.proteins[which(!is.na(PEAKS.proteins$rPCP)),]
antigens = antigens[, c("Accession", "X.Unique", "protein_length", "rPCP")]

print("ADD TRANSCRIPT IDs")
# add transcript IDs
for (a in 1:nrow(antigens)) {
  acc = str_split(antigens$Accession[a], coll("|"), simplify = T)
  antigens[a, "ensembl_transcript"] = gene.tr.prot_info[which(acc[,1] == gene.tr.prot_info$protein |
                                                    acc[,1] == gene.tr.prot_info$TXNAME),"TXNAME"]
}

# convert ENSEMBL into UniPot IDs
print("ADD UNIPROT IDS")
antigens.gencode = left_join(antigens, gencode_annot)
biomart_annot = biomart_annot[,c("ensembl_transcript", "uniprotsptrembl")]
colnames(biomart_annot) = c("ensembl_transcript", "UniProtID")
antigens.biomart = left_join(antigens, biomart_annot)

# format table
print("FORMAT OUTPUT")
master.table = rbind(antigens.gencode, antigens.biomart)
master.table = na.omit(master.table)
master.table = master.table[-which(duplicated(master.table$ensembl_transcript)),]
colnames(master.table) = c("Accession", "unique_matches", "protein_length", "rPCP",
                           "ENSEMBL_transcriptID", "TrEMBLID")

### OUTPUT ###
# table with rPCP values and protein/transcript names and sequences of protein and IP, ...
write.csv(master.table, file = unlist(snakemake@output[["rPCP"]]), row.names = F)
