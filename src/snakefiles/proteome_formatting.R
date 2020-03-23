### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  format whole mouse proteome and assign rPCP to antigens
# input:        mouse UniProt (NOT SwissProt) proteome, rPCP table
# output:       table with: UniProtID, description, source (sp = SwissProt, tr = TrEMBL),
#               whether antigen or not, (mean) rPCP, sequence
# author:       HR

print("### PROTEOME FORMATTING ###")

library(dplyr)
library(seqinr)
library(stringr)
library(protr)

# tmp !!!
# proteome = read.fasta(file = "data/peptidome/SwissProt_mouse_canonicalAndIsoforms.fasta",
#                       seqtype = "AA", strip.desc = T, whole.header = F)
# rPCP = read.csv(file = "data/peptidome/rPCP_03_2020.csv", stringsAsFactors = F, header = T)
# gencode_annot = read.table("data/peptidome/gencode_vM24_TrEMBL_annot.txt", stringsAsFactors = F, header = F)
# biomart_annot = read.csv(file = "data/peptidome/biomaRt_annot.csv", stringsAsFactors = F, header = T)

### INPUT ###
proteome = read.fasta(file = snakemake@input[["UniProt_filtered"]],
                                 seqtype = "AA", strip.desc = T, whole.header = F)
rPCP = read.csv(file = snakemake@input[["rPCP"]], stringsAsFactors = F, header = T)
gencode_annot = read.table(file = snakemake@input[["gencode_annot"]], stringsAsFactors = F, header = F)
biomart_annot = read.csv(file = snakemake@input[["biomart_annot"]], stringsAsFactors = F, header = T)

if (nrow(gencode_annot[which(!gencode_annot$V2==gencode_annot$V3),]) == 0){
  gencode_annot$V3 = NULL
  colnames(gencode_annot) = c("ensembl_transcript", "UniProtID")
} else {
  colnames(gencode_annot) = c("ensembl_transcript", "UniProtID", "")
}

### MAIN PART ###
# extract sequences and annotations from FASTA file
print("EXTRACT SEQUENCES AND ANNOTATION FROM FASTA FILE")
seqs = c()
origin = c()
for (e in 1:length(proteome)) {
  seqs = c(seqs, paste(proteome[[e]], sep = "", collapse = ""))
  origin = c(origin, getAnnot(proteome[[e]]))
}

# split annotation into UniProtID and description
print("FORMAT REFERENCE TABLE")
ref.table = as.data.frame(cbind(origin, seqs))
for (r in 1:nrow(ref.table)) {
  tmp = str_split(ref.table$origin[r], coll("|"), simplify = T)
  ref.table[r, "source"] = tmp[,1]
  ref.table[r, "UniProtID"] = tmp[,2]
  ref.table[r, "description"] = tmp[,3]
}
ref.table$origin = NULL
ref.table[which(ref.table$source == "sp"), "source"] = "SwissProt"
ref.table[which(ref.table$source == "tr"), "source"] = "TrEMBL"

# add ENSEMBL gene IDs
annotation_master = left_join(biomart_annot, gencode_annot)
annotation_master$uniprotsptrembl = NULL
annotation_master[which(is.na(annotation_master$UniProtID)), "UniProtID"] = annotation_master[which(is.na(annotation_master$UniProtID)), "uniprotswissprot"]
annotation_master$uniprotswissprot = NULL
annotation_master[which(annotation_master$UniProtID == ""), "UniProtID"] = annotation_master[which(annotation_master$UniProtID == ""), "ensembl_peptide_id_version"]
colnames(annotation_master) = c("gene_id", "transcript_id", "peptide_id", "UniProtID_reduced")

for (r in 1:nrow(ref.table)) {
  ref.table[r, "UniProtID_reduced"] = str_split(ref.table$UniProtID[r], coll("-"), simplify = T)[1]
}
ref.table = left_join(ref.table, annotation_master, by = "UniProtID_reduced")
ref.table = ref.table[-which(duplicated(ref.table$seqs)), ]

# iterate reference proteome to identify antigens
print("ASSIGN rPCP TO REFERENCE TABLE")
progressBar = txtProgressBar(min = 0, max = nrow(ref.table), style = 3)
for (r in 1:nrow(ref.table)) {
  setTxtProgressBar(progressBar, r)
  # get rid of protein version
  currentProt_UniProt = str_split(ref.table$UniProtID[r], coll("-"), simplify = T)[1]
  currentProt_ENSEMBL = ref.table$peptide_id[r]
  
  # search for current protein in rPCP table
  tmp = rPCP[which(currentProt_UniProt %in% rPCP$TrEMBLID | currentProt_ENSEMBL %in% rPCP$Accession), ]
  if (nrow(tmp) == 0) { # if no search results the current protein is not an antigen
    ref.table[r, "rPCP"] = 0
    ref.table[r, "unique_rPCP"] = 0
    ref.table[r, "class"] = "protein"
    
  } else if (nrow(tmp) == 1){ # if it is an antigen add rPCPs:
    ref.table[r, "rPCP"] = tmp$rPCP[1]
    ref.table[r, "unique_rPCP"] = tmp$unique_rPCP[1]
    ref.table[r, "class"] = "antigen"
    
  } else {
    print(paste0("WARNING: found ", nrow(tmp), " hits for protein ", currentProt_UniProt, " in antigen list"))
    ref.table[r, "rPCP"] = tmp$rPCP[1]
    ref.table[r, "unique_rPCP"] = tmp$unique_rPCP[1]
    ref.table[r, "class"] = "antigen"
  }
}

ref.table = unique(ref.table)
ref.table$gene_id = NULL
ref.table$transcript_id = NULL
ref.table[which(is.na(ref.table$peptide_id)), "peptide_id"] = ""
ref.table = na.omit(ref.table)

# apply protcheck()
ref.table$seqs = as.character(ref.table$seqs)
a <- sapply(toupper(ref.table$seqs), protcheck)
names(a) <- NULL

print(paste0("found ",length(which(a==F)) , " proteins that are failing the protcheck() and is removing them"))

# clean data sets
ref.table = ref.table[which(a==T), ]

### OUTPUT ###
write.csv(ref.table, file = unlist(snakemake@output[["formatted_proteome"]]), row.names = F)

# tmp!
# write.csv(ref.table, file = "data/peptidome/formatted_proteome.csv", row.names = F)
