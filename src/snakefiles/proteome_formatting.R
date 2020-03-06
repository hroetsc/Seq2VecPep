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

# tmp !!!
# proteome = read.fasta(file = "data/peptidome/UniProt_mouse_canonicalAndIsoforms.fasta",
#                       seqtype = "AA", strip.desc = T, whole.header = F)
# rPCP = read.csv(file = "data/peptidome/rPCP_01_2020.csv", stringsAsFactors = F, header = T)

### INPUT ###
proteome = read.fasta(file = snakemake@input[["UniProt_unfiltered"]],
                                 seqtype = "AA", strip.desc = T, whole.header = F)
rPCP = read.csv(file = snakemake@input[["rPCP"]], stringsAsFactors = F, header = T)

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
# contains protein versions!

# assign rPCP to antigens
print("ASSIGN rPCP TO REFERENCE TABLE")
progressBar = txtProgressBar(min = 0, max = nrow(ref.table), style = 3)
for (r in 1:nrow(ref.table)) {
  setTxtProgressBar(progressBar, r)
  # get rid of protein version
  currentProt = str_split(ref.table$UniProtID[r], coll("-"), simplify = T)
  currentProt = currentProt[,1]

  # search for current protein in rPCP table
  # ... if it is a reviewed protein (SwissProt)
  tmp = rPCP[which(rPCP$TrEMBLID %in% currentProt), ]
  if (nrow(tmp) == 0) { # if no search results the current protein is not an antigen
    ref.table[r, "unique_rPCP"] = 0
    ref.table[r, "shared_rPCP"] = 0
    ref.table[r, "total_rPCP"] = 0
    ref.table[r, "class"] = "protein"
    
  } else { # if it is an antigen calculate rPCPs as follows:
    shared_rPCP = mean(unique(tmp$rPCP))
    ref.table[r, "unique_rPCP"] = paste(unique(tmp$rPCP), collapse = coll(" | "))
    ref.table[r, "shared_rPCP"] = shared_rPCP
    ref.table[r, "total_rPCP"] = paste(rep(shared_rPCP, length(unique(tmp$rPCP))) + unique(tmp$rPCP), collapse = coll(" | "))
    ref.table[r, "class"] = "antigen"
  }
}

ref.table = unique(ref.table)
ref.table = na.omit(ref.table)
### OUTPUT ###
write.csv(ref.table, file = unlist(snakemake@output[["formatted_proteome"]]), row.names = F)
