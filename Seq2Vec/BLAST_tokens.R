### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  run BALSTp search for long tokens --
# input:        tokens
# output:       search results
# author:       HR

library(dplyr)
library(Biostrings)
library(seqinr)
library(readr)
library(stringr)

### INPUT ###
indices = read.csv("ids_hp_v50k_w5.csv", stringsAsFactors = F, header = F)


### MAIN PART ###
# select long tokens
indices$V1 = indices$V1 %>% toupper()
indices = indices[nchar(indices$V1) >= 10, ]

writeFasta<-function(data, filename){
  fastaLines = c()
  for (rowNum in 1:nrow(data)){
    fastaLines = c(fastaLines, as.character(paste(">", data[rowNum,"name"], sep = "")))
    fastaLines = c(fastaLines,as.character(data[rowNum,"seq"]))
  }
  fileConn<-file(filename)
  writeLines(fastaLines, fileConn)
  close(fileConn)
}

# export as fasta
names(indices) = c("seq", "name")
writeFasta(indices, "blastp_tokens.fasta")


# run blastp, swissprot as search set
system("blastp -db swissprot -query blastp_tokens.fasta -outfmt '6 sscinames scomnames sskingdoms sseqid ssac qstart qend sstart send qseq evalue pident' -out blastp_results.csv -remote")

# load results
results = read.table("blastp_results.csv", sep = "\t", stringsAsFactors = F)
names(results) = c("sscinames", "scomnames", "sskingdoms", "sseqid", "qstart", "qend", "sstart", "send", "qseq", "evalue", "pident")
results$sscinames = NULL
results$scomnames = NULL
results$sskingdoms = NULL

# reference table
ref = read.fasta("../../../files/SwissProt_canonicalAndIsoforms.fasta", whole.header = T, seqtype = "AA")

ref.master = data.frame(sseqid = rep(NA, length(ref)),
                        meta = rep(NA, length(ref)))

ref.master$sseqid = str_split_fixed(names(ref), coll("|"), Inf)[, 2] %>% as.character()
ref.master$meta = str_split_fixed(names(ref), coll("|"), Inf)[, 3] %>% as.character()

results$sseqid = str_split_fixed(results$sseqid, coll("|"), Inf)[, 2] %>% as.character()
results$sseqid = str_split_fixed(results$sseqid, coll("."), Inf)[, 1]

# join
results.master = left_join(results, ref.master)
results.master$organism = str_sub(results.master$meta,
                                  str_locate(results.master$meta, "OS")[,1],
                                  nchar(results.master$meta))

# evalue cutoff
results.master = results.master[results.master$evalue <= 0.001, ]

### OUTPUT ###
write.csv(results.master, "blastp_results_meta.csv", row.names = F)




