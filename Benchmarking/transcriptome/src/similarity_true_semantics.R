### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  "true" similarity between protein sequences based on GO-term similarity
# input:        sequences
# output:       semantic similarity matrix
# author:       HR

library(biomaRt)
library(protr)
library(parallel)
library(plyr)
library(dplyr)


### INPUT ###
# formatted sequences
sequences = read.csv("transcriptome/data/opt_transcriptome_human.csv", stringsAsFactors = F, header = T)
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)

### MAIN PART ###
# order accessions alphabetically
sequences = sequences[order(sequences$Accession), ]

# retrieve GO terms for every UniProtID
ensembl = useMart("ensembl", dataset = "hsapiens_gene_ensembl")
GOterms = getBM(mart = ensembl,
                attributes = c("go_id", "ensembl_transcript_id_version"))

colnames(GOterms) = c("GOterm", "Accession")

# filter data set
sequences = left_join(sequences, GOterms)
sequences = na.omit(sequences)

print(paste0("found GO terms for ", length(unique(sequences$Accession)), " transcripts"))

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

# gene_list = list()
# for (t in 1:nrow(sequences)){
#   gene_list[[t]] = sequences$Accession[t]
# }

# calculate mean GO-term similarity
alig = parGOSim(golist = GOterms,
                type = "go", organism = "human",
                measure = "Wang")

res = matrix(ncol = ncol(alig)+1, nrow = nrow(alig))
res[, 1] = proteins
res[, c(2:ncol(res))] = alig
colnames(res) = c("Accession", seq(1, ncol(alig)))

### OUTPUT ###
write.csv(res, file = unlist(snakemake@output[["semantics"]]), row.names = T)