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
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)

### MAIN PART ###
# order accessions alphabetically
sequences = sequences[order(sequences$Accession), ]

# retrieve GO terms for every UniProtID
ensembl = useMart("ensembl", dataset = "mmusculus_gene_ensembl")
GOterms = getBM(mart = ensembl,
                attributes = c("go_id", "uniprotswissprot", "uniprotsptrembl"))

# filter data set
GO_terms_sp = GOterms[, c(1:2)]
colnames(GO_terms_sp) = c("GOterm", "Accession")
sequences_sp = left_join(sequences, GO_terms_sp)
sequences_sp = na.omit(sequences_sp)

GO_terms_tr = GOterms[, c(1,3)]
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

# calculate mean GO-term similarity
alig = parGOSim(terms,
                type = "go", organism = "mouse",
                measure = "Wang")

# scale all values between 0 and 1
alig = (alig - min(alig)) / (max(alig) - min(alig))

res = matrix(ncol = ncol(alig)+1, nrow = nrow(alig))
res[, 1] = sequences$Accession
res[, c(2:ncol(res))] = alig
colnames(res) = c("Accession", seq(1, ncol(alig)))

### OUTPUT ###
write.csv(res, file = unlist(snakemake@output[["semantics"]]), row.names = T)