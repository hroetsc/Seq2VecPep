### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  "true" similarity between protein sequences based on pairwise sequence alignment
#               (Clustal Omega)
# input:        sequences
# output:       syntactic similarity matrix
# author:       HR

library(protr)
library(parallel)
library(Biostrings)
library(ape)
library(msa)
 
# library(foreach)
# library(doParallel)
# library(doMC)
# library(plyr)
# 
# cl <- makeCluster(16)
# registerDoParallel(cl)
# registerDoMC(16)

### INPUT ###
# formatted sequences
sequences = read.csv(snakemake@input[["sequence"]], stringsAsFactors = F, header = T)
# sequences = read.csv("proteome/data/red_formatted_proteome.csv", stringsAsFactors = F, header = T)
# data("BLOSUM62")

### MAIN PART ###
# order accessions alphabetically
sequences = sequences[order(sequences$Accession), ]

# create matrix
# alig = matrix(ncol = nrow(sequences), nrow = nrow(sequences))
# colnames(alig) = sequences$Accession
# rownames(alig) = sequences$Accession
# 
# alig = foreach(i = 1:10, .combine = "rbind") %dopar% {
#   
#   for (j in 1:nrow(sequences)){
#     tmp = msaClustalOmega(sequences$seqs[c(i,j)], type = "protein")
#     x = msaConservationScore(tmp, BLOSUM62)
#     alig[i,j] = sum(x) / length(x)
#   }
#   
# }


# parallelised protein sequence similarity calculation
cores = detectCores()
alig = parSeqSim(sequences$seqs,
                 cores = cores,
                 batches = 100, # to save memory
                 verbose = T,
                 type = "local",
                 gap.opening = -2,
                 gap.extension = -8,
                 submat = "BLOSUM50")


res = matrix(ncol = ncol(alig)+1, nrow = nrow(alig))
res[, 1] = sequences$Accession
res[, c(2:ncol(res))] = alig
colnames(res) = c("Accession", seq(1, ncol(alig)))


### OUTPUT ###
write.csv(res, file = unlist(snakemake@output[["syntax"]]), row.names = T)