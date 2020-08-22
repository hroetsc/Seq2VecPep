### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  "true" similarity between protein sequences based on pairwise sequence alignment
#               (Needleman - Wunsch)
# input:        sequences
# output:       syntactic similarity matrix
# author:       HR

library(dplyr)
library(tidyr)
library(Biostrings)
library(protr)
library(stringr)

library(future)
library(foreach)
library(doParallel)


print("### TRUE SYNTACTIC SIMILARITY")

### INPUT ###
# formatted sequences
seqs = read.csv(snakemake@input[["batch_sequence"]], stringsAsFactors = F, header = T)
accessions = read.csv(snakemake@input[["batch_accessions"]], stringsAsFactors = F, header = T)


### MAIN PART ###
# order accessions alphabetically
seqs = seqs[order(seqs$Accession), ]


# parallelised protein sequence similarity calculation
threads = 4

cl <- makeCluster(threads)
registerDoParallel(cl)

accessions$similarity = NULL

res = matrix(ncol = 3, nrow = nrow(accessions)) %>% as.data.frame()
res = foreach (n = 1:nrow(accessions), .combine = "rbind") %dopar% {
  
  prot1 = seqs[which(seqs$Accession == accessions$acc1[n]), "seqs"]
  prot2 = seqs[which(seqs$Accession == accessions$acc2[n]), "seqs"]
  
  s = Biostrings::pairwiseAlignment(Biostrings::AAString(prot1),
                                    Biostrings::AAString(prot2),
                                    type = "local",
                                    gapOpening = 8,
                                    gapExtension = -8,
                                    substitutionMatrix = "BLOSUM50",
                                    scoreOnly = T)
  
  res[n,] = c(accessions$acc1[n], accessions$acc2[n], s)
  
} %>% as.data.frame()

stopCluster(cl)
stopImplicitCluster()

colnames(res) = c("acc1", "acc2", "similarity")

### OUTPUT ###
write.csv(res, file = unlist(snakemake@output[["syntax"]]), row.names = F)
