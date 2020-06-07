### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  embedding based on trained weights
#               (sequence representation based on average token weights)
# input:        sequences, word-IDs, weights
# output:       seq2vec embedding
# author:       HR

library(dplyr)

print("### SEQ2VEC EMBEDINGS ###")


### INPUT ###
currentSeqs = read.csv(file = snakemake@input[["formatted_sequence"]],
                       stringsAsFactors = F, header = T)
SEQ = read.csv(file = snakemake@input[["SEQ"]],
                       stringsAsFactors = F, header = T)

### MAIN PART ###
sequence.repres = SEQ[which(SEQ$Accession %in% currentSeqs$Accession), ]

### OUTPUT ###
# vector representation of sequences
write.csv(sequence.repres, file = unlist(snakemake@output[["embedding_seq2vec"]]), row.names = F)