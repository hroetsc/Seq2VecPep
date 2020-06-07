### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  weight seq2vec embeddings by TF-IDF and SIF
# input:        sequence embeddings
# output:       embeddings weighted by TF-IDF, embeddings weighted by SIF
# author:       HR

library(dplyr)

print("### WEIGHTING OF SEQ2VEC EMBEDDINGS ###")

### INPUT ###
currentSeqs = read.csv(file = snakemake@input[["formatted_sequence"]],
                     stringsAsFactors = F, header = T)

SIF = read.csv(file = snakemake@input[["SIF"]],
               stringsAsFactors = F, header = T)
TFIDF = read.csv(file = snakemake@input[["TFIDF"]],
                 stringsAsFactors = F, header = T)

### MAIN PART ###

SIF = SIF[which(SIF$Accession %in% currentSeqs$Accession), ]
TFIDF = TFIDF[which(TFIDF$Accession %in% currentSeqs$Accession), ]


### OUTPUT ###
write.csv(SIF, file = unlist(snakemake@output[["seq2vec_SIF"]]), row.names = F)
write.csv(TFIDF, file = unlist(snakemake@output[["seq2vec_TFIDF"]]), row.names = F)
