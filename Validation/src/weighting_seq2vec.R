### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  weight seq2vec embeddings by TF-IDF and SIF
# input:        sequence embeddings
# output:       embeddings weighted by TF-IDF, embeddings weighted by SIF
# author:       HR

library(dplyr)

print("### PICK RELEVANT SEQ2VEC EMBEDDINGS ###")

### INPUT ###
currentSeqs = read.csv(file = snakemake@input[["formatted_sequence"]],
                     stringsAsFactors = F, header = T)

seq2vec = read.csv(file = snakemake@input[["seq2vec"]], stringsAsFactors = F, header = T)
seq2vec_TFIDF = read.csv(file = snakemake@input[["seq2vec_TFIDF"]], stringsAsFactors = F, header = T)
seq2vec_SIF = read.csv(file = snakemake@input[["seq2vec_SIF"]], stringsAsFactors = F, header = T)

seq2vec_CCR = read.csv(file = snakemake@input[["seq2vec_CCR"]], stringsAsFactors = F, header = T)
seq2vec_TFIDF_CCR = read.csv(file = snakemake@input[["seq2vec_TFIDF_CCR"]], stringsAsFactors = F, header = T)
seq2vec_SIF_CCR = read.csv(file = snakemake@input[["seq2vec_SIF_CCR"]], stringsAsFactors = F, header = T)

### MAIN PART ###

filter <- function(df = "") {
  df = df[which(df$Accession %in% currentSeqs$Accession), ]
  return(df)
}

seq2vec = filter(seq2vec)
seq2vec_TFIDF = filter(seq2vec_TFIDF)
seq2vec_SIF = filter(seq2vec_SIF)


seq2vec_CCR = filter(seq2vec_CCR)
seq2vec_TFIDF_CCR = filter(seq2vec_TFIDF_CCR)
seq2vec_SIF_CCR = filter(seq2vec_SIF_CCR)


### OUTPUT ###
write.csv(seq2vec, file = unlist(snakemake@output[["seq2vec"]]), row.names = F)
write.csv(seq2vec_TFIDF, file = unlist(snakemake@output[["seq2vec_TFIDF"]]), row.names = F)
write.csv(seq2vec_SIF, file = unlist(snakemake@output[["seq2vec_SIF"]]), row.names = F)

write.csv(seq2vec_CCR, file = unlist(snakemake@output[["seq2vec_CCR"]]), row.names = F)
write.csv(seq2vec_TFIDF_CCR, file = unlist(snakemake@output[["seq2vec_TFIDF_CCR"]]), row.names = F)
write.csv(seq2vec_SIF_CCR, file = unlist(snakemake@output[["seq2vec_SIF_CCR"]]), row.names = F)
