### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  embedding based on sequence order coupling numbers
#               (quasi-sequence order)
# input:        sequences
# output:       sequence embedding using sequence-order coupling numbers
# author:       HR

library(seqinr)
library(protr)
library(Peptides)
library(plyr)
library(dplyr)
library(stringr)
library(readr)

### INPUT ###
# formatted sequences
#sequences = read.csv("Benchmarking/data/red_formatted_proteome.csv", stringsAsFactors = F, header = T)
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)

### MAIN PART ###
QSO = matrix(ncol = 92, nrow = nrow(sequences))
QSO[, 1] = sequences$Accession
QSO[, 2] = sequences$seqs

len = nchar(QSO[,2])
lag = min(len)-1

if(lag >25){
  lag = 25
}

print(paste0("lag: ", lag))

progressBar = txtProgressBar(min = 0, max = nrow(QSO), style = 3)
for (q in 1:nrow(QSO)) {
  setTxtProgressBar(progressBar, q)
  
  x = extractQSO(sequences$seqs[q], nlag = lag)
  QSO[q, 3:(2+length(x))] = x
}

for (p in 3:ncol(QSO)){
  QSO[,p] = as.numeric(as.character(QSO[,p]))
}

if(lag < 25){
  QSO[, which(!is.finite(colSums(QSO[,3:ncol(QSO)])))] = NULL
}

QSO = as.data.frame(QSO)
colnames(QSO)[1:2] = c("Accession", "seqs")

### OUTPUT ###
write.csv(x = QSO, file = unlist(snakemake@output[["embedding_QSO"]]), row.names = F)
