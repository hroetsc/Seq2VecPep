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


print("### QUASI-SEQUENCE-ORDER EMBEDDINGS")

### INPUT ###
# formatted sequences
# sequences = read.csv("downloads/1/current_sequences.csv", stringsAsFactors = F, header = T)
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)

### MAIN PART ###

len = nchar(sequences$seqs)
lag = min(len)-1

if(lag >25){
  lag = 25
}

print(paste0("lag: ", lag))


QSO = matrix(ncol = 40+ 2*lag + 2, nrow = nrow(sequences))
QSO[, 1] = sequences$Accession
QSO[, 2] = sequences$seqs

# protcheck 
# clean sequences
QSO[,2] = as.character(QSO[,2])
a = sapply(toupper(QSO[,2]), protcheck)
names(a) = NULL
print(paste0("found ",length(which(a==F)) , " proteins that are failing the protcheck() and is removing them"))
QSO = QSO[which(a == T), ]


progressBar = txtProgressBar(min = 0, max = nrow(QSO), style = 3)
for (q in 1:nrow(QSO)) {
  setTxtProgressBar(progressBar, q)
  
  x = extractQSO(sequences$seqs[q], nlag = lag)
  QSO[q, 3:(2+length(x))] = x
}

for (p in 3:ncol(QSO)){
  QSO[,p] = as.numeric(as.character(QSO[,p]))
}


QSO = as.data.frame(QSO)
colnames(QSO)[1:2] = c("Accession", "seqs")

### OUTPUT ###
write.csv(x = QSO, file = unlist(snakemake@output[["embedding_QSO"]]), row.names = F)
