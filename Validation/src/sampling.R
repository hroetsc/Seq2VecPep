### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  sample from human proteome for submission to validation pipeline
# author:       HR

library(protr)

### INPUT ###
sequences = read.csv(file = snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)

# sequences = read.csv("data/proteome_human.csv", stringsAsFactors = F, header = T)
# words = read.csv("data/words_hp.csv", stringsAsFactors = F, header = T)


### MAIN PART ###
# same proteins
sequences = sequences[-which(! sequences$Accession %in% words$Accession),]

# sample
k = sample(nrow(sequences), 100)

sequences = sequences[k,]
words =  words[which(words$Accession %in% sequences$Accession),]

if (nrow(words) != nrow(sequences)){
  print("!!!!!!WARNING!!!!!! sequences and words do not match !!!!!!!")
}


### OUTPUT ###
write.csv(sequences, file = unlist(snakemake@output[["batch_sequence"]]), row.names = F)
write.csv(words, file = unlist(snakemake@output[["batch_words"]]), row.names = F)