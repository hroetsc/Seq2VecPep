### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  embedding based on term frequency (padded using R keras package)
# input:        words and tokens
# output:       embedding based on term frequencies
# author:       HR

library(plyr)
library(dplyr)
library(tidytext)
library(stringr)

### INPUT ###
# sequences and encoded sequences
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)
words = read.csv(snakemake@input[["words"]], stringsAsFactors = F, header = T)

### MAIN PART ###
# join tables
sequences = left_join(words, sequences)

# get term frequency
words = unnest_tokens(tbl = words, output = token, input = tokens) %>% # split data frame so that every token gets one row
  count(Accession, token, sort = T) %>% # count how often every token occurs in the same protein (term frequency)
  ungroup()
words$token = toupper(words$token)

# get list of words and term frequencies
words.split = split.data.frame(words, words$Accession)

no_tokens = rep(NA, length(words.split))
Accessions = rep(NA, length(words.split))
progressBar = txtProgressBar(min = 0, max = nrow(sequences), style = 3)
for (w in 1:length(words.split)){
  setTxtProgressBar(progressBar, w)
  
  words.split[[w]][, "token"] = NULL
  no_tokens[w] = nrow(words.split[[w]])
  Accessions[w] = words.split[[w]][, "Accession"][[1]]
}
# make sure that order of proteins is the same
Accessions = as.data.frame(Accessions)
colnames(Accessions) = "Accession"
sequences = left_join(Accessions, sequences)

# pad sequences to get vectors of same length
maxlen = max(no_tokens)
print(paste0("maximum number of tokens per protein is ", maxlen))

dim = 100
print(paste0("truncating, so that vectors are of length ", dim))

term_freq = matrix(ncol = dim+2, nrow = length(words.split))
term_freq[, 1] = sequences$Accession
term_freq[, 2] = sequences$seqs

progressBar = txtProgressBar(min = 0, max = nrow(sequences), style = 3)
for (t in 1:length(words.split)){
  setTxtProgressBar(progressBar, t)
  
  x = words.split[[t]][,2]
  term_freq[t, c(3:(dim+2))] = t(x[c(1:dim),])
}

term_freq[which(is.na(term_freq))] = 0

print("randomize frequency arrangement")
for (t in 1:nrow(term_freq)){
  setTxtProgressBar(progressBar, t)
  term_freq[t, c(3:ncol(term_freq))] = term_freq[t, sample(c(3:ncol(term_freq)))]
}
term_freq = as.data.frame(term_freq)
colnames(term_freq)[1:2] = c("Accession", "seqs")

### OUTPUT ###
write.csv(x = term_freq, file = unlist(snakemake@output[["embedding_termfreq"]]), row.names = F)