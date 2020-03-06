### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  calculate TF-IDF scores (Term Frequency - Document Inverse Frequency) of tokens
# input:        word table generated in generate_tokens
# output:       TF-IDF score for every token in every protein
# author:       HR

print("### CALCULATE TF-IDF SCORES FOR EVERY TOKEN IN ENCODED PROTEOME ###")

#tmp!!!
# setwd("Documents/ProtTransEmbedding/Snakemake/")
# words = read.csv(file = "results/encoded_proteome/words.csv", stringsAsFactors = F, header = T)
# # # for testing!!
# words = words[c(1:100),]

library(stringr)
library(plyr)
library(dplyr)
library(data.table)

### INPUT ###
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)

### MAIN PART ###
print("CALCULATE DOCUMENT FREQUENCY FOR TOKENS IN BPE VOCABULARY")
# document frequency: for every token in subword count how many proteins contain it
vocab = unique(as.character(t(str_split(paste(words$tokens, collapse = ""), coll(" "), simplify = T))))
vocab = data.table(vocab) # bc model vocabulary does not contain all tokens that appear in encoded proteome

progressBar = txtProgressBar(min = 0, max = nrow(vocab), style = 3)
for (i in 1:nrow(vocab)) { # iterate tokens in model vocabulary
  setTxtProgressBar(progressBar, i)
  
  len = 1
  for (j in 1:nrow(words)) { # count how often this token occurs in the proteome
    if (grepl(vocab$vocab[i], words$tokens[j], fixed = T)) {
      len = len + 1
    }
  }
  
  vocab[i,"frequency"] = len
}
#vocab = as.data.frame(vocab)
vocab = na.omit(vocab, cols=seq_along(vocab), invert=F)

print("CALCULATE TERM FREQUENCY FOR EVERY TOKEN IN EVERY PROTEIN")
progressBar = txtProgressBar(min = 0, max = nrow(words), style = 3)

for (i in 1:nrow(words)) {
  setTxtProgressBar(progressBar, i)
  # split tokens
  cnt_tokens = data.table(t(str_split(words$tokens[i], coll(" "), simplify = T)))
  doc_freqs = data.table(table(cnt_tokens))
  
  for (j in 1:nrow(cnt_tokens)) {
    # term frequency: how often does each token occur within the protein?
    cnt_tokens[j, "term_freq"] = doc_freqs[cnt_tokens == as.character(cnt_tokens[j]), "N"]
    
    if (length(which(as.character(cnt_tokens[j,"V1"]) == vocab[,1])) > 0) {
      cnt_tokens[j, "doc_freq"] = vocab[vocab == as.character(cnt_tokens[j,"V1"]), "frequency"]
    } else {
      cnt_tokens[j, "doc_freq"] = NaN
    }
    
  }
  
  cnt_tokens[, "tf_idf"] = as.numeric(cnt_tokens$term_freq) * log(nrow(words) / as.numeric(cnt_tokens$doc_freq))
  words[i, "TF_IDF_score"] = paste(as.character(cnt_tokens$tf_idf), sep = "", collapse = " ")
}

### OUTPUT ###
write.csv(words, file = unlist(snakemake@output[["TF_IDF"]]), row.names = F)

