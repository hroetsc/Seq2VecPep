### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  calculate TF-IDF scores (Term Frequency - Document Inverse Frequency) of tokens
# input:        word table generated in generate_tokens
# output:       TF-IDF score for every token in every protein
# author:       HR, adapted from https://www.tidytextmining.com/tfidf.html

print("### CALCULATE TF-IDF SCORES FOR EVERY TOKEN IN ENCODED PROTEOME ###")

# tmp!!!
# setwd("Documents/QuantSysBios/ProtTransEmbedding/Snakemake/")
# words = read.csv(file = "results/encoded_proteome/words.csv", stringsAsFactors = F, header = T)
# # for testing!!
# words = words[c(1:100),]

#library(plyr)
library(dplyr)
# added
library(tidytext)

### INPUT ###
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)
#words = data.table(words)

### MAIN PART ###
print("using tidytext approach")
print("calculating term frequency - number of occurences of each token in each protein")
words = unnest_tokens(tbl = words, output = token, input = tokens) %>% # split data frame so that every token gets one row
  count(UniProtID, token, sort = F) %>% # count how often every token occurs in the same protein (term frequency)
  ungroup() # print it line by line

print("calculating document frequency - in how many proteins does each token occur?")
doc_freq = words %>%
  group_by(UniProtID) %>% # concatenate word table by UniProtID
  summarize(total = sum(n)) # count how often every UniProtID occurs

words = left_join(words, doc_freq)
# 'words' has the structure one-row-per-token-per-protein
# n are protein token (document term) counts

print("calculating term frequency - document inverse frequency")
TF_IDF = words %>% bind_tf_idf(term = token, document = UniProtID, n)


### OUTPUT ###
write.csv(TF_IDF, file = unlist(snakemake@output[["TF_IDF"]]), row.names = F)

# tmp!
# write.csv(TF_IDF, file = 'results/encoded_proteome/TF_IDF.csv', row.names = F)
