### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  calculate TF-IDF scores (Term Frequency - Document Inverse Frequency) of tokens
# input:        word table generated in generate_tokens
# output:       TF-IDF score for every token in every protein
# author:       HR, adapted from https://www.tidytextmining.com/tfidf.html

print("### CALCULATE TF-IDF SCORES FOR EVERY TOKEN IN ENCODED PROTEOME ###")

library(dplyr)
library(tidytext)

### INPUT ###
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)

### MAIN PART ###
print("using tidytext approach")


print("calculating term frequency - number of occurences of each token in each protein")
words = unnest_tokens(tbl = words, output = token, input = tokens) %>% # split data frame so that every token gets one row
  count(Accession, token, sort = F) %>% # count how often every token occurs in the same protein (term frequency)
  ungroup() # print it line by line


print("calculating document frequency - in how many proteins does each token occur?")
doc_freq = words %>%
  group_by(Accession) %>% # concatenate word table by Accession
  summarize(total = sum(n)) # count how often every Accession occurs

words = left_join(words, doc_freq)
# 'words' has the structure one-row-per-token-per-protein
# n are protein token (document term) counts


print("calculating term frequency - document inverse frequency")
TF_IDF = words %>% bind_tf_idf(term = token, document = Accession, n)


### OUTPUT ###
write.csv(TF_IDF, file = unlist(snakemake@output[["TF_IDF"]]), row.names = F)
