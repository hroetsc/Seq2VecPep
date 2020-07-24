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
# words = read.csv("Seq2Vec/results/encoded_sequence/words_ProteasomeDB.csv",
#                  stringsAsFactors = F)
# words = ext_substr[, c("Accession", "tokens")]
# words = min_substr[, c("Accession", "tokens")]
# words = read.csv("../../Seq2Vec/results/encoded_sequence/words_GENCODEml.csv",
#                  stringsAsFactors = F)

### MAIN PART ###
print("using tidytext approach")


print("calculating term frequency - number of occurences of each token in each protein")
words = tidytext::unnest_tokens(tbl = words, output = token, input = tokens) %>% # split data frame so that every token gets one row
  dplyr::count(Accession, token, sort = F) %>% # count how often every token occurs in the same protein (term frequency)
  dplyr::ungroup() # print it line by line


print("calculating document frequency - in how many proteins does each token occur?")
doc_freq = words %>%
  dplyr::group_by(Accession) %>% # concatenate word table by Accession
  dplyr::summarize(total = sum(n)) # count how often every Accession occurs

words = left_join(words, doc_freq)
# 'words' has the structure one-row-per-token-per-protein
# n are protein token (document term) counts


print("calculating term frequency - document inverse frequency")
TF_IDF = words %>% bind_tf_idf(term = token, document = Accession, n)


### OUTPUT ###
write.csv(TF_IDF, file = unlist(snakemake@output[["TF_IDF"]]), row.names = F)

# write.csv(TF_IDF, "Seq2Vec/results/encoded_sequence/TF_IDF_ProteasomeDB.csv")
# write.csv(TF_IDF, "data/ext_substr_TFIDF.csv", row.names = F)
# write.csv(TF_IDF, "data/min_substr_TFIDF.csv", row.names = F)
# write.csv(TF_IDF, "../../Seq2Vec/results/encoded_sequence/TF_IDF_GENCODEml.csv")
