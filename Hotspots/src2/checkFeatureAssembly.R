
library(dplyr)

tokenWeights = read.csv("data/token_embeddings.csv", stringsAsFactors = F)
tfidf = read.csv("data/TFIDF_training.csv", stringsAsFactors = F)

acc = "P29376-4"
tok = "ED"

tmp = tokenWeights[tokenWeights$subword == tok, c(2:ncol(tokenWeights))] %>% as.numeric()
t = tfidf$tf_idf[(tfidf$Accession == acc & tfidf$token == tok)]

View(tmp*t)
