tfidf = read.csv("../../RUNS/HumanProteome/v_50k/TF_IDF_hp_v50k.csv", stringsAsFactors = F)
summary(tfidf$n)
hist(tfidf$n)

k = which(tfidf$n > 1)
length(k) / nrow(tfidf)

non_unique_tokens = rep(NA, length(k))
for (i in 1:length(k)){
  non_unique_tokens[i] = tfidf$token[k[i]]
}

non_unique_tokens = non_unique_tokens %>% unique()
View(non_unique_tokens)

length(non_unique_tokens) / 50000


windows = read.csv("data/windowTokens_training.csv", stringsAsFactors = F)
u = which(duplicated(windows$tokens))
length(u) / nrow(windows)

u2 = 0
acc = windows$Accession %>% unique()
for (a in acc) {
  tmp = windows[windows$Accession == a, ]
  u2 = u2 + length(which(duplicated(tmp$tokens)))
}
length(u2) / nrow(windows)

