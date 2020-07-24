### HEADER ###
# HOTSPOT REGIONS
# description: compare sequence embedding before and after deep averaging (see Jupyter notebook)
# input: weights file from DAN, original sequence representation
# output: sequence representation DAN, UMAP representations
# author: HR

library(plyr)
library(dplyr)
library(stringr)
library(rhdf5)

library(uwot)
library(ggplot2)
library(paletteer)


### INPUT ###

# extended substring
# seq2vec+TFIDF

weights = h5read("DAN/weights_ext_seq2vec-TFIDF.h5", "/embedding/embedding")
h5ls("DAN/weights_ext_seq2vec-TFIDF.h5")

pre = read.csv("ext_substr_w5_d100_seq2vec-TFIDF.csv", stringsAsFactors = F)

### MAIN PART ###
# extract weights from DAN
weights = plyr::ldply(weights)
weights = t(weights) %>% as.data.frame()
weights = weights[-1,]
colnames(weights) = seq(1, ncol(weights))

post = cbind(pre[, c("Accession", "region", "tokens", "label")], weights) %>% as.data.frame()


# UMAP
grep_weights = function(df = ""){
  c = str_count(df[2,], "\\d+\\.*\\d*")
  return(as.logical(c))
}


UMAP = function(tbl = ""){
  set.seed(42)
  
  for (c in 1:ncol(tbl)){
    tbl[, c] = tbl[, c] %>% as.character() %>% as.numeric()
  }
  
  tbl$Accession = NULL
  
  coord = umap(tbl,
               n_neighbors = 3,
               min_dist = 1,
               n_epochs = 300,
               n_trees = 500,
               metric = "cosine",
               verbose = T,
               approx_pow = T,
               ret_model = T,
               init = "normlaplacian",
               n_threads = availableCores())
  
  um = data.frame(UMAP1 = coord$embedding[,1],
                  UMAP2 = coord$embedding[,2])
  
  return(um)
}

um.pre = UMAP(tbl = pre[, grep_weights(pre)])
um.post = UMAP(post[, grep_weights(post)])


# plotting
col_by = pre$label
nColor = length(levels(as.factor(col_by)))
colors = paletteer_c("viridis::viridis", n = nColor)
colors = c("darkturquoise", "firebrick1")

for (i in 1:length(col_by)){
  if (col_by[i] == "non_hotspot") { rank[i] = colors[1] } else { rank[i] = colors[2] }
}


# prior to DAN
png(filename = "DAN/DAN_pre_seq2vec-TFIDF.png",
    width = 2000, height = 2000, res = 300)

plot(um.pre,
     col = rank,
     cex = 0.3,
     pch = 1,
     xlab = "UMAP 1", ylab = "UMAP 2",
     main = "human proteins: seq2vec + TFIDF",
     sub = "colored by hotspot/non-hotspot")

dev.off()

# after DAN
png(filename = "DAN/DAN_post_seq2vec-TFIDF.png",
    width = 2000, height = 2000, res = 300)

plot(um.post,
     col = rank,
     cex = 0.3,
     pch = 1,
     xlab = "UMAP 1", ylab = "UMAP 2",
     main = "human proteins: seq2vec + TFIDF",
     sub = "colored by hotspot/non-hotspot")

dev.off()


### OUTPUT ###
write.csv(post, "DAN/DAN_ext_substr_w5_d100_seq2vec-TFIDF.csv", row.names = F)

