### SPLICING PREDICTOR ###
# description: check whether embeddings of same sequence generated using different tokens are comparable
# input: token embeddings (human proteome embedding)
# output:
# author: HR

library(plyr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)
library(caret)

library(rhdf5)

library(foreach)
library(doParallel)
library(doMC)
library(future)

registerDoParallel(availableCores())

### INPUT ###

indices = read.csv("RUNS/HumanProteome/ids_hp_w5_new.csv", stringsAsFactors = F, header = F)
weight_matrix = h5read("RUNS/HumanProteome/word2vec_model/hp_model_w5_d100/weights.h5", "/embedding/embedding")

embeddingDim = 100

### MAIN PART ###
{
# assign tokens to weight matrix
if (ncol(indices) == 3){
  indices$V1 = NULL
}
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)

# extract weights
weight_matrix = plyr::ldply(weight_matrix)
weight_matrix = t(weight_matrix) %>% as.data.frame()

weight_matrix = weight_matrix[-1,]
colnames(weight_matrix) = seq(1, ncol(weight_matrix))

weight_matrix["word_ID"] = seq(0, (nrow(weight_matrix)-1), 1)
weight_matrix$word_ID = weight_matrix$word_ID + 1


# merge indices and weights
weights = full_join(weight_matrix, indices) %>% na.omit() %>% unique()

for (c in 1:(ncol(weights)-2)){
  weights[, c] = weights[, c] %>% as.character() %>% as.numeric()
}

}

########## data mining ########## 

matmult = function(v1 = "", v2 = ""){
  return(as.numeric(v1) %*% as.numeric(v2))
}

dot_product = function(v1 = "", v2 = ""){
  p = matmult(v1, v2)/(sqrt(matmult(v1, v1)) * sqrt(matmult(v2, v2)))
  return(p)
}


check_cor = function(token1 = "", token2a = "", token2b = ""){
  v1 = weights[which(weights$subword == token1), c(1:embeddingDim)] %>% as.numeric()
  
  v2 = rowMeans(cbind(weights[which(weights$subword == token2a), c(1:embeddingDim)] %>% as.numeric(),
                      weights[which(weights$subword == token2b), c(1:embeddingDim)] %>% as.numeric()))
  
  corr_pearson = cor(v1, v2, method = "pearson")
  corr_spearman = cor(v1, v2, method = "spearman")
  dot = dot_product(v1, v2)
  eucl = dist(rbind(v1, v2), method = "euclidean")
  
  return(c(corr_pearson, corr_spearman, dot, eucl))
}

summary(nchar(weights$subword))

# pick tokens of length > 1 aa
# split tokens into sub-tokens and check if they are correlating

# always only one splice site in token
weights.l = weights[which(nchar(weights$subword) > 1), ]

res.l = matrix(ncol = 7, nrow = nrow(weights.l))

pb = txtProgressBar(min = 0, max = nrow(weights.l), style = 3)

for (r in 1:nrow(weights.l)) {
  
  setTxtProgressBar(pb, r)
  
  str = weights.l$subword[r]
  str.split = str_split(str, "", simplify = T) %>% paste()
  
  # select random position to paste two tokens together
  pos = sample(nchar(str)-1, 1)
  
  # calculate correlation between original and splitted embedding
  token2a = paste(str.split[1:pos], collapse = "")
  token2b = paste(str.split[(pos+1):nchar(str)], collapse = "")
  
  res = check_cor(str, token2a, token2b)
  
  # add to results matrix
  res.l[r, ] = c(str, token2a, token2b, res)
  
}

##########  analysis ########## 

res.l = as.data.frame(res.l) %>% na.omit()
colnames(res.l) = c("token", "split1", "split2",
                    "pearson", 'spearman', "dot", "euclidean")

res.l$pearson = res.l$pearson %>% as.character() %>% as.numeric()
res.l$spearman = res.l$spearman %>% as.character() %>% as.numeric()
res.l$dot = res.l$dot %>% as.character() %>% as.numeric()
res.l$euclidean = res.l$euclidean %>% as.character() %>% as.numeric()

summary(res.l$pearson)
summary(res.l$spearman)
summary(res.l$dot)
summary(res.l$euclidean)

# correlation of random vectors as baseline
k = 1e03
res.rnd = matrix(ncol = 7, nrow = k)

for (r in 1:k) {
  
  t1 = sample(nrow(weights), 1)
  t2 = sample(nrow(weights), 1)
  
  str = weights$subword[t1]
  
  token2a = weights$subword[t2]
  token2b = token2a
  
  res = check_cor(str, token2a, token2b)
  
  # add to results matrix
  res.rnd[r, ] = c(str, token2a, token2b, res)
  
}
res.rnd = as.data.frame(res.rnd) %>% na.omit()

colnames(res.rnd) = c("token", "split1", "split2",
                    "pearson", 'spearman', "dot", "euclidean")

res.rnd$pearson = res.rnd$pearson %>% as.character() %>% as.numeric()
res.rnd$spearman = res.rnd$spearman %>% as.character() %>% as.numeric()
res.rnd$dot = res.rnd$dot %>% as.character() %>% as.numeric()
res.rnd$euclidean = res.rnd$euclidean %>% as.character() %>% as.numeric()

summary(res.rnd$pearson)
summary(res.rnd$spearman)
summary(res.rnd$dot)
summary(res.rnd$euclidean)

# plot
res.l$len = nchar(as.character(res.l$token))


### OUTPUT ###

{
p = ggplot(res.l, aes(as.factor(len), pearson)) +
  geom_violin(scale = "width", trim = F,
              draw_quantiles = c(0.25, 0.5, 0.75)) +
  #geom_jitter(height = 0, width = 0.0005) +
  scale_fill_viridis_d(option = "inferno", direction = -1) +
  stat_summary(fun=mean, geom="point", size=1, color="red") +
  geom_hline(yintercept = mean(res.rnd$pearson))
p = p +
  ggtitle("embeddings of same sequence generated by token and sub-tokens",
          subtitle = "Pearson corellation") +
  ylab("correlation") +
  xlab("token length") +
  theme_bw() +
  theme(legend.position = "none")

p
ggsave("Predictor/sub-tokens/pearson.png", p, device = "png", dpi = "retina")


s = ggplot(res.l, aes(as.factor(len), spearman)) +
  geom_violin(scale = "width", trim = F,
              draw_quantiles = c(0.25, 0.5, 0.75)) +
  #geom_jitter(height = 0, width = 0.0005) +
  scale_fill_viridis_d(option = "inferno", direction = -1) +
  stat_summary(fun=mean, geom="point", size=1, color="red") +
  geom_hline(yintercept = mean(res.rnd$spearman))
s = s +
  ggtitle("embeddings of same sequence generated by token and sub-tokens",
          subtitle = "Spearman corellation") +
  ylab("correlation") +
  xlab("token length") +
  theme_bw() +
  theme(legend.position = "none")

s
ggsave("Predictor/sub-tokens/spearman.png", s, device = "png", dpi = "retina")


d = ggplot(res.l, aes(as.factor(len), dot)) +
  geom_violin(scale = "width", trim = F,
              draw_quantiles = c(0.25, 0.5, 0.75)) +
  #geom_jitter(height = 0, width = 0.0005) +
  scale_fill_viridis_d(option = "inferno", direction = -1) +
  stat_summary(fun=mean, geom="point", size=1, color="red") +
  geom_hline(yintercept = mean(res.rnd$dot))
d = d +
  ggtitle("embeddings of same sequence generated by token and sub-tokens",
          subtitle = "dot product") +
  ylab("correlation") +
  xlab("token length") +
  theme_bw() +
  theme(legend.position = "none")

d
ggsave("Predictor/sub-tokens/dot.png", d, device = "png", dpi = "retina")


e = ggplot(res.l, aes(as.factor(len), euclidean)) +
  geom_violin(scale = "width", trim = F,
              draw_quantiles = c(0.25, 0.5, 0.75)) +
  #geom_jitter(height = 0, width = 0.0005) +
  scale_fill_viridis_d(option = "inferno", direction = -1) +
  stat_summary(fun=mean, geom="point", size=1, color="red") +
  geom_hline(yintercept = mean(res.rnd$euclidean))
e = e +
  ggtitle("embeddings of same sequence generated by token and sub-tokens",
          subtitle = "euclidean distance") +
  ylab("correlation") +
  xlab("token length") +
  theme_bw() +
  theme(legend.position = "none")

e
ggsave("Predictor/sub-tokens/euclidean.png", e, device = "png", dpi = "retina")
}


##########  token length distribution in database ##########

DB.enc = read.csv("RUNS/ProteasomeDB/ids_ProteasomeDB_w5.csv", stringsAsFactors = F,
                  header = F)
DB.tokens = toupper(DB.enc$V2)

summary(nchar(DB.tokens))
hist(nchar(DB.tokens), freq = F)
