### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  concatenate and visualize similarity scores for different embedding/post-processing methods
# input:        similarity scores
# output:       some nice plots
# author:       HR

library(stringr)
library(plyr)
library(dplyr)

library(ggplot2)
library(ggthemes)

### INPUT ###
fs = list.files(path = "similarity/scores", pattern = ".txt", full.names = T)

scores = as.data.frame(matrix(ncol = 3, nrow = length(fs)))
scores[,1] = fs

for (s in 1:nrow(scores)){
  tmp = read.table(file = scores[s,1], sep = " ", stringsAsFactors = F, header = T)
  scores[s, c(2:3)] = tmp[1,c(1:2)]
}


### MAIN PART ###
scores = as.data.frame(scores)
colnames(scores) = c("file", "syntax", "semantics")

# split filenames to retrieve conditions
names = str_split_fixed(scores$file, coll("/"), Inf) %>% as.data.frame()
names = names[, ncol(names)]
names = str_split_fixed(names, coll("."), Inf)[,1]
names = str_split_fixed(names, coll("_"), Inf)

# complete master table
scores[, "embedding"] = names[,1]
scores[, "weighting"] = names[,2]
scores[which(scores$weighting == ""), "weighting"] = "none"
scores[, "PC1_removal"] = names[,3]
scores[which(scores$weighting == "CCR"), "PC1_removal"] = "CCR"
scores[which(scores$weighting == "CCR"), "weighting"] = "none"
scores[which(scores$PC1_removal == ""), "PC1_removal"] = "none"
scores$file = NULL

# take reciprocal value to make scores more intuitive
#scores$syntax = 1/scores$syntax
#scores$semantics = 1/scores$semantics

# make it suitable for mirrored bar plot
scores = rbind(scores, scores)
scores[, "property"] = c(rep("syntax", nrow(scores)*0.5), rep(rep("semantics", nrow(scores)*0.5)))
scores[c(1:nrow(scores)*0.5), "semantics"] = NA
scores[c(((nrow(scores)*0.5)+1):nrow(scores)), "syntax"] = NA
scores[c(1:nrow(scores)*0.5), "value"] = scores$syntax[c(1:nrow(scores)*0.5)]
scores[c(((nrow(scores)*0.5)+1):nrow(scores)), "value"] = -1* scores$semantics[c(((nrow(scores)*0.5)+1):nrow(scores))]
scores$syntax = NULL
scores$semantics = NULL

# plotting
p = ggplot(scores, aes(x = embedding, y = value, fill = weighting, alpha = 0.8)) +
  geom_bar(stat = "identity", position = "dodge",
           aes(color = PC1_removal))+
  scale_y_continuous(breaks = seq(-1, 1, 0.1)) +
  scale_fill_manual(values = c("cornflowerblue", "aquamarine", "chartreuse2")) +
  scale_color_manual(values = c("black", "firebrick1")) +
  ggtitle("ability of embedding/postprocessing methods \nto capture sequence similarity",
          subtitle = "small values indicate higher ability") +
  ylab("semantics (negative scale) and syntax (positive scale)") +
  theme_minimal()

p

### OUTPUT ###
ggsave(filename = "result.png", plot = p, device = "png", dpi = "retina")
