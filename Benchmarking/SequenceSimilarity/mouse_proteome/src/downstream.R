### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  concatenate and visualize similarity scores for different embedding/post-processing methods
# input:        similarity scores
# output:       some nice plots
# author:       HR

library(stringr)
library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggthemes)

### INPUT ###
fs = list.files(path = "proteome/similarity/scores", pattern = ".txt", full.names = T)

scores = as.data.frame(matrix(ncol = 5, nrow = length(fs)))
scores[,1] = fs

for (s in 1:nrow(scores)){
  tmp = read.table(file = scores[s,1], sep = " ", stringsAsFactors = F, header = T)
  scores[s, c(2:3)] = tmp[c(1:2),1]
  scores[s, c(4:5)] = tmp[c(1:2),2]
  scores[s, c(6:7)] = tmp[c(1:2),3]
  scores[s, c(8:9)] = tmp[c(1:2),4]
}


### MAIN PART ###
scores = as.data.frame(scores)
colnames(scores) = c("file", "syntax", "SD_syntax",
                     "semantics_MF", "SD_semantics_MF",
                     "semantics_BP", "SD_semantics_BP",
                     "semantics_CC", "SD_semantics_CC")

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


# plotting
scores$cat = apply( scores[ ,c("embedding", "weighting", "PC1_removal")] , 1 , paste , collapse = "_" )

# tidy format
master = scores %>% gather(key = "label", value = "score", c("syntax", "semantics_MF",
                                                             "semantics_BP", "semantics_CC"))
master$score = log2(as.numeric(as.character(master$score)))

p = ggplot(master, aes(x = cat, y = score, fill = label)) +
  geom_dotplot(binaxis = "y", stackdir = "center") +
  geom_hline(yintercept = min(master$score), linetype = "dashed") +
  geom_hline(yintercept = max(master$score), linetype = "dashed") +
  ggtitle("ability of embedding/postprocessing methods \nto capture sequence similarity - mouse proteins",
          subtitle = "squared mean difference to true similarities - small values indicate high ability") +
  ylab("score on log2 scale") +
  xlab("embedding and postprocessing method") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90))


p

### OUTPUT ###
ggsave(filename = "proteome/result_proteome.png", plot = p, device = "png", dpi = "retina",
       width = 12.3, height = 7.54)
write.csv(scores, "proteome/result_proteome.csv", row.names = F)
