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
fs = list.files(path = "transcriptome/similarity/scores", pattern = ".txt", full.names = T)

scores = as.data.frame(matrix(ncol = 5, nrow = length(fs)))
scores[,1] = fs

for (s in 1:nrow(scores)){
  tmp = read.table(file = scores[s,1], sep = " ", stringsAsFactors = F, header = T)
  scores[s, c(2:3)] = tmp[1,c(1:2)]
  scores[s, c(4:5)] = tmp[2,c(1:2)]
}


### MAIN PART ###
scores = as.data.frame(scores)
colnames(scores) = c("file", "syntax", "semantics", "SD_syntax", "SD_semantics")

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


# abs of scores
# scores$syntax = abs(scores$syntax)
# scores$semantics = abs(scores$semantics)

# make it suitable for mirrored bar plot
scores = rbind(scores, scores)
scores[, "property"] = c(rep("syntax", nrow(scores)*0.5), rep(rep("semantics", nrow(scores)*0.5)))
scores[c(1:nrow(scores)*0.5), "semantics"] = NA
scores[c(((nrow(scores)*0.5)+1):nrow(scores)), "syntax"] = NA
scores[c(1:nrow(scores)*0.5), "value"] = scores$syntax[c(1:nrow(scores)*0.5)]
scores[c(((nrow(scores)*0.5)+1):nrow(scores)), "value"] = -1* scores$semantics[c(((nrow(scores)*0.5)+1):nrow(scores))]
scores$syntax = NULL
scores$semantics = NULL

scores[c(1:nrow(scores)*0.5), "SD_semantics"] = NA
scores[c(((nrow(scores)*0.5)+1):nrow(scores)), "SD_syntax"] = NA
scores[c(1:nrow(scores)*0.5), "SD"] = scores$SD_syntax[c(1:nrow(scores)*0.5)]
scores[c(((nrow(scores)*0.5)+1):nrow(scores)), "SD"] = scores$SD_semantics[c(((nrow(scores)*0.5)+1):nrow(scores))]
scores$SD_syntax = NULL
scores$SD_semantics = NULL


# more understandable group labels
scores[which(scores$embedding == "DNADCC"), "embedding"] = "dinucleotide-based\ncross-covariance"
scores[which(scores$embedding == "DNAPse"), "embedding"] = "pseudo-\ndinucleotide-\ncomposition"
scores[which(scores$embedding == "seq2vec"), "embedding"] = "Seq2Vec"
scores[which(scores$embedding == "termfreq"), "embedding"] = "term\nfrequency"

# take reciprocal value to make scores more intuitive
# scores$value = 1/scores$value

# plotting
p = ggplot(scores, aes(x = embedding, y = value, fill = weighting, alpha = 0.8)) +
  geom_bar(stat = "identity", position = "dodge",
           aes(color = PC1_removal)) +
  scale_y_continuous(limits = c(-1.2,1.2), breaks = seq(-1, 1, 0.1)) +
  scale_fill_manual(values = c("cornflowerblue", "aquamarine", "chartreuse2"),
                    label = c("none", "Smooth Inverse Freq.", "Term Freq. - Inverse Doc. Freq."),
                    name = "weighting of tokens") +
  scale_color_manual(values = c("black", "firebrick1"),
                     label = c("yes", "no"),
                     name = "removal of PC1") +
  ggtitle("ability of embedding/postprocessing methods \nto capture sequence similarity - human transcrips",
          subtitle = "mean difference to true similarities - small values indicate high ability") +
  ylab("semantics (negative scale) and syntax (positive scale)") +
  theme_minimal()

p


### OUTPUT ###
ggsave(filename = "transcriptome/result_transcriptome.png", plot = p, device = "png", dpi = "retina",
       width = 12.3, height = 7.54)
write.csv(scores, "transcriptome/result_transcriptome.csv", row.names = F)
