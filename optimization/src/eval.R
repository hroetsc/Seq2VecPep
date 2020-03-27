setwd("Documents/ProtTransEmbedding/Snakemake/")
tar = read.table("results/embedded_proteome/opt_target_1000.txt", stringsAsFactors = F, header = F)
con = read.table("results/embedded_proteome/opt_context_1000.txt", stringsAsFactors = F, header = F)
lab = read.table("results/embedded_proteome/opt_label_1000.txt", stringsAsFactors = F, header = F)

skipgrams = cbind(tar, con, lab)
skipgrams = as.data.frame(skipgrams)
colnames(skipgrams) = c("target", "context", "label")

skipgrams[971,]
skipgrams[which(skipgrams$target == 0 & skipgrams$context == 0), ]
skipgrams[which(skipgrams$target == 971), ]
