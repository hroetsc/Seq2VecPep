### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  reduce number of skip-grams to improve model efficiency
# input:        target, context and label integer vectors, generated in skip_gram_NN_1.py
# output:       smaller vectors
# author:       HR

print("IMPROVE MODEL EFFICIENCY BY REDUCING THE NUMBER OF SKIP-GRAMS")

library(readr)
library(data.table)
library(plyr)
library(dplyr)

# tmp!!!
#setwd("/home/hanna/Documents/QuantSysBios/ProtTransEmbedding/Snakemake/")
#target = read.table("./results/embedded_proteome/target.txt", stringsAsFactors = F, header = F)
#context = read.table("./results/embedded_proteome/context.txt", stringsAsFactors = F, header = F)
#label = read.table("./results/embedded_proteome/label.txt", stringsAsFactors = F, header = F)

### INPUT ###
# Snakemake stuff
skip_grams = read.table(snakemake@input[["skip_grams"]], stringsAsFactors = F, header = F)

### MAIN PART ###
skip_grams = data.table(skip_grams)
colnames(skip_grams) = c("target", "context", "label")
head(skip_grams)

# define keep-fraction
keep = 0.2
print(paste0("keeping random ", keep*100, " % of the original amount of skip-grams per target word per protein"))

# isolate target word block (same target word may occur multiple times in different proteins, so "unique" does not work)
progressBar = txtProgressBar(min = 0, max = nrow(skip_grams), style = 3)

keep_idx = rep(NA, nrow(skip_grams))
counter_prev = 0
counter = 0

for (i in 1:nrow(skip_grams)) {
  setTxtProgressBar(progressBar, i)

  # counter is the length of the current target word block
  if (!skip_grams$target[i] == skip_grams$target[i+1]){
    # as long as the target word is the same, increase counter
    counter =  i - counter_prev
    keep_idx[c((counter_prev+1):(counter_prev + ceiling(counter*keep)))] = sample(c((counter_prev+1):i), ceiling(counter*keep))
    counter_prev = i
  }

}
print("this error is fine, since it is reaching the last line of the data frame")

keep_idx = na.omit(keep_idx)
# take into account that it could not be assessed whether the last target word is different from the ones before
keep_idx[length(keep_idx)+1] = nrow(skip_grams)

# shuffle skip-grams to make model training more robust
skip_grams = skip_grams[sample(c(keep_idx)), ]

### OUTPUT ###
# Snakemake stuff
write.csv(skip_grams, file = unlist(snakemake@output[["skip_grams_reduced"]]), row.names = F)

# tmp!!!
#write.csv(skip_grams, "./results/embedded_proteome/skipgrams_reduced.csv", row.names = F)
