### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  generation of data set with uniform distribution of true (syntactic similarity
# input:        human proteome + sequences
# output:       subset of sequences for current iteration of validation pipeline
# author:       HR

library(dplyr)
library(tidyr)
library(Biostrings)

library(future)
library(foreach)
library(doParallel)


print("DATA SET GENERATION")


### INPUT ###
seqs = read.csv(file = snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)


### MAIN PART ###
cl <- makeCluster(availableCores())
registerDoParallel(cl)


N = 5e04
M = 1e04

# same sequences
if (nrow(seqs) != nrow(words)){
  seqs = seqs[-which(! seqs$Accession %in% words$Accession),]
}

# sample N pairs of sequences
# calculate pairwise sequence similarity

sims = matrix(ncol = 3, nrow = N) %>% as.data.frame()

sims = foreach (n = 1:N, .combine = "rbind") %dopar% {
  k = sample(nrow(seqs), 2)
  
  s = Biostrings::pairwiseAlignment(Biostrings::AAString(seqs$seqs[k[1]]),
                                    Biostrings::AAString(seqs$seqs[k[2]]),
                                    type = "local",
                                    # gapOpening = -2,
                                    # gapExtension = -8,
                                    # substitutionMatrix = "BLOSUM50",
                                    scoreOnly = T)
  
  sims[n,] = c(seqs$Accession[k[1]], seqs$Accession[k[2]], s)
  
} %>% as.data.frame()
colnames(sims) = c("acc1", "acc2", "similarity")

# stats
sims$similarity = as.numeric(as.character(sims$similarity))
summary(sims$similarity)
plot(density(sims$similarity))
# hist(sims$similarity)

# group by similarity and sample from groups
num_groups = 500000
sims.ls = sims %>% group_by(sims$similarity %/%
                           (n()/num_groups)) %>%
  nest %>%
  pull(data)


sub.seqs = matrix(ncol = 4, nrow = 0) %>% as.data.frame()


for (i in 1:length(sims.ls)){
  
  if (nrow(sims.ls[[i]]) >= M/length(sims.ls)){
    sub.seqs = rbind(sub.seqs, sims.ls[[i]][sample(nrow(sims.ls[[i]]), size = M/length(sims.ls)), ])
 
  } else {
    sub.seqs = rbind(sub.seqs, sims.ls[[i]][sample(nrow(sims.ls[[i]]), size = nrow(sims.ls[[i]])), ])
  }
  
}

plot(density(sub.seqs$similarity))


# pick current sequence-batch
acc = c(as.character(sub.seqs$acc1), as.character(sub.seqs$acc2)) %>%
  unique()

seqs = seqs[which(seqs$Accession %in% acc), ]
words = words[which(words$Accession %in% acc), ]

### OUTPUT ####
# same files!
write.csv(sub.seqs, file = unlist(snakemake@output[["batch_accessions"]]), row.names = F)
write.csv(sub.seqs, file = unlist(snakemake@output[["true_syntax"]]), row.names = F)

write.csv(seqs, file = unlist(snakemake@output[["batch_sequence"]]), row.names = F)
write.csv(words, file = unlist(snakemake@output[["batch_words"]]), row.names = F)


