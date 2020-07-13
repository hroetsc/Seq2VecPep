### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  generation of data set with uniform distribution of true (syntactic similarity
# input:        human proteome + sequences
# output:       subset of sequences for current iteration of validation pipeline
# author:       HR

library(dplyr)
library(tidyr)
library(Biostrings)
library(protr)
library(stringr)

library(future)
library(foreach)
library(doParallel)


print("DATA SET GENERATION")


### INPUT ###
seqs = read.csv(file = snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)
words = read.csv(file = snakemake@input[["words"]], stringsAsFactors = F, header = T)

# seqs = read.csv("data/proteome_human.csv", stringsAsFactors = F)
# words = read.csv("data/words_hp.csv", stringsAsFactors = F)

### MAIN PART ###
threads = 4

cl <- makeCluster(threads)
registerDoParallel(cl)

N = 10e04
M = 10e02

# same sequences
if (nrow(seqs) != nrow(words)){
  seqs = seqs[-which(! seqs$Accession %in% words$Accession),]
}

# remove non-standard amino acids (similarity measurment would fail)
seqs$seqs = as.character(seqs$seqs) %>% toupper()
a = sapply(seqs$seqs, protcheck)
names(a) = NULL
print(paste0("found ",length(which(a==F)) , " proteins that are failing the protcheck() and is removing them"))
seqs = seqs[which(a == T), ]

# same sequences
if (nrow(seqs) != nrow(words)){
  words = words[-which(! words$Accession %in% seqs$Accession),]
}


# sample N pairs of sequences
# calculate pairwise sequence similarity

sims = matrix(ncol = 3, nrow = N) %>% as.data.frame()

sims = foreach (n = 1:N, .combine = "rbind") %dopar% {
  k = sample(nrow(seqs), 2)
  
  s = Biostrings::pairwiseAlignment(Biostrings::AAString(seqs$seqs[k[1]]),
                                    Biostrings::AAString(seqs$seqs[k[2]]),
                                    type = "local",
                                    gapOpening = 8,
                                    gapExtension = -8,
                                    substitutionMatrix = "BLOSUM50",
                                    scoreOnly = T)
  
  sims[n,] = c(seqs$Accession[k[1]], seqs$Accession[k[2]], s)
  
} %>% as.data.frame()

stopCluster(cl)
stopImplicitCluster()

colnames(sims) = c("acc1", "acc2", "similarity")

# stats
sims$similarity = as.numeric(as.character(sims$similarity))
summary(sims$similarity)
plot(density(sims$similarity))
# hist(sims$similarity)

# group by similarity and sample from groups
# divide into groups

sims$tags = cut(sims$similarity, M, labels = F)
sims.ls = split.data.frame(sims, sims$tags)

sub.seqs = matrix(ncol = 5, nrow = 0) %>% as.data.frame()

for (t in 1:length(sims.ls)){ 
  
  sub.seqs = rbind(sub.seqs,
                   sims.ls[[t]][sample(nrow(sims.ls[[t]]), size = 1), ])
  
}

hist(sub.seqs$similarity, breaks = seq(1, max(ceiling(sub.seqs$similarity))))
plot(density(sub.seqs$similarity))

sub.seqs$tags = NULL

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

# tmp!!!
# write.csv(sub.seqs, file = "data/current_accessions.csv", row.names = F)
# write.csv(seqs, file = "data/current_sequences.csv", row.names = F)
# write.csv(words, file = "data/current_words.csv", row.names = F)

