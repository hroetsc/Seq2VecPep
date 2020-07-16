### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  concatenate BPE training .fasta file
# input:        .fasta file on which the BPE algorithm should be trained
# output:       concatenated sequences
# author:       HR

print("### PREPARATION OF TRAINING DATA SET FOR BYTE-PAIR ENCODING ALGORITHM ###")

library(tibble)
library(dplyr)
library(rlist)
library(stringr)
library(seqinr)
library(berryFunctions)
library(tokenizers.bpe)

### INPUT ###
# data set to train bpe algorithm is specified by the user
params = read.csv(snakemake@input[["params"]], stringsAsFactors = F, header = T)

trainFASTA = read.fasta(params[which(params$parameter == "BPEinput"), "value"],
                        seqtype = "AA",
                        whole.header = T)

# trainFASTA = read.fasta("files/SwissProt_canonicalAndIsoforms.fasta",
#                         seqtype = "AA",
#                         whole.header = T)


### MAIN PART ###
seqs = rep(NA, length(trainFASTA))

# format sequences
progressBar = txtProgressBar(min = 0, max = length(trainFASTA), style = 3)
for (e in 1:length(trainFASTA)) {
  setTxtProgressBar(progressBar, e)
  seqs[e] = paste(trainFASTA[[e]], sep = "", collapse = "")
}

# merge all sequences in the fasta file
seqs = paste(seqs, sep = "", collapse = "")


### OUTPUT ###

write.table(seqs, file = unlist(snakemake@output[["conc_UniProt"]]), sep = "\t",
            row.names = T, col.names = T)

# write.table(seqs, file = "Seq2Vec/data/concatenated_UniProt_hp.txt", sep = "\t",
#             row.names = T, col.names = T)
