### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  segment antigen sequences into variable length fragments using bite-pair encoding algorithm
# input:        antigen dataset from src_antigens.R, mouse SwissProt sequences
# output:       encoded antigen sequences, tokens('words') for word2vec
# author:       HR

library(tibble)
library(dplyr)
library(rlist)
#library(Rcpi)
library(stringr)
library(seqinr)
library(berryFunctions)
library(tokenizers.bpe)

### INPUT ###
# load antigen datasets
antigens = read.csv("./data/immunopeptidome/src_antigens.csv", stringsAsFactors = F, header = T)
antigens = as.data.frame(antigens)
# load the model
bpeModel = bpe_load_model("./results/encoded_antigens/BPE_model.bpe",
                          threads = 14)
ModelVocab = bpeModel$vocabulary
ModelVocab = tibble::as_tibble(ModelVocab)

### MAIN PART ###
# peptide pair encoding
antigens.Encoded = matrix(ncol = ncol(antigens)+1) 
colnames(antigens.Encoded) = c(colnames(antigens), "SegmentedSeq")

progressBar = txtProgressBar(min = 0, max = nrow(antigens), style = 3)
for (n in 1:nrow(antigens)) {
  setTxtProgressBar(progressBar, n)
  
  # encode protein sequence
  PepEncoded = bpe_encode(model = bpeModel, x = antigens[n,"antigenSeq"], type = "subwords")
  PepEncoded = unlist(PepEncoded)[-1]
  
  currentPeptide = as.data.frame(matrix(ncol = ncol(antigens), nrow = length(PepEncoded)))
  colnames(currentPeptide) = colnames(antigens)
  
  currentPeptide[,"Peptide"] = as.character(t(rep(antigens[n,"Peptide"], length(PepEncoded))))
  currentPeptide[,"Length"] = as.character(t(rep(antigens[n,"Length"], length(PepEncoded))))
  currentPeptide[,"Accession"] = as.character(t(rep(antigens[n,"Accession"], length(PepEncoded))))
  currentPeptide[,"Source.File"] = as.character(t(rep(antigens[n,"Source.File"], length(PepEncoded))))
  currentPeptide[,"Sample.code"] = as.character(t(rep(antigens[n,"Sample.code"], length(PepEncoded))))
  currentPeptide[,"Group"] = as.character(t(rep(antigens[n,"Group"], length(PepEncoded))))
  currentPeptide[,"UniProtID"] = as.character(t(rep(antigens[n,"UniProtID"], length(PepEncoded))))
  currentPeptide[,"antigenSeq"] = as.character(t(rep(antigens[n,"antigenSeq"], length(PepEncoded))))
  currentPeptide[,"mean_rPCP"] = as.character(t(rep(antigens[n,"mean_rPCP"], length(PepEncoded))))
  currentPeptide[,"SegmentedSeq"] = as.character(t(PepEncoded))
  
  colnames(currentPeptide) = colnames(antigens.Encoded)
  antigens.Encoded = rbind(antigens.Encoded, currentPeptide)
}

# evaluation
# keep only words longer than 1 aa
antigens.Encoded = na.omit(antigens.Encoded)
antigens.Encoded = antigens.Encoded[-which(nchar(antigens.Encoded$SegmentedSeq) == 1),]
# words
words = antigens.Encoded$SegmentedSeq
words_as_text = paste(words, collapse = " ")

### OUTPUT ###
# save model vocabulary
write.csv(ModelVocab, "./results/encoded_antigens/model_vocab.csv")
# save encoded antigens
save(file = snakemake@output[["encoded_antigend_R"]], antigens.Encoded)
write.csv(antigens.Encoded, snakemake@output[["encoded_antigens"]], row.names = F)
# save words
write(words_as_text, snakemake@output[["words"]])
