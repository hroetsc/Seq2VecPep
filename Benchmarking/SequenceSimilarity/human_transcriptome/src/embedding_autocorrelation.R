### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  embedding based on autocorrelation of dinucleotides (BioMedR package)
# input:        sequences
# output:       embedded sequences
# author:       HR

library(seqinr)
library(plyr)
library(dplyr)
library(stringr)
library(readr)
library(BioMedR)


library(parallel)
library(foreach)
library(doParallel)
library(doMC)
library(plyr)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
registerDoMC(detectCores())


### INPUT ###
# formatted sequences
#sequences = read.csv("transcriptome/data/red_transcriptome_human.csv", stringsAsFactors = F, header = T)
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)


### MAIN PART ###

# physicochemical indices for dinucleotides
idx = c("Base stacking", "Propeller twist", "DNA denaturation", "Aida_BA_transition",
        "Electron_interaction", "Lisser_BZ_transition", "SantaLucia_dS", "Sugimoto_dG",
        "disruptenergy", "Ivanov_BA_transition", "Watson-Crick_interaction", "Tilt",
        "Slide", "Roll", "Rose", "Twist", "Shift", "DNA induced deformability", "Bending stiffness",
        "Breslauer_dG", "Hartman_trans_free_energy", "Polar_interaction", "Sarai_flexibility",
        "Sugimoto_dH", "Stabilising energy of Z-DNA", "SantaLucia_dH", "Dinucleotide GC content",
        "B-DNA twist", "freeenergy", "DNA DNA twist", "Breslauer_dH", "Helix-Coil_transition",
        "SantaLucia_dG", "Stability", "Sugimoto_dS", "Breslauer_dS", "Stacking_energy")

# subset
idx_sel = c("Base stacking", "Propeller twist", "DNA denaturation", "Electron_interaction",
            "Watson-Crick_interaction", "DNA induced deformability", "Stacking_energy",
            "Tilt", "Slide", "Roll", "Rose", "Twist", "Shift")

DNADCC.master = foreach(i = 1:nrow(sequences), .combine = "rbind") %dopar% {
  extrDNADCC(sequences$seqs[i], index = idx_sel, nlag = 2)
}

# clean
DNADCC.master = as.matrix(DNADCC.master)
DNADCC.master[which(!is.finite(DNADCC.master))] = 0

master = cbind(sequences, DNADCC.master)

### OUTPUT ###
write.csv(x = master, file = unlist(snakemake@output[["embedding_DNADCC"]]), row.names = F)
