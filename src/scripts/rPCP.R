### HEADER ###
# 
# description:  calculate rate of antigen representation (rPCP)
# input:        PEAKS search results, proteome.fasta , IP group information
# output:       rPCP per protein per condition
# author:       YH & HR
print("### rPCP CALCULATION ###")

library(dplyr)
library(stringr)
library(seqinr)
library(Biostrings)
library(tidyr)
library(ggraptR)

### INPUT ###
IP_info <- read.csv("/home/yhorokh/data/Mouse_lymphoma/IP/IP_info.csv")

PEAKS.proteins = read.csv("/home/yhorokh/SNAKEMAKE_pipelines/Mouse_lymphoma_analysis/data/PEAKS/Mouse_lymphoma_IP_02_2020_GENCODE_M24_PEAKS_86_top/proteins.csv", stringsAsFactors = F, header = T)
PEAKS.peptides = read.csv("/home/yhorokh/SNAKEMAKE_pipelines/Mouse_lymphoma_analysis/data/PEAKS/Mouse_lymphoma_IP_02_2020_GENCODE_M24_PEAKS_86_top/protein-peptides.csv", stringsAsFactors = F, header = T)
ref.proteins = readDNAStringSet("/home/yhorokh/SNAKEMAKE_pipelines/Mouse_lymphoma_analysis/data/reference/expr_prot_incl_deep_nodup.fasta")
# gene.tr.prot_info = read.csv("/home/yhorokh/SNAKEMAKE_pipelines/Mouse_lymphoma_analysis/data/gene_tr_prot.csv", stringsAsFactors = F, header = T)

### MAIN PART ###
# rPCP = number of unique matches in PEAKS.proteins divided by protein length (ref.proteins)
# extract sequences and annotations from ref.proteins
print("EXTRACT SEQUENCES AND ANNOTATION FROM FASTA FILE")
prot.ref <- data.frame(seq = ref.proteins,
                       origin = names(ref.proteins), 
                       row.names = NULL)
protein <- str_split_fixed(prot.ref$protein, " ", Inf) %>% as.data.frame()
prot.ref$protein <- protein$V1

# split annotation into UniProtID and description
print("FORMAT REFERENCE TABLE AND RETRIEVE PROTEIN LENGTH")
{
  tmp <- str_split_fixed(names(ref.proteins), "[[|]]", Inf) %>% as.data.frame()
  header <- str_split_fixed(names(ref.proteins[1]), "[[|]]", Inf) %>% as.character()
  
  protein = str_detect(string = header, "ENSMUSP") %>% which()
  tr = str_detect(string = header, "ENSMUST") %>% which()
  gene = str_detect(string = header, "ENSMUSG") %>% which()
  
  prot.ref <- data.frame(seq = prot.ref$seq,
                         origin = prot.ref$origin,
                         protein = tmp[,protein] %>% as.character(),
                         gene = tmp[,gene] %>% as.character(),
                         transcript = tmp[,tr] %>% as.character(), 
                         protein_length = nchar(as.character(prot.ref$seq)))
  
  x <- str_split_fixed(prot.ref$protein, " ", Inf) %>% as.data.frame()
  prot.ref$protein <- x$V1
}

# Pre-filter PEAKS peptides
PEAKS.peptides <- PEAKS.peptides[PEAKS.peptides$Length %in% c(8:15),]
PEAKS.peptides$Source.File <- str_replace(PEAKS.peptides$Source.File, pattern = ".raw", "")

# Remove PTMs and cleavage sites from peptides
PEAKS.peptides$pep_seq <- PEAKS.peptides$Peptide
PEAKS.peptides$pep_seq <- gsub("\\(.*?\\)", "", PEAKS.peptides$pep_seq) 
PEAKS.peptides$pep_seq <- str_split_fixed(PEAKS.peptides$pep_seq, "[.]", Inf)[,2]

# Cut the Accession
Protein.Accession <- str_split_fixed(PEAKS.peptides$Protein.Accession, "[|]", Inf) %>% as.data.frame() %>% select("V1")
PEAKS.peptides$Protein.Accession <- Protein.Accession$V1

# Add the condition information
PEAKS.peptides <- left_join(PEAKS.peptides, IP_info)

# Add the protein length
length <- prot.ref[which(as.character(prot.ref$protein)  %in% PEAKS.peptides$Protein.Accession), c("protein", "protein_length")]
colnames(length)[1] <- "Protein.Accession"
PEAKS.peptides <- left_join(PEAKS.peptides, length)

# get number of unique matches
print("CALCULATE rPCP")
pseudocount = 1
min_pep_length = 8
max_pep_length = 15

# Find all unique peptides
rPCP.pep <- PEAKS.peptides
# rPCP.pep <- rPCP.pep[rPCP.pep$Group == "G3",]
rPCP_u <- table(rPCP.pep$pep_seq) %>% 
  as.data.frame() %>%
  mutate(unique_pep = ifelse(!Freq == 1, F, T))
colnames(rPCP_u)[1] <- "pep_seq"
rPCP.pep <- left_join(rPCP.pep, rPCP_u)

{
  keep <- c("Protein.Group","Protein.ID","Protein.Accession","Peptide","Unique","X.10lgP","Scan","Source.File","pep_seq","Sample.code","Group","protein_length", "unique_pep")
  rPCP.pep <- rPCP.pep[,keep]
  
  # Count the total rPCP and unique rPCP
  rPCP <- rPCP.pep %>%
    group_by(Protein.Accession, Group, add = T) %>%
    summarise(n_pep = length(unique(pep_seq)),
              n_unique_pep = length(which(unique_pep == T))) 
  
  rPCP <- left_join(rPCP, length)
  rPCP$rPCP <- c(rPCP$n_pep + pseudocount) / rPCP$protein_length
  rPCP$unique_rPCP <- rPCP$n_unique_pep / rPCP$protein_length
  rPCP <- na.omit(rPCP)
  
  # Normalize by # of all possible PCPs in a given length range 
  X = ((max_pep_length * (max_pep_length + 1))/2) - (((min_pep_length - 1) * ((min_pep_length - 1) + 1))/2)
  rPCP$length_factor <- ( (rPCP$protein_length + 1) * (max_pep_length - min_pep_length + 1) ) - X
  
  rPCP$rPCP <- c(rPCP$n_pep + pseudocount) / rPCP$length_factor
  rPCP$unique_rPCP <- rPCP$n_unique_pep / rPCP$length_factor
  rPCP <- na.omit(rPCP)
  
  # # Estimate the library sizes per 1000 peptides
  # lib_size <- rPCP.pep %>%
  #   group_by(Sample.code, add = T) %>%
  #   summarise(lib_size = n())
  # 
  # rPCP <- left_join(rPCP, lib_size)
  # rPCP$rPCP <- rPCP$rPCP / rPCP$lib_size
  # rPCP$unique_rPCP <- rPCP$unique_rPCP / rPCP$lib_size
  
  # Log10 - scale for plotting
  rPCP$log10_protein_length <- log10(rPCP$protein_length)
  rPCP$log10_rPCP <- log10(rPCP$rPCP)
  rPCP$log10_unique_rPCP <- log10(rPCP$unique_rPCP)
  rPCP$log10_unique_rPCP[which(rPCP$log10_unique_rPCP == -Inf)] <- 0
}

# Estimate the rPCP difference
{
  rPCP$Group <- as.character(rPCP$Group)
  rPCP$rPCP <- as.numeric(rPCP$rPCP)
  rPCP.delta <- rPCP %>% tidyr::spread(Group, rPCP, fill = NA)
  
  # Replace the NA with pseudocount/protein length wherever the protein is not identified
  missing <- which(is.na(rPCP.delta$G1))
  rPCP.delta$G1[missing] <- pseudocount / rPCP.delta$protein_length[missing]
  
  missing <- which(is.na(rPCP.delta$G2))
  rPCP.delta$G2[missing] <- pseudocount / rPCP.delta$protein_length[missing]
  
  missing <- which(is.na(rPCP.delta$G3))
  rPCP.delta$G3[missing] <- pseudocount / rPCP.delta$protein_length[missing]
  
  missing <- which(is.na(rPCP.delta$G4))
  rPCP.delta$G4[missing] <- pseudocount / rPCP.delta$protein_length[missing]
  rm(missing)
  
  # Log-10 rPCP
  rPCP.delta$log10_G1_rPCP <- rPCP.delta$G1 %>% log10()
  rPCP.delta$log10_G2_rPCP <- rPCP.delta$G2 %>% log10()
  rPCP.delta$log10_G3_rPCP <- rPCP.delta$G3 %>% log10()
  rPCP.delta$log10_G4_rPCP <- rPCP.delta$G4 %>% log10()
  
  # rPCP change in Dox treatment
  rPCP.delta$G12 <- rPCP.delta$G2 - rPCP.delta$G1
  rPCP.delta$G34 <- rPCP.delta$G4 - rPCP.delta$G3
  
  # rPCP change in Dox treatment
  rPCP.delta$log10_G12 <- log10(rPCP.delta$G2) - log10(rPCP.delta$G1)
  rPCP.delta$log10_G34 <- log10(rPCP.delta$G4) - log10(rPCP.delta$G3)
  
  rPCP.delta <- unique(rPCP.delta)
}



### PLOTS ###

# rPCP by group vs protein length
ggplot(rPCP, aes(y=protein_length, x=rPCP)) + 
  geom_point(stat="identity", position="identity", alpha=0.7) + 
  facet_grid(. ~ Group) + 
  coord_flip() + 
  theme_bw() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5)) + 
  xlab("rPCP") + 
  ylab("protein_length")

# log10-rPCP by group vs protein length, color by n_pep, size by unique rPCP
ggplot(rPCP, aes(y=log10_rPCP, x=protein_length)) + 
  geom_point(aes(size=unique_rPCP, colour=as.factor(n_pep)), stat="identity", position="identity", alpha=0.7) + 
  facet_grid(. ~ Group) + 
  theme_bw() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5)) + 
  guides(colour=guide_legend(title="n_pep")) + 
  ggtitle("Antigen presentation rates") + 
  xlab("protein_length") + 
  ylab("log10_rPCP")

# log10-rPCP by group vs log10 protein length, color by n_pep, size by unique rPCP
ggplot(rPCP, aes(y=log10_rPCP, x=log10_protein_length)) + 
  geom_point(aes(size=unique_rPCP, colour=as.factor(n_pep)), stat="identity", position="jitter", alpha=0.6) + 
  facet_grid(. ~ Group) + 
  theme_bw() + 
  theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5)) + 
  guides(colour=guide_legend(title="n_pep")) + 
  ggtitle("Antigen presentation rates") + 
  xlab("log10_protein_length") + 
  ylab("log10_rPCP")

### OUTPUT ###
write.csv(rPCP, file = "/home/yhorokh/SNAKEMAKE_pipelines/Mouse_lymphoma_analysis/results/rPCP/rPCP.csv", row.names = F)
write.csv(rPCP.pep, file = "/home/yhorokh/SNAKEMAKE_pipelines/Mouse_lymphoma_analysis/results/rPCP/rPCP_pep.psv", row.names = F)
write.csv(rPCP.delta, file = "/home/yhorokh/SNAKEMAKE_pipelines/Mouse_lymphoma_analysis/results/rPCP/rPCP_delta.csv", row.names = F)

