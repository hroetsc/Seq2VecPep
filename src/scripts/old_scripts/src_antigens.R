### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  retrieve the source antigens and their sequences from immunopeptidome
# input:        immunopeptidome (mouse lymphoma), list of rPCP
# output:       immunopeptidome concatenated with source antigens, antigen sequences and rPCP values
# author:       HR

library(tibble)
library(dplyr)
library(rlist)
#library(Rcpi)
library(stringr)
library(seqinr)
library(berryFunctions)

setwd("/home/hroetsc/Documents/ProtTransEmbedding/Snakemake/")

### INPUT ###
# immunopeptidome
IP = read.csv("./data/immunopeptidome/Mouse_lymphoma_PEAKS_IP.csv", stringsAsFactors = F, header = T)
IP$X = NULL
IP$X.1 = NULL
IP = na.omit(IP)
IP = unique(IP)

# rPCP
rPCP = read.csv("./data/immunopeptidome/rPCP_final.csv", stringsAsFactors = F, header = T)
rPCP = na.omit(rPCP)
rPCP = unique(rPCP)

### FUNCTIONS ###
# get amino acid from UniProt (from Rcpi package, Snakemake cannot load it)
library(RCurl)
library(doParallel)

getSeqFromUniProt = function (id, parallel = 5) 
{
  fastaTxt = getFASTAFromUniProt(id, parallel)
  tmpfile = tempfile(pattern = paste0(id, "-"), fileext = "fasta")
  for (i in 1:length(id)) write(fastaTxt[[i]], tmpfile[i])
  AASeq = lapply(tmpfile, readFASTA)
  unlink(tmpfile)
  return(AASeq)
}

getFASTAFromUniProt = function (id, parallel = 5) 
{
  fastaURL = paste0("https://www.uniprot.org/uniprot/", id, ".fasta")
  fastaTxt = getURLAsynchronous(url = fastaURL, perform = parallel)
  return(fastaTxt)
}

### MAIN PART ###
# get source antigens
antigens = as.data.frame(matrix(ncol = ncol(IP)+2))
colnames(antigens) = c(colnames(IP), "UniProtID", "antigenSeq")
progressBar = txtProgressBar(min = 0, max = nrow(IP), style = 3)
counter = 1

for (l in 1:nrow(IP)) {
  setTxtProgressBar(progressBar, l)
  UniProtID = unlist(strsplit(IP[l,"Accession"], ":")) # split accessions by UniProtID
  
  tmp = as.data.frame(matrix(ncol = ncol(antigens), nrow = length(UniProtID)))
  colnames(tmp) = colnames(antigens)
  tmp[,"Peptide"] = as.character(t(rep(IP[l,"Peptide"], length(UniProtID))))
  tmp[,"Length"] = as.character(t(rep(IP[l,"Length"], length(UniProtID))))
  tmp[,"Accession"] = as.character(t(rep(IP[l,"Accession"], length(UniProtID))))
  tmp[,"Source.File"] = as.character(t(rep(IP[l,"Source.File"], length(UniProtID))))
  tmp[,"Sample.code"] = as.character(t(rep(IP[l,"Sample.code"], length(UniProtID))))
  tmp[,"Group"] = as.character(t(rep(IP[l,"Group"], length(UniProtID))))
  tmp[,"UniProtID"] = as.character(t(UniProtID))
  
  for (m in 1:nrow(tmp)) { # get antigen sequence
    if(!is.error(getSeqFromUniProt(tmp[m,"UniProtID"], parallel = 20))) {
      AntigenSeq = getSeqFromUniProt(tmp[m,"UniProtID"], parallel = 20)
      tmp[m,"antigenSeq"] = as.character(AntigenSeq[[1]][[1]])
    } else {
      tmp[m,"antigenSeq"] = NA
    }
  }
  antigens[c(counter:(counter+nrow(tmp)-1)),] = tmp
  counter = counter+nrow(tmp)
}
# unique and NA-free dataset
antigens = na.omit(antigens)
antigens = unique(antigens)

# merge rPCP and IP
rPCP[,"Accession"] = rPCP$source
rPCP$source = NULL
antigens = inner_join(antigens, rPCP[,c("Accession","mean_rPCP")])

### OUTPUT ###
#save(file = snakemake@output[["src_antigens_R"]], antigens)
write.csv(antigens, file = "./data/immunopeptidome/src_antigens.csv", row.names = F)
