### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  "true" similarity between protein sequences based on GO-term similarity
# input:        sequences
# output:       semantic similarity matrices (MF, BP, CC)
# author:       HR


library(plyr)
library(dplyr)
library(stringr)
library(GOSemSim)
library(AnnotationHub)

print("TRUE SEMANTIC SIMILARITY")

### INPUT ###
# formatted sequences
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)


### MAIN PART ###

# order accessions alphabetically
sequences = sequences[order(sequences$Accession), ]
acc = sequences$Accession

GO_sim = function(ont = ""){
  
  print(paste0("calculating semantic similarity for ", ont, " ontology"))
  
  hsGO = godata('org.Hs.eg.db', keytype = "UNIPROT",ont = ont, computeIC = F)
  annot = hsGO@geneAnno
  
  alig = matrix(ncol = length(acc), nrow = length(acc))
  
  pb = txtProgressBar(min = 0, max = length(acc), style = 3)
  
  rm = c()
  
  for(a in 1:nrow(alig)) {
    
    setTxtProgressBar(pb, a)
    
    for (b in 1:ncol(alig)){
      
      # no isoform discrimination in GO terms!
      prot1 = str_split(acc[a], coll("-"), simplify = T)[,1]
      prot2 = str_split(acc[b], coll("-"), simplify = T)[,1]
      
      go1 = as.character(annot[which(annot$UNIPROT == prot1), "GO"])
      go2 = as.character(annot[which(annot$UNIPROT == prot2), "GO"])
      
      if(length(go1) & length(go2)){
        alig[a,b] = mgoSim(go1, go2,
                           semData = hsGO,measure="Wang", combine="BMA")
        
      } else {
        
        alig[a,b] = 0
  
      }
      
    }
    
  }
  
  # remove all columns that don't have a GO term
  
  k = which(colSums(alig) == 0)
  
  acc = acc[-k]
  alig = alig[-k,]
  alig = alig[,-k]
  
  if (ncol(alig) == 0){
    
    print("!!!WARNING!!! NO GO TERM FOUND FOR ANY PROTEIN !!!")
    
  }
  
  # add accessions
  res = matrix(ncol = ncol(alig)+1, nrow = nrow(alig))
  res[, 1] = acc
  res[, c(2:ncol(res))] = alig
  colnames(res) = c("Accession", seq(1, ncol(alig)))
  
  res = as.data.frame(res)
  
  return(res)
  
}


res_MF = GO_sim(ont = "MF")
res_BP = GO_sim(ont = "BP")
res_CC = GO_sim(ont = "CC")


### OUTPUT ###
write.csv(res_MF, file = unlist(snakemake@output[["semantics_MF"]]), row.names = T)
write.csv(res_BP, file = unlist(snakemake@output[["semantics_BP"]]), row.names = T)
write.csv(res_CC, file = unlist(snakemake@output[["semantics_CC"]]), row.names = T)
