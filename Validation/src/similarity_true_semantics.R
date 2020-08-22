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


print("TRUE SEMANTIC SIMILARITY")

### INPUT ###
# formatted sequences
sequences = read.csv(snakemake@input[["batch_sequence"]], stringsAsFactors = F, header = T)
accessions = read.csv(snakemake@input[["batch_accessions"]], stringsAsFactors = F, header = T)


### MAIN PART ###

GO_sim = function(ont = ""){
  
  print(paste0("calculating semantic similarity for ", ont, " ontology"))
  
  hsGO = godata('org.Hs.eg.db', keytype = "UNIPROT", ont = ont, computeIC = F)
  annot = hsGO@geneAnno
  
  alig = accessions %>% as.data.frame()
  alig$similarity = NULL
  
  pb = txtProgressBar(min = 0, max = nrow(accessions), style = 3)
  
  rm = c()
  
  for(a in 1:nrow(accessions)) {
    
    setTxtProgressBar(pb, a)
    
    # no isoform discrimination in GO terms!
    prot1 = str_split(accessions$acc1[a], coll("-"), simplify = T)[,1]
    prot2 = str_split(accessions$acc2[a], coll("-"), simplify = T)[,1]
    
    go1 = as.character(annot[which(annot$UNIPROT == prot1), "GO"])
    go2 = as.character(annot[which(annot$UNIPROT == prot2), "GO"])
    
    if(length(go1) & length(go2)){
      alig[a, "similarity"] = mgoSim(go1, go2,
                                  semData = hsGO,measure="Wang", combine="BMA")
      
    } else {
      
      alig[a, "similarity"] = NA
      
    }
    
  }
  
  alig = na.omit(alig)
  
  return(alig)
  
}


res_MF = GO_sim(ont = "MF")
res_BP = GO_sim(ont = "BP")
res_CC = GO_sim(ont = "CC")


plot(density(res_MF$similarity))
plot(density(res_BP$similarity))
plot(density(res_CC$similarity))


### OUTPUT ###
write.csv(res_MF, file = unlist(snakemake@output[["semantics_MF"]]), row.names = F)
write.csv(res_BP, file = unlist(snakemake@output[["semantics_BP"]]), row.names = F)
write.csv(res_CC, file = unlist(snakemake@output[["semantics_CC"]]), row.names = F)

