
cleaning = function(tbl = ""){
  # sort in protein order
  if ("Accession" %in% colnames(tbl)){
    tbl = tbl[order(tbl[, "Accession"]), ]
    tbl$Accession = NULL
  } else {
    tbl = tbl[order(tbl[,1]), ]
    tbl[,1] = NULL
  }
  
  # remove redundant columns
  if ("seqs" %in% colnames(tbl)){
    tbl$seqs = NULL
  }
  
  if ("X" %in% colnames(tbl)){
    tbl$X = NULL
  }
  
  return(as.matrix(tbl))
}


fs = list.files("similarity", ".csv", recursive = F, full.names = T)


for (i in 1:length(fs)){
  tbl = read.csv(fs[i], stringsAsFactors = F)
  tbl = cleaning(tbl)
  
  plot(density(tbl), main = fs[i])
  
  tbl = tbl / mean(tbl)
  tbl = (tbl - mean(tbl)) / sd(tbl)
  
  plot(density(tbl), main = fs[i])
  
}


