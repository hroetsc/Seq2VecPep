library(stringr)

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
  
  ti = str_split(fs[i], coll("/"), simplify = T)[,-1] %>% as.character()
  ti = str_split(ti, coll("."), simplify = T)[,1]
  
  png(paste0("tmp/", ti, ".png"))
  plot(density(tbl), main = ti)
  dev.off()
  
  tbl = (tbl - min(tbl)) / (max(tbl) - min(tbl))
  
  tbl = (tbl - mean(tbl)) / sd(tbl)
  
  
  #plot(density(tbl), main = ti)
  tbl = pnorm(tbl)
  
  png(paste0("tmp/", ti, "_norm.png"))
  plot(density(tbl), main = ti)
  dev.off()
  
  print(fs[i])
  print(min(tbl))
  print(max(tbl))
  print(mean(tbl))
}


