### SPLICING PREDICTOR ###
# description: generate all possible products from tokens of substrate
# input: database, trained BPE model
# output: all possible products
# author: HR

library(plyr)
library(dplyr)
library(tidyr)
library(stringr)
library(tokenizers.bpe)

# no fragment length restrictions
# no intervening sequence length restrictions

#### INPUT ###
setwd("Documents/QuantSysBios/ProtTransEmbedding/Predictor/")
DB = read.csv("../../Database/15_post-processing/OUTPUT/ProteasomeDB.csv",
              stringsAsFactors = F)

bpeModel = bpe_load_model("../Seq2Vec/results/encoded_sequence/BPE_model_hp.bpe")

### MAIN PART ###
########## tokenize substrates ##########
subs = DB$substrateSeq %>% unique()
subs.encoded = rep(NA, length(subs))

for (s in seq_along(subs)){
  subs.encoded[s] = bpe_encode(bpeModel, subs[s], type = "subwords") %>%
    unlist() %>%
    paste(collapse = " ")
}

subs.encoded = as.data.frame(subs.encoded)
subs.encoded = cbind(DB$substrateID %>% unique(),
                     subs,
                     subs.encoded)

names(subs.encoded) = c("substrateID", "substrateSeq", "substrateEnc")
subs.encoded$substrateEnc = str_split_fixed(subs.encoded$substrateEnc, coll("▁ "), Inf)[, 2]


# tmp
# subEnc = subs.encoded$substrateEnc[1]
# subSeq = subs.encoded$substrateSeq[1] %>% as.character()

{
########## all token combinations ##########


allComb = function(subEnc = ""){
  
  print("find all possible combinations of tokens")
  
  subEnc = str_split(subEnc, coll(" "), simplify = T) %>%
    as.vector()
  
  # generate all possible token combinations
  allPoss = list()
  
  for (i in 1:(length(subEnc) - 1)) {
    
    # forward string - only cis and PCP
    cnt = combn(subEnc, i) %>%
      t() %>%
      as.data.frame() %>%
      unite(col = "product", sep = " ")
    
    allPoss[i] = cnt
    
  }
  
  
  # flatten list
  allPoss = unlist(allPoss) %>%
    as.data.frame() %>%
    unique()
  
  names(allPoss) = "product"
  allPoss$product = as.character(allPoss$product)
  
  return(allPoss)
}

########## assign types ##########

assTypes = function(allPoss = "", subEnc = "") {
  
  print("assign product types")
  
  allPoss$productType = NA
  allPoss$location = NA
  
  pb = txtProgressBar(min = 0, max = nrow(allPoss), style = 3)
  
  for (i in 1:nrow(allPoss)) {
    
    setTxtProgressBar(pb, i)
    
    cnt.Pro = allPoss$product[i] %>%
      str_split(coll(" "), simplify = T) %>%
      paste()
    
    
    # locate product tokens in substrate
    loc = rep(NA, length(cnt.Pro))
    for (c in seq_along(cnt.Pro)) {
      loc[c] = which(subEnc == cnt.Pro[c])
    }
    allPoss$location[i] = loc %>% paste(collapse = " ")
    
    # differences between positions
    D = diff(loc)
    
    # PCP if all are sequential
    if (all(D == 1)) {
      
      allPoss$productType[i] = "PCP"
      
      # if there is exactly one "break" --> PSP
    } else if (length(which(D > 1)) == 1){
      
      allPoss$productType[i] = "cis"
      
      if (any(D < 1)) {
        
        allPoss$productType[i] = "revCis"
        
      }
      
    } else {
      
      allPoss$productType[i] = "none"
      
    }
    
  }
  
  allPoss = allPoss[-which(allPoss$productType == "none"), ]
  
  return(allPoss)
}
}

########## PCPs ##########
# find location of tokens in substrate

getIdx = function(product = "", substrate = "") {
  
  res = rep(NA, length(product))
  
  for (p in seq_along(product)) {
    
    cnt.pro = str_split(product[p], coll(" "), simplify = T) %>%
      paste()
    
    loc = rep(NA, length(cnt.pro))
    
    for (c in seq_along(cnt.pro)) {
      loc[c] = which(substrate == cnt.pro[c])
    }
    
    res[p] = loc %>% paste(collapse = " ")
    
  }
  
  return(res)
}

# 2 cleavage sites at any position

genPCP = function(subEnc = ""){
  
  print("all possible PCPs")
  
  subEnc = str_split(subEnc, coll(" "), simplify = T) %>%
    as.vector()
  
  pcp.products = c()
  
  pos1 = 1
  
  for (o in 1:length(subEnc)) {
    
    pos2 = 1
    
    for (p in 1:length(subEnc)) {
      
      cnt.pcp = paste(subEnc[pos1:pos2], collapse = " ")
      
      pcp.products = c(pcp.products,
                       cnt.pcp)
      
      pos2 = pos2 + 1
      
    }
    
    pos1 = pos1 + 1
    
  }
  
  pcp.products = pcp.products[-which(pcp.products == paste(subEnc, collapse = " "))] %>%
    as.data.frame()
  
  pcp.products$type = rep("PCP", nrow(pcp.products))
  
  names(pcp.products) = c("product", "type")
  pcp.products$location = getIdx(product = pcp.products$product %>% as.character(),
                                 substrate = subEnc)
  
  return(pcp.products)
}

########## PSPs ##########
# 4 splice / cleavage sites

genPSP = function(subEnc = "") {
  
  print("all possible PSPs")
  
  subEnc = str_split(subEnc, coll(" "), simplify = T) %>%
    as.vector()
  
  psp.products = c()
  
  pos1 = 1
  
  for (o in seq_along(subEnc)) {
    
    pos2 = 1
    
    for (p in seq_along(subEnc)) {
      
      pos3 = 1
      
      for (q in seq_along(subEnc)) {
        
        pos4 = 1
        
        for (r in seq_along(subEnc)) {
          
          SR1 = subEnc[pos1:pos2] %>%
            paste(collapse = " ")
          
          SR2 = subEnc[pos3:pos4] %>%
            paste(collapse = " ")
          
          cnt.psp = paste(SR1, SR2, sep = " ")
          
          psp.products = c(psp.products,
                           cnt.psp)
          pos4 = pos4 + 1
          
        }
        
        pos3 = pos3 + 1
        
      }
      
      pos2 = pos2 + 1
      
    }
    
    pos1 = pos1 + 1
    
  }
  
  psp.products = as.data.frame(psp.products) %>%
    unique()
  psp.products$type = NA
  names(psp.products) = c("product", "type")
  
  # filter
  pb = txtProgressBar(min = 0, max = nrow(psp.products), style = 3)
  
  # location of tokens in substrate
  psp.products$location = getIdx(product = psp.products$product %>% as.character(),
                                 substrate = subEnc)
  
  for (i in 1:nrow(psp.products)) {
    setTxtProgressBar(pb, i)
    
    # assign product type
    # get location and calculate pairwise differences
    loc = psp.products$location[i] %>% 
      str_split(coll(" "), simplify = T) %>% 
      paste() %>% 
      as.numeric()
    D = diff(loc)
    
    # PCP if all are sequential (can this happen??)
    if (all(D == 1) &
        !any(duplicated(loc))) {
      
      psp.products$type[i] = "PCP"
      
      # if there is exactly one "break" --> cis PSP
    } else if (length(which(D > 1)) == 1 &
               !any(D < 1)&
               !any(duplicated(loc))){
      
      psp.products$type[i] = "cis"
      
      # revCis PSP
    } else if (length(which(D < 1)) == 1 &
               !any(D > 1)&
               !any(duplicated(loc))) {
      
      psp.products$type[i] = "revCis"
      
      # trans PSP:
      # token must occur multiple times
      # only one "gap" in sequence
    } else if (any(duplicated(loc)) & 
               length(which(! D == 1)) == 1) {
      
      psp.products$type[i] = "trans"
      
      # no meaningful product
    } else {
      
      psp.products$type[i] = "none"
      
    }
    
  }
  
  psp.products = psp.products[-which(psp.products$type == "none"), ]
  
  return(psp.products)
}



########## match extended and minimal substring ##########

# loop over concatenated table
# check if there is minimal substring to existing substring
# store in table if true

matchSubstr = function(pcp.products = "", psp.products = "", subSeq = "") {
  
  print("match extended and minimal substrings, if present")
  
  master = rbind(pcp.products, psp.products) %>%
    unique()
  
  table(master$type)
  
  ext_min.match = data.frame(ext_substring = rep(NA, nrow(master)),
                             min_substring = rep(NA, nrow(master)),
                             type = rep(NA, nrow(master)),
                             location = rep(NA, nrow(master)),
                             present = rep(NA, nrow(master)))

  
  pb = txtProgressBar(min = 0, max = nrow(master), style = 3)
  for (m in 1:nrow(master)) {
    
    setTxtProgressBar(pb, m)
    
    # current product as extended substring
    cnt.ext = master$product[m] %>%
      as.character() %>%
      str_split(coll(" "), simplify = T) %>%
      as.character()
    
    # complete table
    ext_min.match$ext_substring[m] = cnt.ext %>% paste(collapse = " ")
    ext_min.match$type[m] = master$type[m] %>% as.character()
    ext_min.match$location[m] = master$location[m]
    
    # get location
    loc = master$location[m] %>% 
      str_split(coll(" "), simplify = T) %>% 
      paste() %>% 
      as.numeric()
    D = diff(loc)
    
    
    if (length(D) > 1) {
      
      # PCPs should differ by tokens at start/end
      if (master$type[m] == "PCP" &
          length(cnt.ext) > 2){
        
        cnt.min = cnt.ext[2:(length(cnt.ext) - 1)] %>%
          paste(collapse = " ")
        
        {
          
          master.line = master[which(cnt.min == master$product), ]
          
          if (nrow(master.line) == 1 &
              master.line$type[1] == master$type[m]) {
            
            ext_min.match$min_substring[m] = master.line$product[1] %>% as.character()
            ext_min.match$present[m] = T
            
          } else {
            
            ext_min.match$min_substring[m] = cnt.min
            ext_min.match$present[m] = F
            
          }
          
          }
        
      # PSPs can differ by tokens at start/end + 2 tokens at splice site
      } else if (master$type[m] %in% c("cis", "revCis", "trans") &
                 length(cnt.ext) > 4 &
                 which(!D == 1) %in% seq(3, length(cnt.ext) - 2)){
        
        SR1 = cnt.ext[2:(which(D != 1)-1)]%>%
          paste(collapse = " ")
        SR2 = cnt.ext[(which(D != 1)+2):(length(cnt.ext)-1)]%>%
          paste(collapse = " ")
        
        cnt.min = paste(SR1, SR2, sep = " ")
        
        {
          
          master.line = master[which(cnt.min == master$product), ]
          
          if (nrow(master.line) == 1 &
              master.line$type[1] == master$type[m]) {
            
            ext_min.match$min_substring[m] = master.line$product[1] %>% as.character()
            ext_min.match$present[m] = T
            
          } else {
            
            ext_min.match$min_substring[m] = cnt.min
            ext_min.match$present[m] = F
            
          }
          
        }
        
      } else {
        
        ext_min.match$min_substring[m] = NA
        ext_min.match$present[m] = F
        
      }
    
    } else {
      
      ext_min.match$min_substring[m] = NA
      ext_min.match$present[m] = F
      
    }
    
  }
  
  ext_min.match %>% na.omit() %>% nrow()
  ext_min.match[ext_min.match$present == F, ] %>% nrow()
  ext_min.match[!is.na(ext_min.match$min_substring) & ext_min.match$present == F, ] %>% nrow()
  
  return(ext_min.match)
}

# location and position refer to extended substring
# location: indices of tokens in encoded substate
# position: exact aa position in original substrate


########## map to substrate sequence ##########

mapSubs = function(ext_min.match = "", subSeq = "") {
  
  print("map products to substrate")
  pb = txtProgressBar(min = 0, max = nrow(ext_min.match), style = 3)
  
  for (e in 1:nrow(ext_min.match)) {
    
    setTxtProgressBar(pb, e)
    
    if (ext_min.match$type[e] == "PCP") {
      
      pos = str_locate(subSeq,
                       ext_min.match$ext_substring[e] %>%
                         str_replace_all(coll(" "), coll("")) %>%
                         as.character())
      
      ext_min.match$position[e] = pos %>% as.numeric() %>% paste(collapse = "_")
      
    } else {
      
      cnt.pro = ext_min.match$ext_substring[e] %>%
        str_split(coll(" "), simplify = T) %>%
        paste()
      
      loc = ext_min.match$location[e] %>% 
        str_split(coll(" "), simplify = T) %>% 
        paste() %>% 
        as.numeric()
      D = diff(loc)
      
      SR1 = cnt.pro[1:which(D != 1)]%>%
        paste(collapse = " ")
      SR2 = cnt.pro[(which(D != 1)+1):length(cnt.pro)]%>%
        paste(collapse = " ")
      
      pos1 = str_locate(subSeq,
                        SR1 %>%
                          str_replace_all(coll(" "), coll("")) %>%
                          as.character()) %>%
        as.numeric() %>%
        paste(collapse = "_")
      
      pos2 = str_locate(subSeq,
                        SR2 %>%
                          str_replace_all(coll(" "), coll("")) %>%
                          as.character()) %>%
        as.numeric() %>%
        paste(collapse = "_")
      
      ext_min.match$position[e] = paste(pos1, pos2, sep = "_")
    }
    
    
  }
  
  # remove products with inverse order
  ext_min.match = ext_min.match[-which(ext_min.match$position == "NA_NA"), ]
  
  ext_min.match = ext_min.match[, c("ext_substring", "type", "location", "position",
                                    "min_substring", "present")]
  names(ext_min.match) = c("product", "type", "tokenIDs", "aaPositions", "min_version", "present_in_DB")
  
  return(ext_min.match)
}


########## run ##########
results = list()

for (i in 1:nrow(subs.encoded)) {
  
  print(paste0("SUBSTRATE: ", subs.encoded$substrateID[i]))
  subEnc = subs.encoded$substrateEnc[i]
  subSeq = subs.encoded$substrateSeq[i] %>% as.character()
  
  # gen PCP, PSP
  pcp.products = genPCP(subEnc = subEnc)
  psp.products = genPSP(subEnc = subEnc)
  
  # match extended and minimal substring
  ext_min.match = matchSubstr(pcp.products = pcp.products,
                              psp.products = psp.products)
  
  # map to substrate
  mapped = mapSubs(ext_min.match = ext_min.match,
                   subSeq = subSeq)
  
  # store
  results[i] = mapped
}



### OUTPUT ###


