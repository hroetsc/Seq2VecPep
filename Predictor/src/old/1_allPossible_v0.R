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
print(bpeModel$vocab_size)


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


# tmp!
i = 1
print(paste0("SUBSTRATE: ", subs.encoded$substrateID[i]))
subEnc = subs.encoded$substrateEnc[i]
subSeq = subs.encoded$substrateSeq[i] %>% as.character()


########## token location in substrate ##########
# location and position refer to extended substring
# location: indices of tokens in encoded substate
# position: exact aa position in original substrate

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


########## generate PCPs ##########
genPCP = function(subEnc = ""){
  
  print("all possible PCPs")
  
  subEnc = str_split(subEnc, coll(" "), simplify = T) %>%
    as.vector()
  
  pcp.products = c()
  
  for (o in 1:length(subEnc)) {
    
    for (p in 1:length(subEnc)) {
      
      if (! p < o) {
        
        cnt.pcp = paste(subEnc[o:p], collapse = " ")
        
        pcp.products = c(pcp.products,
                         cnt.pcp)
      }
      
    }
    
  }
  
  pcp.products = pcp.products[-which(pcp.products == paste(subEnc, collapse = " "))] %>%
    as.data.frame()
  
  pcp.products$type = rep("PCP", nrow(pcp.products))
  
  names(pcp.products) = c("product", "type")
  pcp.products$location = getIdx(product = pcp.products$product %>% as.character(),
                                 substrate = subEnc)
  
  return(pcp.products)
}



########## generate PSPs ##########
genPSP = function(subEnc = "", subSeq = "", pcp.products = "") {
  
  print("all possible PSPs")
  
  # combine all possible PCPs
  reactants = pcp.products$product %>% as.character()
  
  cis.products = c()
  revCis.products = c()
  trans.products = c()
  
  for (a in 1:length(reactants)) {
    
    for (b in 1:length(reactants)) {
      
      cnt.product = paste(reactants[a], reactants[b], collapse = " ")
      
      # get reactant position in substrate
      i1.j1 = str_locate(subEnc %>% str_replace_all(" ", ""),
                         reactants[a] %>% str_replace_all(" ", ""))
      i2.j2 = str_locate(subEnc %>% str_replace_all(" ", ""),
                         reactants[b] %>% str_replace_all(" ", ""))
      pos = c(i1.j1, i2.j2)
      
      # cis --> pos[2] < pos[3]
      if (pos[2]+1 < pos[3]) {
        cis.products = c(cis.products, cnt.product)
        
        # revCis --> pos[4] <= pos[1]
      } else if (pos[4] <= pos[1]) {
        revCis.products = c(revCis.products, cnt.product)
        
        # trans --> pos[1] <= pos[3]
      } else if (pos[1] <= pos[3] & pos[2]+1 != pos[3]) {
        trans.products = c(trans.products, cnt.product)
      }
    }
    
  }
  
  # concatenate
  psp.products = data.frame(product = c(cis.products, revCis.products, trans.products),
                            type = c(rep("cis", length(cis.products)),
                                     rep("revCis", length(revCis.products)),
                                     rep("trans", length(trans.products))))
  
  # get positions
  psp.products$location = getIdx(product = psp.products$product %>% as.character(),
                                 substrate = subEnc %>% str_split(coll(" "), simplify = T) %>%
                                   as.vector())
  
  return(psp.products)
}

 
########### match etended and minimal substring --> control ##########
# matchSubstr = function(products = "", subSeq = "") {
#   
#   print("match extended and minimal substrings, if present")
#   
#   ext_min.match = data.frame(ext_substring = rep(NA, nrow(products)),
#                              min_substring = rep(NA, nrow(products)),
#                              type = rep(NA, nrow(products)),
#                              location = rep(NA, nrow(products)),
#                              present = rep(NA, nrow(products)))
#   
#   
#   pb = txtProgressBar(min = 0, max = nrow(products), style = 3)
#   for (m in 1:nrow(products)){
#     
#     setTxtProgressBar(pb, m)
#     
#     # current product as extended substring
#     cnt.ext = products$product[m] %>%
#       as.character() %>%
#       str_split(coll(" "), simplify = T) %>%
#       as.character()
#     
#     # complete table
#     ext_min.match$ext_substring[m] = cnt.ext %>% paste(collapse = " ")
#     ext_min.match$type[m] = products$type[m] %>% as.character()
#     ext_min.match$location[m] = products$location[m]
#     
#     # get location
#     loc = products$location[m] %>% 
#       str_split(coll(" "), simplify = T) %>% 
#       paste() %>% 
#       as.numeric()
#     D = diff(loc)
#     
#     if (length(D) > 1) {
#       
#       # PCPs should differ by tokens at start/end
#       if (products$type[m] == "PCP" &
#           length(cnt.ext) > 2) {
#         
#         cnt.min = cnt.ext[2:(length(cnt.ext) - 1)] %>%
#           paste(collapse = " ")
#         
#         {
#           
#           products.line = products[which(cnt.min == products$product), ]
#           
#           if (nrow(products.line) == 1 &
#               products.line$type[1] == products$type[m]) {
#             
#             ext_min.match$min_substring[m] = products.line$product[1] %>% as.character()
#             ext_min.match$present[m] = T
#             
#           } else {
#             
#             ext_min.match$min_substring[m] = cnt.min
#             ext_min.match$present[m] = F
#             
#           }
#           
#           }
#         
#       # PSPs can differ by tokens at start/end + 2 tokens at splice site
#       } else if (products$type[m] %in% c("cis", "revCis", "trans") &
#                  length(cnt.ext) > 4 &
#                  which(!D == 1) %in% seq(3, length(cnt.ext) - 2)){
#         
#         SR1 = cnt.ext[2:(which(D != 1)-1)]%>%
#           paste(collapse = " ")
#         SR2 = cnt.ext[(which(D != 1)+2):(length(cnt.ext)-1)]%>%
#           paste(collapse = " ")
#         
#         cnt.min = paste(SR1, SR2, sep = " ")
#         
#         {
#           
#           products.line = products[which(cnt.min == products$product), ]
#           
#           if (nrow(products.line) == 1 &
#               products.line$type[1] == products$type[m]) {
#             
#             ext_min.match$min_substring[m] = products.line$product[1] %>% as.character()
#             ext_min.match$present[m] = T
#             
#           } else {
#             
#             ext_min.match$min_substring[m] = cnt.min
#             ext_min.match$present[m] = F
#             
#           }
#           
#         }
#       }else {
#         
#         ext_min.match$min_substring[m] = NA
#         ext_min.match$present[m] = F
#         
#       }
#       
#     } else {
#       
#       ext_min.match$min_substring[m] = NA
#       ext_min.match$present[m] = F
#       
#     }
#     
#     
#   }
#   
#   return(ext_min.match)
# }

########## map all possible products to substrate positions ##########
mapToSubs = function(products = "", subSeq = "") {
  
  print("map products to substrate positions")
  pb = txtProgressBar(min = 0, max = nrow(products), style = 3)
  
  for (e in 1:nrow(products)) {
    
    setTxtProgressBar(pb, e)
    
    if (products$type[e] == "PCP") {
      
      pos = str_locate(subSeq,
                       products$product[e] %>%
                         str_replace_all(coll(" "), coll("")) %>%
                         as.character())
      
      products$position[e] = pos %>% as.numeric() %>% paste(collapse = "_")
      
    } else {
      
      cnt.pro = products$product[e] %>%
        str_split(coll(" "), simplify = T) %>%
        paste()
      
      loc = products$location[e] %>% 
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
      
      products$position[e] = paste(pos1, pos2, sep = "_")
    }
    
    
  }
  
  return(products)
}

########## get labels ##########
getLabels = function(mapped = "", subSeq = "") {
  
  print("for every theoretical product, determine if it is present in the database")
  
  cnt.DB = DB[DB$substrateSeq == subSeq & DB$digestTime %in% c("4", "20", "24"), ]
  
  ## tokenize true products using substrate encoding
  # token position in substrate
  tokens = str_split(subEnc, coll(" "), simplify = T) %>%
    t() %>%
    as.data.frame()
  tokens$V1 = as.character(tokens$V1)
  
  end = 0
  for (t in 1:nrow(tokens)) {
    tokens$start[t] = end + 1
    end = end + nchar(tokens$V1[t])
    tokens$end[t] = end
  }
  
  # for every real product --> build all possible token products
  # check if/where they are in the all possible table
  
  for (p in 1:nrow(cnt.DB)) {
    pos = str_split(cnt.DB$positions[p], "_", simplify = T) %>% as.vector()
    
    if (cnt.DB$productType[p] == "PCP") {
      pos = seq(pos[1], pos[2])
      
      cnt.ext = tokens$V1[tokens$start %in% pos | tokens$end %in% pos] %>%
        paste(collapse = " ")
      cnt.min = tokens$V1[tokens$start %in% pos & tokens$end %in% pos] %>%
        paste(collapse = " ")
      
      mapped$label[mapped$product == cnt.ext] = 1
      mapped$label[mapped$product == cnt.min] = 1
      mapped$true_product[mapped$product == cnt.ext] = cnt.DB$pepSeq[p]
      mapped$true_product[mapped$product == cnt.min] = cnt.DB$pepSeq[p]
      
      # for spliced, it is more complicated
    } else if (cnt.DB$spliceType[p] %in% c("cis", "revCis", "trans")) {
      SR1 = seq(pos[1], pos[2])
      SR2 = seq(pos[3], pos[4])
      
      ext_SR1 = tokens$V1[tokens$start %in% SR1 | tokens$end %in% SR1] %>%
        paste(collapse = " ")
      ext_SR2 = tokens$V1[tokens$start %in% SR2 | tokens$end %in% SR2] %>%
        paste(collapse = " ")
      
      min_SR1 = tokens$V1[tokens$start %in% SR1 & tokens$end %in% SR1] %>%
        paste(collapse = " ")
      min_SR2 = tokens$V1[tokens$start %in% SR2 & tokens$end %in% SR2] %>%
        paste(collapse = " ")
      
      # all possible combinations of SRs
      cnt1 = paste(ext_SR1, min_SR2)
      cnt2 = paste(ext_SR1, min_SR2)
      cnt3 = paste(min_SR1, min_SR2)
      cnt4 = paste(ext_SR1, ext_SR2)
      
      mapped$label[mapped$product == cnt1] = 1
      mapped$label[mapped$product == cnt2] = 1
      mapped$label[mapped$product == cnt3] = 1
      mapped$label[mapped$product == cnt4] = 1
      
      mapped$true_product[mapped$product == cnt1] = cnt.DB$pepSeq[p]
      mapped$true_product[mapped$product == cnt2] = cnt.DB$pepSeq[p]
      mapped$true_product[mapped$product == cnt3] = cnt.DB$pepSeq[p]
      mapped$true_product[mapped$product == cnt4] = cnt.DB$pepSeq[p]
      
      
    }
  }
  
  mapped$label[is.na(mapped$label)] = 0
  mapped$true_product[is.na(mapped$true_product)] = ""
  
  table(mapped$label)
  cnt.DB$pepSeq %>% unique() %>% length()
  
  peps = cnt.DB$pepSeq %>% unique()
  peps[! peps %in% mapped$true_product]
  # ~ 100 peptides could not be found
  return(mapped)
}


########## run ##########
results = list()

for (i in 1:nrow(subs.encoded)) {
  
  print(paste0("SUBSTRATE: ", subs.encoded$substrateID[i]))
  subEnc = subs.encoded$substrateEnc[i]
  subSeq = subs.encoded$substrateSeq[i] %>% as.character()
  
  # get PCP, PSP
  pcp.products = genPCP(subEnc)
  psp.products = genPSP(subEnc, subSeq, pcp.products)
  
  # concatenate
  products = rbind(pcp.products, psp.products) %>%
    unique()
  products$product = as.character(products$product)
  barplot(products$type %>% table())
  print(paste0(nrow(products), " theoretical products"))
  
  # map to substrate
  mapped = mapToSubs(products, subSeq)
  master = getLabels(mapped, subSeq)
  
  # store
  results[[i]] = master
}



