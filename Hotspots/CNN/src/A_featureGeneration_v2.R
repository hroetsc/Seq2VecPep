### HEADER ###
# HOTSPOT PREDICTION
# description: generate feature set by applying a sliding window over a protein sequence
# input: windowCounts (from Juliane), human proteome (encoded)
# output: feature set (counts and tokens for every sequence)
# author: HR

library(plyr)
library(dplyr)
library(stringr)

library(foreach)
library(doParallel)
library(future)
registerDoParallel(availableCores())

### INPUT ###
load("../HOTSPOTS/accU.RData")
load("../HOTSPOTS/RESULTS/windowCounts.RData")

# human proteome
prots = read.csv("../../files/proteome_human.csv", stringsAsFactors = F, header = T)
words = read.csv("../../RUNS/HumanProteome/v_50k/words_hp_v50k.csv", stringsAsFactors = F, header = T)

### MAIN PART ###

tokPerWindow = 8

# slide over tokens (not amino acids)
# take the mean count as count for sliding window

# keep only proteins with hotspots
prots = prots[which(prots$Accession %in% accU), ]
prots = left_join(prots, words)

names(windowCounts) = accU

  
# loop over proteins
get_windows_counts = function(extension = "", outfile = ""){
  
  windowTokens = list()
  
  windowTokens = foreach (i = 1:length(windowCounts)) %dopar% {
    
    cnt.Prot = prots[prots$Accession == names(windowCounts)[i], ]
    
    cnt.Tokens = str_split(cnt.Prot$tokens, coll(" "), simplify = T) %>%
      t() %>%
      as.data.frame()
    names(cnt.Tokens) = "tok"
    cnt.Tokens$tok = as.character(cnt.Tokens$tok)
    
    # enumerate all tokens
    cnt.Tokens$idx = seq(1, nrow(cnt.Tokens)) %>% as.numeric()
    
    # apply sliding window and check which tokens are in current window
    # store all window tokens for current protein
    wnd.Tokens = data.frame(Accession = rep(cnt.Prot$Accession, nrow(cnt.Tokens) - tokPerWindow + 1),
                            tokens = rep(NA, nrow(cnt.Tokens) - tokPerWindow + 1),
                            counts = rep(NA, nrow(cnt.Tokens) - tokPerWindow + 1))
    
    pos1 = 1
    pos2 = pos1 + tokPerWindow - 1
    
    while (pos2 <= nrow(cnt.Tokens)) {
      
      # check which tokens are in current sliding window
      cnt.Wnd = cnt.Tokens$tok[pos1:pos2] %>% paste(collapse = " ")
      wnd.Tokens$tokens[pos1] = cnt.Wnd
      
      
      # different extensions --> refine window that is used to get counts
      if (extension == "none") {
        Wnd.for.counts = cnt.Tokens$tok[pos1:pos2] %>% paste(collapse = " ")
      } else if (extension == "N") {
        Wnd.for.counts = cnt.Tokens$tok[(pos1+1):pos2] %>% paste(collapse = " ")
      } else if (extension == "C") {
        Wnd.for.counts = cnt.Tokens$tok[pos1:(pos2-1)] %>% paste(collapse = " ")
      } else if (extension == "NandC") {
        Wnd.for.counts = cnt.Tokens$tok[(pos1+1):(pos2-1)] %>% paste(collapse = " ")
      }
      
      
      # get mean counts of current window
      loc = str_locate(cnt.Prot$seqs,
                       str_replace_all(Wnd.for.counts, coll(" "), coll(""))) %>%
        as.numeric()
      
      wnd.Tokens$counts[pos1] = windowCounts[[i]][c(loc[1]:loc[2])] %>% mean()
      
      
      # increment positions of sliding window
      pos1 = pos1 + 1
      pos2 = pos2 + 1
      
    }
    
    
    windowTokens[[i]] = wnd.Tokens
    
    
  }
  
  
  ### OUTPUT ###
  save(windowTokens, file = paste0(outfile, ".RData"))
  
  # reduce feature space
  # average counts of neighbouring identical token sets - should not happen anymore
  # plus:
  # flatten and save list in chunks
  # load(outfile)
  
  out = paste0(outfile, ".csv")
  system(paste0("rm ", out))
  
  pb = txtProgressBar(min = 0, max = length(windowTokens), style = 3)
  
  for ( i in 1:length(windowTokens) ) {
    
    setTxtProgressBar(pb, i)
    
    wT = windowTokens[[i]] %>% as.data.frame()
    
    if(file.exists(out)) {
      write.table(wT, out, sep = ",", row.names = F, append = T, col.names = F)
      
    } else {
      write.table(wT, out, sep = ",", row.names = F, append = F, col.names = T)
      
    }
  }
  
  
}

# apply
get_windows_counts(extension = "none", outfile = "data/windowTokens")
get_windows_counts(extension = "N", outfile = "data/Next_windowTokens")
get_windows_counts(extension = "C", outfile = "data/Cext_windowTokens")
get_windows_counts(extension = "NandC", outfile = "data/NandCext_windowTokens")

