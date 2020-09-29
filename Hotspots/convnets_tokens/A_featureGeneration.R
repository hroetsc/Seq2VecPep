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
registerDoParallel(8)

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
  wnd.Tokens = rep(NA, nchar(cnt.Prot$seqs))
  
  pos1 = 1
  pos2 = pos1 + tokPerWindow
  
  while (pos1 <= (nchar(cnt.Prot$seqs) - 20)) {
    
    # check which tokens are in current sliding window
    wnd.Idx = which(cnt.Tokens$start %in% seq(pos1, pos2) &
                      cnt.Tokens$end %in% seq(pos1, pos2))
    
    # if last window does not match any start position for tokens
    if (length(wnd.Idx) == 0) {
      wnd.Idx = nrow(cnt.Tokens)
    }
    
    # if number of tokens matches the desired length, keep them
    if (length(wnd.Idx) == tokPerWindow) {
      
      wnd.Tokens[pos1] = cnt.Tokens$tok[wnd.Idx] %>% paste(collapse = " ")
      
      # if smaller, add tokens until size is reached, beginning with
      # C-term (unless the window is at the end position of the protein)
    
    } else if (length(wnd.Idx) < tokPerWindow) {
      
      k = 0
      
      while (length(wnd.Idx) < tokPerWindow) {
        # start at C-term if possible
        if (k %% 2 == 0 & (! wnd.Idx[length(wnd.Idx)] + 1 > nrow(cnt.Tokens))) {
          wnd.Idx = c(wnd.Idx,
                      wnd.Idx[length(wnd.Idx)] + 1)
          # else, start extending at N-term
        } else if (k %% 2 == 0 & wnd.Idx[length(wnd.Idx)] + 1 > nrow(cnt.Tokens)) {
          wnd.Idx = c(wnd.Idx[1] - 1,
                      wnd.Idx)
          # continue at N-term if possible
        } else if (k %% 2 != 0 & (! wnd.Idx[1] - 1 < 1)) {
          wnd.Idx = c(wnd.Idx[1] - 1,
                      wnd.Idx)
          # else, continue at C-term
        } else if (k %% 2 != 0 & wnd.Idx[1] - 1 < 1) {
          wnd.Idx = c(wnd.Idx,
                      wnd.Idx[length(wnd.Idx)] + 1)
        }
        
        k = k + 1
        
        
      }
      
      wnd.Tokens[pos1] = cnt.Tokens$tok[wnd.Idx] %>% paste(collapse = " ")
      
      # if greater, remove tokens until size is reached, beginning with N-term
    } else if (length(wnd.Idx) > tokPerWindow) {
      
      k = 0
      
      while (length(wnd.Idx) > tokPerWindow) {
        
        # start removing at N-term
        if (k %% 2 == 0) {
          wnd.Idx = wnd.Idx[-1]
          
          # continue at C-term
        } else {
          wnd.Idx = wnd.Idx[-length(wnd.Idx)]
          
        }
        
        k = k + 1
        
      }
      
      wnd.Tokens[pos1] = cnt.Tokens$tok[wnd.Idx] %>% paste(collapse = " ")
      
    }
    
    # save all window tokens in output together with counts
    
    
    # increment window positions
    # sliding window at the end of protein will become smaller
    pos1 = pos1 + 1
    
    if (pos1 + windowSize <= nchar(cnt.Prot$seqs)) {
      pos2 = pos1 + windowSize
    } else {
      pos2 = nchar(cnt.Prot$seqs)
    }
    
      
  }
  
  # store all window tokens for current protein
  # some sliding windows contain same token sets and same scores --> remove to reduce size
  out = cbind(rep(cnt.Prot$Accession, length(wnd.Tokens)),
              wnd.Tokens,
              windowCounts[[i]]) %>% 
    as.data.frame() %>%
    unique()
  
  names(out) = c("Accession","tokens", "counts")
  
  windowTokens[[i]] = out
  
  
}


### OUTPUT ###
save(windowTokens, file = "results/windowTokens.RData")

# reduce feature space
# average counts of neighbouring identical token sets
# plus:
# flatten and save list in chunks
load("data/windowTokens.RData")

out = "data/windowTokens.csv"
system(paste0("rm ", out))

pb = txtProgressBar(min = 0, max = length(windowTokens), style = 3)

for ( i in 1:length(windowTokens) ) {
  
  setTxtProgressBar(pb, i)
  
  cnt.Prot = windowTokens[[i]]
  unique.pairs = cnt.Prot$tokens[duplicated(cnt.Prot$tokens)] %>%
    as.character() %>%
    unique()
  
  cnt.Prot$counts = cnt.Prot$counts %>% as.character() %>% as.numeric()
  
  if (length(unique.pairs) > 0) {
    
    
    for (j in 1:length(unique.pairs)) {
      
      dup.tok = which(cnt.Prot$tokens == unique.pairs[j])
      
      # which of the duplicates are sequential?
      dup.diff.1 = c(0, diff(dup.tok))
      dup.diff.2 = c(diff(dup.tok), 0)
      
      # get mean count of all duplicates
      idx = dup.tok[dup.diff.1 == 1 | dup.diff.2 == 1]
      m = cnt.Prot$counts[idx] %>%
        as.numeric() %>%
        mean()
      
      # keep only one of the duplicates
      cnt.Prot$counts[idx[1]] = m
      cnt.Prot$counts[idx[2:length(idx)]] = NA
      
      
    }
    
  }
  
  windowTokens[[i]] = cnt.Prot
  
  # write to outfile
  wT = cnt.Prot %>% na.omit()
  
  
  if(file.exists(out)) {
    write.table(wT, out, sep = ",", row.names = F, append = T, col.names = F)
    
  } else {
    write.table(wT, out, sep = ",", row.names = F, append = F, col.names = T)
    
  }
}



