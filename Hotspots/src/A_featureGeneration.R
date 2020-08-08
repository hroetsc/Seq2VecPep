### HEADER ###
# HOTSPOT PREDICTION
# description: generate feature set by applying a sliding window over a protein sequence
# input: windowCounts20 (from Juliane), human proteome (encoded)
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
load("HOTSPOTS/accU.RData")
load("HOTSPOTS/RESULTS/windowCounts20.RData")

# human proteome
prots = read.csv("../files/proteome_human.csv", stringsAsFactors = F, header = T)
words = read.csv("../RUNS/HumanProteome/words_hp.csv", stringsAsFactors = F, header = T)

### MAIN PART ###

tokPerWindow = 8
windowSize = 20

# keep only proteins with hotspots
prots = prots[which(prots$Accession %in% accU), ]
prots = left_join(prots, words)

names(windowCounts20) = accU

# normalize counts by maximum count in each protein to get scores between 0 and 1
# without underestimating densities in sequences with many hotspots
countsNormMax20 = list()
for (w in 1:length(windowCounts20)){
  countsNormMax20[[w]] = windowCounts20[[w]] / max(windowCounts20[[w]]) %>% round(4)
}
names(countsNormMax20) = accU


# loop over proteins
windowTokens = list()

windowTokens = foreach (i = 1:length(countsNormMax20)) %dopar% {
  
  cnt.Prot = prots[prots$Accession == names(countsNormMax20)[i], ]
  
  cnt.Tokens = str_split(cnt.Prot$tokens, coll(" "), simplify = T) %>%
    t() %>%
    as.data.frame()
  names(cnt.Tokens) = "tok"
  cnt.Tokens$tok = as.character(cnt.Tokens$tok)
  
  # get all token positions in the substrate
  end = 0
  for (j in 1:nrow(cnt.Tokens)){
    cnt.Tokens$start[j] = end + 1
    end = end + nchar(cnt.Tokens$tok[j])
    cnt.Tokens$end[j] = end
    
  }
  
  # apply sliding window and check which tokens are in current window
  wnd.Tokens = rep(NA, nchar(cnt.Prot$seqs))
  
  pos1 = 1
  pos2 = pos1 + windowSize
  
  while (pos1 <= nchar(cnt.Prot$seqs)) {
    
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
              countsNormMax20[[i]]) %>% 
    as.data.frame() %>%
    unique()
  
  names(out) = c("Accession","tokens", "counts")
  
  windowTokens[[i]] = out
  
  
}

# flatten and save list in chunks

out = "results/windowTokens.csv"
system(paste0("rm ", out))

counter = 1
increment = 50

while(counter <= (length(windowTokens)-increment+1)) {
  
  wT = windowTokens[c(counter:(counter+increment-1))] %>%
    plyr::ldply()
  
  if(file.exists(out)) {
    write.table(wT, out, sep = ",", row.names = F, append = T, col.names = F)
    
  } else {
    write.table(wT, out, sep = ",", row.names = F, append = F, col.names = T)
    
  }
  
  counter = counter + increment
}


### OUTPUT ###
save(windowTokens, file = "results/windowTokens.RData")





