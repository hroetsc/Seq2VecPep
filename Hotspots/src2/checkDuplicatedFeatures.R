### HEADER ###
# HOTSPOT PREDICTION
# description: check how counts differ for identical tokens in same protein
# input: windowTokens.RData
# output: distribution
# author: HR

library(plyr)
library(dplyr)
library(stringr)


### INPUT ###
load("data/windowTokens.RData")
wT = read.csv("data/windowTokens.csv", stringsAsFactors = F)

### MAIN PART ###

########## how many tokens are non-unique? ########## 

# get count differences and standard deviations

differences = list()
stand.dev = list()
mean.count = list()

pb = txtProgressBar(min = 0, max = length(windowTokens), style = 3)
for (i in 1:length(windowTokens)) {
  
  setTxtProgressBar(pb, i)
  
  cnt.Prot = windowTokens[[i]]
  unique.pairs = cnt.Prot$tokens[duplicated(cnt.Prot$tokens)] %>%
    as.character() %>%
    unique()
  
  if (length(unique.pairs) > 0) {
    
    D = rep(NA, length(unique.pairs))
    SD = rep(NA, length(unique.pairs))
    M = rep(NA, length(unique.pairs))
    
    for (j in 1:length(unique.pairs)) {
      
      cnt.Counts = cnt.Prot$counts[cnt.Prot$tokens == unique.pairs[j]] %>%
        as.character() %>%
        as.numeric()
      
      # in case there are more than 2 duplicates, sample a difference
      k = sample(length(cnt.Counts), 2)
      
      D[j] = abs(cnt.Counts[k[1]] - cnt.Counts[k[2]])
      SD[j] = sd(cnt.Counts)
      M[j] = mean(cnt.Counts)
      
    }
    
    differences[[i]] = D
    stand.dev[[i]] = SD
    mean.count[[i]] = M
    
  } else {
    
    differences[[i]] = NA
    stand.dev[[i]] = NA
    mean.count[[i]] = NA
    
  }
  
  
}


# flatten lists
differences.flat = unlist(differences) %>%
  na.omit()

stand.dev.flat = unlist(stand.dev) %>%
  na.omit()

mean.flat = unlist(mean.count) %>%
  na.omit()

# plot
{
png(filename = "countDiff_identTokens.png", width = 2000, height = 2000, res = 300)
plot(density(log2(differences.flat)),
     main = "count differences between identical token sets",
     sub = paste0("mean: ", mean(log2(differences.flat)) %>% round(3),
                  " (", mean(differences.flat) %>% round(3), " counts)"),
     xlab = "log2 absolute count difference",
     ylab = "density")
dev.off()


png(filename = "countSD_identTokens.png", width = 2000, height = 2000, res = 300)
plot(density(log2(stand.dev.flat)),
     main = "standard deviation of counts of identical token sets",
     sub = paste0("mean: ", mean(log2(stand.dev.flat)) %>% round(3),
                  " (", mean(stand.dev.flat) %>% round(3), " units)"),
     xlab = "log2 standard deviation",
     ylab = "density")
dev.off()

png(filename = "countCV_identTokens.png", width = 2000, height = 2000, res = 300)
plot(density(stand.dev.flat / mean.flat),
     main = "CV of counts of identical token sets",
     sub = paste0("mean: ", mean(stand.dev.flat / mean.flat) %>% round(3)),
     xlab = "coefficient of variance",
     ylab = "density")
dev.off()

png(filename = "countMeanVsSD_identTokens.png", width = 2000, height = 2000, res = 300)
plot(stand.dev.flat ~ mean.flat,
     main = "mean vs. standard deviation",
     xlab = "mean counts of duplicates",
     ylab = "SD of duplicates",
     cex = 0.2)
dev.off()


}

########## only take neighbour tokens into account ##########
differences = list()
stand.dev = list()
mean.count = list()

pb = txtProgressBar(min = 0, max = length(windowTokens), style = 3)
for (i in 1:length(windowTokens)) {
  
  setTxtProgressBar(pb, i)
  
  cnt.Prot = windowTokens[[i]]
  unique.pairs = cnt.Prot$tokens[duplicated(cnt.Prot$tokens)] %>%
    as.character() %>%
    unique()
  
  if (length(unique.pairs) > 0) {
    
    D = rep(NA, length(unique.pairs))
    SD = rep(NA, length(unique.pairs))
    M = rep(NA, length(unique.pairs))
    
    for (j in 1:length(unique.pairs)) {
      
      loc.dup = which(cnt.Prot$tokens == unique.pairs[j])
      diff.loc.dup = diff(loc.dup)
      
      if (all(diff.loc.dup == 1)) {
        
        cnt.Counts = cnt.Prot$counts[loc.dup] %>%
          as.character() %>%
          as.numeric()
        
        # in case there are more than 2 duplicates, sample a difference
        k = sample(length(cnt.Counts), 2)
        
        D[j] = abs(cnt.Counts[k[1]] - cnt.Counts[k[2]])
        SD[j] = sd(cnt.Counts)
        M[j] = mean(cnt.Counts)
        
      } else {
        
        D[j] = NA
        SD[j] = NA
        M[j] = NA
        
      }
      
      
    }
    
    differences[[i]] = D
    stand.dev[[i]] = SD
    mean.count[[i]] = M
    
  } else {
    
    differences[[i]] = NA
    stand.dev[[i]] = NA
    mean.count[[i]] = NA
    
  }
  
  
}


# flatten lists
differences.flat = unlist(differences) %>%
  na.omit()

stand.dev.flat = unlist(stand.dev) %>%
  na.omit()

mean.flat = unlist(mean.count) %>%
  na.omit()


# plot
{
  png(filename = "countDiff_identTokens_neighbour.png", width = 2000, height = 2000, res = 300)
  plot(density(log2(differences.flat)),
       main = "count differences between identical token sets \n (neighbouring tokens in same protein)",
       sub = paste0("mean: ", mean(log2(differences.flat)) %>% round(3),
                    " (", mean(differences.flat) %>% round(3), " counts)"),
       xlab = "log2 absolute count difference",
       ylab = "density")
  dev.off()
  
  
  png(filename = "countSD_identTokens_neighbour.png", width = 2000, height = 2000, res = 300)
  plot(density(log2(stand.dev.flat)),
       main = "standard deviation of counts of identical token sets \n (neighbouring tokens in same protein)",
       sub = paste0("mean: ", mean(log2(stand.dev.flat)) %>% round(3),
                    " (", mean(stand.dev.flat) %>% round(3), " units)"),
       xlab = "log2 standard deviation",
       ylab = "density")
  dev.off()
  
  png(filename = "countCV_identTokens_neighbour.png", width = 2000, height = 2000, res = 300)
  plot(density(stand.dev.flat / mean.flat),
       main = "CV of counts of identical token sets \n (neighbouring tokens in same protein)",
       sub = paste0("mean: ", mean(stand.dev.flat / mean.flat) %>% round(3)),
       xlab = "coefficient of variance",
       ylab = "density")
  dev.off()
  
  png(filename = "countMeanVsSD_identTokens_neighbour.png", width = 2000, height = 2000, res = 300)
  plot(stand.dev.flat ~ mean.flat,
       main = "mean vs. standard deviation \n (neighbouring tokens in same protein)",
       xlab = "mean counts of duplicates",
       ylab = "SD of duplicates",
       cex = 0.2)
  dev.off()
  
  
}



########## in all proteins ##########

non.unique.tokens = wT$tokens[duplicated(wT$tokens)] %>%
  as.character() %>%
  unique()

non.unique.tokens = non.unique.tokens[sample(length(non.unique.tokens), 1e04)]

pb = txtProgressBar(min = 0, max = length(non.unique.tokens), style = 3)

D.all = rep(NA, length(non.unique.tokens))
SD.all = rep(NA, length(non.unique.tokens))
M.all = rep(NA, length(non.unique.tokens))


for (k in 1:length(non.unique.tokens)) {
  
  setTxtProgressBar(pb, k)
  
  cnt.Counts = wT$counts[wT$tokens == non.unique.tokens[k]] %>%
    as.character() %>%
    as.numeric()
  
  # in case there are more than 2 duplicates, sample a difference
  s = sample(length(cnt.Counts), 2)
  
  D.all[k] = abs(cnt.Counts[s[1]] - cnt.Counts[s[2]])
  SD.all[k] = sd(cnt.Counts)
  M.all[k] = mean(cnt.Counts)
  
}

D.all = na.omit(D.all)
SD.all = na.omit(SD.all)
M.all = na.omit(M.all)
CV.all = (SD.all / (M.all+1)) %>% na.omit()

# plot
{
  png(filename = "countDiff_identTokens_allProt.png", width = 2000, height = 2000, res = 300)
  plot(density(D.all),
       main = "count differences between identical token sets (all proteins)",
       sub = paste0("mean: ", mean(D.all) %>% round(3),
                    " (", mean(2^D.all) %>% round(3), " counts)"),
       xlab = "log2 absolute count difference",
       ylab = "density")
  dev.off()
  
  
  png(filename = "countSD_identTokens_allProt.png", width = 2000, height = 2000, res = 300)
  plot(density(SD.all),
       main = "standard deviation of counts of identical token sets (all proteins)",
       sub = paste0("mean: ", mean(SD.all) %>% round(3),
                    " (", mean(2^SD.all) %>% round(3), " units)"),
       xlab = "log2 standard deviation",
       ylab = "density")
  dev.off()
  
  
  png(filename = "countCV_identTokens_allProt.png", width = 2000, height = 2000, res = 300)
  plot(density(CV.all),
       main = "CV of counts of identical token sets (all proteins)",
       sub = paste0("mean: ", mean(CV.all) %>% round(3), ", CV = sd / (mean +1)"),
       xlab = "coefficient of variance",
       ylab = "density")
  dev.off()
  
  
  png(filename = "countMeanVsSD_identTokens_allProt.png", width = 2000, height = 2000, res = 300)
  plot(SD.all ~ M.all,
       main = "mean vs. standard deviation (all proteins)",
       xlab = "mean counts of duplicates",
       ylab = "SD of duplicates",
       cex = 0.2)
  dev.off()
}


