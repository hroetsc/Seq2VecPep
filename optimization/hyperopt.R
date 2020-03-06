setwd("optimization/results/")

library(stringr)

sizes = c(5000, 10000, 50000, 100000)

gp.5000 = read.csv("gp_5000.csv", stringsAsFactors = F, header = F)
gp.10000 = read.csv("gp_10000.csv", stringsAsFactors = F, header = F)
gp.50000 = read.csv("gp_50000.csv", stringsAsFactors = F, header = F)
gp.100000 = read.csv("gp_100000.csv", stringsAsFactors = F, header = F)

gbrt.5000 = read.csv("gbrt_5000.csv", stringsAsFactors = F, header = F)
gbrt.10000 = read.csv("gbrt_10000.csv", stringsAsFactors = F, header = F)
gbrt.50000 = read.csv("gbrt_50000.csv", stringsAsFactors = F, header = F)
gbrt.100000 = read.csv("gbrt_100000.csv", stringsAsFactors = F, header = F)

format_table = function(df = "") {
  df$V1 = NULL
  colnames(df) = c("value", "hyperparams")
  hyperparams = str_split(df$hyperparams, coll("["), simplify = T)
  hyperparams = str_split(hyperparams[,2], coll("]"), simplify = T)
  df = cbind(df, str_split(hyperparams[,1], coll(","), simplify = T))
  df$hyperparams = NULL
  colnames(df) = c("fitness_value", "learning_rate", "embedding_size", "activation_function", "batch_size", "epochs", "adam_decay")
  return(df)
}

# fitness values are negated classification accuracies

gp.5000 = format_table(gp.5000)
gp.10000 = format_table(gp.10000)
gp.50000 = format_table(gp.50000)
gp.100000 = format_table(gp.100000)

gbrt.5000 = format_table(gbrt.5000)
gbrt.10000 = format_table(gbrt.10000)
gbrt.50000 = format_table(gbrt.50000)
gbrt.100000 = format_table(gbrt.100000)


v = c()
for (i in 1:length(sizes)) {
  v = c(v, rep(sizes[i], 3))
}

gp.5000[c(1:3),]
gp.10000[c(1:3),]
gp.50000[c(1:3),]
gp.100000[c(1:3),]

gbrt.5000[c(1:3),]
gbrt.10000[c(1:3),]
gbrt.50000[c(1:3),]
gbrt.100000[c(1:3),]

top = data.frame(optimization = c(rep("gp", 12), rep("gbrt", 12)),
                 vocab_size = rep(v, 2))
top = cbind(top,
            rbind(gp.5000[c(1:3),],
            gp.10000[c(1:3),],
            gp.50000[c(1:3),],
            gp.100000[c(1:3),],
            
            gbrt.5000[c(1:3),],
            gbrt.10000[c(1:3),],
            gbrt.50000[c(1:3),],
            gbrt.100000[c(1:3),]))
