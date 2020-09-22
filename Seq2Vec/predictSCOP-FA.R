### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  use seq2vec embeddings to predict SCOP protein family
# input:        protein matrices, SCOP data
# output:       classification results
# author:       HR

library(dplyr)
library(stringr)
library(caret)
library(e1071)
library(uwot)
library(future)


### INPUT ###
embeddingDim = 128

# proteome
prots = read.csv("../files/proteome_human.csv", stringsAsFactors = F)

# Load SCOP classification
{
scop <- readr::read_delim("../files/scop_cla_latest.txt",skip = 5, delim = " ")
header = colnames(scop)
header = header[-1]

scop$SCOPCLA = NULL
colnames(scop) = header

scop_class <- scop$SCOPCLA %>%
  str_remove_all(pattern = "TP=") %>%
  str_remove_all(pattern = "CL=") %>%
  str_remove_all(pattern = "CF=") %>%
  str_remove_all(pattern = "SF=") %>%
  str_remove_all(pattern = "FA=") %>% 
  str_split_fixed(pattern = ",", Inf)
scop_class <- as_tibble(scop_class) %>% as.data.frame()
scop_class = cbind(scop$`FA-UNIID`, scop_class)
colnames(scop_class) <- c("Accession","TP", "CL", "CF", "SF", "FA")
scop_class$Accession = as.character(scop_class$Accession)
# TP=protein type, CL=protein class, CF=fold, SF=superfamily, FA=family
scop_des <- readr::read_delim("../files/scop_des_latest.txt", delim = " ")
}

# embedings
emb.fs = list.files(path = "../RUNS/HumanProteome/v_50k", pattern = "sequence_repres",
                    full.names = T)


### MAIN PART ###
########## preprocessing ##########
# extract proteins listed in SCOP
prots$Accession = str_split_fixed(prots$Accession, coll("-"), Inf)[,1]
scop_class = scop_class[, c("Accession", "CL")] %>%
  unique()

master = full_join(prots, scop_class) %>% 
  na.omit() %>%
  unique()


## examine classes
perClass = table(master$CL)
barplot(perClass)
sum(perClass) == nrow(master)  # --> every protein has 1 class

classes = data.frame(CL = as.character(c("1000000", "1000001", "1000002", "1000003", "1000004")),
                     name = as.character(c("all_alpha", "all_beta", "alpha_and_beta", "alpha_and_beta", "small_proteins")),
                     label = as.character(c(0, 1, 2, 3, 4)))
classes
master = full_join(master, classes) %>%
  unique()

# --> data set contains isoforms
master = master[-which(duplicated(master$seqs) & duplicated(master$CL)), ]
acc = master$Accession %>% unique()

# split into training and valiation data
idx = sample(length(acc), floor(length(acc)*0.7))
master.train = master[master$Accession %in% acc[idx], ]
master.test = master[!master$Accession %in% acc[idx], ]

# unique identifiers
master$Accession = NULL
master.train$Accession = NULL
master.test$Accession = NULL

# make sure that both data sets contain all classes
length(levels(as.factor(master.train$CL)))
length(levels(as.factor(master.test$CL)))


# load embeddings, normalise and decorrelate them
normalise_features = function(mat = ""){
  
  mat = as.matrix(mat)
  mat = t(apply(mat, 1, function(x) (x - mean(x))/sd(x) )) %>%
    as.data.frame()
  
  crl = cor(mat)
  hc = findCorrelation(crl, cutoff = 0.8) %>% sort()
  
  if (length(hc) > 0) {
    mat = mat[, -c(hc)]
  }
  
  return(mat)
}


load_features = function(path = "", normalise = T){
  
  emb = read.csv(path, stringsAsFactors = F)
  emb = inner_join(master, emb) %>% na.omit()
  
  emb.train = emb[emb$seqs %in% master.train$seqs, ]
  emb.test = emb[emb$seqs %in% master.test$seqs, ]
  
  if (normalise == T){
    mat.train = normalise_features(emb.train[, c((ncol(emb.train)-embeddingDim+1) : ncol(emb.train))])
    mat.test = normalise_features(emb.test[, c((ncol(emb.test)-embeddingDim+1) : ncol(emb.test))])
    
  } else {
    mat.train = emb.train[, c((ncol(emb.train)-embeddingDim+1) : ncol(emb.train))]
    mat.test = emb.test[, c((ncol(emb.test)-embeddingDim+1) : ncol(emb.test))]
    
  }
  
  out = list()
  out[[1]] = cbind(emb.train$label,
                   mat.train)
  names(out[[1]])[1] = "label"
  out[[2]] = cbind(emb.test$label,
                   mat.test)
  names(out[[2]])[1] = "label"
  names(out) = c("train", "test")
  
  return(out)
}

# pure seq2vec
features = load_features(path = emb.fs[6])
train_dt = features[["train"]]
test_dt = features[["test"]]



########## SVM ##########

svm1 = svm(as.factor(label)~., data = train_dt,
           type = "C-classification",
           kernel = "radial",
           cost = 10)
summary(svm1)
pred1 = predict(svm1, test_dt)
tbl1 = table(test$label, pred1)
print(paste0("accuracy: ", diag(tbl1) %>% sum / nrow(test)))



########## caret + other built-in classifiers ##########
control = trainControl(method = "cv", number = 10,
                    savePredictions = T)

# linear discriminant analysis
fit.lda = train(label~., data = train_dt, method = "lda", metric = "Accuracy", trControl = control)
tbl.lda = table(fit.lda$pred$obs, fit.lda$pred$pred)
print(paste0("accuracy: ", diag(tbl.lda) %>% sum / nrow(fit.lda$pred)))

# k-nearest neighbours
fit.knn = train(label~., data = train_dt, method = "knn", metric = "Accuracy", trControl = control)
tbl.knn = table(fit.knn$pred$obs, fit.knn$pred$pred)
print(paste0("accuracy: ", diag(tbl.knn) %>% sum / nrow(fit.knn$pred)))

# random forest
fit.rf = train(label~., data = train_dt, method = "rf", metric = "Accuracy", trControl = control)
tbl.rf = table(fit.rf$pred$obs, fit.rf$pred$pred)
print(paste0("accuracy: ", diag(tbl.rf) %>% sum / nrow(fit.rf$pred)))

# Monotone Multi-Layer Perceptron Neural Network
fit.monmlp = train(label~., data = train_dt, method = "monmlp", metric = "Accuracy", trControl = control)
tbl.monmlp = table(fit.monmlp$pred$obs, fit.monmlp$pred$pred)
print(paste0("accuracy: ", diag(tbl.monmlp) %>% sum / nrow(fit.monmlp$pred)))

# summary
results = resamples(list(lda=fit.lda, knn=fit.knn, rf=fit.rf, momlp=fit.monmlp))
summary(results)

write.csv(results$values, "../HORIZONS/SCOP_classifiers.csv", row.names = F)

########## UMAP and random forest ##########

UMAP = function(tbl) {
  
  coord = umap(tbl[, c(2:ncol(tbl))],
               n_neighbors = 5,
               min_dist = 0.8,
               n_epochs = 300,
               n_trees = 15,
               metric = "cosine",
               verbose = T,
               approx_pow = T,
               ret_model = T,
               init = "spca",
               n_threads = availableCores(),
               n_components = 10)
  
  emb = normalise_features(as.matrix(coord$embedding))
  um = cbind(tbl$label, emb) %>%
    as.data.frame()
  
  names(um)[1] = "label"
  
  return(um)
  
}


train_dt_UMAP = UMAP(tbl = train_dt)

# random forest
fit.rf_UMAP = train(as.factor(label)~., data = train_dt_UMAP, method = "rf",
                    metric = "Accuracy", trControl = control)
tbl.rf_UMAP = table(fit.rf_UMAP$pred$obs, fit.rf_UMAP$pred$pred)
print(paste0("accuracy: ", diag(tbl.rf_UMAP) %>% sum / nrow(fit.rf_UMAP$pred)))
## --> not better


########## random forest for all embeddings ##########
control = trainControl(method = "cv", number = 10,
                       savePredictions = T)

for (e in 1:length(emb.fs)) {
  
  {
  nm = str_split(emb.fs[e], coll("/"), simplify = T)
  nm = str_split(nm[, ncol(nm)], coll("_"), simplify = T)
  nm = paste0(nm[, ncol(nm)-1], nm[, ncol(nm)])
  nm = str_split(nm, coll("."), simplify = T)[, 1]
  }
  
  print(nm)
  features = load_features(path = emb.fs[e])
  
  train_dt = features[["train"]]
  test_dt = features[["test"]]
  
  # random forest
  fit.rf = train(label~., data = train_dt, method = "rf", metric = "Accuracy", trControl = control)
  tbl.rf = table(fit.rf$pred$obs, fit.rf$pred$pred)
  print(paste0("accuracy: ", diag(tbl.rf) %>% sum / nrow(fit.rf$pred)))
  
  fpath = paste0("../HORIZONS/random-forest/", nm, ".RData")
  save(fit.rf, file = fpath)
}

########## random forest without normalization ##########
control = trainControl(method = "cv", number = 10,
                       savePredictions = T)

# pure seq2vec
features = load_features(path = emb.fs[6], normalise = F)
train_dt = features[["train"]]
test_dt = features[["test"]]

# random forest
fit.rf = train(label~., data = train_dt, method = "rf", metric = "Accuracy", trControl = control)
tbl.rf = table(fit.rf$pred$obs, fit.rf$pred$pred)
print(paste0("accuracy: ", diag(tbl.rf) %>% sum / nrow(fit.rf$pred)))


