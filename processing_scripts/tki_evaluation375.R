setwd("/home/bioinf/bhklab/julia/projects/xAI")

library(data.table)

set.seed(123)


#############################################################
# Load in data
#############################################################

load("results/corr_H4H.RData")
load("results/dr_pset.RData")

train <- fread("results/train_375.csv") |> as.data.frame()
test <- fread("data/test.csv") |> as.data.frame()
targets <- fread("results/targets_375.csv") |> as.data.frame()


#############################################################
# Subset training and test data
#############################################################

train <- train[,match(rownames(gdsc_rna), colnames(train))]
test <- test[,match(rownames(gdsc_rna), colnames(test))]


#############################################################
# Create TKI drug response predictor from GDSC
#############################################################

#tkis <- c("Sapatinib", "Gefitinib", "Erlotinib", "Afatinib", "Osimertinib", "Selumetinib", "Lapatinib")
tkis <- c("Sapatinib", "Gefitinib", "Erlotinib", "Osimertinib")

gdsc_aac <- gdsc_aac[rownames(gdsc_aac) %in% tkis,]
tki_response <- data.frame(sample = colnames(gdsc_aac), AAC = colMeans(gdsc_aac), tissue = NA)

# subset for only cells with both gdsc_rna and tki response
common <- intersect(tki_response$sample, colnames(gdsc_rna))
tki_response <- tki_response[tki_response$sample %in% common,]


# remove cell lines with NA values across gene expression
gdsc_rna <- gdsc_rna[,match(tki_response$sample, colnames(gdsc_rna))]
gdsc_rna <- gdsc_rna[,colSums(is.na(gdsc_rna))<nrow(gdsc_rna)]

# subset tki response
tki_response <- tki_response[tki_response$sample %in% colnames(gdsc_rna),]


# remove cells with NA value in TKI response
tki_response <- tki_response[!is.na(tki_response$AAC),]
gdsc_rna <- gdsc_rna[,match(tki_response$sample, colnames(gdsc_rna))]

#############################################################
# Append GDSC and TKI response info to train datasets
#############################################################

new_train <- rbind(train, as.data.frame(t(gdsc_rna)))
new_targets <- rbind(targets, tki_response)

write.csv(new_train, file = "results/train375_tki.csv", quote = F, row.names = F)
write.csv(new_targets, file = "results/targets375_tki.csv", quote = F, row.names = F)


#############################################################
# Evaluate predictions
#############################################################

y_pred <- read.csv("results/en_tki.csv")
y_true <- read.csv("results/RESULTS_gefitinib.csv")

# remove cells with NA value in TKI response
y_true$sampleId = y_pred$sampleId
y_true <- y_true[!is.na(y_true$AAC),]
y_pred <- y_pred[match(y_true$sampleId, y_pred$sampleId),]

cor(y_pred$AAC, y_true$AAC, method = "spearman")