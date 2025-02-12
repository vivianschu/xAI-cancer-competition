setwd("/home/bioinf/bhklab/julia/projects/xAI")

library(data.table)

set.seed(123)


#############################################################
# Load in data
#############################################################

load("results/corr_H4H.RData")
load("results/dr_pset.RData")

test <- fread("data/test.csv") |> as.data.frame()
rownames(test) <- test$V1
test$V1 <- NULL

# remove cell lines with NA values across gene expression
gdsc_rna <- gdsc_rna[,colSums(is.na(gdsc_rna))<nrow(gdsc_rna)]
gdsc_aac <- gdsc_aac[,colnames(gdsc_aac) %in% colnames(gdsc_rna)]

#############################################################
# Create TKI drug response datasets
#############################################################

# function to subset gdsc_aac by drug
aac_tki <- function(drug) {
    gdsc_aac <- gdsc_aac[rownames(gdsc_aac) == drug,]
    response <- data.frame(sample = colnames(gdsc_aac), AAC = as.numeric(gdsc_aac))
    response <- response[!is.na(response$AAC),]
    return(response)
}

#tkis <- c("Sapatinib", "Gefitinib", "Erlotinib", "Afatinib", "Osimertinib", "Selumetinib", "Lapatinib")

gefitinib <- aac_tki("Gefitinib")
sapatinib <- aac_tki("Sapatinib")
erlotinib <- aac_tki("Erlotinib")
osimertinib <- acc_tki("Osimertinib")


#############################################################
# Create TKI gene expression datasets
#############################################################

# function to subset gdsc_rna by samples with drug response
rna_tki <- function(drug) {
    rna <- gdsc_rna[,colnames(gdsc_rna) %in% drug$sample]
    rna <- t(rna) |> as.data.frame()
    return(rna)
}

s_rna <- rna_tki(sapatinib)
test <- test[,match(colnames(s_rna), colnames(test))]


#############################################################
# Save files
#############################################################

write.csv(s_rna, file = "results/train_sap.csv", quote = F, row.names = F)
write.csv(sapatinib, file = "results/targets_sap.csv", quote = F, row.names = F)
write.csv(test, file = "results/test_sap.csv", quote = F, row.names = F)


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