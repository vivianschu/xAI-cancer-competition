setwd("/home/bioinf/bhklab/julia/projects/xAI")

library(data.table)
library(sva)

set.seed(123)


#############################################################
# Load in data
#############################################################

train <- fread("data/train.csv") |> as.data.frame()
test <- fread("data/test.csv") |> as.data.frame()
targets <- fread("data/train_targets.csv") |> as.data.frame()

rownames(train) <- train$V1
rownames(test) <- test$V1

train$V1 <- test$V1 <- NULL


#############################################################
# Pepare inputs for batch correctlion
#############################################################

combined_data <- rbind(train, test) |> t() 

label <- c(rep("Train", nrow(train)), rep("Test", nrow(test)))


#############################################################
# Batch correction using ComBat
#############################################################

adj_data <- ComBat_seq(combined_data, batch=label, group = NULL)

#############################################################
# Extract new train and test datasets
#############################################################

new_train <- adj_data[,1:nrow(train)] |> t() |> as.data.frame()
new_test <- adj_data[,nrow(train)+1:nrow(test)] |> t() |> as.data.frame()

rownames(new_test) <- paste0("CL", 1:nrow(test))

write.csv(new_train, file = "results/train_combat.csv", quote = F, row.names = F)
write.csv(new_test, file = "results/test_combat.csv", quote = F, row.names = F)


#############################################################
# Evaluate predictions
#############################################################

y_pred <- read.csv("results/en_combat.csv")
y_true <- read.csv("results/RESULTS_gefitinib.csv")
y_true <- read.csv("results/RESULTS_sapatinib.csv")

# remove cells with NA value in TKI response
y_true$sampleId = y_pred$sampleId
y_true <- y_true[!is.na(y_true$AAC),]
y_pred <- y_pred[match(y_true$sampleId, y_pred$sampleId),]

cor(y_pred$AAC, y_true$AAC, method = "spearman")