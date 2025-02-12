# did not finish running

setwd("/home/bioinf/bhklab/julia/projects/xAI")

library(data.table)
library(ggplot2)

set.seed(123)

train <- fread("data/train.csv")
test <- fread("data/test.csv")
targets <- fread("data/train_targets.csv")


# create matrix to store results
res <- data.frame(matrix(nrow = nrow(train) * nrow(test), ncol = 3))
colnames(res) <- c("Train", "Test", "Corr")
res$Train <- rep(train$V1, nrow(test))
res$Test <- rep(test$V1, each = nrow(train))

# remove cell line labels
rownames(train) <- train$V1
rownames(test) <- test$V1
train$V1 <- test$V1 <- NULL

# correlate each test sample with each train sample
train_matrix <- as.matrix(train)
test_matrix <- as.matrix(test)

train_std <- scale(train_matrix, center = TRUE, scale = TRUE)
test_std <- scale(test_matrix, center = TRUE, scale = TRUE)
cor_matrix <- train_std %*% t(test_std) / (ncol(train_matrix) - 1)

# Convert to long format if needed
cor_df <- as.data.table(as.table(cor_matrix))
colnames(cor_df) <- c("Train_Row", "Test_Row", "Correlation")


## remove later
good_samples <- res[res$Corr > 0.85,]$Train |> unique()
res <- res[-which(res$Train) %in% good_samples,]



write.csv(train_new, file = "results/train_corr.csv", quote = F, row.names = F)
write.csv(targets_new, file = "results/targets_corr.csv", quote = F, row.names = F)

