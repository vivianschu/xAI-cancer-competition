setwd("/home/bioinf/bhklab/julia/projects/xAI")

library(data.table)
library(umap)
library(ggplot2)

#set.seed(101) or set.seed(123) <- 3.75
set.seed(123)

train <- fread("data/train.csv")
test <- fread("data/test.csv")
targets <- fread("data/train_targets.csv")

### ===== UMAP ===== ###

train$label <- "Train"
test$label <- "Test"

merge <- rbind(train, test) |> as.data.frame()
labels <- merge$label
cells <- merge$V1
merge <- merge[,-which(colnames(merge) %in% c("V1", "label"))]

# create umap projection
umap_df <- umap(merge)$layout |> as.data.frame()
colnames(umap_df) <- c("UMAP1", "UMAP2")
umap_df$label <- labels
umap_df$cells <- cells
umap_df$tissue <- c(targets$tissue, rep("Test", 304))

# plot umap by threshold
png("results/umap.png", width = 5, height = 4, res = 600, units = "in")
ggplot(data = umap_df, aes(x = UMAP1, y = UMAP2, shape = label, fill = label)) + geom_point(size = 2.5) +
    scale_shape_manual(values = c(22, 21)) +
    geom_hline(yintercept = c(3.75, 5), linetype = "dotted") +
    theme_classic() + 
    theme(panel.border = element_rect(color = "black", fill = NA, size = 0.5), legend.key.size = unit(0.7, 'cm')) +
    labs(x = "UMAP1", y = "UMAP2", shape = "Label", fill = "Label")
dev.off()

# plot umap by tissue
png("results/umap_tissue.png", width = 8, height = 4, res = 600, units = "in")
ggplot(data = umap_df[umap_df$label == "Train",], aes(x = UMAP1, y = UMAP2, color = tissue)) + geom_point(size = 2.5) +
    geom_hline(yintercept = c(3.75, 5), linetype = "dotted") +
    theme_classic() + 
    theme(panel.border = element_rect(color = "black", fill = NA, size = 0.5), legend.key.size = unit(0.7, 'cm')) +
    labs(x = "UMAP1", y = "UMAP2", color = "Tissue")
dev.off()


# identify samples with UMAP > 3.75
tmp <- umap_df[umap_df$UMAP2 > 3.75,]
to_remove <- tmp[tmp$label == "Train",]$cells

# remove samples from training dataset
train_new <- train[-which(train$V1 %in% to_remove),]
train_new[, c("V1", "label") := NULL]
targets_new <- targets[-which(targets$sample %in% to_remove),]

write.csv(train_new, file = "results/train_375.csv", quote = F, row.names = F)
write.csv(targets_new, file = "results/targets_375.csv", quote = F, row.names = F)



### ===== UMAP after filtering ===== ###

train_new$label <- "Train"
test$label <- "Test"

test <- as.data.frame(test)

merge <- rbind(train_new, test[,-c(1)]) |> as.data.frame()
labels <- merge$label
merge <- merge[,-which(colnames(merge) == "label")]

# create umap projection
umap_df <- umap(merge)$layout |> as.data.frame()
colnames(umap_df) <- c("UMAP1", "UMAP2")
umap_df$label <- labels
umap_df$cells <- cells


# plot umap by threshold
png("results/umap_new.png", width = 5, height = 4, res = 600, units = "in")
ggplot(data = umap_df, aes(x = UMAP1, y = UMAP2, shape = label, fill = label)) + geom_point(size = 2.5) +
    scale_shape_manual(values = c(22, 21)) +
    theme_classic() + 
    theme(panel.border = element_rect(color = "black", fill = NA, size = 0.5), legend.key.size = unit(0.7, 'cm')) +
    labs(x = "UMAP1", y = "UMAP2", shape = "Label", fill = "Label")
dev.off()


# identify samples with UMAP > 5
tmp <- umap_df[umap_df$UMAP2 > 5,]
to_remove <- tmp[tmp$label == "Train",]$cells

# remove samples from training dataset
train_new <- train[-which(train$V1 %in% to_remove),]
train_new[, c("V1", "label") := NULL]
targets_new <- targets[-which(targets$sample %in% to_remove),]

write.csv(train_new, file = "results/train_5.csv", quote = F, row.names = F)
write.csv(targets_new, file = "results/targets_5.csv", quote = F, row.names = F)



### ===== UMAP after filtering ===== ###

train_new$label <- "Train"
test$label <- "Test"

test <- as.data.frame(test)

merge <- rbind(train_new, test[,-c(1)]) |> as.data.frame()
labels <- merge$label
merge <- merge[,-which(colnames(merge) == "label")]

# create umap projection
umap_df <- umap(merge)$layout |> as.data.frame()
colnames(umap_df) <- c("UMAP1", "UMAP2")
umap_df$label <- labels
umap_df$cells <- cells


# plot umap by threshold
png("results/umap_new5.png", width = 5, height = 4, res = 600, units = "in")
ggplot(data = umap_df, aes(x = UMAP1, y = UMAP2, shape = label, fill = label)) + geom_point(size = 2.5) +
    scale_shape_manual(values = c(22, 21)) +
    theme_classic() + 
    theme(panel.border = element_rect(color = "black", fill = NA, size = 0.5), legend.key.size = unit(0.7, 'cm')) +
    labs(x = "UMAP1", y = "UMAP2", shape = "Label", fill = "Label")
dev.off()

