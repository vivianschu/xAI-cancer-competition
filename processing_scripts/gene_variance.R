setwd("/home/bioinf/bhklab/julia/projects/xAI")

library(data.table)
library(ggplot2)

set.seed(123)

# load in data
train <- fread("data/train.csv") |> as.data.frame()
train <- train[,-c(1)]

# compute coefficient of variance
mean <- apply(train, 2, mean)
std <- apply(train, 2, sd)

coef <- (std / mean) * 100

# create dataframe for plotting
toPlot <- data.frame(gene = colnames(train), coef = coef)

# plot
png("results/var.png", width = 5, height = 4, res = 600, units = "in")
ggplot(data = toPlot, aes(x = coef)) + 
    geom_density(fill = 'gray', alpha = 0.5) +
    geom_vline(xintercept = 1000, linetype = 'dotted') +
    theme_classic() + 
    theme(panel.border = element_rect(color = "black", fill = NA, size = 0.5), legend.key.size = unit(0.7, 'cm')) +
    labs(x = "Coefficient of Variance", y = "Density")
dev.off()

# identify high variance genes
hvg <- toPlot[toPlot$coef > 1000,]$gene

# subset training data
train_new <- train[,which(colnames(train) %in% hvg)]

# save file
write.csv(train_new, file = "results/gene_var/train_gv.csv", quote = F, row.names = F)