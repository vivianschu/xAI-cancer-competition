setwd("/home/bioinf/bhklab/julia/projects/xAI")

library(data.table)
library(edgeR)
library(ggplot2)

set.seed(123)

train <- fread("data/train.csv")
targets <- fread("data/train_targets.csv")


# format gene counts
cells <- train$V1
train$V1 <- NULL
train <- as.data.frame(t(train))

# binarize drug response
targets$bin <- ifelse(targets$AAC > median(targets$AAC), "Sen", "Res")

# set the lfc threshold
lfc_thres = 1.5

# differential gene expression analysis
group <- factor(targets$bin)
y <- DGEList(counts = train, group = group)
design <- model.matrix(~group)
y <- calcNormFactors(y)
y <- estimateDisp(y,design)
fit <- glmQLFit(y,design)
tr <- glmTreat(fit,coef=2, lfc=lfc_thres)
DEG <- as.data.frame(topTags(tr, n=nrow(tr$table)))


# subset training dataset
keep <- rownames(DEG[which(abs(DEG$logFC) > 2),])
train <- train[rownames(train) %in% keep,]

train <- as.data.frame(t(train))
write.csv(train, file = "results/train_deg.csv", quote = F, row.names = F)