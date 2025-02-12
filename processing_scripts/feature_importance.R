setwd("/home/bioinf/bhklab/julia/projects/xAI")

# load in libraries
suppressPackageStartupMessages({
    library(ggplot2)
    library(ggrepel)
    library(edgeR)
})


#############################################################
# Load in data
#############################################################

pred <- read.csv("results/pred/optimized_en-ensemble-pred.csv")
test <- read.csv("data/test.csv")
rownames(test) <- paste0("TS", 1:nrow(test))
test$X <- NULL

#############################################################
# Subset groups of 5 samples
#############################################################

# order pred
pred <- pred[order(pred$AAC, decreasing = T),]

# subset for top and bottom 3 cells
top3 <- pred$sampleId[1:3]
bot3 <- pred$sampleId[302:304]

# subset expression data
top3 <- test[match(top3, rownames(test)),]
bot3 <- test[match(bot3, rownames(test)),]

#############################################################
# Bar plots expression of each genes of interest
#############################################################

# genes of interest
genes <- c("EGFR", "CEACAM6", "VIM", "KRT5", "KRT17", "SPARC")


# plot each gene expression
for (gene in genes) {
    toPlot <- data.frame(Sample = c(rownames(top3), 
                                    rownames(bot3)),
                         Expr = c(top3[,colnames(top3) == gene],
                                  bot3[,colnames(bot3) == gene]))
    toPlot$Group <- rep(c("Top3", "Bottom3"), each = 3)
    toPlot$Sample <- factor(toPlot$Sample, levels = toPlot$Sample)
    toPlot$Group <- factor(toPlot$Group, levels = unique(toPlot$Group))

    filename = paste0("results/explainability/indiv_GE/", gene, ".png")
    png(filename, width = 3, height = 3, res = 600, units = "in")
    print({ggplot(toPlot, aes(x = Sample, y = log2(Expr), fill = Group)) +
        geom_bar(stat = "identity", position = position_dodge(width = 0.8), size = 0.5, color = "black") +
        theme_classic() + 
        scale_fill_manual(values = c("#679289", "#F4C095")) +
        theme(panel.border = element_rect(color = "black", fill = NA, size = 0.5), 
              legend.key.size = unit(0.7, 'cm'),
              axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        labs(y = "log2-Normalized Expression", title = gene)})
    dev.off()
}

#############################################################
# Faceted bar plots expression of each genes of interest
#############################################################

# genes of interest
genes <- c("EGFR", "CEACAM6", "VIM", "KRT5", "KRT17", "SPARC")
    
# create dataframe for plotting
toPlot <- rbind(top3[,colnames(top3) %in% genes],
                bot3[,colnames(bot3) %in% genes])
toPlot$Group <- rep(c("Top3", "Bottom3"), each = 3)
toPlot$Sample <- factor(rownames(toPlot), levels = rownames(toPlot))
toPlot$Group <- factor(toPlot$Group, levels = unique(toPlot$Group))

toPlot <- reshape2::melt(toPlot)
toPlot$variable <- factor(toPlot$variable, levels = genes)

# plot
png("results/explainability/indiv_GE/facets.png", width = 6, height = 4, res = 600, units = "in")
ggplot(toPlot, aes(x = Sample, y = log2(value), fill = Sample)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), size = 0.5, color = "black") +
    theme_classic() + 
    facet_wrap(.~variable) +
    scale_fill_manual(values = c("#628D56", "#5AAA46", "#94C47D", "#7BA8D0", "#317EC2", "#3A68AE")) +
    theme(panel.border = element_rect(color = "black", fill = NA, size = 0.5), 
            legend.key.size = unit(0.7, 'cm'),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(y = expression(log[2]*Normalized_Expression))
    #labs(y = "log2-Normalized Expression")
dev.off()

#############################################################
# Heatmap expression of genes of interest
#############################################################

# create dataframe for plotting
toPlot <- rbind(top3[,colnames(top3) %in% genes],
                bot3[,colnames(bot3) %in% genes])
toPlot$Group <- rep(c("Top3", "Bottom3"), each = 3)
toPlot$Sample <- factor(rownames(toPlot), levels = rownames(toPlot))
toPlot$Group <- factor(toPlot$Group, levels = unique(toPlot$Group))
toPlot <- reshape2::melt(toPlot)

# heatmap
png("results/explainability/indiv_GE_heatmap.png", width = 4, height = 3.5, res = 600, units = "in")
ggplot(toPlot, aes(x = Sample, y = variable, fill = log2(value+1))) +
    geom_tile() +
    theme_classic() +
    scale_fill_gradient2(high = "#316BC2", mid = "white", low = "#D15050", 
                       midpoint = median(log2(toPlot$value+1))) +
    geom_vline(xintercept = 3.5) +
    theme(panel.border = element_rect(color = "black", fill = NA, size = 0.5), 
            legend.key.size = unit(0.7, 'cm'),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(y = "Gene", fill = "log2-\nNormalized\nExpression")
dev.off()


#############################################################
# Differential gene expression
#############################################################

n = 10

# subset for top and bottom 10 cells
top10 <- pred$sampleId[1:n]
bot10 <- pred$sampleId[(304-n+1):304]

# subset expression data
top10 <- test[match(top10, rownames(test)),]
bot10 <- test[match(bot10, rownames(test)),]

# create dataframe for DEG analysis
df <- rbind(top10, bot10) |> t()

# create metadata 
meta <- data.frame(Group = rep(c("Top", "Bottom"), each = n))
meta$Group <- factor(meta$Group, levels = c("Bottom", "Top"))
meta$Sample <- factor(rownames(df), levels = rownames(df))

# differential gene expression analysis
y = DGEList(counts = df, group = meta$Group)
design <- model.matrix(~meta$Group)
y <- calcNormFactors(y)
y <- estimateDisp(y,design)
fit <- glmQLFit(y,design)
tr <- glmTreat(fit,coef=2, lfc=1)
DEG <- as.data.frame(topTags(tr, n=nrow(tr$table)))

DEG[rownames(DEG) %in% genes,]

#############################################################
# Format dataframe for volcano plot
#############################################################

DEG$sig <- DEG$label <- ifelse(DEG$FDR < 0.05, TRUE, FALSE)
DEG[DEG$sig == TRUE,]$label <- ifelse(DEG[DEG$sig == TRUE,]$logFC > 0, "Upregulated", "Downregulated")
DEG$label <- factor(DEG$label, levels = c("Upregulated", "Downregulated", FALSE))

nUp <- nrow(DEG[DEG$label == "Upregulated",])
nDown <- nrow(DEG[DEG$label == "Downregulated",])

print(paste("Upregulated:", nUp))
print(paste("Downregulated:", nDown))


#############################################################
# Create volcano plot
#############################################################

DEG_labeled <- DEG[rownames(DEG) %in% genes, ]

png("results/explainability/volcano.png", width = 8, height = 5, res = 600, units = "in")
ggplot() +
    geom_point(data = DEG, aes(x = logFC, y = -log10(FDR)), color = "black") +
    geom_point(data = DEG[DEG$sig == TRUE,], aes(x = logFC, y = -log10(FDR), color = label)) +
    geom_text_repel(data = DEG_labeled, aes(x = logFC, y = -log10(FDR), label = rownames(DEG_labeled)),
                    size = 5, box.padding = 1, point.padding = 0.5, max.overlaps = 20) +  
    geom_hline(yintercept = -log10(0.05), linetype = "dotted", size = 0.75) +
    geom_vline(xintercept = c(1.5, -1.5), linetype = "dotted", size = 0.75) +
    scale_color_manual("", labels = c("Overexpressed in\nTop Samples", "Underexpressed in\nTop Samples"), 
                        values = c("#48A9A6", "#C1666B")) +
    theme_classic() +
    theme(panel.border = element_rect(color = "black", fill = NA, size = 0.5),   
        text = element_text(size = 17), 
        legend.key.size = unit(1, 'cm'),
        axis.text.x = element_text(size=17, vjust = 0.5), 
        axis.text.y = element_text(size=17)) +
    labs(x = "logFC", y = expression(-log[10]*FDR))   
dev.off()
