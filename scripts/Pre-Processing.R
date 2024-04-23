#This script details the pre-processing steps of our project. 
#The same workflow is applied to both testing and training datasets, but only the workflow for training dataset is shown here.
library(dplyr)

#######STEP 1: Merge Molecular Features and Gene Essentiality Score Datasets#####
#Specify training files path.
scores_file <- '/Users/a.h./Documents/University/Comp Bio/CS4220/Project/Data/Training/Achilles_v2.11_training_phase3.gct'
copy_num_file <- '/Users/a.h./Documents/University/Comp Bio/CS4220/Project/Data/Training/CCLE_copynumber_training_phase3.gct'
exp_data_file <- '/Users/a.h./Documents/University/Comp Bio/CS4220/Project/Data/Training/CCLE_expression_training_phase3.gct'
mutation_file <- '/Users/a.h./Documents/University/Comp Bio/CS4220/Project/Data/Training/CCLE_hybridmutation_training_phase3.gct'

#Read files.
scores <- read.delim(scores_file, skip = 2)
copy_num <- read.delim(copy_num_file, skip = 2)
exp_data <- read.delim(exp_data_file, skip = 2)
mutation_data <- read.delim(mutation_file, skip = 2)

#As the 4 sets of datasets have different gene lists, we find common genes across all files.
scores_genes <- scores$Description
copy_num_genes <- copy_num$Description
exp_data_genes <- exp_data$Description
mutation_genes <- mutation_data$Description

genes1 <- intersect(scores_genes, copy_num_genes)
genes2 <- intersect(exp_data_genes, genes1)
final_genes <- intersect(mutation_genes, genes2)

#Retain only common genes across all files.
scores_final <- scores[match(final_genes, scores$Description),]
copy_num_final <- copy_num[match(final_genes, copy_num$Description),]
exp_data_final <- exp_data[match(final_genes, exp_data$Description),]
mutation_final <- mutation_data[match(final_genes, mutation_data$Description),]

#Choose training cell-line that has the most significant correlation between gene expression and gene essentiality scores.
#We then combine molecular features data with gene essentiality scores data into one table.
final <- cbind(scores_final$Description, exp_data_final$CHAGOK1, copy_num_final$CHAGOK1, mutation_final$CHAGOK1, scores_final$CHAGOK1)
final <- data.frame(final)
colnames(final) <- c("Gene", "Exp_Data", "Copy_Num", "Mutation","Score")
final[,c(2:5)] <- final[,c(2:5)] %>% mutate_all(as.numeric)

#Before proceeding, we quantile normalise gene expression data to account for technical variance.
final$Exp_Data <- log2(final$Exp_Data + 1)

#######STEP 2: Retain Significant Genes for Dataset#####
#Find functional enrichment clusters that are significant (FDR <= 0.05).
functional_enrichment <- read.csv("/Users/a.h./Documents/University/Comp Bio/CS4220/Project/Data/Functional Enrichment/Functional_Enrichment_Training.csv")
significant_clusters <- functional_enrichment[sort(functional_enrichment$FDR <= 0.05),]
rownames(significant_clusters) <- NULL
significant_clusters$Cluster_id <- rownames(significant_clusters)

#Specify which cluster each gene appear in: 
#Within each cluster, values 1 indicates that the gene belongs to the cluster and 0 indicates that the gene do not belong to the cluster.
genes_clusters <- data.frame("Gene" = final$Gene)
cluster_function <- for (cluster in 1:nrow(significant_clusters)) {
  sig_genes <- unlist(strsplit(significant_clusters[cluster,"Genes"], split = ", "))
  column_name <- paste("Cluster", significant_clusters[cluster,"Cluster_id"], sep = "_")
  genes_clusters[, column_name] <- ifelse(genes_clusters$Gene %in% sig_genes, 1, 0)
}

#Merge cluster information with the table containing molecular features and gene essentiality score data.
cleaned_data <- cbind(final, genes_clusters[,2:(nrow(significant_clusters)+1)])

#####STEP 3: MOLECULAR FEATURE SELECTION#####
#Not all molecular features provided are relevant to predicting gene essentiality scores.
#Assuming good predictive molecular features are significantly correlated with gene essentiality scores, perform pearson correlation between each molecular feature and gene essentiality score.
corr_exp <- cor.test(cleaned_data$Exp_Data, cleaned_data$Score)
corr_copy_num <- cor.test(cleaned_data$Copy_Num, cleaned_data$Score)
corr_mutation <- cor.test(cleaned_data$Mutation, cleaned_data$Score)

#Retain Gene Expression feature because it shows significant correlation with gene essentiality scores.
cleaned_data <- subset(cleaned_data, select = -c(Copy_Num, Mutation))

#####STEP 4: CLUSTER SELECTION#####
#Not all significant clusters contribute to gene essentiality. Only clusters where genes have significant correlated gene expression with gene essentiality scores are predictive. 
#Perform pearson correlation between gene expression and gene essentiality scores within each significant cluster.
corr_table <- data.frame("Cluster" = c(1:nrow(significant_clusters)), "Exp" = 0)
corr_function <- for(cluster in 1:nrow(significant_clusters)){
  cluster_id <- paste("Cluster", cluster, sep = "_")
  genes_in_cluster <- cleaned_data[cleaned_data[,cluster_id] == 1,]
  corr_exp <- cor.test(genes_in_cluster$Exp_Data, genes_in_cluster$Score)
  corr_table$Exp[cluster] <- corr_exp$p.value
}

#Retain only clusters where genes show significant correlation (p-val <= 0.05) with gene essentiality scores.
noncorrelated_clusters <- corr_table[corr_table$Exp > 0.05,]
noncorrelated_clusters$Name <- paste("Cluster", noncorrelated_clusters$Cluster, sep = "_")
`%notIn%` <- Negate(`%in%`)
final_data <- cleaned_data[,colnames(cleaned_data) %notIn% noncorrelated_clusters$Name]

#######STEP 5: DISCRETIZING VARIABLES AFTER FEATURE SELECTION#####
#Discretization of gene expression is required for calculation of weights in the Weighted Naive Bayes Model.
#We discretize according to quartile ranges as expression data follow a normal distribution.
#Gene expression within the top 25% of total gene expression is labelled as "High" and the rest as "Low".
exp_ranges <- quantile(final_data$Exp_Data)
final_data$Disc_Exp <- ifelse(final_data$Exp_Data >= exp_ranges[4], "High", "Low")

#Discretization of gene essentiality scores is required for the Weighted Naive Bayes Model to classify genes, and for calculation of weights in Weighted Naive Bayes Model.
#We discretize according to quartile ranges as score data follow a normal distribution.
#Gene essentiality scores within the bottom 25% of overall gene essentiality is labelled as "Essential" and the rest as "Non-Essential".
score_ranges <- quantile(final_data$Score)
final_data$Disc_Scores <-  ifelse(final_data$Score <= score_ranges[2], "Essential", "Non-Essential")

#######STEP 6: OBTAINING MODEL INPUT AND WEIGHT DATA#####
#INPUT DATA: We obtain the training data input by retaining 
#1.gene expression 2.discretized gene essentiality scores.
Training_CHAGOK1 <- subset(final_data, select = c(Gene,Exp_Data,Disc_Scores))

#WEIGHT DATA: We obtain the dataset required for calculating weights by retaining 
#1.discretised gene expression 2. cluster information 3.discretized gene essentiality scores.
Weight_CHAGOK1 <- subset(final_data, select = -c(Exp_Data,Score))

