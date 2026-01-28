# Install GSEApy if you haven't already
# !pip install gseapy

import pandas as pd
import gseapy as gp

def prepare_ranking(csv_file):
    # Load the CSV file containing Ensembl names, symbols, and ranks
    #csv_file = "/home/karen/Documents/GitHub/Identify_muscle_age_genes/deep_learning/Results/feature_selection/RNAseq/ridge_L2_0/Experiment_GSE60590_feature_selection_Symbols.csv"
    gene_rank_df = pd.read_csv(csv_file)
    # check the first rows of the dataframe, if it starts with "","coef","abs_coef","NULL", replace NULL with "Symbol" and "" with "Ensembl"
    gene_rank_df.columns = ["Ensembl","coef","abs_coef","Symbol"]
    if gene_rank_df.columns[0] == "V1":
        gene_rank_df.columns = ["Ensembl","coef","abs_coef","Symbol"]
        # remove first row
        gene_rank_df = gene_rank_df.iloc[1:]
    # Add a column rank form 1 to len(gene_rank_df) with the rank of the gene depending on the abs_coef
    gene_rank_df['rank'] = gene_rank_df['abs_coef'].rank(ascending=False)
    return gene_rank_df


def get_GSEA(rank_dict, database='KEGG_2019_Human', outdir='gsea_results', min_size=10, max_size=1000,):
    rank_file_path = "rank_file.txt"

    with open(rank_file_path, 'w') as file:
        for gene, rank in rank_dict.items():
            file.write(f"{gene}\t{rank}\n")


    results = gp.prerank(rnk=rank_file_path, gene_sets=database, outdir=outdir, min_size=min_size, max_size=max_size, permutation_num=1000)
    return results

def main(file_name, csv_path = "Results/feature_selection/RNAseq/ridge_L2_0/"):
    
    #file_name = "Experiment_GSE60590_feature_selection_Symbols.csv"
    gene_rank_df = prepare_ranking(csv_path+file_name)
    rank_dict = dict(zip(gene_rank_df["Symbol"], gene_rank_df["rank"]))
    results = get_GSEA(rank_dict)
    enrichments = results.res2d
    best_enrichments = enrichments[enrichments['FDR q-val']<0.1].sort_values('FDR q-val', ascending=True)
    # save the results to a csv file
    best_enrichments.to_csv(f"{csv_path}GSEA/{file_name[:-4]}_KEGG_2019.csv")

# for file in os.listdir("Results/feature_selection/RNAseq/ridge_L2_0/"), run the main function
import os

def run():
    csv_path = "Results/feature_selection/RNAseq/ridge_L2_0/"
    file_list = os.listdir(csv_path)
    for file in file_list:
        if file.endswith("_Symbols.csv"):
            print(file)
            main(file, csv_path)
