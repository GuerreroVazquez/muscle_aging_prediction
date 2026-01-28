import gseapy
import numpy as np
import pandas as pd
import gseapy as gp
import sys
import re
orig_stdout = sys.stdout
default_pathway_file = '../../Results/Pathway_A_def_expressed_gene_names.csv'
databases_names = gseapy.get_library_name()


def remove_version_from_list(gene_names):
    """
    From a list of gene names, return a list with the same names but without version.
    :param gene_names: List of gene names
    :return: List of gene names without version
    """
    return [re.sub('\..*$', '', gene) for gene in gene_names]

def remove_version(genes_data_frame, column_names, diff_express=True,
                   output="gene_ensmbl_no_version_names"):
    """
    Creates a new dataframe with the same genes but without the version in the gene names.
    :param genes_data_frame: The dataframe with the gene names
    :param column_names: List of experiment names
    :param diff_express: bool If the genes are differentially expressed or not
    :param output: The prefix of the name the output file will have
    :return: New dataframe with gene names without version
    """
    new_df = genes_data_frame
    for name in column_names:
        transcripts = genes_data_frame[name]
        gene_names_1 = remove_version_from_list(transcripts)
        new_df[name] = gene_names_1
    if output:
        if diff_express:
            new_df.to_csv(f"{output}NDE.csv")
        else:
            new_df.to_csv(f"{output}DE.csv")
    return new_df

def get_pathways(genes_file='../../Results/def_expressed_gene_names.csv', sel_databases_names=None, remove_version_=False, cutoff=0.5):
    """
     From the gene_files csv, gets the pathways in  the databases in databases_names
      of ALL the experiments
     :param sel_databases_names: List of string with the subset of gseapy.get_library_name()
     :param genes_file: CSV file with the experiments df genes in each column. The column names are the sources of the genes
     :return:
     """
    if sel_databases_names is "all":
        sel_databases_names = databases_names
    if sel_databases_names is None:
        sel_databases_names = ['GO_Molecular_Function_2021',
                               'GO_Cellular_Component_2021',
                               'GO_Biological_Process_2021',
                               'Reactome_2022',
                               'KEGG_2021_Human']
    df = pd.read_csv(genes_file)
    names = list(df.head())
    df_results = pd.DataFrame()
    if remove_version_:
        df = remove_version(genes_data_frame=df, column_names=names, output=None)

    for experiment in names:
        DEGs = df[experiment].tolist()
        DEGs = [x for x in DEGs if pd.notnull(x)]
        experiment_dic = pd.DataFrame()
        for database in sel_databases_names:
            try:
                enr_GOBP_up = gp.enrichr(gene_list=DEGs,
                                         gene_sets=[database],
                                         organism='Human',
                                         outdir=f'test/{experiment}',
                                         cutoff=cutoff
                                         )
                if not enr_GOBP_up.results.empty:
                    experiment_dic = pd.concat([experiment_dic, enr_GOBP_up.results])
            except Exception as e:
                print(f"Error", e)
        experiment_dic["Experiment"] = experiment
        df_results = pd.concat([df_results, experiment_dic])
    df_results.rename(columns={'P-value': 'PValue'}, inplace=True)
    df_results.rename(columns={'Adjusted P-value': 'APValue'}, inplace=True)

    return df_results



def filter_pathways(df=None, diff_expression=default_pathway_file, pvalue_threshold=0.1, apvalue_threshold=1,
                    ratio_treshold=0):
    """
    This function will get the csv of the pathways generated previously and get the patways that
    have a criteria of pvalue, adjusted pvalue and ratio
    :param df: dataframe The dataframe output from the gseapy
    :param apvalue_threshold: float Threshold that marks up to which value to take the adjusted P value
    :param pvalue_threshold:  float Threshold that marks up to which value to take the P value
    :param ratio_treshold: float What is the minimun ratio (genes found/ genes total in pathway)
    :param diff_expression: str the CSV file that holds the output of gseapy

    :return: dataframe with the filters applied
    """
    if df is None and diff_expression is not None:
        df = pd.read_csv(diff_expression)

    ratio = [int(x.split('/')[0]) / int(x.split('/')[1]) for x in df['Overlap']]
    df['Ratio'] = ratio
    df = df.query(f"PValue < {pvalue_threshold}")
    df = df.query(f"APValue < {apvalue_threshold}")
    df = df.query(f"Ratio > {ratio_treshold}")
    return df

