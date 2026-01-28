import scanpy as sc
import decoupler as dc

# Only needed for processing
import numpy as np
import pandas as pd
from anndata import AnnData



def filter_nameless_genes(ensembl_names, annot):
    return [
        item if item.split('.')[0] in annot.index else np.nan 
        for item in ensembl_names
    ]
def convert_to_symbol(ensembl_names, annot):
    
    ensembl_names = [item.split('.')[0] if item is not np.nan else np.nan for item in ensembl_names]
    symbol_names = [
        annot.loc[ensembl_id, 'external_gene_name'] if ensembl_id is not np.nan else np.nan
        for ensembl_id in ensembl_names
    ]
    return symbol_names

def convert_file_to_symbol():
    pass