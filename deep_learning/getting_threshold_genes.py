from matplotlib.pylab import LinAlgError
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from kneed import KneeLocator


def get_gene_rank(feature_scoreing:str):
    gene_rank_df = pd.read_csv(feature_scoreing)
    if len(gene_rank_df.columns)==3:
        column_names = ["Ensembl","coef","abs_coef"]
    elif len(gene_rank_df.columns)==4:
        column_names = ["Ensembl","coef","abs_coef","Symbol"]
    elif len(gene_rank_df.columns)==2: # if it is shap value, it will be like this
        column_names = ["Ensembl","coef","abs_coef"]
        gene_rank_df["abs_coef"] = gene_rank_df[gene_rank_df.columns[1]].abs()
    # check the first rows of the dataframe, if it starts with "","coef","abs_coef","NULL", replace NULL with "Symbol" and "" with "Ensembl"
    gene_rank_df.columns = column_names
    if gene_rank_df.columns[0] == "V1":
        gene_rank_df.columns = column_names
        # remove first row
        gene_rank_df = gene_rank_df.iloc[1:]
    # check if the first row value is NaN
    if gene_rank_df.iloc[0][1] == "coef":
        gene_rank_df = gene_rank_df.iloc[1:]

    gene_rank_df = gene_rank_df.fillna(0)
    gene_rank_df["abs_coef"] = pd.to_numeric(gene_rank_df["abs_coef"], errors='coerce')
    return gene_rank_df


def get_gene_rank_format_RF(feature_scoreing:str):
    gene_rank_df = pd.read_csv(feature_scoreing)
    # check the first rows of the dataframe, if it starts with "","coef","abs_coef","NULL", replace NULL with "Symbol" and "" with "Ensembl"
    #gene_rank_df.columns = ["Ensembl","coef","abs_coef","Symbol"]
    if gene_rank_df.columns[0] == "V1":
        gene_rank_df.columns = ["coef","Ensembl","Symbol"]
        # remove first row
        gene_rank_df = gene_rank_df.iloc[1:]
    if gene_rank_df.iloc[0][1] == "coef":
        gene_rank_df = gene_rank_df.iloc[1:]
    # check if the first row value is NaN
    gene_rank_df["abs_coef"] = gene_rank_df["coef"].abs()

    gene_rank_df = gene_rank_df.fillna(0)
    gene_rank_df["abs_coef"] = pd.to_numeric(gene_rank_df["abs_coef"], errors='coerce')
    return gene_rank_df
def get_gene_rank_format_cat_temporal(feature_scoreing:str):
    gene_rank_df = pd.read_csv(feature_scoreing)
    # check the first rows of the dataframe, if it starts with "","coef","abs_coef","NULL", replace NULL with "Symbol" and "" with "Ensembl"
    #gene_rank_df.columns = ["Ensembl","coef","abs_coef","Symbol"]
    if gene_rank_df.columns[0] == "V1":
        gene_rank_df.columns = ["Ensembl","coef","Symbol"]
        # remove first row
        gene_rank_df = gene_rank_df.iloc[1:]
    if gene_rank_df.iloc[0][1] == "Ensembl":
        gene_rank_df = gene_rank_df.iloc[1:]
    # check if the first row value is NaN
    gene_rank_df["coef"] = pd.to_numeric(gene_rank_df["coef"], errors='coerce')
    gene_rank_df["abs_coef"] = gene_rank_df["coef"].abs()

    gene_rank_df = gene_rank_df.fillna(0)
    gene_rank_df["abs_coef"] = pd.to_numeric(gene_rank_df["abs_coef"], errors='coerce')
    return gene_rank_df
def get_elbow_point_from_gene_rank(gene_rank_df):
    x = gene_rank_df.index
    y = gene_rank_df["abs_coef"]
    alpha = get_elbow_point(x,y)
    return alpha

def get_inflexion_point_from_gene_rank(gene_rank_df):
    x = gene_rank_df.index
    y = gene_rank_df["abs_coef"]
    alpha = get_inflexion_point(x,y)
    return alpha
# Define a function for the curve fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def get_inflexion_point(x,y):
    
    # Fit a logarithmic curve
    try:
        coefficients = np.polyfit(np.log(x), y, 1)  # Fit a first-degree polynomial (straight line) to the logarithm of x

        # Generate curve points
        x_curve = np.linspace(min(x), max(x), 1000)
        y_curve = np.polyval(coefficients, np.log(x_curve))
        dy_dx = np.gradient(np.gradient(y_curve))
        d2y_dx2 = np.gradient(np.gradient(dy_dx))
        inflection_point = []
        for i in range(1,len(d2y_dx2)-10):
            if d2y_dx2[i-2] > d2y_dx2[i+1] + d2y_dx2[i+2]:
                inflection_point.append(i)
        max_change = inflection_point[-1]
        alpha=y_curve[max_change]
    except LinAlgError as e:
        y = y.sort_values(ascending=False)
        kn = KneeLocator(y,x, curve='convex', direction='decreasing')
        
        alpha = kn.knee
    return alpha

def get_elbow_point(x,y):
    
    y = y.sort_values(ascending=False)
    kn = KneeLocator(y,x, curve='convex', direction='decreasing')
    
    alpha = kn.knee
    return alpha


def get_df(csv_file, inflexion=True, alpha=None):
    """
    inflexion: if to use inflexion point, if false, it uses elbow
    alpha: if there is already a desired alpha
    """
    rank_genes = get_gene_rank(csv_file)
    if alpha:
        gene_rank_df = rank_genes[rank_genes["abs_coef"]>alpha]
        return gene_rank_df
    if inflexion:
        alpha = get_inflexion_point_from_gene_rank(rank_genes)
    else:
        alpha = get_elbow_point_from_gene_rank(rank_genes)

    print(alpha)
    gene_rank_df = rank_genes[rank_genes["abs_coef"]>alpha]
    return gene_rank_df



def get_df_format_cat_temporal(csv_file):
    rank_genes = get_gene_rank_format_cat_temporal(csv_file)
    gene_rank_df = rank_genes[rank_genes["abs_coef"]>0]
    return gene_rank_df


def get_df_format_RF(csv_file):
    rank_genes = get_gene_rank_format_RF(csv_file)
    alpha = get_inflexion_point_from_gene_rank(rank_genes)
    gene_rank_df = rank_genes[rank_genes["abs_coef"]>alpha]
    return gene_rank_df