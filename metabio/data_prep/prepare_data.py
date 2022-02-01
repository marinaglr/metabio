import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

def calculate_fingerprint(smiles, fp_size, radius, metab_num=1):
    """
    Calculates count Morgan fingerprints as pandas columns
    Input:
        smiles: str - smiles from one compound
        fp_size: int - size of the Morgan fingerprint
        radius: int - radius of the Morgan fingerprint
        metab_num: int - metabolite number from the list of best scored metabolites. Number used to name the fingerprint columns of each metabolite.
    
    Output:
        return: dataframe (with one row) containing each fingerprint byte in one column
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache(False)
    Chem.GetSSSR(mol)

    ### Count fingerprint
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=fp_size)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    fp = array

    fp_column_names = [f"metab_{metab_num}_bit_{i}" for i in range(len(fp))]
    fp = pd.DataFrame([fp], columns=fp_column_names)

    return fp

def prepare_datasets(data_df, endpoint, metabolites_df=pd.DataFrame(), missing_val=-1, score_threshold=0,
                        detox_phaseII=False, logP=-99, morgan_metab=False, physchem_metab=False):
    """
    Prepare following data sets:
        chem: with molecular physchem descriptors and fingerprint
        metab: physchem + fingerprint + descriptors from metabolites
    Input:
        data_df: df - data set containing the parent compounds
        endpoint: str - name of the endpoint -> class column in data set must be "Toxicity_{endpoint}"
        metabolites_df: dataframe - contains the metabolites predicted with Meteor and the corresponding parent smiles (in a column named "parent_SMILES")
        score_threshold: int - score threshold of the considered metabolites
        detox_phaseII: bool - if True -> filter out compounds further metabolized by phase II reactions or products from those reactions
        logP: float - minimum logP used to filter metabolites
        morgan_metab: bool - if True -> calculate the Morgan count fingerprint from metabolites
        physchem_metab: bool - if True -> calculate the RDKit physchem properties from metabolites
    
    Output:
        return: dictionary containing: metab (df with metabolite descriptors), chem (only parent descriptors), class_endp, smiles_endp
    """
    
    cols = [c for c in data_df.columns if 'smiles' in c.lower() or 'CAS' in c]
    dfraw = data_df.drop(cols, axis=1)

    endpoint_col = f"Toxicity_{endpoint}"
        
    ## endpoint smiles
    smilesCol = [c for c in data_df.columns if c.lower() == 'smiles' or c.lower() == "smiles (canonical)"]
    smiles_endp = data_df[data_df[endpoint_col] != missing_val][smilesCol]

    df_endp_smiles = data_df[data_df[endpoint_col] != missing_val]
    df_endp = dfraw[dfraw[endpoint_col] != missing_val]

    # Calculate descriptors for the 5 top ranked metabolites
    if morgan_metab == True or physchem_metab == True:
        desc_all_parents = pd.DataFrame()
        if physchem_metab == True:
            ### get RDKit descriptor list
            mol_descriptors = [x[0] for x in Descriptors._descList]
            calculator = MolecularDescriptorCalculator(mol_descriptors)

        for parent_smiles in smiles_endp[smilesCol].values:
            # get metabolites for parent (with given filters)
            metabolites = get_metabolites_from_parent(parent_smiles[0], metabolites_df, score_threshold=score_threshold,
                                                            detox_phaseII=detox_phaseII, logP=logP)
            all_desc_df = pd.DataFrame()
            all_fp_df = pd.DataFrame()
            metab_num = 1

            # Calculate descriptors for the top 5 metabolites of each parent
            for metabolite_smi in metabolites["SMILES (Canonical)"].values[:5]:
                
                if morgan_metab == True:
                    ### Calculate Morgan (count) fingerprint of metabolites
                    fp = calculate_fingerprint(metabolite_smi, 1024, radius=2, metab_num=metab_num)
                    all_fp_df = pd.concat([all_fp_df, fp], axis=1)
                
                if physchem_metab == True:
                    ### Calculate RDKit descriptors of metabolites
                    desc = calculator.CalcDescriptors(mol=Chem.MolFromSmiles(metabolite_smi))
                    desc = pd.DataFrame([desc], columns=[f"metab_{metab_num}_{j}" for j in mol_descriptors])
                    all_desc_df = pd.concat([all_desc_df, desc], axis=1)
                
                metab_num = metab_num + 1

            # Concatenate the descriptors of the metabolites from each parent compound
            if morgan_metab == True and physchem_metab == True:
                all_input_desc = pd.concat([all_fp_df, all_desc_df], axis=1)
            elif morgan_metab == False and physchem_metab == True:
                all_input_desc = all_desc_df.copy()
            elif morgan_metab == True and physchem_metab == False:
                all_input_desc = all_fp_df.copy()
            all_input_desc["SMILES (Canonical)"] = parent_smiles
            desc_all_parents = pd.concat([desc_all_parents, all_input_desc], axis=0)
        
        if morgan_metab == True:
            fing_cols = [c for c in desc_all_parents.columns if "bit_" in c]
            desc_all_parents[fing_cols].fillna(value=0, inplace=True) # fill fps for parents with less than 5 metabolites with 0
        if physchem_metab == True:
            desc_all_parents.replace([np.inf, -np.inf], np.nan, inplace=True) # avoid infinite numbers
            desc_all_parents.fillna(desc_all_parents.mean(), inplace=True) # fill the rest of descriptors for parents with less than 5 metabolites with the average of the column

        desc_all_parents.reset_index(drop=True, inplace=True)
        
        # concatenate fps to the other descriptors
        df_desc_metab = df_endp_smiles.merge(desc_all_parents, how="left", on="SMILES (Canonical)")
        drop_invivocols = [c for c in df_desc_metab.columns if 'vivo' in c.lower() or 'toxicity' in c.lower() or 'smiles' in c.lower()]
        df_desc_metab = df_desc_metab.drop(drop_invivocols, axis=1) # with fps of metabolites
    

    else:
        drop_invivocols = [c for c in df_endp.columns if 'vivo' in c.lower() or 'toxicity' in c.lower()]
        df_desc_metab = df_endp.drop(drop_invivocols, axis=1) # with fps of metabolites

    
    ## Parent compound descriptors + fingerprint
    drop_cols = [c for c in df_endp.columns if 'vivo' in c.lower() or 'vitro' in c.lower() or 'toxicity' in c.lower() or 'metab' in c.lower() or 'p0' in c or 'p1' in c or "Split" in c or "info_" in c.lower()]
    df_chem_endp = df_endp.drop(drop_cols, axis=1)

    ## endpoint class label
    class_endp = df_endp[endpoint_col]
    encoder = LabelEncoder()
    encoder.fit(class_endp)
    class_endp = encoder.transform(class_endp)
    
    endp_data = {"metab": df_desc_metab, "chem": df_chem_endp, "class": class_endp, "smiles": smiles_endp}
        
    return endp_data

def get_metabolites_from_parent(parent_smiles, metabolites_df, score_threshold=0, detox_phaseII=False, logP=-99):
    """
    Get the metabolites predicted for a parent compound and matching the indicated criteria
    Input:
        parent_smiles: str - smiles for the current parent compound
        metabolites_df: df - dataframe containing all metabolites and a column with the smiles of their corresponding parent compound ("parent_SMILES")
        score_threshold: int - minimum score of the metabolites to be returned
        detox_phaseII: bool - if True -> filter out compounds further metabolized by phase II reactions or products from those reactions
        logP: float - minimum logP used to filter metabolites
    
    Output:
        dataframe containing the set of metabolites corresponding to the parent_smiles
    """
        
    # Filter out compounds that are further metabolized in phase II reactions
    if detox_phaseII == True:
        metabolites_df = metabolites_df[metabolites_df["info_Detox_phaseII"] == False]
        metabolites_df = metabolites_df[metabolites_df["info_Phase"] != "Phase II"]
    
    if logP != -99:
        metabolites_df = metabolites_df[metabolites_df["info_LogP"] > logP]
        #metabolites_df = metabolites_df[metabolites_df["SlogP"] > logP]
        
    if score_threshold != 0: # all predicted metabolites
        metabolites_df = metabolites_df[metabolites_df["info_Absolute Score"] > score_threshold]
    
    metabolites_parent_df = metabolites_df[metabolites_df["parent_SMILES"] == parent_smiles]
    
    info_cols = [c for c in metabolites_parent_df.columns if "info_" in c]
    metabolites_parent_df = metabolites_parent_df.drop(info_cols, axis=1)

    return metabolites_parent_df

def prepare_descriptors(X_df, parent_or_metab, endpoint, model_path, desc_type="chem", save_normalizer=True, save_columns=True):
    """
    Apply low variance filter and normalize features
    
    Input:
        X_df: dataframe - input dataframe with descriptors
        parent_or_metab: str - 'parent' or 'metab': indicate whether the X_df data contains only the parent compounds or metabolites
        endpoint: str - name of the endpoint
        model_path: str - path to save the feature scaler model and the remaining columns after variance filter
        desc_type: str - input feature set: 'chem', 'metab' or 'cddd'
        save_normalizer: bool - save the model used to scale the features
        save_columns: bool - save a CSV file containing the name of the remaining columns after the variance filter
    
    Output:
        X_df: dataframe - X data after variance filter and feature normalization
        selected_cols: list - columns after variance filter
        scaler: model fited on X_df to normalize the features
    """
    ### Variance filter
    num_all_cols = len(X_df.columns)
    selector = VarianceThreshold(0.001)
    selector.fit_transform(X_df)
    X_df = X_df[X_df.columns[selector.get_support(indices=True)]]
    X = X_df.values
    selected_cols = X_df.columns
    num_selected_cols = len(X_df.columns)
    print(f"Variance filter removed {num_all_cols-num_selected_cols} columns")
    # Save remaining columns after filter in a CSV file
    if save_columns == True:
        if os.path.isdir(f"{model_path}") == False:
            os.mkdir(f"{model_path}")
        columns_df = pd.DataFrame(X_df.columns, 
                                columns=["Feature"]).to_csv(f'{model_path}/model_{endpoint}_{desc_type}_{parent_or_metab}_variance_filter_columns.csv', index=False)

    
    ### Normalize all descriptors
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)
    X_df = pd.DataFrame(X, columns=X_df.columns)
    print(X_df.shape)
    # Save the scaler
    if save_normalizer == True:
        # check if directory exists
        if os.path.isdir(f"{model_path}/normalizer") == False:
            if os.path.isdir(f"{model_path}") == False:
                os.mkdir(f"{model_path}")
            os.mkdir(f"{model_path}/normalizer")

        scalerfile = f'{model_path}/normalizer/model_{endpoint}_{desc_type}_{parent_or_metab}.pkl'
        pickle.dump(scaler, open(scalerfile, 'wb'))
    #X_df.to_csv(f"{self.mainPath}/data/internal/{endpoint}_training_set_normalized.csv")
    
    return X_df, selected_cols, scaler