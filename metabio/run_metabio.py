import argparse
import pandas as pd
import os
from pathlib import Path
from metabio.modeling.probability_combiner import ProbabilityCombiner
import metabio.modeling.train as train
import metabio.modeling.evaluate as evaluate
import metabio.data_prep.prepare_data as prepare_data
import metabio.data_prep.split_data as split_data

# Define main path where the data and results folders are contained
MAIN_PATH = Path(__file__).parent.parent
DATA_PATH = f'{MAIN_PATH}/data'
RESULTS_PATH = f'{MAIN_PATH}/results'
assert os.path.isdir(DATA_PATH), f'Data path does not exist.'
assert os.path.isdir(RESULTS_PATH), f'Results path does not exist.'


#Parse arguments
parser = argparse.ArgumentParser(description='Toxicity prediction with consideration of metabolism.')
parser.add_argument('-e','--endpoint', type=str, default='none', help='endpoint for model training')
parser.add_argument('-s','--score', type=int, default=0, help='filter out compounds with a score below this threshold')
parser.add_argument('-l','--logp', type=int, default=-99, help='filter compounds with a logP below this threshold')
parser.add_argument('--detox', action='store_true', default=False, help='filter out metabolites detoxified by phase II reactions')
parser.add_argument('--lasso', action='store_true', default=False, help='feature selection with lasso')
parser.add_argument('--oversampling', action='store_true', help='train baseline models for specified endpoint (without metabolism information)')
parser.add_argument('--desc', type=str, default='chem', help='a) "chem": train models on Morgan fingerprint and RDKit physchem of parent cmpds, b) "meta": include chem descriptors of metabolites, c) "cddd": use CDDD descriptors')
parser.add_argument('--splits', action='store_true', default=False, help='create CV split data sets')
parser.add_argument('-i', '--split_num', type=int, help='number of the CV round for which to create the split data. To create all -> -1')
parser.add_argument('--train', action='store_true', help='train baseline models for specified endpoint (without metabolism information)')
parser.add_argument('--train_metab', action='store_true', help='train models with the data including the labeled metabolite')
parser.add_argument('--method', type=str, default='rf', help='ML method for model training. Options: rf, knn, gb and svm.')
parser.add_argument('--grid', action='store_true', help='perform a grid search to optimize the hyperparameters. By default they are optimized for all methods but for RF.')
parser.add_argument('--eval', action='store_true', help='evaluate models and calculate performance metrics')
parser.add_argument('--combine', action='store_true', help='calculate the combined predicted probabilities for parent compounds and their respective metabolites')
parser.add_argument('--approach', type=str, default='hybrid', help='approach used to calculate the predictions: a) "baseline" (baseline model for the prediction of both parent compounds and metabolites) or b) "hybrid" (baseline model for the prediction of parent compounds plus metabolism-aware model for the prediction of metabolites)')

args = parser.parse_args()
endpoint = args.endpoint
endpoint_col = f'Toxicity_{endpoint}'
cv = 5 # Number of folds in crossvalidation
method = args.method
metabolites_df = pd.read_csv(f'{MAIN_PATH}/data/metabolites/{endpoint}_metabolites.csv')

# load data
if args.splits == True or args.train_metab == True or args.train == True:
    parent_df = pd.read_csv(f'{MAIN_PATH}/data/parents/{endpoint}_dataset.csv', header=0)
    labeled_metab = pd.read_csv(f'{MAIN_PATH}/data/metabolites/{endpoint}_labeled_metabolites.csv', index_col=0)

# create data splits used for training and evaluating the models
if args.splits == True:
    print(f'\nCreating CV splits for {endpoint}')
    split_data.create_CV_splits(parent_df, labeled_metab, cv, endpoint, endpoint_col, desc_type='chem', split_num=args.split_num)
    
if args.train == True:
    print(f'\nTraining models for {endpoint}')
    
    # Train models using the prepared split files
    models, mean_results = train.train_models_from_CV_files(cv, endpoint, desc_type=args.desc, method=method, feat_sel=args.lasso, oversampling=args.oversampling, 
                                        grid_search=args.grid, num_trees=500, splits_path=f'{MAIN_PATH}/data/splits/')
    mean_results.to_csv(f'{MAIN_PATH}/results/{endpoint}_{method}_desc={args.desc}_oversampling={args.oversampling}_featSel={args.lasso}.csv')                                    
    
if args.train_metab == True:
    print(f'\nTraining models for {endpoint}')
    models, mean_results = train.train_models_metabolite_label(parent_df, labeled_metab, cv, endpoint, endpoint_col, desc_type=args.desc, create_splits=False, method=method, 
                                        feat_sel=args.lasso, oversampling=args.oversampling, grid_search=args.grid, num_trees=500)
    mean_results.to_csv(f'{MAIN_PATH}/results/{endpoint}_metab_{method}_desc={args.desc}_oversampling={args.oversampling}_featSel={args.lasso}.csv')                                    

    
if args.eval == True:
    print(f'Evaluating models for {endpoint}')

    # Define location of test sets (from CV) and prepared metabolites
    test_parents_path = f'{MAIN_PATH}/data/splits/'

    modes = ['baseline', 'mean', 'median', 'max_all', 'max_metab']

    results_df = evaluate.evaluate_test_sets(endpoint, metabolites_df, test_parents_path, metab_labels=args.metab_model_metab, 
                                           metab_model_for_parent=args.metab_model_parent, score_threshold=args.score, modes=modes, cv_runs=5,
                                           feat_sel=False, silent=True, detox_phaseII=args.detox, logP=args.logp)
    
    if os.path.isdir(f'{MAIN_PATH}/data') == False:
        os.mkdir(f'{MAIN_PATH}/results/{args.approach}')
    
    results_df.to_csv(f'{MAIN_PATH}/results/{args.approach}/{endpoint}_metab_scoreThreshold={args.score}_detoxPhaseII={args.detox}_logP={args.logp}.csv')
    print(results_df)          

if args.combine == True:
    print('Evaluating model on combined probabilities')
    
    # Define location of test sets (from CV) and prepared metabolites
    test_parents_path = f'{MAIN_PATH}/data/splits/'

    if args.approach == 'hybrid':
        metab_model_for_metab=True
        metab_model_for_parent=False
    elif args.approach == 'baseline':
        metab_model_for_metab=False
        metab_model_for_parent=False

    combiner = ProbabilityCombiner(endpoint, metabolites_df, desc_type=args.desc)
    combiner.evaluate_combined_probabilities(test_parents_path, metab_model_for_metab=metab_model_for_metab, metab_model_for_parent=metab_model_for_parent,
                                            modes=['baseline', 'only_parents', 'mean', 'median', 'max_all', 'max_metab'], 
                                            cv_runs=5, feat_sel=args.lasso, score_threshold=args.score, detox_phaseII=args.detox, logP=args.logp, method=args.method)

