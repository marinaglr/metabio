import argparse
import pandas as pd
import os
import metabio.modeling.train as train
import metabio.data_prep.prepare_data as prepare_data
import metabio.data_prep.split_data as split_data


#Parse arguments
parser = argparse.ArgumentParser(description='Toxicity prediction with consideration of metabolism.')
parser.add_argument('-e','--endpoint', type=str, default='none', help='endpoint for model training')
parser.add_argument('-s','--score', type=int, default=0, help='filter out compounds with a score below this threshold')
parser.add_argument('-r','--reactive', type=str, default='extreme', help='filter out compounds with reactive groups: "extreme", "all" or ""')
parser.add_argument('-d','--detox', type=str, default='0', help='filter out metabolites detoxified by phase II reactions: 0 (no) or 1 (yes)')
parser.add_argument('-l','--logp', type=int, default=-99, help='filter compounds with a logP below this threshold')
parser.add_argument('-c','--combination', type=int, default=0, help='index of the combination from the grid search to use')
parser.add_argument('--lasso', action='store_true', default=False, help='feature selection with lasso')
parser.add_argument('--train', action='store_true', help='train baseline models for specified endpoint (without metabolism information)')
parser.add_argument('--trainmetab', action='store_true', help='train models with the data including the (extrapolated) metabolite labels')
parser.add_argument('--method', type=str, default='rf', help='ML method for model training. Options: rf, knn, gb and svm.')
parser.add_argument('--splits', action='store_true', help='create CV split data sets')
parser.add_argument('-i', '--split_num', type=int, help='number of the CV round for which to create the split data. To create all -> -1')
parser.add_argument('--oversampling', action='store_true', help='train baseline models for specified endpoint (without metabolism information)')
parser.add_argument('--eval', action='store_true', help='evaluate models and calculate performance metrics')
parser.add_argument('--metab_model_parent', action='store_true', help='apply the model trained on the labeled metabolites to calculate the probability of the parent compounds')
parser.add_argument('--metab_model_metab', action='store_true', help='apply the model trained on the labeled metabolites to calculate the probability of the metabolites')
parser.add_argument('--join', action='store_true', help='join all output results in a single file')

args = parser.parse_args()
endpoint = args.endpoint

'''
if args.detox == 0:
    detox_phaseII == False
elif args.detox == 1:
    detox_phaseII == True
'''

parameters = {'score_threshold': [0, 100, 200, 300], 'filter_reactive': [''], 'detox_phaseII': [True, False], 'logP':[0, 3, -99]}

#num = 0
combination = {}
grid = []
for score_threshold in parameters['score_threshold']:
    for filter_reactive in parameters['filter_reactive']:
        for detox_phaseII in parameters['detox_phaseII']:
            for logP in parameters['logP']:
                combination = {'score_threshold': score_threshold, 'filter_reactive': filter_reactive, 'detox_phaseII': detox_phaseII, 'logP': logP}
                grid.append(combination)
                #num = num + 1

param = grid[args.combination]
filter_reactive = param['filter_reactive']
detox_phaseII = param['detox_phaseII']
score_threshold = param['score_threshold']
logP = param['logP']

missingVal = -1
num_trees = 500 # Number of trees built in random forest
cv = 5 # Number of folds in crossvalidation
method = args.method

if args.lasso == False:
    feat_imp = True
else:
    feat_imp = False

# Define which models are used to make the predictions of the parent compounds and the metabolites
if args.metab_model_metab == False and args.metab_model_parent == False:
    model_combination = 'parent_model'
elif args.metab_model_metab == True and args.metab_model_parent == True:
    model_combination = 'metab_model'
elif args.metab_model_metab == True and args.metab_model_parent == False:
    model_combination = 'combined_model'
        
# Load data
MAIN_PATH = '/home/garcim64/Documents/toxMetabolites/'

# Read the corresponding data file and initialize the class
raw_data = pd.read_csv(f'{MAIN_PATH}/data/raw_data/{endpoint}_dataset.csv', header=0) # baseline
#raw_data = pd.read_csv(f'{MAIN_PATH}/data/meteor_metabolites/biotransformation_descriptor/{endpoint}_biotransformation_descriptor.csv', index_col=0)
  
    
info_cols = [c for c in raw_data.columns if 'info_' in c]
raw_data.drop(info_cols, axis=1, inplace=True)

#models = toxMetabolites.toxMetabolites(raw_data, MAIN_PATH, missingVal)

    
if args.train == True:
    
    print(f'\nTraining models for {endpoint}')
    desc_type = 'chem'
    
    # Prepare data sets
    if desc_type == 'metab':
        metabolites_df = pd.read_csv(f'{MAIN_PATH}/data/meteor_metabolites/processed/{endpoint}_metabolites.csv')
        data_endp = prepare_data.prepare_datasets(endpoint, calc_metab_maccs=True, metabolites_df=metabolites_df)
    else:
        data_endp = prepare_data.prepare_datasets(endpoint, calc_metab_maccs=False)
    datasets = {endpoint: data_endp}
    

    # Train models
    #models.train_models(datasets[endpoint][desc_type], datasets[endpoint]['class'], datasets[endpoint]['smiles'].values, cv, endpoint, desc_type=desc_type, 
    #                    num_trees=num_trees, method=method, feat_imp=False, feat_sel=args.lasso, class_weight='balanced', oversampling=args.oversampling)

    train.train_models_from_CV_files(cv, endpoint, desc_type=desc_type, num_trees=num_trees, method=method, 
                                  feat_imp=False, feat_sel=args.lasso, class_weight='balanced', oversampling=args.oversampling)
if args.splits == True:
    endpoint_col = f'Toxicity_{endpoint}'
    parent_df = pd.read_csv(f'{MAIN_PATH}/data/raw_data/{endpoint}_dataset.csv', header=0)
    metab_labels = pd.read_csv(f'{MAIN_PATH}/data/meteor_metabolites/labeled_metabolites/no_duplicates/{endpoint}_label_metabolites_unknown=active.csv')

    print(f'\nCreating CV splits for {endpoint}')

    # Create splits
    split_data.create_CV_splits(parent_df, metab_labels, cv, endpoint, endpoint_col, desc_type='chem', split_num=args.split_num)
    
    
if args.trainmetab == True:
    
    parent_df = pd.read_csv(f'{MAIN_PATH}/data/raw_data/{endpoint}_dataset.csv', header=0)
    metab_labels = pd.read_csv(f'{MAIN_PATH}/data/meteor_metabolites/labeled_metabolites/no_duplicates/{endpoint}_label_metabolites_unknown=active.csv', index_col=0)

    endpoint_col = f'Toxicity_{endpoint}'
    
    print(f'\nTraining models for {endpoint}')
    desc_type = 'chem'
    
    # Train models
    train.train_models_metabolite_label(parent_df, metab_labels, cv, endpoint, endpoint_col, create_splits=False, desc_type=desc_type, 
                        num_trees=num_trees, method=method, feat_imp=False, feat_sel=args.lasso, class_weight='balanced', oversampling=args.oversampling)

    
if args.eval == True:
    print(f'Evaluating...')
    # Define location of test sets (from CV) and prepared metabolites
    test_parents_path = f'{MAIN_PATH}/data/splits/'
    metabolites_df = pd.read_csv(f'{MAIN_PATH}/data/meteor_metabolites/processed/{endpoint}_metabolites.csv')

    all_results_run_glory = pd.DataFrame()
    modes = ['baseline', 'mean', 'median', 'max_all', 'max_metab']

    results_df = evaluate.evaluate_test_sets(endpoint, metabolites_df, test_parents_path, metab_labels=args.metab_model_metab, 
                                           metab_model_for_parent=args.metab_model_parent, score_threshold=score_threshold, modes=modes, cv_runs=5,
                                           feat_sel=False, silent=True, filter_reactive=filter_reactive, detox_phaseII=detox_phaseII, logP=logP)
    
        
    results_df.to_csv(f'{MAIN_PATH}/results/evaluation/{model_combination}/{endpoint}_metab_scoreThreshold={score_threshold}_filterReact={filter_reactive}_detoxPhaseII={detox_phaseII}_logP={logP}.csv')
    print(results_df)
        
        
if args.join == True:

    all_results = pd.DataFrame()
    for score_threshold in parameters['score_threshold']:
        for filter_reactive in parameters['filter_reactive']:
            for detox_phaseII in parameters['detox_phaseII']:
                for logP in parameters['logP']:
                    file_name = f'{MAIN_PATH}/results/evaluation/{model_combination}/{endpoint}_metab_scoreThreshold={score_threshold}_filterReact={filter_reactive}_detoxPhaseII={detox_phaseII}_logP={logP}.csv'
                    if os.path.isfile(file_name):
                        result = pd.read_csv(file_name, index_col=0)
                        all_results = pd.concat([all_results, result], axis=0)
                    else:
                        print(f'File {file_name} is missing.')
                    
    all_results.to_csv(f'{MAIN_PATH}/results/evaluation/{model_combination}/{endpoint}_all_results.csv')             


