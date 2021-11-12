from src.CoreManagement import CoreComponent

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ensemble', choices=['mlp', 'lasso', 'ridge', 'nnet', 'ensemble', 'adaboost', "Gboost"], help='type of model')
    parser.add_argument('--imputer', type=str, 
                        default='random_forest', 
                        choices=['gain', 'knn', 'mice', 'mean', 'median', 'else', 'random_forest'], help='type of imputer')
    parser.add_argument('--outlier', type=str,
                        default='else',
                        choices=['zscore', 'iqr', 'local', 'isolationforest', 'winsorize', 'else'], help='type of outlier')
    parser.add_argument('--pca', type=str,
                        default='kbest', #kbest
                        choices=['pca', 'kbest', 'tree', 'lsvc', 'lassoCV', 'lasso', 'SelectPercentile'
                                 'else'], help='type of feature selection')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='type of device')

    args = parser.parse_args()
    model = args.model
    imputer = args.imputer
    outlier = args.outlier
    pca = args.pca
    device = args.device

    core_managemnet = CoreComponent(model=model, imputer=imputer, outlier=outlier, pca=pca, device=device)
    core_managemnet.initialization()
    core_managemnet.run()
    core_managemnet.kill()