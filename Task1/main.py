from src.CoreManagement import CoreComponent

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ridge', choices=['lasso', 'ridge', 'nnet'], help='type of model')
    parser.add_argument('--imputer', type=str,
                        default='mice',
                        choices=['gain', 'knn', 'mice', 'mean', 'else'], help='type of imputer')
    parser.add_argument('--outlier', type=str,
                        default='isolationforest',
                        choices=['zscore', 'local', 'isolationforest', 'else'], help='type of outlier')
    parser.add_argument('--pca', type=str,
                        default='tree',
                        choices=['pca', 'tree', 'lasso', 'else'], help='type of feature selection')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='type of device')

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