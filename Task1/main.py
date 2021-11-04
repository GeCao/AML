from src.CoreManagement import CoreComponent

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imputer', type=str, default='knn', choices=['knn', 'mean', 'else'], help='type of imputer')
    parser.add_argument('--outlier', type=str, default='zscore', choices=['zscore', 'else'], help='type of outlier')
    parser.add_argument('--pca', type=str, default='pca', choices=['pca', 'else'], help='type of outlier')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='type of device')

    args = parser.parse_args()
    imputer = args.imputer
    outlier = args.outlier
    pca = args.pca
    device = args.device

    core_managemnet = CoreComponent(model='lasso', imputer=imputer, outlier=outlier, pca=pca, device=device)
    core_managemnet.initialization()
    core_managemnet.run()
    core_managemnet.kill()