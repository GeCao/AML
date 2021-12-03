from src.CoreManagement import CoreComponent

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='nnet', choices=["svm", "ensemble", "nnet"], help='type of model')
    parser.add_argument('--imputer', type=str, 
                        default='median',
                        choices=['knn', 'mice', 'mean', 'median', 'else', 'random_forest'], help='type of imputer')
    parser.add_argument('--outlier', type=str,
                        default='else',
                        choices=['zscore', 'iqr', 'local', 'isolationforest', 'winsorize', 'else'], help='type of outlier')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='type of device')
    parser.add_argument('--no_checkpoint', type=bool, default=False, choices=[True, False],
                        help='选择False会从data/*_checkpoint.csv读取之前保存过的数据，否则会重新进行一遍耗时的数据处理')

    args = parser.parse_args()
    model = args.model
    imputer = args.imputer
    outlier = args.outlier
    device = args.device
    no_checkpoint = args.no_checkpoint

    core_managemnet = CoreComponent(model=model, imputer=imputer, outlier=outlier, device=device,
                                    no_checkpoint=no_checkpoint)
    core_managemnet.initialization()
    core_managemnet.run(seed=66)
    core_managemnet.kill()