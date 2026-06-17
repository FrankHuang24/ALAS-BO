import torch
from sklearn.datasets import load_digits,load_breast_cancer

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from .optimization_problem import OptimizationProblem
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

from sklearn.datasets import fetch_openml



class XGBoost_HPO(OptimizationProblem):
    def __init__(self):
        self.min = -0.99  # Estimated best achievable accuracy (98%)
        self.minimum = torch.tensor([-2.0, 300, 6, 3, 0.9, 0.9, 0.1, 0.0, 0.0])  # Estimated optimal params
        self.dim = 9
        self.lb = torch.tensor([-3.0, 50.0, 3.0, 1.0, 0.5, 0.5, 0.0, -3.0, -3.0])
        self.ub = torch.tensor([0.0, 500.0, 10.0, 10.0, 1.0, 1.0, 5.0, 1.0, 1.0])
        self.int_var = torch.tensor([1, 2, 3], dtype=torch.int32)  # Indices of integer params
        self.cont_var = torch.tensor([0, 4, 5, 6, 7, 8])  # Indices of continuous params
        self.info = (
            "9-dimensional XGBoost hyperparameter optimization\n"
            "Objective: maximize 5-fold cross-validation accuracy on breast cancer dataset"
        )

        # Load dataset
        bank = fetch_openml('bank-marketing', version=1, as_frame=True)
        X, y = bank.data, bank.target
        y = y.astype(int)
        y = y-1
        self.X, self.y = X, y

        # data = load_digits()
        # self.X, self.y = data.data, data.target

        # data = load_breast_cancer()
        # self.X, self.y = data.data, data.target

    def eval(self, x):
        x = x.clone()
        if len(self.int_var) > 0:
            x[:, self.int_var] = torch.round(x[:, self.int_var])

        x = self._denormalize(x)
        scores = []
        for params in x:
            model_params = {
                'learning_rate': 10 ** params[0].item(),
                'n_estimators': int(params[1]),
                'max_depth': int(params[2]),
                'min_child_weight': int(params[3]),
                'subsample': params[4].item(),
                'colsample_bytree': params[5].item(),
                'gamma': params[6].item(),
                'reg_alpha': 10 ** params[7].item(),
                'reg_lambda': 10 ** params[8].item(),
            }

            model = XGBClassifier(**model_params, enable_categorical=True, use_label_encoder=False, eval_metric="logloss")
            score = cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
            scores.append(-score)

        return torch.tensor(scores).to(torch.float32).unsqueeze(1)

    def _denormalize(self, x):
        """Clamp denormalized parameters to their valid bounds."""
        x_denorm = x.clone()
        for i in range(self.dim):
            x_denorm[:, i] = torch.clamp(
                x[:, i] * (self.ub[i] - self.lb[i]) + self.lb[i],
                self.lb[i], self.ub[i]
            )
        return x_denorm


class XGBoost_HPO_14D(OptimizationProblem):
    def __init__(self):
        self.min = -0.99  # Estimated best achievable accuracy (98%)
        self.minimum = torch.tensor([-2.0, 300, 6, 3, 0.9, 0.9, 0.1, 0.0, 0.0, 0.01, 0.1, 1.0, 0.5, 0.3, 0.2])
        self.dim = 15
        self.lb = torch.tensor([-3.0, 50.0, 3.0, 1.0, 0.5, 0.5, 0.0, -3.0, -3.0, 0.0, 0.01, 0.1, 0.2, 0.1, 0.0])
        self.ub = torch.tensor([0.0, 500.0, 10.0, 10.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 0.9, 0.5])
        self.int_var = torch.tensor([1, 2, 3, 11], dtype=torch.int32)  # Indices of integer params
        self.cont_var = torch.tensor([0, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14])  # Indices of continuous params
        self.info = (
            "15-dimensional XGBoost hyperparameter optimization\n"
            "Objective: maximize 5-fold cross-validation accuracy on bank marketing dataset"
        )

        # Load dataset
        bank = fetch_openml('bank-marketing', version=1, as_frame=True)
        X, y = bank.data, bank.target
        y = y.astype(int)
        y = y - 1
        self.X, self.y = X, y

    def eval(self, x):
        x = x.clone()
        if len(self.int_var) > 0:
            x[:, self.int_var] = torch.round(x[:, self.int_var])

        x = self._denormalize(x)
        scores = []
        for params in x:
            model_params = {
                'learning_rate': 10 ** params[0].item(),
                'n_estimators': int(params[1]),
                'max_depth': int(params[2]),
                'min_child_weight': int(params[3]),
                'subsample': params[4].item(),
                'colsample_bytree': params[5].item(),
                'gamma': params[6].item(),
                'reg_alpha': 10 ** params[7].item(),
                'reg_lambda': 10 ** params[8].item(),
                'colsample_bylevel': params[9].item(),
                'colsample_bynode': params[10].item(),
                'num_parallel_tree': int(params[11]),
                'max_delta_step': params[12].item(),
                'scale_pos_weight': params[13].item(),
            }

            model = XGBClassifier(**model_params, enable_categorical=True, use_label_encoder=False, eval_metric="logloss")
            score = cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
            scores.append(-score)

        return torch.tensor(scores).to(torch.float32).unsqueeze(1)

    def _denormalize(self, x):
        x_denorm = x.clone()
        for i in range(self.dim):
            x_denorm[:, i] = torch.clamp(
                x[:, i] * (self.ub[i] - self.lb[i]) + self.lb[i],
                self.lb[i], self.ub[i]
            )
        return x_denorm


class LightGBM_HPO(OptimizationProblem):
    """16-dimensional LightGBM hyperparameter optimization

    Global optimum: Unknown (real-world problem)

    :ivar dim: 16
    :ivar int_var: Indices of integer parameters
    :ivar cont_var: Indices of continuous parameters
    :ivar min: Estimated best accuracy (0.98)
    """

    def __init__(self):
        self.min = -0.98  # Targeting 98% accuracy
        self.dim = 16
        self.lb = torch.zeros(16)
        self.ub = torch.ones(16)
        self.int_var = torch.tensor([1, 2, 3, 8, 9, 10], dtype=torch.int32)
        self.cont_var = torch.tensor([0, 4, 5, 6, 7, 11, 12, 13, 14, 15])

        # housing = fetch_california_housing()
        # self.X, self.y = housing.data, housing.target
        bank = fetch_openml('bank-marketing', version=1, as_frame=True)
        self.X, self.y = bank.data, bank.target

        self.info = (
            "16D LightGBM hyperparameter optimization\n"
            "Objective: Maximize 5-fold CV accuracy on digits dataset"
        )

    def eval(self, x):
        x = x.clone()

        if len(self.int_var) > 0:
            x[:, self.int_var] = torch.round(x[:, self.int_var])
        x = torch.clamp(x, 0, 1)

        x = self._denormalize(x)
        scores = []
        for params in x:
            model_params = {
                'learning_rate': 10 ** (params[0] * 3 - 4),  # 1e-4 to 1e-1
                'num_leaves': int(10 + params[1] * 200),  # 10-210
                'max_depth': int(3 + params[2] * 15),  # 3-18
                'min_child_samples': int(5 + params[3] * 45),  # 5-50
                'subsample': 0.5 + params[4] * 0.5,  # 0.5-1.0
                'colsample_bytree': 0.5 + params[5] * 0.5,
                'reg_alpha': 10 ** (params[6] * 7 - 10),  # 1e-10 to 1e-3
                'reg_lambda': 10 ** (params[7] * 7 - 10),
                'min_split_gain': params[8] * 0.2,
                'feature_fraction': 0.5 + params[9] * 0.5,
                'bagging_freq': int(params[10] * 10),
                'bagging_fraction': 0.5 + params[11] * 0.5,
                'lambda_l1': 10 ** (params[12] * 7 - 10),
                'lambda_l2': 10 ** (params[13] * 7 - 10),
                'min_data_in_leaf': int(5 + params[14] * 45),
                'path_smooth': params[15] * 0.5
            }
            model = LGBMClassifier(**model_params)
            score = cross_val_score(model, self.X, self.y, cv=3).mean()
            scores.append(-score)

        return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

    def _denormalize(self, x):
        x_denorm = x.clone()
        ranges = torch.tensor([
            [-4, -1],  # 0: learning_rate (log10)
            [10, 110],  # 1: num_leaves
            [3, 8],  # 2: max_depth
            [2, 20],  # 3: min_child_samples
            [0.5, 1.0],  # 4: subsample
            [0.5, 1.0],  # 5: colsample_bytree
            [-3, -1],  # 6: reg_alpha (log10)
            [-3, -1],  # 7: reg_lambda (log10)
            [0, 0.2],  # 8: min_split_gain
            [0.5, 0.95],  # 9: feature_fraction
            [0, 5],  # 10: bagging_freq
            [0.5, 0.95],  # 11: bagging_fraction
            [-3, -1],  # 12: lambda_l1 (log10)
            [-3, -1],  # 13: lambda_l2 (log10)
            [5, 30],  # 14: min_data_in_leaf
            [0, 0.5]  # 15: path_smooth
        ])
        for i in range(self.dim):
            if i in [0, 6, 7, 12, 13]:
                x_denorm[:, i] = 10 ** (x[:, i] * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0])
            else:
                x_denorm[:, i] = x[:, i] * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
        return x_denorm



class SVM_HPO(OptimizationProblem):
    """3-dimensional SVM hyperparameter optimization

    Global optimum: Unknown (real-world problem)
    Search space:
    - C (log): [1e-3, 1e3]
    - gamma (log): [1e-4, 1e1]
    - kernel: {0: 'linear', 1: 'rbf', 2: 'poly'}
    """

    def __init__(self):
        self.min = -0.99  # Targeting 99% accuracy
        self.dim = 3
        self.lb = torch.tensor([0.0, 0.0, 0.0])
        self.ub = torch.tensor([1.0, 1.0, 2.99])  # 3 kernel choices
        self.int_var = torch.tensor([2], dtype=torch.int32)  # Kernel index
        self.cont_var = torch.tensor([0, 1])

        housing = fetch_california_housing()
        self.X, self.y = housing.data, housing.target

        self.info = (
            "3D SVM hyperparameter optimization\n"
            "Kernels: 0=linear, 1=rbf, 2=poly\n"
            "Objective: Maximize 5-fold CV accuracy"
        )

    def eval(self, x):
        x = x.clone()

        x[:, 2] = torch.clamp(torch.round(x[:, 2]), 0, 2)

        scores = []

        for params in x:
            C = 10 ** (params[0] * 6 - 3)
            gamma = 10 ** (params[1] * 5 - 4)
            kernel_idx = int(params[2])
            model = SVC(
                C=float(C),
                gamma=float(gamma),
                kernel=['linear', 'rbf', 'poly'][kernel_idx],
                random_state=42
            )
            score = cross_val_score(model, self.X, self.y, cv=5).mean()
            scores.append(-score)

        return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)