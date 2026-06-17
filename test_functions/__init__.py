from importlib import import_module

from .optimization_problem import OptimizationProblem
from .ackley import Ackley
from .branin import Branin
from .exponential import Exponential
from .griewank import Griewank
from .hartmann3 import Hartmann3
from .hartmann6 import Hartmann6
from .hartmann24 import Hartmann24
from .levy import Levy
from .perm import Perm
from .rastrigin import Rastrigin
from .rosenbrock import Rosenbrock
from .schwefel import Schwefel
from .six_hump_camel import SixHumpCamel
from .sphere import Sphere
from .weierstrass import Weierstrass
from .zakharov import Zakharov

__all__ = [
    "OptimizationProblem",
    "Ackley",
    "Branin",
    "Exponential",
    "Griewank",
    "Hartmann3",
    "Hartmann6",
    "Hartmann24",
    "Levy",
    "Perm",
    "Rastrigin",
    "Rosenbrock",
    "Schwefel",
    "SixHumpCamel",
    "Sphere",
    "Weierstrass",
    "Zakharov",
]

_OPTIONAL_IMPORTS = {
    "Robot3": ("robot", "Robot3", "Box2D and pygame"),
    "Robot4": ("robot", "Robot4", "Box2D and pygame"),
    "PortfolioSurrogate": ("portfolio_surrogate", "PortfolioSurrogate", "the bundled portfolio surrogate data"),
    "XGBoost_HPO": ("hyperparameter_tuning", "XGBoost_HPO", "scikit-learn and xgboost"),
    "XGBoost_HPO_14D": ("hyperparameter_tuning", "XGBoost_HPO_14D", "scikit-learn and xgboost"),
    "LightGBM_HPO": ("hyperparameter_tuning", "LightGBM_HPO", "scikit-learn and lightgbm"),
    "SVM_HPO": ("hyperparameter_tuning", "SVM_HPO", "scikit-learn"),
}


def __getattr__(name):
    if name not in _OPTIONAL_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name, requirement = _OPTIONAL_IMPORTS[name]
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ImportError as exc:
        raise ImportError(
            f"Optional benchmark {name!r} requires {requirement}. "
            "Install the optional dependencies or choose a synthetic benchmark."
        ) from exc

    value = getattr(module, attr_name)
    globals()[name] = value
    return value
