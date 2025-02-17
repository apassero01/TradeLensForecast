# Enums to define configuration mappings
from enum import Enum


class OptimizerEnum(Enum):
    # Maps strings to TF optimizers
    OPTIMIZER_ADAM = "adam"

class CriterionEnum(Enum):
    # Maps strings to TF loss functions
    CRITERION_MSE = "mse"
    MIN_SEQ = "min_seq"
    EXP_MSE = "exp_mse"
    GUISS = "guiss"