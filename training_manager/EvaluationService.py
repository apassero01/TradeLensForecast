from abc import ABC

import numpy as np

from training_manager.models import Evaluation


class EvaluationService(ABC):

    def create_evaluation(self, trainer, train_or_test, index_range, pred, actual):
        eval_obj = Evaluation.objects.create(trainer=trainer,
                                  train_or_test=train_or_test,
                                  index_range=index_range,
                                  pred=pred,
                                  actual=actual)
        return eval_obj


    def get_evaluation(self, eval_obj):
        pass


    def compute_dir_accruacy(self, y_pred, y_true):
        '''
        Compute the directional accuracy of the predictions
        '''

        sign_pred = np.sign(y_pred)
        sign_true = np.sign(y_true)

        directional_match = (sign_pred == sign_true).astype(float)
        directional_accuracy = directional_match.mean(axis=0)

        return directional_accuracy

    def compute_rmse(self, y_pred, y_true):
        '''
        Compute the root mean squared error of the predictions
        '''
        return np.sqrt(np.mean((y_pred - y_true)**2))

    def compute_median_distribution(self, y_pred, y_true):
        '''
        Compute the median distribution of the predictions
        '''
        pass


class StockEvalulation(EvaluationService):

    def __init__(self, eval_obj):
        pass


class StockEvaluationPctChg(EvaluationService):
    pass


class StockEvaluationRawPrice(EvaluationService):
    pass
