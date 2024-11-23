from shared_utils.strategy.BaseStrategy import Strategy


class ModelStrategy(Strategy):
    '''
    The ModelStrategy class is a base class for all strategies that are

    ModelStrategy components can manipulate data within the Model.

    For example, a Trainer strategy could create a new model.
    Or a Trainer strategy could perform a fit operation on the model.
    '''
    def __init__(self, strategy_request):
        super().__init__(strategy_request)

    def apply(self, data_object_name):
        pass

