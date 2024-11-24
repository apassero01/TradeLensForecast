from shared_utils.strategy.BaseStrategy import Strategy


class FeatureSetStrategy(Strategy):
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, feature_set_entity):
        NotImplementedError("Child classes must implement the 'apply' method.")

    def verify_executable(self, entity, strategy_request):
        raise NotImplementedError("Child classes must implement the 'verify_executable' method.")

    @staticmethod
    def get_request_config():
        return {}



