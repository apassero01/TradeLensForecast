from markdown_it.rules_inline import entity

from training_session.strategy.TrainingSessionStrategy import CreateModelStageStrategy


class TrainingSessionStrategyService:
    registered_strategies = [
        CreateModelStageStrategy,
    ]
    @staticmethod
    def get_available_strategies():
        available_stragies = []
        for strategy in TrainingSessionStrategyService.registered_strategies:
            available_stragies.append({'name': strategy.name, 'config': strategy.get_request_config()})
        return available_stragies
