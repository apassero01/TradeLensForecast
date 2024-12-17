from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.VisualizationTypeEnum import VisualizationTypeEnum
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy.BaseStrategy import Strategy
import numpy as np

class VisualizationStrategy(Strategy):
    entity_type = EntityEnum.VISUALIZATION
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, entity: Entity):
        raise NotImplementedError("VisualizationStrategy apply method not implemented")
    
    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        return True
    
    def get_parent_attributes(self, entity: Entity): 
        array = entity._parent.get_attribute(self.strategy_request.param_config.get('parent_data_attribute_name'))
        entity.set_attribute('data', {'array': array})

    @staticmethod
    def get_request_config():
        return {
            'parent_data_attribute_name': 'X'
        }
    
class HistogramStrategy(VisualizationStrategy):
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)
        self.visualization_type = VisualizationTypeEnum.HISTOGRAM

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        pass
    
    def apply(self, entity: Entity):
        self.get_parent_attributes(entity)
        self.verify_executable(entity, self.strategy_request)

        array = entity.get_attribute('data').get('array')
        num_bins = self.strategy_request.param_config.get('num_bins')
        bin_width = self.strategy_request.param_config.get('bin_width')
        x_axis_label = self.strategy_request.param_config.get('x_axis_label')
        y_axis_label = self.strategy_request.param_config.get('y_axis_label')
        title = self.strategy_request.param_config.get('title')

        # Create histogram
        counts, bin_edges = np.histogram(array, bins=num_bins)

        visualization = {
            "type": VisualizationTypeEnum.HISTOGRAM.value,
            "config": {
                "title": title,
                "xAxisLabel": x_axis_label,
                "yAxisLabel": y_axis_label
            },
            "data": {
                "bins": bin_edges.tolist(),
                "counts": counts.tolist()
            }
        }

        entity.set_attribute('visualization', visualization)

        return self.strategy_request
    @staticmethod
    def get_request_config() -> dict:
        return {
            'num_bins': 10,
            'bin_width': 10,
            'x_axis_label': 'X Axis Label',
            'y_axis_label': 'Y Axis Label',
            'title': 'Histogram Title',
            'parent_data_attribute_name': 'X'
        }


        


