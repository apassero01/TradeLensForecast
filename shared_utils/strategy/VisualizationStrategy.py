from shared_utils.models import StrategyRequest
from shared_utils.strategy.BaseStrategy import Strategy, GetAttributesStrategy
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
        self.visualization_type = None

    def apply(self, entity: Entity):
        self.get_parent_attributes(entity)
        self.visualization_type = VisualizationTypeEnum(self.strategy_request.param_config.get('visualization_type'))
        data = entity.get_attribute(self.strategy_request.param_config.get('parent_data_attribute_name'))
        data = data.tolist() if isinstance(data, np.ndarray) else data
        visualization = {
            "type": self.visualization_type.value,
            "config": {},
            "data": data
        }
        entity.set_attribute('visualization', visualization)

        return self.strategy_request
    
    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        return True
    
    def get_parent_attributes(self, entity: Entity):
        parent_data_name = self.strategy_request.param_config.get('parent_data_attribute_name')
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = GetAttributesStrategy.__name__
        strategy_request.target_entity_id = entity.get_parents()[0]
        strategy_request.param_config = {
            'attribute_names': [
                parent_data_name
            ]
        }

        strategy_request = self.executor_service.execute_request(strategy_request)

        entity.set_attribute(parent_data_name, strategy_request.ret_val[parent_data_name])

    @staticmethod
    def get_request_config():
        return {
            'visualization_type': 'stockchart',
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

        array = entity.get_attribute(self.strategy_request.param_config.get('parent_data_attribute_name'))
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


        
class LineGraphStrategy(VisualizationStrategy):
    """
    A strategy for creating a line graph visualization from a 3D array of shape:
        (batch, seq_length, features).
    Outputs metadata about the array's shape to allow flexible front-end grouping.
    """
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)
        self.visualization_type = VisualizationTypeEnum.MULTILINE

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        """
        Ensure the input is a valid 3D NumPy array.
        """
        pass

    def apply(self, entity: Entity):
        # 1. Retrieve parent array and configuration
        self.get_parent_attributes(entity)
        self.verify_executable(entity, self.strategy_request)

        array_3d = entity.get_attribute(self.strategy_request.param_config.get('parent_data_attribute_name'))
        title = self.strategy_request.param_config.get('title', 'Line Graph')
        x_axis_label = self.strategy_request.param_config.get('x_axis_label', 'X Axis')
        y_axis_label = self.strategy_request.param_config.get('y_axis_label', 'Values')

        if not isinstance(array_3d, np.ndarray) or len(array_3d.shape) != 3:
            raise ValueError("Input must be a 3D NumPy array of shape (batch, seq_length, features).")

        batch_size, seq_length, num_features = array_3d.shape

        # 2. Prepare x-axis (seq_length dimension)
        x_axis = list(range(seq_length))

        # 3. Create the visualization object
        visualization = {
            "type": self.visualization_type.value,
            "config": {
                "title": title,
                "xAxisLabel": x_axis_label,
                "yAxisLabel": y_axis_label
            },
            "data": {
                "x": x_axis,
                "lines": array_3d.tolist(),  # Pass raw data as a list
                "shape": [batch_size, seq_length, num_features],  # Include original shape metadata
            }
        }

        # 4. Store the visualization in the entity
        entity.set_attribute('visualization', visualization)
        return self.strategy_request

    @staticmethod
    def get_request_config() -> dict:
        """
        Default params for the line graph strategy.
        """
        return {
            'title': 'Line Graph Title',
            'x_axis_label': 'X Axis (Seq Length)',
            'y_axis_label': 'Y Axis (Values)',
            'parent_data_attribute_name': 'X',
            'visualization_type': 'multiline'
        }

