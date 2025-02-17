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

    def __init__(self, strategy_executor: 'StrategyExecutor', strategy_request: 'StrategyRequestEntity'):
        super().__init__(strategy_executor, strategy_request)
        self.visualization_type = None

    def apply(self, entity: 'Entity'):
        # Retrieve the necessary attributes from the parent entity.
        self.get_parent_attributes(entity)

        # Set the visualization type (e.g. "stockchart", etc.)
        self.visualization_type = VisualizationTypeEnum(self.strategy_request.param_config.get('visualization_type'))

        # Build the data dictionary.
        # First try the new key "parent_data_attributes" (which may be a dict or a single string),
        # then fall back to the older "parent_data_attribute_name" key.
        data = {}
        if 'parent_data_attributes' in self.strategy_request.param_config:
            attr_config = self.strategy_request.param_config['parent_data_attributes']
            if isinstance(attr_config, dict):
                # For each mapping, the key is the parent's attribute name and the value is
                # the key to use in the data dictionary.
                for parent_attr, data_key in attr_config.items():
                    value = entity.get_attribute(parent_attr)
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    data[data_key] = value
            else:
                # If the provided value is not a dict, assume it is a single attribute name.
                for attr in attr_config:
                    value = entity.get_attribute(attr)
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    data[attr] = value
        elif 'parent_data_attribute_name' in self.strategy_request.param_config:
            # Backward compatibility: use the old key.
            parent_attr = self.strategy_request.param_config['parent_data_attribute_name']
            value = entity.get_attribute(parent_attr)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            data[parent_attr] = value

        visualization = {
            "type": self.visualization_type.value,
            "config": {},
            "data": data
        }
        entity.set_attribute('visualization', visualization)
        return self.strategy_request

    def verify_executable(self, entity: 'Entity', strategy_request: 'StrategyRequestEntity'):
        # You can add further verification logic here if needed.
        return True

    def get_parent_attributes(self, entity: 'Entity'):
        """
        Retrieve the parent attributes from the parent entity.
        If the configuration uses the new key "parent_data_attributes", we expect either:
          - A dict: in which case we retrieve all keys of that dict.
          - A single string.
        Otherwise, if "parent_data_attribute_name" is used, we retrieve that single attribute.
        """
        parent_attrs = []
        param_config = self.strategy_request.param_config
        if 'parent_data_attributes' in param_config:
            attr_config = param_config['parent_data_attributes']
            if isinstance(attr_config, dict):
                parent_attrs = list(attr_config.keys())
            else:
                # If it's not a dict, assume it is a single attribute name.
                parent_attrs = attr_config
        elif 'parent_data_attribute_name' in param_config:
            parent_attrs = [param_config['parent_data_attribute_name']]

        if not parent_attrs:
            return  # Nothing to retrieve.

        # Create a request to retrieve all the attributes from the parent.
        parent_request = StrategyRequestEntity()
        parent_request.strategy_name = GetAttributesStrategy.__name__
        # Assume the parent entity id is the first in the list returned by entity.get_parents()
        parent_request.target_entity_id = entity.get_parents()[0]
        parent_request.param_config = {
            'attribute_names': parent_attrs
        }
        parent_request = self.executor_service.execute_request(parent_request)

        # Set each retrieved attribute on the current entity.
        for attr in parent_attrs:
            entity.set_attribute(attr, parent_request.ret_val[attr])

    @staticmethod
    def get_request_config():
        # For backward compatibility, the default is still a single attribute.
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

        if isinstance(array_3d, list):
            array_3d = np.array(array_3d)

        # Ensure the array is at least 3D.
        array_3d = np.atleast_3d(array_3d)

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

