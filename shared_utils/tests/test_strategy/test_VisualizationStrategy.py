from xxlimited import Error

import numpy as np
from django.test import TestCase

from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.entities.Entity import Entity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.VisualizationTypeEnum import VisualizationTypeEnum
from shared_utils.entities.VisualizationEntity import VisualizationEntity

# Import the HistogramStrategy
from shared_utils.strategy.VisualizationStrategy import HistogramStrategy


class ParentDataEntity(Entity):
    """A parent entity that holds the data array."""

    def __init__(self, entity_id=None):
        super().__init__(entity_id)
        # By default, let's assume it doesn't have 'array' set; we can set it in test methods as needed.
        # self.set_attribute('array', [])


class HistogramStrategyTestCase(TestCase):
    def setUp(self):
        self.executor = StrategyExecutor()
        # If your setup requires registration, uncomment and adjust accordingly:
        # self.executor.register_strategy("HistogramStrategy", HistogramStrategy)

        # Create a parent entity that will hold the data
        self.parent_data_entity = ParentDataEntity()

        # Create a visualization entity and link it as a child of the parent_data_entity
        self.vis_entity = VisualizationEntity()
        self.parent_data_entity.add_child(self.vis_entity)

    def create_strategy_request(self, param_config=None):
        """Helper to create a fresh HistogramStrategy request."""
        request = StrategyRequestEntity()
        request.strategy_name = "HistogramStrategy"
        if param_config is None:
            # Use default config provided by HistogramStrategy
            config = HistogramStrategy.get_request_config()
        else:
            config = param_config
        request.param_config = config
        return request

    def test_histogram_creation(self):
        """Test creating a histogram visualization with valid data from the parent entity."""
        # Set array data in the parent entity
        test_array = [1, 2, 3, 4, 5, 6]
        self.parent_data_entity.set_attribute('array', test_array)

        # Create request and strategy, specifying the attribute name in the parent
        strategy_request = self.create_strategy_request({
            'num_bins': 3,
            'x_axis_label': 'Custom X',
            'y_axis_label': 'Custom Y',
            'title': 'Custom Histogram',
            'parent_data_attribute_name': 'array'
        })
        strategy = HistogramStrategy(self.executor, strategy_request)

        # Apply the strategy to the visualization entity
        result = strategy.apply(self.vis_entity)

        # Verify the visualization attribute is set on the visualization entity
        visualization = self.vis_entity.get_attribute('visualization')
        self.assertIsNotNone(visualization)
        self.assertEqual(visualization['type'], VisualizationTypeEnum.HISTOGRAM.value)
        self.assertEqual(visualization['config']['title'], 'Custom Histogram')
        self.assertEqual(visualization['config']['xAxisLabel'], 'Custom X')
        self.assertEqual(visualization['config']['yAxisLabel'], 'Custom Y')

        # Check that histogram bins and counts were computed
        bins = visualization['data']['bins']
        counts = visualization['data']['counts']
        self.assertEqual(len(bins), 4)  # For 3 bins, we should have 4 bin edges
        self.assertEqual(len(counts), 3)  # counts should match the number of bins

        # Ensure result is the same StrategyRequestEntity passed in
        self.assertEqual(result, strategy_request)

    def test_default_config(self):
        """Test that default configuration from get_request_config works."""
        test_array = [10, 20, 30, 40, 50]
        # Set the 'array' attribute on the parent entity
        self.parent_data_entity.set_attribute('array', test_array)

        # No custom param_config passed, so defaults should be used
        # However, we must ensure the default 'parent_data_attribute_name' matches what's in the parent entity.
        strategy_request = self.create_strategy_request()
        default_config = HistogramStrategy.get_request_config()
        if default_config.get('parent_data_attribute_name') != 'array':
            # Update the request's config to point to 'array' in the parent entity
            default_config['parent_data_attribute_name'] = 'array'
            strategy_request.param_config = default_config

        strategy = HistogramStrategy(self.executor, strategy_request)
        strategy.apply(self.vis_entity)

        visualization = self.vis_entity.get_attribute('visualization')
        self.assertEqual(visualization['config']['title'], 'Histogram Title')
        self.assertEqual(visualization['config']['xAxisLabel'], 'X Axis Label')
        self.assertEqual(visualization['config']['yAxisLabel'], 'Y Axis Label')
        # With default num_bins = 10, ensure we have 11 bin edges
        self.assertEqual(len(visualization['data']['bins']), 11)
        self.assertEqual(len(visualization['data']['counts']), 10)