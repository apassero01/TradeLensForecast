from unittest.mock import patch, MagicMock

from django.test import TestCase

from training_session.models import ModelSet
from training_session.services.StrategyService import ModelSetStrategyService
from training_session.services.TrainingSessionService import TrainingSessionService


class StrategyServiceTestCase(TestCase):

    def setUp(self):
        # Sample ordered strategies for testing
        self.ordered_strategies = [
            {"name": "CreateFeatureSets", "config": {"step_number": 0, "is_applied": True}},
            {"name": "ScaleByFeatures", "config": {"step_number": 1, "is_applied": True}},
            {"name": "TrainTestSplitDate", "config": {"step_number": 2, "is_applied": True}},
        ]

    def test_strategy_hist_diff_with_existing_strategy(self):
        """
        Test that strategy_hist_diff returns the correct step_number for a matching strategy.
        """
        new_strategy = {"name": "ScaleByFeatures", "config": {}}
        result = ModelSetStrategyService.strategy_hist_diff(self.ordered_strategies, new_strategy)
        self.assertEqual(result, 1, "Expected step_number 1 for strategy with name 'ScaleByFeatures'")

    def test_strategy_hist_diff_with_nonexistent_strategy(self):
        """
        Test that strategy_hist_diff returns None when no matching strategy name is found.
        """
        new_strategy = {"name": "NonExistentStrategy", "config": {}}
        result = ModelSetStrategyService.strategy_hist_diff(self.ordered_strategies, new_strategy)
        self.assertIsNone(result, "Expected None for a non-existent strategy name")

    def test_amend_strategy_hist_replace_strategy_and_deactivate_subsequent(self):
        """
        Test amend_strategy_hist to replace a strategy at a given step_number and deactivate subsequent strategies.
        """
        # New strategy to replace step_number 1
        new_strategy = {"name": "NewScaleFeature", "config": {"step_number": 1, "is_applied": True}}

        # Apply amend_strategy_hist
        updated_strategies = ModelSetStrategyService.amend_strategy_hist(
            self.ordered_strategies, new_strategy, step_number=1
        )

        # Verify that the strategy at step_number 1 is replaced
        self.assertEqual(updated_strategies[1], new_strategy, "Expected new strategy to replace step_number 1")

        # Verify that subsequent strategies have is_applied set to False
        self.assertFalse(updated_strategies[2]["config"]["is_applied"],
                         "Expected is_applied to be False for strategies after step_number 1")


    def test_apply_model_set_strategy(self):
        training_session_service = TrainingSessionService()
        session = training_session_service.create_training_session()
        session.X_features = ['feature1', 'feature2']
        session.y_features = ['feature3']

        model_set = ModelSet()
        model_set.X = [[1, 2], [3, 4]]
        model_set.y = [[5], [6]]

        session.model_sets = [model_set]

        config = {
            'm_service': 'training_session',
            'type': 'CreateFeatureSetsStrategy',
            'parent_strategy': 'ModelSetsStrategy',
            'feature_set_configs': [
                {
                    'scaler_config': {
                        'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                    },
                    'feature_list': ['feature1', 'feature2'],
                    'do_fit_test': False,
                    'feature_set_type': 'X'
                },
                {
                    'scaler_config': {
                        'scaler_name': 'MEAN_VARIANCE_SCALER_3D',
                    },
                    'feature_list': ['feature3'],
                    'do_fit_test': False,
                    'feature_set_type': 'y'
                }
            ]
        }

        strategy_json = {
            "name": "CreateFeatureSets",
            "config": config
        }

        session, _ = ModelSetStrategyService.apply_model_set_strategy(session, strategy_json)

        self.assertEqual(len(session.model_sets), 1)
        self.assertEqual(len(session.ordered_model_set_strategies), 1)
        self.assertEqual(session.ordered_model_set_strategies[0]['config']['step_number'], 0)

        self.assertEqual(len(session.model_sets[0].X_feature_sets), 1)
        self.assertEqual(len(session.model_sets[0].y_feature_sets), 1)

        self.assertEqual(session.model_sets[0].X_feature_sets[0].feature_list, ['feature1', 'feature2'])
        self.assertEqual(session.model_sets[0].y_feature_sets[0].feature_list, ['feature3'])


    @patch('training_session.services.StrategyService.ModelSetStrategyService.get_strategy_instance')
    def test_apply_model_set_strategy_replace_middle_strategy(self, mock_get_strategy_instance):
        """
        Test applying a new strategy that replaces a middle strategy, confirming
        that subsequent strategies have is_applied set to False and the amended
        strategy has is_applied set to True.
        """
        training_session_service = TrainingSessionService()
        self.session = training_session_service.create_training_session()

        # Initial ordered strategies history
        self.session.ordered_model_set_strategies = [
            {"name": "InitialStrategy", "config": {"step_number": 0, "is_applied": True}},
            {"name": "MidStrategy", "config": {"step_number": 1, "is_applied": True}},
            {"name": "FinalStrategy", "config": {"step_number": 2, "is_applied": True}}
        ]

        new_strategy_json = {
            "name": "MidStrategy",
            "config": {
                "step_number": 1,
                "is_applied": True,
                "additional_config": "new_config_value"
            }
        }

        # Mock the `get_strategy_instance` method to return a mock strategy with the defined config
        mock_strategy = MagicMock()
        mock_strategy.config = new_strategy_json["config"]  # Set config to match new_strategy_json
        mock_strategy.apply.return_value = []  # Mock apply method to prevent data transformations
        mock_get_strategy_instance.return_value = mock_strategy

        # Apply the new strategy
        session, _ = ModelSetStrategyService.apply_model_set_strategy(self.session, new_strategy_json)

        # Verify that the history is updated correctly
        # The amended strategy at step_number 1 should have is_applied set to True
        self.assertEqual(session.ordered_model_set_strategies[1]["config"]["is_applied"], True)

        # All strategies after step_number 1 should have is_applied set to False
        for strategy in session.ordered_model_set_strategies[2:]:
            self.assertFalse(strategy["config"]["is_applied"], f"Expected is_applied to be False for {strategy['name']}")

        # Verify that the number of strategies has not changed
        self.assertEqual(len(session.ordered_model_set_strategies), 3)

        # Ensure that `get_strategy_instance` was called with the correct populated configuration
        mock_get_strategy_instance.assert_called_once_with(new_strategy_json["config"])

        # Verify that apply was called on the mock strategy
        mock_strategy.apply.assert_called_once_with(self.session.model_sets)

    @patch('training_session.services.StrategyService.ModelSetStrategyService.get_strategy_instance')
    def test_apply_model_set_strategy_with_final_strategy(self, mock_get_strategy_instance):
        """
        Test applying a new strategy that is marked as final, confirming that the
        session attributes are updated correctly and the strategy is applied.
        """
        training_session_service = TrainingSessionService()
        self.session = training_session_service.create_training_session()

        # Initial ordered strategies history
        self.session.ordered_model_set_strategies = [
            {"name": "InitialStrategy", "config": {"step_number": 0, "is_applied": True}},
            {"name": "MidStrategy", "config": {"step_number": 1, "is_applied": True}},
            {"name": "FinalStrategy", "config": {"step_number": 2, "is_applied": True}}
        ]

        new_strategy_json = {
            "name": "FinalStrategy",
            "config": {
                "step_number": 2,
                "is_applied": True,
                "additional_config": "new_config_value",
                "is_final": True
            }
        }

        # Mock the `get_strategy_instance` method to return a mock strategy with the defined config
        mock_strategy = MagicMock()
        mock_strategy.config = new_strategy_json["config"]
        mock_strategy.apply.return_value = ([1], [2], [3], [4], [5], [6])

        # Mock the `get_strategy_instance` method to return a mock strategy with the defined config
        mock_strategy = MagicMock()
        mock_strategy.config = new_strategy_json["config"]  # Set config to match new_strategy_json
        mock_strategy.apply.return_value = [[1],[2],[3],[4],[5],[6]]  # Mock apply method to prevent data transformations
        mock_get_strategy_instance.return_value = mock_strategy

        # Apply the new strategy
        session, _ = ModelSetStrategyService.apply_model_set_strategy(self.session, new_strategy_json)
        self.assertListEqual(session.X_train, [1])
        self.assertListEqual(session.X_test, [2])
        self.assertListEqual(session.y_train, [3])
        self.assertListEqual(session.y_test, [4])
        self.assertListEqual(session.train_row_ids, [5])
        self.assertListEqual(session.test_row_ids, [6])


    def test_populate_strategy_config(self):
        training_session_service = TrainingSessionService()
        session = training_session_service.create_training_session()
        session.X_features = ['feature1', 'feature2']
        session.y_features = ['feature3']
        session.ordered_model_set_strategies = ['strategy1', 'strategy2']

        config = {
            'X_features': None,
            'y_features': None,
        }

        config = ModelSetStrategyService.populate_strategy_config(session, config)
        self.assertDictEqual(config, {'X_features': ['feature1', 'feature2'], 'y_features': ['feature3']})