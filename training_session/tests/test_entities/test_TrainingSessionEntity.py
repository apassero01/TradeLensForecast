from django.test import TestCase
from django.utils import timezone
from training_session.models import TrainingSession
from shared_utils.models import StrategyRequest
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity
import uuid

class TestTrainingSessionEntity(TestCase):
    def setUp(self):
        """Set up test data"""
        # Create a test session
        self.entity_id = str(uuid.uuid4())
        self.test_session = TrainingSession.objects.create(
            entity_id=self.entity_id,
            created_at=timezone.now()
        )

        # Create test strategy request
        self.strategy_request = StrategyRequest.objects.create(
            strategy_name="TestStrategy",
            strategy_path="path.to.strategy",
            param_config={"param": "value"},
            training_session=self.test_session,
            parent_request=None
        )

    def test_from_db(self):
        """Test converting from database model to entity"""
        entity = TrainingSessionEntity.from_db(self.test_session)
        
        # Verify basic attributes
        self.assertEqual(entity.id, self.test_session.pk)
        self.assertEqual(entity.entity_id, self.test_session.entity_id)
        self.assertEqual(entity.created_at, self.test_session.created_at)
        
        # Verify strategy history
        self.assertEqual(len(entity.strategy_history), 1)
        strategy = entity.strategy_history[0]
        self.assertEqual(strategy.strategy_name, "TestStrategy")
        self.assertEqual(strategy.strategy_path, "path.to.strategy")
        self.assertEqual(strategy.param_config, {"param": "value"})

    def test_to_db(self):
        """Test converting from entity to database model"""
        # Create an entity
        entity = TrainingSessionEntity(entity_id=self.entity_id)
        entity.id = self.test_session.pk
        entity.created_at = timezone.now()

        # Convert to model
        model = TrainingSessionEntity.to_db(entity)
        
        # Verify basic attributes
        self.assertEqual(model.entity_id, self.entity_id)
        self.assertEqual(model.created_at, entity.created_at)
        
        # Verify strategy requests were saved
        self.assertEqual(model.strategy_requests.count(), 1)
        saved_strategy = model.strategy_requests.first()
        self.assertEqual(saved_strategy.strategy_name, "TestStrategy")
        self.assertEqual(saved_strategy.strategy_path, "path.to.strategy")
        self.assertEqual(saved_strategy.param_config, {"param": "value"})

    def test_to_db_update_existing(self):
        """Test updating existing database model"""
        # First create an entity from existing model
        entity = TrainingSessionEntity.from_db(self.test_session)
        
        # Modify entity
        new_strategy = StrategyRequestEntity()
        new_strategy.strategy_name = "UpdatedStrategy"
        new_strategy.strategy_path = "updated.path"
        new_strategy.param_config = {"updated": "config"}
        entity.add_to_strategy_history(new_strategy)
        
        # Update model
        updated_model = TrainingSessionEntity.to_db(entity, self.test_session)
        
        # Verify updates
        self.assertEqual(updated_model.strategy_requests.count(), 2)
        strategies = list(updated_model.strategy_requests.all())
        self.assertEqual(strategies[1].strategy_name, "UpdatedStrategy")
        self.assertEqual(strategies[1].strategy_path, "updated.path")
        self.assertEqual(strategies[1].param_config, {"updated": "config"})