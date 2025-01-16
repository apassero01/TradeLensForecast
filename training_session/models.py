from django.db import models
class FeatureSet:
    def __init__(self, scaler_config, feature_list, do_fit_test, secondary_feature_list=None):
        self.scaler_config = scaler_config
        self.feature_list = feature_list
        self.secondary_feature_list = secondary_feature_list
        self.do_fit_test = do_fit_test

class TrainingSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    entity_id = models.CharField(max_length=255, null=True, blank=True)

