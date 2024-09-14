from django.db import models

class FeatureFactoryConfig(models.Model):
    name = models.CharField(max_length=50)
    description = models.CharField(max_length=200)
    parameters = models.JSONField()
    version = models.CharField(max_length=10, blank=True, null=True)
    feature_names = models.JSONField(blank=True, null=True)
    class_path = models.CharField(max_length=200, blank=True, null=True)

    def __str__(self):
        return self.name    

class DataSet(models.Model):
    dataset_type = models.CharField(max_length=50)
    start_timestamp = models.DateTimeField()
    end_timestamp = models.DateTimeField()
    features = models.JSONField()
    metadata = models.JSONField()

class DataRow(models.Model):
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    features = models.JSONField()
