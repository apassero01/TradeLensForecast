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

class StockData(models.Model):
    TIMEFRAME_CHOICES = [
        ('5m', '5m'),
        ('15m', '15m'),
        ('30m', '30m'),
        ('1h', '1h'),
        ('4h', '4h'),
        ('1d', '1d'),
        ('1w', '1w'),
        ('1M', '1M'),
    ]
    ticker = models.CharField(max_length=10)
    timestamp = models.DateTimeField()
    timeframe = models.CharField(max_length=3, choices=TIMEFRAME_CHOICES)
    features = models.JSONField()


class FeatureTracker(models.Model):

    features = models.JSONField()

class DataSetTracker(models.Model):
    ticker = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=3)
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    features = models.JSONField()





