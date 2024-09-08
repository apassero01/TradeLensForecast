from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.
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
class StockSequence(models.Model):

    ticker = models.CharField(max_length=10)
    start_timestamp = models.DateTimeField()
    end_timestamp = models.DateTimeField()
    sequence_length = models.IntegerField()
    sequence_data = ArrayField(ArrayField(models.FloatField()))
    timeframe = models.CharField(max_length=3, choices=TIMEFRAME_CHOICES)


class FeatureDict(models.Model):

    ticker = models.CharField(max_length=10)
    feature_dict = models.JSONField()
    timeframe = models.CharField(max_length=3, choices=TIMEFRAME_CHOICES)

class SequenceSetTracker(models.Model):
    ticker = models.CharField(max_length=10)
    sequence_length = models.IntegerField()
    timeframe = models.CharField(max_length=3, choices=TIMEFRAME_CHOICES)
    start_timestamp = models.DateTimeField()
    end_timestamp = models.DateTimeField()


    def __str__(self):
        return f"{self.ticker} - {self.sequence_length} - {self.timeframe}"