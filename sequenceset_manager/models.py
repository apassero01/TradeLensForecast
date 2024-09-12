from django.db import models
from django.contrib.postgres.fields import ArrayField

class SequenceSet(models.Model):
    dataset_type = models.CharField(max_length=10)
    sequence_length = models.IntegerField()
    start_timestamp = models.DateTimeField()
    end_timestamp = models.DateTimeField()
    feature_dict = models.JSONField()
    metadata = models.JSONField()



class Sequence(models.Model):
    sequence_set = models.ForeignKey(SequenceSet, on_delete=models.CASCADE)
    start_timestamp = models.DateTimeField()
    end_timestamp = models.DateTimeField()
    sequence_length = models.IntegerField()
    sequence_data = ArrayField(ArrayField(models.FloatField()))
