from django.db import models
from django.contrib.postgres.fields import ArrayField
from training_session.models import TrainingSession


# Create your models here.
class Trainer(models.Model):
    model_params = models.JSONField()
    model_weights_dir = models.CharField(max_length=100)
    training_session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)


class Evaluation(models.Model):
    trainer = models.ForeignKey(Trainer, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    train_or_test = models.CharField(max_length=10)
    index_range = models.JSONField(default=None, null=True)
    pred = ArrayField(ArrayField(models.FloatField()))
    actual = ArrayField(ArrayField(models.FloatField()))
    std_pred = ArrayField(ArrayField(models.FloatField()), default=None, null=True)
    general_metrics = models.JSONField(default=None, null=True)
    sequence_metrics = models.JSONField(default=None, null=True)
    sequence_ids = ArrayField(models.IntegerField(), default=None, null=True)