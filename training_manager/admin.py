from django.contrib import admin

# Register your models here.
from .models import TrainingSession
from .models import Trainer
from .models import Evaluation
from .models import FeatureSet


admin.site.register(TrainingSession)
admin.site.register(Trainer)
admin.site.register(Evaluation)
admin.site.register(FeatureSet)

