from django.contrib import admin
from .models import StockSequence, FeatureDict, SequenceSetTracker

# Register your models here.

admin.site.register(StockSequence)
admin.site.register(FeatureDict)
admin.site.register(SequenceSetTracker)
