from django.contrib import admin
from .models import StockData, FeatureFactoryConfig, FeatureTracker, DataSetTracker

# Register your models here.
admin.site.register(StockData)
admin.site.register(FeatureFactoryConfig)
admin.site.register(FeatureTracker)
admin.site.register(DataSetTracker)