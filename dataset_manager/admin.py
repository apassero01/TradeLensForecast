from django.contrib import admin
from .models import StockData, FeatureFactoryConfig

# Register your models here.
admin.site.register(StockData)
admin.site.register(FeatureFactoryConfig)