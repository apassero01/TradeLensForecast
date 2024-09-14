from django.contrib import admin
from .models import FeatureFactoryConfig, DataSet, DataRow

# Register your models here.
admin.site.register(FeatureFactoryConfig)
admin.site.register(DataSet)
admin.site.register(DataRow)
