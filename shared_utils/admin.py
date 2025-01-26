from django.contrib import admin

from .entities.EntityModel import EntityModel
from .models import StrategyRequest
# Register your models here.

admin.site.register(StrategyRequest)
admin.site.register(EntityModel)