from django.contrib import admin
from .models import SequenceSet, Sequence, FeatureSequence

# Register your models here.

admin.site.register(SequenceSet)
admin.site.register(Sequence)
admin.site.register(FeatureSequence)
