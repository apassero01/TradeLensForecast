from django.db import migrations
from pgvector.django import VectorExtension

class Migration(migrations.Migration):

    dependencies = [
        ("shared_utils", "00001_initial"),   # ← point to the real initial
    ]

    operations = [
        VectorExtension(),
    ]