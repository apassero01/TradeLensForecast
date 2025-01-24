from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('shared_utils', '0011_rename_entity_models_strategyrequest_entity_model_and_more'),  # Replace with the actual latest migration
    ]

    operations = [
        # Drop the foreign key constraint on `parent_request_id` (if it exists)
        migrations.RunSQL(
            "ALTER TABLE shared_utils_strategyrequest DROP CONSTRAINT IF EXISTS shared_utils_strategyrequest_parent_request_id_fkey;"
        ),
        # Change `parent_request_id` from bigint to uuid
        migrations.RunSQL(
            "ALTER TABLE shared_utils_strategyrequest ALTER COLUMN parent_request_id TYPE uuid USING parent_request_id::uuid;"
        ),
        # Re-add the foreign key constraint with the correct type
        migrations.AddField(
            model_name='strategyrequest',
            name='parent_request',
            field=models.ForeignKey(
                to='shared_utils.strategyrequest',
                on_delete=models.CASCADE,
                null=True,
                blank=True,
                related_name='nested_requests',
                to_field='entity_id',
            ),
        ),
    ]