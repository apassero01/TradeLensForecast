#!/bin/bash

# Quick PostgreSQL backup script for TradeLens
# Simple one-liner backup without advanced features

DB_NAME="tradelens"
DB_USER="postgres"
DB_HOST="${Postgres_HOST:-localhost}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="tradelens_backup_${TIMESTAMP}.sql"

echo "Creating quick backup of TradeLens database..."

# Set password from environment if available
export PGPASSWORD="${Postgres_PASSWORD:-postgres}"

# Perform backup
pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" > "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Backup completed: $BACKUP_FILE"
    echo "📁 File size: $(du -h "$BACKUP_FILE" | cut -f1)"
    echo "🗜️  Compressing..."
    gzip "$BACKUP_FILE"
    echo "✅ Compressed backup: ${BACKUP_FILE}.gz"
    echo "📁 Compressed size: $(du -h "${BACKUP_FILE}.gz" | cut -f1)"
else
    echo "❌ Backup failed!"
    exit 1
fi

unset PGPASSWORD
