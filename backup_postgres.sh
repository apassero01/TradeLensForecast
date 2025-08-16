#!/bin/bash

# PostgreSQL Database Backup Script for TradeLens
# This script creates a backup of the TradeLens PostgreSQL database

# Set default values
DB_NAME="tradelens"
DB_USER="postgres"
DB_HOST="${Postgres_HOST:-localhost}"
DB_PORT="5432"
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/tradelens_backup_${TIMESTAMP}.sql"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if PostgreSQL is installed and accessible
check_postgres() {
    if ! command -v pg_dump &> /dev/null; then
        print_error "pg_dump command not found. Please install PostgreSQL client tools."
        exit 1
    fi
}

# Function to create backup directory
create_backup_dir() {
    if [ ! -d "$BACKUP_DIR" ]; then
        print_status "Creating backup directory: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
    fi
}

# Function to perform the backup
perform_backup() {
    print_status "Starting backup of database: $DB_NAME"
    print_status "Backup file: $BACKUP_FILE"
    
    # Set password from environment variable if available
    if [ -n "$Postgres_PASSWORD" ]; then
        export PGPASSWORD="$Postgres_PASSWORD"
    else
        print_warning "Postgres_PASSWORD environment variable not set. You may be prompted for password."
    fi
    
    # Perform the backup
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
            --verbose \
            --no-password \
            --clean \
            --create \
            --format=plain \
            --file="$BACKUP_FILE"
    
    # Check if backup was successful
    if [ $? -eq 0 ]; then
        print_status "Backup completed successfully!"
        print_status "Backup saved to: $BACKUP_FILE"
        
        # Show file size
        file_size=$(du -h "$BACKUP_FILE" | cut -f1)
        print_status "Backup file size: $file_size"
        
        # Compress the backup
        print_status "Compressing backup..."
        gzip "$BACKUP_FILE"
        compressed_size=$(du -h "${BACKUP_FILE}.gz" | cut -f1)
        print_status "Compressed backup saved to: ${BACKUP_FILE}.gz"
        print_status "Compressed file size: $compressed_size"
        
    else
        print_error "Backup failed!"
        exit 1
    fi
    
    # Unset password variable for security
    unset PGPASSWORD
}

# Function to clean up old backups (keep last 7 days)
cleanup_old_backups() {
    print_status "Cleaning up old backups (keeping last 7 days)..."
    find "$BACKUP_DIR" -name "tradelens_backup_*.sql.gz" -type f -mtime +7 -delete
    remaining_backups=$(find "$BACKUP_DIR" -name "tradelens_backup_*.sql.gz" -type f | wc -l)
    print_status "Remaining backups: $remaining_backups"
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST     Database host (default: localhost)"
    echo "  -d, --database DB   Database name (default: tradelens)"
    echo "  -u, --user USER     Database user (default: postgres)"
    echo "  -p, --port PORT     Database port (default: 5432)"
    echo "  --no-cleanup        Don't clean up old backups"
    echo "  --help              Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  Postgres_PASSWORD   Database password"
    echo "  Postgres_HOST       Database host (overrides default)"
    echo ""
    echo "Examples:"
    echo "  $0                              # Basic backup with defaults"
    echo "  $0 --host myserver.com          # Backup from remote host"
    echo "  $0 --no-cleanup                 # Backup without cleaning old files"
}

# Parse command line arguments
NO_CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            DB_HOST="$2"
            shift 2
            ;;
        -d|--database)
            DB_NAME="$2"
            shift 2
            ;;
        -u|--user)
            DB_USER="$2"
            shift 2
            ;;
        -p|--port)
            DB_PORT="$2"
            shift 2
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
print_status "TradeLens PostgreSQL Backup Script"
print_status "Database: $DB_NAME on $DB_HOST:$DB_PORT"
print_status "User: $DB_USER"

# Run all functions
check_postgres
create_backup_dir
perform_backup

if [ "$NO_CLEANUP" = false ]; then
    cleanup_old_backups
fi

print_status "Backup process completed!"

