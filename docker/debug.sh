#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}TradeLens Docker Debug Setup${NC}"
echo "=================================="

# Check if PyCharm is running (basic check)
if ! pgrep -f "pycharm" > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: PyCharm doesn't appear to be running${NC}"
    echo "Make sure to:"
    echo "1. Open PyCharm"
    echo "2. Configure Python Debug Server (port 5678)"
    echo "3. Start the debug server before running this script"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if debug port is available
if lsof -Pi :5678 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Debug port 5678 is in use (likely PyCharm debug server)${NC}"
else
    echo -e "${RED}✗ Debug port 5678 is not in use${NC}"
    echo "Please start PyCharm debug server first!"
    exit 1
fi

# Navigate to docker directory
cd "$(dirname "$0")"

echo -e "${BLUE}Starting TradeLens in debug mode...${NC}"
echo "Debug features enabled:"
echo "  • PyCharm remote debugging"
echo "  • Single-threaded execution"
echo "  • Live code reloading"
echo "  • Detailed logging"
echo ""

# Build and run with debug configuration
docker-compose -f docker-compose.yml -f docker-compose.debug.yml down 2>/dev/null || true
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up --build

echo -e "${GREEN}Debug session ended${NC}" 