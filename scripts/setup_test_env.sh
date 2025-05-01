#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up local test environment...${NC}"

# Function to check if a process is running
check_process() {
    pgrep -f "$1" > /dev/null
    return $?
}

# Function to wait for a service to be ready
wait_for_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    echo -n "Waiting for $service to be ready"
    while ! curl -s "$url" > /dev/null; do
        if [ $attempt -gt $max_attempts ]; then
            echo -e "\n${RED}$service failed to start after $max_attempts attempts${NC}"
            return 1
        fi
        echo -n "."
        sleep 1
        ((attempt++))
    done
    echo -e "\n${GREEN}$service is ready${NC}"
    return 0
}

# Function to cleanup on exit
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    # Kill background processes
    pkill -f "ipfs daemon" || true
    pkill -f "anvil" || true
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set up trap for cleanup on script exit
trap cleanup EXIT

# Check and start IPFS daemon
if ! check_process "ipfs daemon"; then
    echo -e "${YELLOW}Starting IPFS daemon...${NC}"
    ipfs daemon &
    if ! wait_for_service "IPFS" "http://localhost:5001/api/v0/version"; then
        echo -e "${RED}Failed to start IPFS daemon${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}IPFS daemon is already running${NC}"
fi

# Check and start Anvil
if ! check_process "anvil"; then
    echo -e "${YELLOW}Starting Anvil...${NC}"
    # Start Anvil with default local settings
    anvil &
    if ! wait_for_service "Anvil" "http://localhost:8545"; then
        echo -e "${RED}Failed to start Anvil${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Anvil is already running${NC}"
fi

# Check if contracts are deployed
if [ ! -f ".contract_addresses" ]; then
    echo -e "${YELLOW}Deploying contracts...${NC}"
    # Deploy contracts using Foundry
    forge script script/Deploy.s.sol --rpc-url http://localhost:8545 --broadcast
    
    # Store contract addresses
    echo "Storing contract addresses..."
    CONTRACT_ADDRESS=$(cat broadcast/Deploy.s.sol/*/run-latest.json | jq -r '.transactions[0].contractAddress')
    echo "CONTRACT_ADDRESS=$CONTRACT_ADDRESS" > .contract_addresses
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to deploy contracts${NC}"
        exit 1
    fi
    echo -e "${GREEN}Contracts deployed successfully${NC}"
else
    echo -e "${GREEN}Contracts already deployed${NC}"
fi

# Load contract addresses
source .contract_addresses

# Export environment variables
export CONTRACT_ADDRESS
export IPFS_URL="http://localhost:5001/api/v0"
export NODE_URL="http://127.0.0.1:8545"

echo -e "${GREEN}Local test environment setup complete!${NC}"
echo "Contract Address: $CONTRACT_ADDRESS"
echo "IPFS URL: $IPFS_URL"
echo "Node URL: $NODE_URL"

# Keep the script running to maintain the services
echo -e "${YELLOW}Press Ctrl+C to stop the services and cleanup...${NC}"
while true; do sleep 1; done 