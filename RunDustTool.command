#!/bin/bash

# Go to the folder where this script lives
cd "$(dirname "$0")"

# Run the Python script
/usr/bin/env python3 dust_analysis.py

echo
echo "Done. Outputs are in the 'results' folder."
echo "Press any key to close this window..."
read -n 1 -s
