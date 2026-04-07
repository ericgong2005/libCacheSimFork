#!/bin/bash

# Clear or create results.txt
> results.txt

# Loop through all .py files in the current directory
for file in *.py; do
    # Check if at least one .py file exists
    [ -e "$file" ] || continue

    echo "$file" >> results.txt
    python3 "$file" >> results.txt 2>&1
    echo "" >> results.txt
done
