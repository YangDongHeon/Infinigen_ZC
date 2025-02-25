#!/bin/bash

# Check if a filename was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <python_script>"
    exit 1
fi

SCRIPT="$1"

# Extract the base name of the file (remove directory path and .py extension)
BASE=$(basename "$SCRIPT" .py)

# Set the output names based on the input file's base name
OUTPUT_PROFILE="${BASE}_profile"
OUTPUT_TEXT="output.txt"

# Run the nsys profiling command using the derived names
nsys profile -o "$OUTPUT_PROFILE" -t cuda,nvtx --force-overwrite true --stats=true python "$SCRIPT"

echo "Profiling complete. Output saved to:"
echo "  - Text output: $OUTPUT_TEXT"
echo "  - NSYS profile files: ${OUTPUT_PROFILE}.qdrep and ${OUTPUT_PROFILE}.sqlite (and related files)"

