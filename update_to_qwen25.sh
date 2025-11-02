#!/bin/bash
# Script to update all Qwen2-VL references to Qwen2.5-VL

echo "Updating all Qwen2-VL references to Qwen2.5-VL..."

# Update in markdown files
for file in *.md; do
    if [ -f "$file" ]; then
        sed -i 's/Qwen2-VL/Qwen2.5-VL/g' "$file"
        sed -i 's/Qwen\/Qwen2-VL/Qwen\/Qwen2.5-VL/g' "$file"
        echo "Updated: $file"
    fi
done

echo "All files updated!"
echo "Note: Python files were already updated manually."
