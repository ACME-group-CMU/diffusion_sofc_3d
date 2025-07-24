#!/bin/bash

for version in {14..15}; do
    sbatch --job-name="generate_${version}_true" \
           --output="./Outputs/generate_${version}_true_%j.out" \
           --error="./Outputs/generate_${version}_true_%j.err" \
           generate.sh $version false
done
