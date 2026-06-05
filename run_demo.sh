#!/usr/bin/env bash
set -e
python -m src.generate_sample_data
python -m src.train --data-dir data/sample --output-dir outputs/demo
