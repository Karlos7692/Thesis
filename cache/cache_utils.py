import os
import re
"""
This module provides general functionality for cache validation and manipulation of cached objects
"""

DEFAULT_CACHE_DIR = os.path.join("/Users/Karl/Projects/Thesis/cache")
ID_TEMPLATE = "{algorithm}-{stock}-{start}-{end}-{best_model}-{unique_id}"
ID_BM_PATTERN_TEMPLATE = r'{algorithm}-{stock}-{start}-{end}-{best_model}-\w+.json'
ID_PATTERN_TEMPLATE = "{algorithm}-{stock}-{start}-{end}.*"
UNIQUE_ID_INDEX = 5


def get_matching_ids(algorithm, stock, start_period, end_period, best_model, cache_dir=DEFAULT_CACHE_DIR):
    pattern = ID_PATTERN_TEMPLATE.format(algorithm=algorithm, stock=stock, start=start_period, end=end_period)
    if best_model:
        pattern = ID_BM_PATTERN_TEMPLATE.format(algorithm=algorithm, stock=stock, start=start_period, end=end_period,
                                                best_model="bm" if best_model else "m")

    return [f for f in os.listdir(cache_dir) if re.match(pattern, os.path.basename(f))]


def get_new_cache_name(algorithm, stock, start_period, end_period, best_model, cache_dir=DEFAULT_CACHE_DIR):
    # Get the unique identifier for all models with the same key.
    files = get_matching_ids(algorithm, stock, start_period, end_period, best_model, cache_dir=cache_dir)
    unique_identifiers = [int(os.path.basename(f).split("-")[UNIQUE_ID_INDEX].replace(".json", "")) for f in files]
    new_id = max(unique_identifiers) + 1 if len(unique_identifiers) > 0 else 1
    return ID_TEMPLATE.format(algorithm=algorithm, stock=stock, start=start_period, end=end_period,
                              best_model="bm" if best_model else "m", unique_id=new_id)