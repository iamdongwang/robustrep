"""Rater profiling via Blockscout v2 at a conservative rate (public limit is ~1 req/s sustained)."""
import logging, sys
from robustrep.sources.rater_profile import BlockscoutV2Client, enrich_raters
from robustrep.store import Store
logging.basicConfig(level=logging.WARNING)
rps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
with Store(sys.argv[1]) as store:
    n = enrich_raters(store, BlockscoutV2Client(rps=rps, retries=5))
    print(f"raters profiled: {n}; rater profile mode: {store.get_sync('rater_profile_mode')}", flush=True)
