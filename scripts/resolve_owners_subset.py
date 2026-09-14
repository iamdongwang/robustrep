"""Resolve IdentityRegistry owners only for agents whose evidence carries tx hashes (level-2 URIs).
Owners are needed solely to upgrade level 2 -> 3 (tx party match), so this is the minimal set.
Rotates across several public RPCs on failure (per-endpoint rate limits)."""
import sys, time, logging
from robustrep.config import Config
from robustrep.sources import base_erc8004 as base
from robustrep.sources.rpc import RpcClient
from robustrep.store import Store

logging.basicConfig(level=logging.WARNING)
db = sys.argv[1]
urls = ["https://mainnet.base.org", "https://base-rpc.publicnode.com", "https://1rpc.io/base", "https://base.drpc.org"]
with Store(db) as store:
    ids = [r[0] for r in store.conn.execute(
        "select distinct f.agent_id from feedback f join evidence_cache e on e.uri=f.feedback_uri "
        "where e.level=2 and not exists (select 1 from agents a where a.agent_id=f.agent_id and a.owner<>'')")]
    print(f"agents to resolve: {len(ids)}", flush=True)
    rpc = RpcClient(urls, user_agent=Config().user_agent, retries=8)
    t0 = time.time(); done = 0
    for i, agent_id in enumerate(ids, 1):
        owner = base.owner_of(rpc, agent_id)
        store.upsert_agent_owner(agent_id, owner or "")
        done += 1 if owner else 0
        if i % 50 == 0:
            print(f"{i}/{len(ids)} resolved={done} {time.time()-t0:.0f}s", flush=True)
    print(f"done: {done}/{len(ids)} owners in {time.time()-t0:.0f}s", flush=True)
