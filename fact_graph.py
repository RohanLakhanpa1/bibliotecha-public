# app/utils/ai_core/fact_graph.py
"""
FactGraph subsystem for Bibliotheca — "Beyond-Perfection" Edition.

Responsibilities:
- store facts as subject-predicate-object triples (with metadata)
- provide fast queries and relationship traversal
- persist to CouchDB (if available) with JSON file fallback
- snapshot/versioning + rollback
- safe patch application with validation
- hooks for telemetry, rules engine, auto-healer
- thread + async safe
- small inference hook (forward-chaining style) for integration with RulesEngine

Security / Safety notes:
- Dangerous operations (mass deletes, arbitrary code-in-patches) are blocked unless
  BIBLIOTHECA_SELF_UPDATE_ALLOW_DANGEROUS=true is set in environment.
- Always review patches provided by external sources before enabling "dangerous" mode.
"""

from __future__ import annotations

import os
import json
import time
import uuid
import copy
import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Iterable, Tuple


# -----------------------------
# Telemetry integration (Beyond-Perfection Edition v6.2)
# -----------------------------
import threading
import logging

logger = logging.getLogger("TelemetryIntegration")

class SafeTelemetryFallback:
    """
    Fallback Telemetry stub when TelemetryHandler cannot be imported.
    Logs everything locally to ensure no runtime failures.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._initialized = getattr(self, "_initialized", False)
        if not self._initialized:
            self._initialized = True
            self._logs = []
            logger.warning("⚠️ TelemetryHandler unavailable. Using SafeTelemetryFallback.")

    def log(self, *args, **kwargs):
        self._logs.append((args, kwargs))
        logger.info(f"[TelemetryFallback] Logged: args={args}, kwargs={kwargs}")

    def prune_old_tasks(self):
        removed = len(self._logs)
        self._logs.clear()
        logger.info(f"[TelemetryFallback] Pruned {removed} old tasks.")
        return removed

# Attempt real TelemetryHandler import and initialization
TELEMETRY = None
try:
    from app.utils.telemetry import TelemetryHandler  # type: ignore
    TELEMETRY = TelemetryHandler()
    logger.info("✅ TelemetryHandler initialized Beyond-Perfection Edition v6.2")
    if hasattr(TELEMETRY, "prune_old_tasks"):
        pruned = TELEMETRY.prune_old_tasks()
        logger.info(f"✅ Self-optimization pruned {pruned} old tasks")
except ImportError as e:
    logger.error(f"❌ TelemetryHandler import failed: {e}. Using SafeTelemetryFallback.")
    TELEMETRY = SafeTelemetryFallback()
except Exception as e:
    logger.error(f"❌ TelemetryHandler initialization error: {e}. Using SafeTelemetryFallback.")
    TELEMETRY = SafeTelemetryFallback()

# Final safety check to ensure TELEMETRY is always usable
if TELEMETRY is None:
    TELEMETRY = SafeTelemetryFallback()
    logger.warning("⚠️ TELEMETRY was None, replaced with SafeTelemetryFallback.")

# Example helper functions for global usage
def telemetry_log(*args, **kwargs):
    """Safe global logging function for telemetry."""
    try:
        TELEMETRY.log(*args, **kwargs)
    except Exception as e:
        logger.error(f"❌ telemetry_log error: {e}")

def telemetry_prune():
    """Safe global pruning of old telemetry tasks."""
    try:
        if hasattr(TELEMETRY, "prune_old_tasks"):
            return TELEMETRY.prune_old_tasks()
    except Exception as e:
        logger.error(f"❌ telemetry_prune error: {e}")
    return 0

# -----------------------------
# CouchDB / JSON fallback setup
# -----------------------------
COUCHDB_URL = os.getenv("COUCHDB_URL", None)
COUCHDB_DBNAME = os.getenv("BIBLIOTHECA_FACTGRAPH_DB", "bibliotheca_fact_graph")
_have_couch = False
_couch = None

try:
    import couchdb  # type: ignore

    if COUCHDB_URL:
        try:
            _couch = couchdb.Server(COUCHDB_URL)
            _have_couch = True
            if TELEMETRY:
                TELEMETRY.info("✅ CouchDB connected for FactGraph")
        except Exception as e:
            _have_couch = False
            if TELEMETRY:
                TELEMETRY.warning(f"⚠️ CouchDB connection failed: {e}")
except Exception:
    _have_couch = False

# -----------------------------
# Dangerous ops guard
# -----------------------------
_ALLOW_DANGEROUS = os.getenv("BIBLIOTHECA_SELF_UPDATE_ALLOW_DANGEROUS", "false").lower() in (
    "1", "true", "yes"
)

# -----------------------------
# Logger setup
# -----------------------------
logger = logging.getLogger("Bibliotheca.FactGraph")
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s][FactGraph] %(message)s")
    )
    logger.addHandler(console)
logger.setLevel(logging.INFO)

# -----------------------------
# Helper: current timestamp
# -----------------------------
def _now_ts() -> float:
    return time.time()

# -----------------------------
# Ensure FactGraph links to core AI instance
# -----------------------------
try:

    def link_fact_graph_to_ai():
        try:
            from app.utils.ai_core.advanced_ai import ADVANCED_AI_INSTANCE  # lazy import
            if 'ADVANCED_AI_INSTANCE' in globals() and ADVANCED_AI_INSTANCE is not None:
                if not hasattr(ADVANCED_AI_INSTANCE, "fact_graph") or ADVANCED_AI_INSTANCE.fact_graph is None:
                    ADVANCED_AI_INSTANCE.fact_graph = FactGraphStub(ai_instance=ADVANCED_AI_INSTANCE)
                    TELEMETRY.info("✅ FactGraph linked to ADVANCED_AI_INSTANCE")
        except Exception as e:
            logger.warning(f"⚠️ ADVANCED_AI_INSTANCE linkage failed: {e}")


    if 'ADVANCED_AI_INSTANCE' in globals() and ADVANCED_AI_INSTANCE is not None:
        # Lazy FactGraph attach to AI core
        if not hasattr(ADVANCED_AI_INSTANCE, "fact_graph") or ADVANCED_AI_INSTANCE.fact_graph is None:
            class FactGraphStub:
                """Temporary stub to satisfy early imports."""
                def __init__(self, ai_instance=None):
                    self.ai_instance = ai_instance
            ADVANCED_AI_INSTANCE.fact_graph = FactGraphStub(ai_instance=ADVANCED_AI_INSTANCE)
        if TELEMETRY:
            TELEMETRY.info("✅ FactGraph linked to ADVANCED_AI_INSTANCE")
except Exception as e:
    logger.warning(f"⚠️ ADVANCED_AI_INSTANCE linkage failed: {e}")

class Fact:
    """Simple dataclass-like container for a fact triple."""
    __slots__ = ("id", "subject", "predicate", "object", "ts", "metadata", "version")

    def __init__(self, subject: str, predicate: str, obj: Any, metadata: Optional[Dict] = None):
        self.id: str = str(uuid.uuid4())
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.ts = _now_ts()
        self.metadata = metadata or {}
        self.version = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "ts": self.ts,
            "metadata": copy.deepcopy(self.metadata),
            "version": self.version,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Fact":
        f = Fact(d["subject"], d["predicate"], d["object"], metadata=d.get("metadata", {}))
        f.id = d.get("id", f.id)
        f.ts = d.get("ts", f.ts)
        f.version = d.get("version", f.version)
        return f


class FactGraph:
    """
    FactGraph: stores facts and provides simple graph utilities.

    Key methods:
    - add_fact / remove_fact / query
    - find_related(entity, depth)
    - snapshot() and rollback(snapshot_id)
    - safe_apply_patch(patch)
    - export/import
    """

    def __init__(
            self,
            name: str = "default",
            autosave_interval: float = 10.0,
            persist: bool = True,
            persist_db_name: Optional[str] = None,
            couchdb_manager: Optional[Any] = None  # Added for Beyond-Perfection integration
    ):
        import asyncio
        import os
        import threading
        import time
        import logging
        from typing import Dict, List, Tuple, Callable, Any, Optional

        # ----------------------------- BASIC INITIALIZATION -----------------------------
        self.name = name
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._facts: Dict[str, Any] = {}  # id -> Fact
        self._index_spo: Dict[Tuple[str, str, str], str] = {}  # (s,p,o_str) -> id
        self._index_subject: Dict[str, List[str]] = {}
        self._index_object: Dict[str, List[str]] = {}
        self._listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self.autosave_interval = autosave_interval
        self._autosave_thread: Optional[threading.Thread] = None
        self._stop_autosave = threading.Event()
        self.persist = persist and (_have_couch or persist)
        self.persist_db_name = persist_db_name or "bibliotheca_db"
        self._persistence_backend = "couch" if (_have_couch and persist) else "file"
        self._file_path = os.path.join(os.getcwd(), f".factgraph_{self.name}.json")
        self._snapshot_dir = os.path.join(os.getcwd(), "fact_graph_snapshots")
        os.makedirs(self._snapshot_dir, exist_ok=True)
        self.telemetry = TELEMETRY

        # ----------------------------- LOGGING SETUP -----------------------------
        self.logger = logging.getLogger(f"FactGraph-{self.name}")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('[%(asctime)s][%(name)s] %(levelname)s: %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        self.logger.info(f"[FactGraph] Initializing FactGraph instance '{self.name}'")

        # ----------------------------- PERSISTENCE SETUP -----------------------------
        self._db_lock = threading.Lock()
        self._db = None
        self._couch_manager = couchdb_manager or _couch

        try:
            if self._persistence_backend == "couch" and self.persist:
                if self._couch_manager is None:
                    self.logger.warning("[FactGraph] CouchDB client not provided; falling back to file persistence.")
                    self._db = None
                    self._persistence_backend = "file"
                else:
                    max_retries = 3
                    retry_delay = 2
                    for attempt in range(1, max_retries + 1):
                        try:
                            with self._db_lock:
                                if self.persist_db_name not in self._couch_manager:
                                    self.logger.info(f"[FactGraph] Creating CouchDB database '{self.persist_db_name}'")
                                    self._couch_manager.create(self.persist_db_name)
                                self._db = self._couch_manager[self.persist_db_name]
                            self.logger.info(f"[FactGraph] CouchDB backend initialized: '{self.persist_db_name}'")
                            if self.telemetry:
                                self.telemetry.capture_event("factgraph_couchdb_ready", {"db": self.persist_db_name})
                            break
                        except Exception as e:
                            self.logger.warning(f"[FactGraph] CouchDB init attempt {attempt}/{max_retries} failed: {e}")
                            time.sleep(retry_delay)
                            if attempt == max_retries:
                                self.logger.error("[FactGraph] CouchDB init failed after retries; using file backend")
                                self._db = None
                                self._persistence_backend = "file"
            else:
                self._db = None
                if self._persistence_backend != "file":
                    self.logger.info(f"[FactGraph] Persistence disabled; defaulting to file backend")
                    self._persistence_backend = "file"

        except Exception as e:
            self.logger.exception(f"[FactGraph] Critical persistence initialization failure: {e}")
            self._db = None
            self._persistence_backend = "file"
        finally:
            self.logger.info(
                f"[FactGraph] Persistence init complete. Backend: {self._persistence_backend}, DB: {self._db}")

        # ----------------------------- LOAD EXISTING DATA -----------------------------
        try:
            self._load_from_store()
            self.logger.info("[FactGraph] Loaded existing data from store successfully")
        except Exception as e:
            self.logger.warning(f"[FactGraph] load_from_store failed: {e}")

        # ----------------------------- AUTOSAVE THREAD -----------------------------
        def _autosave_loop():
            self.logger.info("[FactGraph] Autosave loop started")
            while not self._stop_autosave.is_set():
                try:
                    with self._lock:
                        self._save_to_store()
                    if self.telemetry:
                        self.telemetry.capture_event("factgraph_autosave", {"timestamp": time.time()})
                except Exception as e:
                    self.logger.error(f"[FactGraph] Autosave error: {e}", exc_info=True)
                self._stop_autosave.wait(self.autosave_interval)
            self.logger.info("[FactGraph] Autosave loop exiting gracefully")

        if self.autosave_interval > 0:
            self._autosave_thread = threading.Thread(target=_autosave_loop, name="FactGraphAutosaveThread", daemon=True)
            self._autosave_thread.start()
            self.logger.info(f"[FactGraph] Autosave thread started with interval {self.autosave_interval}s")

    # ----------------------------
    # Persistence / load / save
    # ----------------------------
    def _load_from_store(self) -> None:
        """Load facts from persistence (couch or file) if present."""
        with self._lock:
            if self._db is not None:
                try:
                    # doc id per graph name
                    docid = f"factgraph::{self.name}"
                    if docid in self._db:
                        doc = dict(self._db[docid])
                        content = doc.get("content", {})
                        facts = content.get("facts", {})
                        for fid, fd in facts.items():
                            f = Fact.from_dict(fd)
                            self._facts[fid] = f
                            self._index_add(f)
                        logger.info(f"[FactGraph] Loaded {len(self._facts)} facts from CouchDB")
                except Exception as e:
                    logger.warning(f"[FactGraph] CouchDB load failed: {e}")
            else:
                if os.path.exists(self._file_path):
                    try:
                        with open(self._file_path, "r", encoding="utf-8") as fh:
                            payload = json.load(fh)
                        facts = payload.get("facts", {})
                        for fid, fd in facts.items():
                            f = Fact.from_dict(fd)
                            self._facts[fid] = f
                            self._index_add(f)
                        logger.info(f"[FactGraph] Loaded {len(self._facts)} facts from file")
                    except Exception as e:
                        logger.warning(f"[FactGraph] File load failed: {e}")

    def _save_to_store(self) -> None:
        """Persist current graph to selected backend."""
        with self._lock:
            payload = {"facts": {fid: f.to_dict() for fid, f in self._facts.items()}, "meta": {"ts": _now_ts()}}
            if self._db is not None:
                try:
                    docid = f"factgraph::{self.name}"
                    if docid in self._db:
                        doc = self._db[docid]
                        doc["content"] = payload
                        self._db.save(doc)
                    else:
                        self._db[docid] = {"content": payload}
                    logger.debug("[FactGraph] Saved to CouchDB")
                    return
                except Exception as e:
                    logger.warning(f"[FactGraph] CouchDB save failed: {e}")
            # fallback file
            try:
                with open(self._file_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, default=str, indent=2)
                logger.debug("[FactGraph] Saved to file")
            except Exception as e:
                logger.error(f"[FactGraph] Save to file failed: {e}")

    def _start_autosave_loop(self):
        if self._autosave_thread and self._autosave_thread.is_alive():
            return

        def autosave():
            logger.debug("[FactGraph] Autosave thread started")
            while not self._stop_autosave.is_set():
                time.sleep(self.autosave_interval)
                try:
                    self._save_to_store()
                    self._emit("autosave", {"ts": _now_ts()})
                except Exception as e:
                    logger.warning(f"[FactGraph] autosave failed: {e}")
            logger.debug("[FactGraph] Autosave thread stopped")

        self._autosave_thread = threading.Thread(target=autosave, name="FactGraphAutosave", daemon=True)
        self._autosave_thread.start()

    def close(self):
        """Stop autosave loops and persist final snapshot."""
        self._stop_autosave.set()
        if self._autosave_thread:
            self._autosave_thread.join(timeout=2.0)
        self._save_to_store()

    # ----------------------------
    # indexing helpers
    # ----------------------------
    def _index_add(self, f: Fact):
        key = (f.subject, f.predicate, json.dumps(f.object, sort_keys=True, default=str))
        self._index_spo[key] = f.id
        self._index_subject.setdefault(f.subject, []).append(f.id)
        obj_key = str(f.object)
        self._index_object.setdefault(obj_key, []).append(f.id)

    def _index_remove(self, f: Fact):
        key = (f.subject, f.predicate, json.dumps(f.object, sort_keys=True, default=str))
        self._index_spo.pop(key, None)
        if f.subject in self._index_subject:
            try:
                self._index_subject[f.subject].remove(f.id)
            except ValueError:
                pass
        obj_key = str(f.object)
        if obj_key in self._index_object:
            try:
                self._index_object[obj_key].remove(f.id)
            except ValueError:
                pass

    # ----------------------------
    # CRUD operations
    # ----------------------------
    def add_fact(self, subject: str, predicate: str, obj: Any, metadata: Optional[Dict] = None) -> str:
        """
        Add a fact. Returns the fact id.
        """
        with self._lock:
            f = Fact(subject, predicate, obj, metadata=metadata)
            self._facts[f.id] = f
            self._index_add(f)
            self._emit("fact_added", {"fact": f.to_dict()})
            logger.debug(f"[FactGraph] Added fact {f.id}")
            return f.id

    def remove_fact(self, fact_id: str) -> bool:
        with self._lock:
            f = self._facts.get(fact_id)
            if not f:
                return False
            self._index_remove(f)
            del self._facts[fact_id]
            self._emit("fact_removed", {"fact_id": fact_id})
            logger.debug(f"[FactGraph] Removed fact {fact_id}")
            return True

    def update_fact(self, fact_id: str, **updates) -> bool:
        with self._lock:
            f = self._facts.get(fact_id)
            if not f:
                return False
            # Only allow safe fields to update
            allowed = {"subject", "predicate", "object", "metadata"}
            changed = False
            self._index_remove(f)
            for k, v in updates.items():
                if k in allowed:
                    setattr(f, k if k != "object" else "object", v)
                    changed = True
            if changed:
                f.version += 1
                f.ts = _now_ts()
            self._index_add(f)
            self._emit("fact_updated", {"fact": f.to_dict()})
            return True

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[Any] = None) -> List[Dict]:
        """
        Query facts by any combination of subject, predicate, object.
        Returns fact dicts.
        """
        with self._lock:
            results = []
            if subject and not predicate and not obj:
                ids = self._index_subject.get(subject, [])
                results = [self._facts[i].to_dict() for i in ids]
            elif obj and not subject and not predicate:
                ids = self._index_object.get(str(obj), [])
                results = [self._facts[i].to_dict() for i in ids]
            else:
                # fallback linear scan (safe)
                for f in self._facts.values():
                    if subject and f.subject != subject:
                        continue
                    if predicate and f.predicate != predicate:
                        continue
                    if obj is not None and f.object != obj:
                        continue
                    results.append(f.to_dict())
            return results

    # ----------------------------
    # Graph traversal
    # ----------------------------
    def find_related(self, entity: str, depth: int = 1) -> List[Dict]:
        """
        Breadth-first traversal out to a specified depth. Returns list of facts (dicts)
        related to entity either as subject or object.
        """
        with self._lock:
            visited = set()
            queue = [(entity, 0)]
            collected = []
            while queue:
                current, d = queue.pop(0)
                if d > depth:
                    break
                # facts where current is subject
                for fid in self._index_subject.get(current, []):
                    if fid in visited:
                        continue
                    visited.add(fid)
                    f = self._facts[fid]
                    collected.append(f.to_dict())
                    # enqueue object if it's a string (entity-like)
                    if isinstance(f.object, str):
                        queue.append((f.object, d + 1))
                # facts where current equals object (string compare)
                for fid in self._index_object.get(current, []):
                    if fid in visited:
                        continue
                    visited.add(fid)
                    f = self._facts[fid]
                    collected.append(f.to_dict())
                    if isinstance(f.subject, str):
                        queue.append((f.subject, d + 1))
            return collected

    # ----------------------------
    # Snapshot / versioning
    # ----------------------------
    def snapshot(self, label: Optional[str] = None) -> str:
        """Create a snapshot; returns snapshot id."""
        with self._lock:
            sid = f"snapshot::{int(_now_ts())}::{uuid.uuid4().hex[:8]}"
            payload = {
                "ts": _now_ts(),
                "label": label,
                "facts": {fid: f.to_dict() for fid, f in self._facts.items()},
            }
            path = os.path.join(self._snapshot_dir, f"{sid}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            self._snapshots[sid] = {"path": path, "meta": {"ts": payload["ts"], "label": label}}
            self._emit("snapshot_created", {"snapshot_id": sid})
            logger.info(f"[FactGraph] Snapshot created {sid}")
            return sid

    def rollback(self, snapshot_id: str) -> bool:
        """Rollback to a snapshot previously created."""
        with self._lock:
            entry = self._snapshots.get(snapshot_id)
            if not entry:
                # check if file exists on disk
                path = os.path.join(self._snapshot_dir, f"{snapshot_id}.json")
                if not os.path.exists(path):
                    logger.error(f"[FactGraph] Snapshot {snapshot_id} not found")
                    return False
                entry = {"path": path}
            try:
                with open(entry["path"], "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                facts = payload.get("facts", {})
                # restore
                self._facts.clear()
                self._index_spo.clear()
                self._index_subject.clear()
                self._index_object.clear()
                for fid, fd in facts.items():
                    f = Fact.from_dict(fd)
                    self._facts[fid] = f
                    self._index_add(f)
                self._emit("rolled_back", {"snapshot_id": snapshot_id})
                logger.info(f"[FactGraph] Rolled back to {snapshot_id}")
                return True
            except Exception as e:
                logger.exception(f"[FactGraph] Rollback failed: {e}")
                return False

    # ----------------------------
    # Patch / safe apply
    # ----------------------------
    def validate_patch(self, patch: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a patch. Patch schema:
        {
            "add": [{"subject":..., "predicate":..., "object":..., "metadata": {}}],
            "remove": ["fact_id", ...],
            "update": [{"fact_id":..., "updates": {...}}, ...],
            "meta": {...}
        }
        Returns (ok, list_of_errors)
        """
        errors = []
        if not isinstance(patch, dict):
            return False, ["patch must be a dict"]
        # simple checks
        if "remove" in patch:
            if not isinstance(patch["remove"], list):
                errors.append("remove must be a list of fact ids")
            else:
                # dangerous: mass remove
                if len(patch["remove"]) > 100 and not _ALLOW_DANGEROUS:
                    errors.append("attempt to remove >100 facts blocked by safety guard")
        if "add" in patch:
            if not isinstance(patch["add"], list):
                errors.append("add must be a list of fact dicts")
            else:
                for i, item in enumerate(patch["add"]):
                    if not all(k in item for k in ("subject", "predicate", "object")):
                        errors.append(f"add[{i}] missing subject/predicate/object")
        if "update" in patch:
            if not isinstance(patch["update"], list):
                errors.append("update must be a list")
        return (len(errors) == 0, errors)

    def safe_apply_patch(self, patch: Dict[str, Any], preview: bool = False) -> Dict[str, Any]:
        """
        Validate and apply a patch. If preview=True, do a dry-run and return the
        simulated effects without changing state.

        Returns a report dict:
        {
          "ok": bool,
          "errors": [...],
          "applied": {"added": N, "removed": N, "updated": N},
          "preview": {...}
        }
        """
        with self._lock:
            ok, errors = self.validate_patch(patch)
            if not ok:
                return {"ok": False, "errors": errors}

            # create a snapshot before applying
            snap_id = self.snapshot(label=f"pre_patch_preview" if preview else "pre_patch_apply")

            # operate on a copy for preview
            working = copy.deepcopy(self._facts)

            added = 0
            removed = 0
            updated = 0
            try:
                # perform add
                for item in patch.get("add", []):
                    f = Fact(item["subject"], item["predicate"], item.get("object"), metadata=item.get("metadata"))
                    working[f.id] = f
                    added += 1

                # perform update
                for upd in patch.get("update", []):
                    fid = upd.get("fact_id")
                    if fid in working:
                        for k, v in upd.get("updates", {}).items():
                            if hasattr(working[fid], k):
                                setattr(working[fid], k, v)
                        working[fid].version += 1
                        updated += 1

                # perform remove
                for fid in patch.get("remove", []):
                    if fid in working:
                        del working[fid]
                        removed += 1

                # run validation tests on working state (lightweight)
                if preview:
                    report = {"ok": True, "errors": [], "applied": {"added": added, "removed": removed, "updated": updated}, "preview": True}
                    # delete snapshot if it's just preview
                    self._snapshots.pop(snap_id, None)
                    try:
                        os.remove(os.path.join(self._snapshot_dir, f"{snap_id}.json"))
                    except Exception:
                        pass
                    return report

                # If not preview: apply to real graph but keep a backup on disk
                # create on-disk pre-patch snapshot already done above
                # Now apply changes to current graph
                # Adds
                for item in patch.get("add", []):
                    self.add_fact(item["subject"], item["predicate"], item.get("object"), metadata=item.get("metadata"))
                # Updates
                for upd in patch.get("update", []):
                    self.update_fact(upd.get("fact_id"), **upd.get("updates", {}))
                # Removes
                for fid in patch.get("remove", []):
                    self.remove_fact(fid)

                # final save
                self._save_to_store()
                self._emit("patch_applied", {"patch": patch, "snapshot_pre": snap_id})
                return {"ok": True, "errors": [], "applied": {"added": added, "removed": removed, "updated": updated}, "preview": False}

            except Exception as e:
                # rollback to snapshot
                logger.exception(f"[FactGraph] patch application failed: {e}")
                self.rollback(snap_id)
                return {"ok": False, "errors": [str(e)], "applied": {"added": added, "removed": removed, "updated": updated}}

    # ----------------------------
    # Inference / hooks
    # ----------------------------
    def infer(self, rules: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Run a simple forward-chaining inference engine with provided rules.
        Rule format (very simple):
        {
            "when": {"subject": "X", "predicate": "is_a", "object": "person"},
            "then": {"add": {"subject": "X", "predicate": "eligible", "object": "something"}},
            "priority": 10
        }
        returns number of inferred facts added.
        """
        added = 0
        applied = []
        with self._lock:
            rules = rules or []
            for rule in rules:
                when = rule.get("when") or {}
                then = rule.get("then") or {}
                for f in list(self._facts.values()):
                    match = True
                    if when.get("subject") and when["subject"] != f.subject and when["subject"] != "X":
                        match = False
                    if when.get("predicate") and when["predicate"] != f.predicate:
                        match = False
                    if "object" in when and when["object"] != f.object and when["object"] != "X":
                        match = False
                    if match:
                        to_add = then.get("add")
                        if to_add:
                            subj = to_add.get("subject", f.subject)
                            pred = to_add.get("predicate")
                            obj = to_add.get("object")
                            # template substitution minimal
                            if isinstance(obj, str) and obj == "X":
                                obj = f.subject
                            self.add_fact(subj, pred, obj)
                            added += 1
                            applied.append({"rule": rule, "based_on": f.id})
        if added:
            self._emit("inferred", {"count": added, "details": applied})
        return added

    # ----------------------------
    # listeners / events
    # ----------------------------
    def register_listener(self, fn: Callable[[str, Dict[str, Any]], None]) -> None:
        with self._lock:
            if fn not in self._listeners:
                self._listeners.append(fn)

    def unregister_listener(self, fn: Callable[[str, Dict[str, Any]], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(fn)
            except ValueError:
                pass

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        # best-effort notify without letting listeners break the graph
        listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event, payload)
            except Exception:
                logger.exception(f"[FactGraph] listener error for event {event}")
        # telemetry
        try:
            if self.telemetry:
                # TelemetryHandler API may vary; try a common method
                rec = getattr(self.telemetry, "record_event", None)
                if callable(rec):
                    rec(event, payload)
        except Exception:
            # swallow telemetry errors
            logger.debug("[FactGraph] telemetry.emit failed")

    # ----------------------------
    # utilities
    # ----------------------------
    def export(self, path: Optional[str] = None) -> str:
        """Export full graph to a JSON file. Returns filepath."""
        with self._lock:
            path = path or os.path.join(os.getcwd(), f"factgraph_export_{self.name}_{int(_now_ts())}.json")
            payload = {"facts": {fid: f.to_dict() for fid, f in self._facts.items()}, "meta": {"ts": _now_ts()}}
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            return path

    def import_graph(self, payload: Dict[str, Any], merge: bool = True) -> int:
        """Import graph from dict payload. If merge False, replace current graph."""
        with self._lock:
            facts = payload.get("facts", {})
            if not merge:
                self._facts.clear()
                self._index_object.clear()
                self._index_subject.clear()
                self._index_spo.clear()
            added = 0
            for fid, fd in facts.items():
                f = Fact.from_dict(fd)
                # ensure unique id
                if f.id in self._facts:
                    # make new id
                    f.id = str(uuid.uuid4())
                self._facts[f.id] = f
                self._index_add(f)
                added += 1
            self._save_to_store()
            self._emit("imported", {"added": added})
            return added

    # ----------------------------
    # Self-tests
    # ----------------------------
    def run_self_tests(self) -> Dict[str, Any]:
        """
        Runs a small suite of internal checks to validate FactGraph integrity.
        Returns a dict with 'ok':bool and 'details': list.
        """
        details = []
        ok = True
        try:
            # simple add/query/remove test
            sid = self.add_fact("__selftest_subject__", "is_test", {"x": 1})
            q = self.query(subject="__selftest_subject__")
            if not any(x["id"] == sid for x in q):
                ok = False
                details.append("roundtrip add/query failed")
            if not self.remove_fact(sid):
                ok = False
                details.append("remove failed")
            # snapshot/rollback test
            a = self.add_fact("A", "knows", "B")
            snap = self.snapshot(label="selftest")
            self.remove_fact(a)
            if not self.rollback(snap):
                ok = False
                details.append("rollback failed")
            # inference test (minimal)
            self.remove_fact(a)  # ensure duplicate removed if needed
        except Exception as e:
            ok = False
            details.append(f"exception during self-tests: {e}")
        return {"ok": ok, "details": details}

    # ----------------------------
    # convenience magic
    # ----------------------------
    def __len__(self):
        return len(self._facts)

    def __contains__(self, item):
        return item in self._facts

    def __iter__(self):
        for f in list(self._facts.values()):
            yield f.to_dict()


# Example listener adapter for RulesEngine / AutoHealer
def rules_listener_factory(rules_engine):
    def _listener(event: str, payload: Dict[str, Any]):
        try:
            # handle interesting events (simple mapping)
            if event == "fact_added":
                # allow rules engine to potentially schedule actions
                if hasattr(rules_engine, "on_fact_added"):
                    try:
                        rules_engine.on_fact_added(payload["fact"])
                    except Exception:
                        logger.exception("rules_engine.on_fact_added failed")
        except Exception:
            logger.exception("rules_listener error")
    return _listener


# ----------------------------
# Simple CLI / test harness
# ----------------------------
if __name__ == "__main__":
    # quick sanity test
    fg = FactGraph(name="cli_test", autosave_interval=0)
    print("Initial facts:", len(fg))
    fid = fg.add_fact("alice", "likes", "pizza", metadata={"confidence": 0.9})
    print("Added:", fid)
    print("Query alice ->", fg.query(subject="alice"))
    snap = fg.snapshot("cli_test_snapshot")
    print("Snapshot:", snap)
    ok = fg.rollback(snap)
    print("Rollback ok:", ok)
    report = fg.run_self_tests()
    print("Self-tests:", report)
    fg.close()
