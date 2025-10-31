# app/utils/memory.py
"""
Compatibility & lazy bridge for MemoryManager.

This file intentionally does NOT re-implement the full MemoryManager logic.
Instead it:
 - exposes a compatibility symbol `MemoryManager` that maps to the real class
   in app.utils.memory_manager when available,
 - exposes `MemoryStore` as an alias for older imports,
 - provides a lazy-instantiated global `MEMORY` singleton (use get_memory())
   which avoids circular-import issues during package import,
 - provides simple helpers to initialize / reload memory safely.

Purpose: keep legacy imports working while centralizing the implementation
in app.utils.memory_manager (so developers only maintain one true implementation).
"""

from __future__ import annotations
import sys
sys.modules.pop('app.utils.memory_manager', None)

import importlib
import logging
import os
import threading
import inspect
from typing import Any, Dict, Optional, Type

# ---------------------------------------------------------------------
# Logging (immediate)
# ---------------------------------------------------------------------
logger = logging.getLogger("BibliothecaAI.Memory")
_log_level = os.getenv("BIBLIOTHECA_LOG_LEVEL", "INFO")
# Accept either numeric or string level
try:
    numeric_level = int(_log_level)
except Exception:
    numeric_level = _log_level
logger.setLevel(numeric_level)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s"))
    logger.addHandler(ch)

# ---------------------------------------------------------------------
# Try to load application config safely (non-fatal)
# ---------------------------------------------------------------------
def _load_app_config() -> Dict[str, Any]:
    """Attempt to load app config from app.config.settings without raising."""
    try:
        from app.config.settings import CONFIG  # type: ignore
        if isinstance(CONFIG, dict):
            return CONFIG
    except Exception:
        try:
            from app.config.settings import load_config  # type: ignore
            cfg = load_config()
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            # Fallback to environment-driven minimal config
            return {}
    return {}

APP_CONFIG: Dict[str, Any] = _load_app_config()

# ---------------------------------------------------------------------
# Lazy import helpers for the real MemoryManager implementation
# ---------------------------------------------------------------------
import threading
import importlib
import logging
from typing import Optional, Type

logger = logging.getLogger("BibliothecaAI.Memory")

# Internal state
_RealMemoryManager: Optional[Type] = None
_MemoryManager_import_attempted = False
_memory_manager_lock = threading.RLock()  # Use reentrant lock for safety

def _import_real_memory_manager() -> Optional[Type]:
    """
    Attempt to import the real MemoryManager class from
    app.utils.memory_manager. Returns the class or None on failure.

    Features:
    ✅ Thread-safe
    ✅ Safe for repeated calls
    ✅ Full logging of import success/failure
    ✅ Deferred import avoids circular dependencies
    ✅ Compatibility fallback
    """
    global _RealMemoryManager, _MemoryManager_import_attempted

    if _RealMemoryManager is not None:
        return _RealMemoryManager

    with _memory_manager_lock:
        # Double-check inside lock
        if _RealMemoryManager is not None:
            return _RealMemoryManager

        if _MemoryManager_import_attempted:
            logger.debug("[Memory] Previous MemoryManager import attempt failed — returning None")
            return None

        _MemoryManager_import_attempted = True
        try:
            mod = importlib.import_module("app.utils.memory_manager")
            cls = getattr(mod, "MemoryManager", None)

            if cls is None:
                logger.warning("[Memory] MemoryManager class not found in app.utils.memory_manager")
                _RealMemoryManager = None
            else:
                _RealMemoryManager = cls
                logger.info("[Memory] Successfully imported real MemoryManager ✅")
            return _RealMemoryManager

        except ModuleNotFoundError as mnf:
            logger.warning(f"[Memory] MemoryManager module not found: {mnf}")
            _RealMemoryManager = None
            return None
        except Exception as exc:
            logger.error(f"[Memory] Error importing MemoryManager: {exc}", exc_info=True)
            _RealMemoryManager = None
            return None

# Compatibility alias for older modules
MemoryManager: Optional[Type] = _import_real_memory_manager()

# MemoryStore alias for older imports or backward compatibility
# Falls back to a minimal stub if MemoryManager not available
if MemoryManager is not None:
    MemoryStore: Type = MemoryManager
else:
    class _MemoryStoreStub:
        """
        Minimal stub to preserve compatibility with legacy imports.
        Raises errors on unsafe operations.
        """
        def __init__(self, *args, **kwargs):
            logger.warning("[MemoryStoreStub] Using stub MemoryStore — real MemoryManager not loaded")
        def __getattr__(self, item):
            raise AttributeError(f"[MemoryStoreStub] Attempted access to '{item}' but real MemoryManager is unavailable")
    MemoryStore = _MemoryStoreStub

# ---------------------------------------------------------------------
# Global lazy singleton accessor
# ---------------------------------------------------------------------
MEMORY: Optional[Any] = None
_memory_singleton_lock = threading.Lock()

def _instantiate_with_compatibility(cls: Type, effective_config: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
    """
    Try to instantiate cls with a variety of common constructor signatures to
    preserve backward compatibility across different versions.
    """
    # Prefer explicit kwarg keys commonly used
    preferred_keys = ["config", "base_dir", "couchdb_url", "db_name", "auto_heal"]
    inst_kwargs = {}

    # If cls.__init__ accepts **kwargs, just pass what we have
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except Exception:
        accepts_var_kw = True

    # Start by trying to pass 'config' if accepted
    if "config" in getattr(cls.__init__, "__code__", {}).co_varnames or accepts_var_kw:
        inst_kwargs.update({"config": effective_config})
    # Merge any explicit kwargs user provided
    inst_kwargs.update(kwargs)

    # Attempt instantiation in a few orders to maximize compatibility
    attempts = []

    # -----------------------------
    # Robust MemoryManager instantiation attempts
    # -----------------------------
    attempts = []

    # 1️⃣ Try full kwargs (but remove 'config' if not accepted)
    def _try_full_kwargs():
        try:
            # inspect constructor signature
            import inspect
            sig = inspect.signature(cls.__init__)
            filtered_kwargs = dict(inst_kwargs)
            if "config" not in sig.parameters:
                filtered_kwargs.pop("config", None)
            return cls(**filtered_kwargs)
        except Exception as e:
            logger.debug(f"[Memory] Attempt 1 failed: {e}")
            raise

    attempts.append(_try_full_kwargs)

    # 2️⃣ Try passing effective_config as first positional arg + kwargs
    def _try_positional_config():
        try:
            import inspect
            sig = inspect.signature(cls.__init__)
            if len(sig.parameters) > 1:  # first is 'self', second accepts config-like input
                return cls(effective_config, **kwargs)
            else:
                raise TypeError("Constructor does not accept positional config")
        except Exception as e:
            logger.debug(f"[Memory] Attempt 2 failed: {e}")
            raise

    attempts.append(_try_positional_config)

    # 3️⃣ Try only **kwargs (without config)
    def _try_kwargs_only():
        try:
            inst_kwargs_no_config = dict(kwargs)
            inst_kwargs_no_config.pop("config", None)
            return cls(**inst_kwargs_no_config)
        except Exception as e:
            logger.debug(f"[Memory] Attempt 3 failed: {e}")
            raise

    attempts.append(_try_kwargs_only)

    # 4️⃣ Last-resort no-arg constructor
    def _try_no_args():
        try:
            return cls()
        except Exception as e:
            logger.debug(f"[Memory] Attempt 4 failed: {e}")

            # fallback to in-memory dummy
            class _FallbackMemoryStore:
                def __init__(self, *a, **kw):
                    logger.info("[Memory] Using fallback in-memory MemoryStore")

            return _FallbackMemoryStore()

    attempts.append(_try_no_args)

    # Execute attempts in order until one succeeds
    for attempt_fn in attempts:
        try:
            MEMORY = attempt_fn()
            if MEMORY:
                logger.info("[Memory] MemoryManager instantiated successfully ✅")
                break
        except Exception:
            continue
    else:
        logger.error("[Memory] All MemoryManager instantiation attempts failed, using in-memory fallback")

        class _FallbackMemoryStore:
            def __init__(self, *a, **kw):
                logger.info("[Memory] Using fallback in-memory MemoryStore")

        MEMORY = _FallbackMemoryStore()

    last_exc = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as te:
            last_exc = te
            continue
        except Exception as e:
            last_exc = e
            # If non-TypeError, still return failure later
            break

    # Raise last exception for debugging if all attempts fail
    raise last_exc or RuntimeError("Unknown instantiation failure")


def init_memory(*, config: Optional[Dict[str, Any]] = None, **kwargs) -> Optional[Any]:
    """
    Initialize and return the global MEMORY singleton using the real MemoryManager.

    Args:
        config: optional config dict (will merge with APP_CONFIG, priority given to arg).
        kwargs: forwarded to MemoryManager constructor (e.g. base_dir, couchdb_url).

    Returns:
        The MEMORY singleton instance (or None on failure).
    """
    global MEMORY, MemoryManager, MemoryStore

    with _memory_singleton_lock:
        if MEMORY is not None:
            return MEMORY

        cls = _import_real_memory_manager()
        if not cls:
            logger.error("[Memory] Cannot initialize MEMORY: MemoryManager implementation not available.")
            return None

        # Merge configs: explicit config overrides APP_CONFIG
        effective_config: Dict[str, Any] = {}
        if APP_CONFIG:
            effective_config.update(APP_CONFIG)
        if isinstance(config, dict):
            effective_config.update(config)

        # Prepare kwargs, but remove 'config' to avoid unexpected argument
        inst_kwargs = dict(kwargs)
        if 'config' in inst_kwargs:
            inst_kwargs.pop('config')

        # Try multiple instantiation methods
        attempts = []
        # 1️⃣ Full kwargs
        attempts.append(lambda: cls(**inst_kwargs))
        # 2️⃣ Positional effective_config + remaining kwargs
        attempts.append(lambda: cls(effective_config, **inst_kwargs))
        # 3️⃣ No-arg fallback
        attempts.append(lambda: cls())

        last_exc = None
        for attempt in attempts:
            try:
                MEMORY = attempt()
                logger.info(f"[Memory] Successfully instantiated MEMORY using {attempt.__name__}")
                break
            except TypeError as e:
                last_exc = e
                logger.debug(f"[Memory] Attempt {attempt.__name__} failed: {e}")
            except Exception as e:
                last_exc = e
                logger.warning(f"[Memory] Attempt {attempt.__name__} failed: {e}")
        else:
            logger.error(f"[Memory] Failed to instantiate MEMORY: {last_exc}")
            MEMORY = None
            return None

        # Ensure .config attribute is set if possible
        try:
            MEMORY.config = effective_config
        except Exception:
            pass

        # Update module-level aliases after successful instantiation
        MemoryManager = cls
        MemoryStore = cls

        logger.info("[Memory] MEMORY singleton instantiated successfully ✅")
        return MEMORY


def get_memory(create_if_missing: bool = True, **kwargs) -> Optional[Any]:
    """
    Return the global MEMORY singleton. If it's not initialized and create_if_missing is True,
    attempt to initialize it with optional kwargs forwarded to init_memory().

    Example:
        get_memory()  # returns singleton or initializes with defaults
        get_memory(config={'X':1}, base_dir='.')  # custom init
    """
    global MEMORY
    if MEMORY is not None:
        return MEMORY
    if not create_if_missing:
        return None
    return init_memory(**kwargs)

def reload_memory_manager() -> bool:
    """
    Force-reload the underlying app.utils.memory_manager module and reinitialize MEMORY.
    Returns True on successful reload and initialization.
    """
    global _RealMemoryManager, MemoryManager, MemoryStore, MEMORY, _MemoryManager_import_attempted

    with _memory_singleton_lock:
        try:
            # Reload module (will raise if import fails)
            mod = importlib.reload(importlib.import_module("app.utils.memory_manager"))
            _RealMemoryManager = getattr(mod, "MemoryManager", None)
            MemoryManager = _RealMemoryManager
            MemoryStore = _RealMemoryManager or MemoryStore
            _MemoryManager_import_attempted = True

            # If an old instance exists, try to stop/cleanup it gracefully
            if MEMORY is not None:
                try:
                    if hasattr(MEMORY, "stop"):
                        try:
                            MEMORY.stop()
                        except Exception:
                            pass
                    if hasattr(MEMORY, "shutdown"):
                        try:
                            MEMORY.shutdown()
                        except Exception:
                            pass
                except Exception:
                    pass
                MEMORY = None

            # Re-init singleton with defaults
            new_mem = init_memory()
            return new_mem is not None
        except Exception as exc:
            logger.exception(f"[Memory] reload_memory_manager failed: {exc}")
            return False

# ---------------------------------------------------------------------
# Backwards-compatible eager init attempt (non-fatal)
# ---------------------------------------------------------------------
# Historically some code expected MEMORY to be ready on import. We will attempt
# one cautious initialization here, but keep it silent on failure (to avoid
# crashing during circular import scenarios).
try:
    if MemoryManager is None:
        # attempt quick import but don't force creation if circular
        _import_real_memory_manager()

    # try a safe, non-intrusive init only when memory import looks clean
    if MemoryManager is not None and MEMORY is None:
        try:
            _maybe_init = os.getenv("BIBLIOTHECA_EAGER_MEMORY_INIT", "1").lower() not in ("0", "false", "no")
            if _maybe_init:
                # small, non-fatal attempt; errors are swallowed
                init_memory()
        except Exception:
            logger.debug("[Memory] eager init attempt failed (non-fatal).")
except Exception:
    logger.debug("[Memory] deferred loading strategy in use.")

# expose minimal public API for legacy code
__all__ = [
    "MemoryManager",
    "MemoryStore",
    "get_memory",
    "init_memory",
    "reload_memory_manager",
    "MEMORY",
    "APP_CONFIG",
]


# ------------------------------------------------------------------
# CouchDB wrapper (Beyond-Perfection)
# ------------------------------------------------------------------
class CouchDBMemory:
    """
    Ultra-robust CouchDB backend wrapper for MemoryManager / MemoryStore.
    Responsibilities:
      - Resolve config from args -> APP_CONFIG -> environment -> defaults
      - Normalize credentials and URL
      - Connect with retries + exponential backoff
      - Auto-create DBs when permitted
      - Provide safe CRUD operations with retries and logging
      - Provide streaming all_docs generator
      - Health check, reconnect, and graceful close
      - Optional telemetry hooks (if self.telemetry exists)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        db_name: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 0.5,
        auto_create: Optional[bool] = None,
    ):
        # basic state
        self.available: bool = False
        self.server = None
        self.db = None
        self.client = None
        self.db_name = None
        self.url = None
        self._lock = threading.RLock()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # load config from APP_CONFIG or env
        cfg = (APP_CONFIG or {}).get("couchdb", {})
        url = url or cfg.get("url") or os.getenv("COUCHDB_URL") or "http://127.0.0.1:5984"
        username = username or cfg.get("username") or os.getenv("COUCHDB_USER") or os.getenv("COUCHDB_USERNAME")
        password = password or cfg.get("password") or os.getenv("COUCHDB_PASSWORD") or os.getenv("COUCHDB_PASS")
        db_name = db_name or cfg.get("db_name") or cfg.get("dbs", {}).get("memory") or os.getenv("COUCHDB_DB_NAME") or os.getenv("COUCHDB_DB") or "bibliotheca_memory"
        if auto_create is None:
            auto_create = cfg.get("auto_create_dbs", True)

        # normalize url
        if url.endswith("/"):
            url = url[:-1]

        # embed credentials into URL if not present and provided
        try:
            if username and password and "@" not in url:
                if url.startswith("https://"):
                    base = url[len("https://"):]
                    url = f"https://{username}:{password}@{base}"
                elif url.startswith("http://"):
                    base = url[len("http://"):]
                    url = f"http://{username}:{password}@{base}"
                else:
                    url = f"http://{username}:{password}@{url}"
        except Exception:
            # defensive fallback: leave URL unchanged if formatting fails
            pass

        self.url = url
        self.db_name = db_name
        self._auto_create = bool(auto_create)

        # validate couchdb library presence
        if couchdb is None:
            logger.warning("[Memory][CouchDBMemory] python-couchdb not installed. CouchDB backend disabled.")
            self.available = False
            return

        # attempt initial connection
        self._connect_with_retries()

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _capture_telemetry_exception(self, exc, context: str = "") -> None:
        """If telemetry object is available, capture exception."""
        try:
            telemetry = getattr(self, "telemetry", None)
            if telemetry and hasattr(telemetry, "capture_exception"):
                telemetry.capture_exception(exc, context=f"CouchDBMemory.{context}")
        except Exception:
            # Never let telemetry errors bubble up
            logger.debug("[Memory][CouchDBMemory] telemetry capture failed", exc_info=True)

    def _connect_with_retries(self) -> None:
        """Try to connect to CouchDB server and DB with retries and exponential backoff."""
        with self._lock:
            self.available = False
            last_exc = None
            for attempt in range(1, max(1, self.max_retries) + 1):
                try:
                    self.server = couchdb.Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")
                    # quick sanity check - server.version() raises on auth/conn issues
                    try:
                        _ = self.server.version()
                    except Exception:
                        # some couchdb.Server implementations expose attribute differently;
                        # treat failures below as connection problems.
                        pass

                    # ensure DB exists or create if allowed
                    if self.db_name in self.server:
                        self.db = self.server[self.db_name]
                    else:
                        if self._auto_create:
                            self.db = self.server.create(self.db_name)
                        else:
                            raise RuntimeError(f"DB '{self.db_name}' not found and auto-create disabled")

                    self.client = self.db
                    self.available = True
                    logger.info(f"[Memory][CouchDBMemory] Connected to CouchDB {self.url} -> {self.db_name}")
                    return
                except couchdb.http.Unauthorized as e:
                    logger.error(f"[Memory][CouchDBMemory] Unauthorized CouchDB access at {self.url}: {e}")
                    last_exc = e
                    self.available = False
                    self._capture_telemetry_exception(e, "unauthorized")
                    break  # no point retrying on auth error
                except Exception as e:
                    last_exc = e
                    logger.warning(f"[Memory][CouchDBMemory] CouchDB connection attempt {attempt} failed: {e}")
                    self._capture_telemetry_exception(e, "connect_attempt")
                    if attempt < self.max_retries:
                        sleep_time = self.retry_delay * (2 ** (attempt - 1))
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"[Memory][CouchDBMemory] Failed to connect after {self.max_retries} attempts: {e}")
                        self.available = False
            # keep server/db None on failure
            self.server = None
            self.db = None
            self.client = None
            if last_exc:
                # capture final exception
                self._capture_telemetry_exception(last_exc, "connect_final")

    def _retryable(self, fn, *args, retries: Optional[int] = None, **kwargs):
        """Generic retry wrapper for CouchDB operations."""
        if retries is None:
            retries = max(1, self.max_retries)
        attempt = 0
        last_exc = None
        while attempt < retries:
            attempt += 1
            try:
                return fn(*args, **kwargs)
            except couchdb.http.Unauthorized as e:
                logger.error(f"[Memory][CouchDBMemory] Unauthorized during operation: {e}")
                self._capture_telemetry_exception(e, "operation_unauthorized")
                # unauthorized -> stop retrying
                raise
            except Exception as e:
                last_exc = e
                logger.debug(f"[Memory][CouchDBMemory] Operation attempt {attempt} failed: {e}", exc_info=True)
                self._capture_telemetry_exception(e, "operation_retry")
                if attempt < retries:
                    time.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    # attempt reconnect before next retry
                    try:
                        self._connect_with_retries()
                    except Exception:
                        pass
                    continue
                else:
                    logger.error(f"[Memory][CouchDBMemory] Operation failed after {retries} attempts: {e}")
                    raise

    # -----------------------------
    # Public operations
    # -----------------------------
    def health_check(self) -> bool:
        """Check server and DB are responsive."""
        with self._lock:
            if not self.available or not self.server:
                return False
            try:
                _ = self.server.version()
                # ensure DB still exists
                if self.db_name not in self.server:
                    if self._auto_create:
                        self.db = self.server.create(self.db_name)
                        self.client = self.db
                    else:
                        return False
                return True
            except Exception as e:
                logger.warning(f"[Memory][CouchDBMemory] health_check failed: {e}")
                self._capture_telemetry_exception(e, "health_check")
                self.available = False
                return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get document by key (doc._id). Returns default if missing or on error."""
        if not self.available or not self.client:
            return default
        try:
            def _op():
                doc = self.client.get(key)
                return doc if doc is not None else default
            return self._retryable(_op)
        except couchdb.http.Unauthorized:
            # bubbled up by _retryable
            self.available = False
            return default
        except Exception as e:
            logger.debug(f"[Memory][CouchDBMemory] get('{key}') failed: {e}", exc_info=True)
            self._capture_telemetry_exception(e, "get")
            return default

    def set(self, key: str, data: dict) -> bool:
        """
        Save a document with id=key. Merges with existing doc if present.
        Returns True on success.
        """
        if not self.available or not self.client:
            logger.debug("[Memory][CouchDBMemory] set skipped - backend unavailable")
            return False

        try:
            def _op():
                existing = self.client.get(key) or {}
                # preserve rev if present
                if isinstance(existing, dict) and existing.get("_rev"):
                    data["_rev"] = existing.get("_rev")
                doc = dict(existing)
                doc.update(data)
                doc["_id"] = key
                # Use save() for softer compatibility across python-couchdb versions
                try:
                    self.client.save(doc)
                except AttributeError:
                    # fallback to __setitem__ style
                    self.client[doc["_id"]] = doc
                return True
            return self._retryable(_op)
        except couchdb.http.Unauthorized:
            self.available = False
            return False
        except Exception as e:
            logger.error(f"[Memory][CouchDBMemory] set('{key}') failed: {e}", exc_info=True)
            self._capture_telemetry_exception(e, "set")
            return False

    def delete(self, key: str) -> bool:
        """Delete a document by id if exists. Returns True if deleted or False."""
        if not self.available or not self.client:
            return False
        try:
            def _op():
                doc = self.client.get(key)
                if not doc:
                    return False
                try:
                    self.client.delete(doc)
                except AttributeError:
                    # python-couchdb older method:
                    del self.client[doc.id]
                return True
            return self._retryable(_op)
        except couchdb.http.Unauthorized:
            self.available = False
            return False
        except Exception as e:
            logger.error(f"[Memory][CouchDBMemory] delete('{key}') failed: {e}", exc_info=True)
            self._capture_telemetry_exception(e, "delete")
            return False

    def bulk_write(self, docs: list, batch_size: int = 100) -> int:
        """
        Write many docs in batches. Each doc should be a dict with optional '_id'.
        Returns number of docs successfully written (best-effort).
        """
        if not self.available or not self.client or not docs:
            return 0
        written = 0
        try:
            # python-couchdb supports db.update(docs) or db.save(doc)
            for i in range(0, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                def _op_batch():
                    try:
                        # prefer update if available
                        if hasattr(self.client, "update"):
                            res = list(self.client.update(batch))
                            # update returns sequence of (success, id, rev)
                            return sum(1 for r in res if r and r[0])
                        else:
                            # fallback: iterate saving individually
                            cnt = 0
                            for d in batch:
                                if "_id" not in d:
                                    d["_id"] = str(uuid.uuid4())
                                try:
                                    self.client.save(d)
                                    cnt += 1
                                except Exception:
                                    # try item assignment fallback
                                    self.client[d["_id"]] = d
                                    cnt += 1
                            return cnt
                    except Exception as e:
                        raise
                written += self._retryable(_op_batch)
            return written
        except Exception as e:
            logger.error(f"[Memory][CouchDBMemory] bulk_write failed: {e}", exc_info=True)
            self._capture_telemetry_exception(e, "bulk_write")
            return written

    def all_docs(self, include_docs: bool = True, limit: Optional[int] = None):
        """
        Generator over docs (streaming). Yields doc dicts.
        Avoids loading entire DB into memory.
        """
        if not self.available or not self.server:
            return
            yield  # generator no-op

        try:
            view = self.server[self.db_name].view("_all_docs", include_docs=include_docs, limit=limit)
            for row in view:
                doc = row.get("doc") or {}
                yield doc
        except Exception as e:
            logger.error(f"[Memory][CouchDBMemory] all_docs failed: {e}", exc_info=True)
            self._capture_telemetry_exception(e, "all_docs")
            return

    # -----------------------------
    # Maintenance helpers
    # -----------------------------
    def compact(self) -> bool:
        """Trigger DB compaction (best-effort)."""
        if not self.available or not self.server:
            return False
        try:
            try:
                # CouchDB admin DB compaction endpoint
                self.server.resource.put(f"/{self.db_name}/_compact")
            except Exception:
                # Older python-couchdb wrappers may need using server[f"{dbname}"].compact()
                try:
                    if hasattr(self.server[self.db_name], "compact"):
                        self.server[self.db_name].compact()
                except Exception:
                    pass
            return True
        except Exception as e:
            logger.warning(f"[Memory][CouchDBMemory] compact failed: {e}", exc_info=True)
            self._capture_telemetry_exception(e, "compact")
            return False

    def reconnect(self) -> None:
        """Force reconnect attempt to CouchDB (synchronous)."""
        with self._lock:
            try:
                self._connect_with_retries()
            except Exception as e:
                logger.debug(f"[Memory][CouchDBMemory] reconnect failed: {e}", exc_info=True)
                self._capture_telemetry_exception(e, "reconnect")

    def close(self) -> None:
        """Gracefully drop references to server/client."""
        with self._lock:
            try:
                # python-couchdb has no explicit close; removing references is sufficient
                self.client = None
                self.db = None
                self.server = None
                self.available = False
                logger.info("[Memory][CouchDBMemory] closed connection and cleared cached handles")
            except Exception as e:
                logger.debug(f"[Memory][CouchDBMemory] close failed: {e}", exc_info=True)

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            pass

# ------------------------------------------------------------------
# AdvancedAI
# ------------------------------------------------------------------

        class AdvancedAI:
            def __init__(self, db_url: str = "http://127.0.0.1:5984", db_name: str = "bibliotheca_ai"):
                self.db_url = db_url
                self.db_name = db_name
                self.server = None
                self.db = None
                self.connect_to_db()

            def connect_to_db(self):
                """Establish connection to CouchDB and ensure database exists."""
                try:
                    self.server = Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")
                    if self.db_name not in self.server:
                        self.db = self.server.create(self.db_name)
                        print(f"[DB] Created new database: {self.db_name}")
                    else:
                        self.db = self.server[self.db_name]
                        print(f"[DB] Connected to existing database: {self.db_name}")
                except Exception as e:
                    print(f"[DB][Error] Failed to connect/create DB: {e}")
                    traceback.print_exc()
                    self.db = None

            def sync_to_db(self, doc_id: str, data: dict, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
                """
                Fully-featured method to sync AI data to CouchDB.

                Args:
                    doc_id (str): The document ID to store/update.
                    data (dict): The data payload to sync.
                    max_retries (int): Maximum retry attempts for conflicts or connection errors.
                    retry_delay (float): Seconds to wait between retries.

                Returns:
                    bool: True if sync succeeded, False otherwise.
                """
                if not self.db:
                    print("[DB][Error] No database connection established.")
                    return False

                attempt = 0
                while attempt < max_retries:
                    try:
                        # Fetch existing document to preserve revision
                        existing_doc = self.db.get(doc_id)
                        if existing_doc:
                            # Merge old data with new data intelligently
                            existing_doc.update(data)
                            doc_to_save = existing_doc
                        else:
                            doc_to_save = data
                            doc_to_save["_id"] = doc_id

                        # Attempt to save document
                        self.db.save(doc_to_save)
                        print(f"[DB] Successfully synced document '{doc_id}' on attempt {attempt + 1}.")
                        return True

                    except ResourceConflict:
                        # Handle CouchDB conflicts by re-fetching and retrying
                        attempt += 1
                        print(f"[DB][Warning] Conflict detected for '{doc_id}', retrying ({attempt}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue

                    except http.ServerError as e:
                        # Server errors (5xx) – retryable
                        attempt += 1
                        print(
                            f"[DB][Warning] Server error on sync for '{doc_id}': {e}. Retrying ({attempt}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue

                    except Exception as e:
                        # Non-recoverable error – log and abort
                        print(f"[DB][Error] Failed to sync document '{doc_id}': {e}")
                        traceback.print_exc()
                        return False

                print(f"[DB][Error] Maximum retry attempts reached for '{doc_id}'. Sync failed.")
                return False

        def sync_to_backend(self):
            """
            Sync in-memory cache to CouchDB if available; always write fallback JSON.
            Non-destructive: creates/updates docs on backend.
            """
            with memory_lock:
                if getattr(self.db, "available", False):
                    for key, val in self.memory_cache.items():
                        try:
                            payload = val if isinstance(val, dict) else {"value": val}
                            self.db.set(key, payload)
                        except Exception:
                            self.logger.debug(f"[Memory] sync key failed: {key} - {traceback.format_exc()}")
                # Always write fallback JSON
                self._write_fallback()
                self.logger.info("[Memory] sync_to_backend complete")

        def _write_fallback(self):
            """Write the current memory cache to a fallback JSON file."""
            try:
                with open(self.fallback_file, "w", encoding="utf-8") as f:
                    json.dump(self.memory_cache, f, indent=4, ensure_ascii=False)
                self.logger.debug(f"[Memory] Fallback written to {self.fallback_file}")
            except Exception:
                self.logger.error(f"[Memory] Failed to write fallback: {traceback.format_exc()}")

        def load_fallback(self):
            """Load memory from fallback JSON if it exists."""
            try:
                with open(self.fallback_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    with memory_lock:
                        self.memory_cache.update(data)
                self.logger.info(f"[Memory] Fallback loaded from {self.fallback_file}")
            except FileNotFoundError:
                self.logger.info("[Memory] No fallback file found, starting fresh")
            except Exception:
                self.logger.error(f"[Memory] Failed to load fallback: {traceback.format_exc()}")

        # -----------------------------
        # CRUD methods (thread-safe)
        # -----------------------------
        def set(self, key: str, value: Any):
            """Set a key in memory cache, push to backend if possible."""
            with memory_lock:
                self.memory_cache[key] = value
                try:
                    self.sync_to_backend()
                except Exception:
                    self.logger.error(f"[Memory] Error saving key {key}: {traceback.format_exc()}")

        def get(self, key: str, default: Any = None):
            """Get a key from memory cache."""
            with memory_lock:
                return self.memory_cache.get(key, default)

        def delete(self, key: str):
            """Delete a key from memory cache, remove from backend if possible."""
            with memory_lock:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                try:
                    self.sync_to_backend()
                except Exception:
                    self.logger.error(f"[Memory] Error deleting key {key}: {traceback.format_exc()}")

    # -----------------------------
    # Task Management
    # -----------------------------
    def create_task(self, task_type: str, task_data: dict):
        task_id = f"{task_type}_{uuid4().hex}"
        payload = {
            "type": task_type,
            "data": task_data,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
        }
        self.set(task_id, payload)
        self.logger.info(f"[TaskManager] Created task {task_id} of type {task_type}")
        return task_id

    def run_pending_tasks(self, timeout_per_task: int = 30):
        threads = []
        for key, value in list(self.memory_cache.items()):
            if isinstance(value, dict) and value.get("status") == "pending":
                t = Thread(target=self._execute_task, args=(key, value), daemon=True)
                t.start()
                threads.append(t)
        for t in threads:
            t.join(timeout=timeout_per_task)

    def _execute_task(self, task_id: str, task_info: dict):
        try:
            task_type = task_info.get("type")
            self.logger.info(f"[TaskManager] Executing {task_type} for {task_id}")
            if task_type.startswith("monetize_"):
                self._handle_monetization(task_info)
            elif task_type.startswith("media_"):
                self._handle_media(task_info)
            # Mark complete
            task_info["status"] = "completed"
            task_info["completed_at"] = datetime.utcnow().isoformat()
            self.set(task_id, task_info)
        except Exception as e:
            task_info["status"] = "failed"
            task_info["error"] = str(e)
            self.set(task_id, task_info)
            self.logger.error(f"[TaskManager] Failed task {task_id}: {traceback.format_exc()}")

    def _handle_monetization(self, task_info: dict):
        self.logger.info(f"[Monetization] Placeholder executing: {task_info}")

    def _handle_media(self, task_info: dict):
        self.logger.info(f"[Media] Placeholder generating: {task_info}")

    # -----------------------------
    # Self-fixing / Self-edit (Safe)
    # -----------------------------
    def _syntax_check_python(self, content: str, filename_hint: str = "<string>"):
        """
        Try to compile python content to check for syntax errors.
        """
        try:
            compile(content, filename_hint, "exec")
            return True, None
        except Exception as e:
            return False, str(e)

    def _run_test_command(self):
        """
        Runs user-provided SELF_TEST_CMD (if set). Returns (True, stdout) on success.
        """
        cmd = os.getenv("SELF_TEST_CMD") or self.config.get("self_test_cmd")
        if not cmd:
            # no tests configured -> consider success
            return True, "no tests configured (treated as success)"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            success = result.returncode == 0
            output = (result.stdout or "") + (result.stderr or "")
            return success, output
        except Exception as e:
            return False, f"test command failed: {e}"

    def _create_backup(self, file_path: str) -> Optional[str]:
        try:
            p = Path(file_path)
            if not p.exists():
                return None
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            bak = p.with_suffix(p.suffix + f".bak_{timestamp}")
            p.replace(bak)  # atomic move
            # restore: caller should rename bak back to original
            # but to keep original name available we create a new file with original content (copy)
            # (note: above replace moved file; create a copy of backup to original if needed)
            # We'll copy backup content back when apply final write.
            return str(bak)
        except Exception:
            self.logger.error(f"[SelfEdit] Backup failed for {file_path}: {traceback.format_exc()}")
            return None

    def _restore_backup(self, file_path: str, backup_path: Optional[str]):
        try:
            if not backup_path:
                return False
            p = Path(file_path)
            bak = Path(backup_path)
            if bak.exists():
                if p.exists():
                    p.unlink()
                bak.replace(p)
                return True
            return False
        except Exception:
            self.logger.error(f"[SelfEdit] Restore failed: {traceback.format_exc()}")
            return False

    def can_apply_with_owner_token(self, owner_token: Optional[str]) -> bool:
        """
        Validate provided owner token against env/config. If the env/config
        token is present, only matching token allows override skipping tests.
        """
        if not self.owner_token_env:
            # no owner token configured -> cannot bypass tests via token (but normal test flow may still succeed)
            return False
        return owner_token is not None and str(owner_token) == str(self.owner_token_env)

    def apply_patch_safe(self, file_path: str, new_content: str, run_tests: bool = True, owner_token: Optional[str] = None):
        """
        Safely attempt to apply a code update:
        - Syntax check
        - Backup
        - Write file
        - Run tests (if configured) OR allow owner_token to bypass tests
        - On test failure: restore backup
        Returns dict with keys: success (bool), reason, tests_output (optional)
        """
        file_path = str(file_path)
        ok, err = self._syntax_check_python(new_content, filename_hint=file_path)
        if not ok:
            return {"success": False, "reason": "syntax_error", "details": err}

        # create backup (save original to .bak_TIMESTAMP)
        backup_path = None
        try:
            if Path(file_path).exists():
                # copy original to bak (do not delete original yet)
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{file_path}.bak_{timestamp}"
                with open(file_path, "rb") as src, open(backup_path, "wb") as dst:
                    dst.write(src.read())
                self.logger.info(f"[SelfEdit] Backup created: {backup_path}")
        except Exception:
            self.logger.error(f"[SelfEdit] Backup creation failed: {traceback.format_exc()}")
            return {"success": False, "reason": "backup_failed"}

        # write new content to file
        try:
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            self.logger.info(f"[SelfEdit] Wrote new content to {file_path} (temp)")
        except Exception:
            # If write fails, attempt restore from backup
            if backup_path:
                self._restore_backup(file_path, backup_path)
            return {"success": False, "reason": "write_failed", "details": traceback.format_exc()}

        # Run tests if requested and not bypassed by owner token
        test_output = None
        if run_tests:
            if self.can_apply_with_owner_token(owner_token):
                self.logger.warning("[SelfEdit] Owner token matched: bypassing tests as requested")
                tests_ok = True
                test_output = "bypassed by owner token"
            else:
                tests_ok, test_output = self._run_test_command()
                self.logger.info(f"[SelfEdit] Tests run -> success={tests_ok}")
        else:
            tests_ok = True
            test_output = "tests_not_requested"

        if not tests_ok:
            # restore backup
            restored = False
            if backup_path:
                restored = self._restore_backup(file_path, backup_path)
            return {"success": False, "reason": "tests_failed", "tests_output": test_output, "restored": restored}

        # success: optionally keep backup, or rotate per policy
        return {"success": True, "tests_output": test_output, "backup": backup_path}

    # -----------------------------
    # Auto-fix orchestration (higher-level)
    # -----------------------------
    def auto_fix(self, task_description: str, apply: bool = True, owner_token: Optional[str] = None):
        """
        High-level auto-fix. This function produces a plan and runs a simulation.
        Real code generation must be provided by caller or a secure generator.
        For safety, `apply` must be True and tests must pass (unless owner_token bypass).
        """
        self.logger.info(f"[AutoFix] {'Applying' if apply else 'Simulating'}: {task_description}")
        plan = self._generate_fix_plan(task_description)
        simulation = self._simulate_fix(plan)
        if not simulation.get("success"):
            self.logger.warning("[AutoFix] Simulation failed - aborting")
            return {"success": False, "reason": "simulation_failed", "details": simulation.get("details")}

        if apply:
            # NOTE: This method does not generate file content by itself.
            # Caller should call apply_patch_safe with actual content.
            # Here we return the plan and signal ready-to-apply.
            return {"success": True, "plan": plan, "ready_to_apply": True}
        return {"success": True, "plan": plan, "ready_to_apply": False}

    def _generate_fix_plan(self, description: str):
        return {
            "description": description,
            "steps": ["analyze", "propose_patch", "syntax_check", "backup", "apply", "run_tests", "verify"]
        }

    def _simulate_fix(self, plan: dict):
        # lightweight simulation hook - override/extend in higher-level modules
        try:
            # For now, always succeed simulation; extend with static analysis as needed.
            return {"success": True}
        except Exception as e:
            return {"success": False, "details": str(e)}

    # -----------------------------
    # Capability analysis helpers
    # -----------------------------
    def analyze_self(self):
        self.logger.info("[SelfAnalysis] Checking capabilities...")
        missing = []
        for feature in ["video_generation", "image_generation", "social_media", "monetization"]:
            if not self.external_feature_exists(feature):
                missing.append(feature)
        for f in missing:
            self.create_task(f"monetize_add_{f}", {"feature": f})
        return missing

    # ---------- Robust feature detection method (replace existing one) ----------
    def external_feature_exists(self, feature_name: str) -> bool:
        """
        Robust detection for optional external features.

        - Caches results on the instance in `_feature_cache`.
        - Checks environment variables, config overrides, and presence of Python packages/binaries.
        - Safe and extendable.
        """
        import importlib
        import importlib.util
        import shutil
        import logging

        logger = getattr(self, "logger", logging.getLogger("BibliothecaAI.Features"))

        # per-instance cache
        cache = getattr(self, "_feature_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_feature_cache", cache)

        if feature_name in cache:
            return bool(cache[feature_name])

        result = False
        try:
            cfg = getattr(self, "config", {}) or globals().get("APP_CONFIG", {}) or {}

            if feature_name == "image_generation":
                # keys or client libs
                result = bool(
                    os.getenv("OPENAI_API_KEY")
                    or os.getenv("HUGGINGFACE_API_KEY")
                    or cfg.get("openai_api_key")
                    or cfg.get("huggingface_api_key")
                    or importlib.util.find_spec("openai")
                    or importlib.util.find_spec("diffusers")
                )

            elif feature_name == "video_generation":
                result = bool(importlib.util.find_spec("moviepy") or shutil.which("ffmpeg"))

            elif feature_name == "social_media":
                result = bool(
                    os.getenv("TELEGRAM_BOT_TOKEN")
                    or os.getenv("TWITTER_API_KEY")
                    or os.getenv("X_API_KEY")
                    or os.getenv("INSTAGRAM_TOKEN")
                    or cfg.get("social", {}).get("telegram")
                )

            elif feature_name == "monetization":
                result = bool(
                    os.getenv("FIVERR_API_KEY")
                    or cfg.get("fiverr_api_key")
                    or cfg.get("monetization_enabled")
                )

            elif feature_name == "couchdb":
                # check for couchdb client availability or configured URL
                result = bool(
                    importlib.util.find_spec("couchdb")
                    or importlib.util.find_spec("couchdb3")
                    or cfg.get("couchdb", {}).get("url")
                    or os.getenv("COUCHDB_URL")
                )

            else:
                # generic: check if a python package or binary exists
                result = bool(importlib.util.find_spec(feature_name) or shutil.which(feature_name))

        except Exception as ex:
            logger.debug(f"[Memory] external_feature_exists({feature_name}) error: {ex}", exc_info=True)
            result = False

        cache[feature_name] = bool(result)
        return bool(result)

    # ---------- Safe MEMORY / MemoryStore bootstrap (replace broken instantiation) ----------
    # Defensive bootstrap that:
    #  - tolerates circular imports (defers module import)
    #  - detects available MemoryManager class (local or external module)
    #  - tries multiple constructor signatures
    #  - falls back to an in-memory safe MemoryStore

    import importlib
    import inspect
    import threading
    import traceback

    # local logger (integrates with existing logger if present)
    logger = globals().get("logger", logging.getLogger("BibliothecaAI.Memory"))

    # prefer any MemoryManager already defined in globals()
    _memory_manager_cls = globals().get("MemoryManager", None)

    # try deferred import from app.utils.memory_manager if not present
    if _memory_manager_cls is None:
        try:
            mm_mod = importlib.import_module("app.utils.memory_manager")
            _memory_manager_cls = getattr(mm_mod, "MemoryManager", None)
            if _memory_manager_cls:
                logger.debug("[Memory] Loaded MemoryManager from app.utils.memory_manager")
        except Exception as e:
            # circular import or not installed — defer gracefully
            logger.debug(f"[Memory] Deferred MemoryManager load: {e}")

    # Fallback minimal in-memory MemoryStore (safe and simple)
    class _MemoryStoreFallback:
        def __init__(self, *args, **kwargs):
            self._store = {}
            self.config = kwargs.get("config", {}) if kwargs else globals().get("APP_CONFIG", {}) or {}
            self._lock = threading.RLock()
            self.logger = logger

        def get(self, key, default=None):
            with self._lock:
                return self._store.get(key, default)

        def set(self, key, value):
            with self._lock:
                self._store[key] = value
                return True

        def delete(self, key):
            with self._lock:
                return self._store.pop(key, None) is not None

        def list_keys(self):
            with self._lock:
                return list(self._store.keys())

        def health_check(self):
            return {"available": True, "backend": "fallback", "items": len(self._store)}

        def sync_to_db(self):
            return False  # no-op for fallback

    # instantiate MEMORY robustly
    MEMORY = globals().get("MEMORY", None)
    MemoryStore = globals().get("MemoryStore", None)

    # -----------------------------
    # MEMORY Singleton Initialization — Beyond Perfection Edition
    # -----------------------------
    import logging
    import traceback
    import inspect
    import importlib
    from typing import Optional, Type

    logger = logging.getLogger("BibliothecaAI.Memory")
    logger.setLevel(logging.DEBUG)

    # Fallback minimal in-memory MemoryStore
    class _MemoryStoreFallback:
        """
        Ultra-lightweight in-memory MemoryStore fallback.
        Supports get/set/delete/exists for safe operation if MemoryManager fails.
        """

        def __init__(self):
            self.memory = {}
            self.logger = logger
            self.ai_helper = None

        def get(self, key, default=None):
            return self.memory.get(key, default)

        def set(self, key, value, ttl=None, meta=None):
            self.memory[key] = {"value": value, "ttl": ttl, "meta": meta}
            return True

        def delete(self, key):
            if key in self.memory:
                del self.memory[key]
                return True
            return False

        def exists(self, key):
            return key in self.memory

    # -----------------------------
    # Safe MemoryManager Instantiation
    # -----------------------------
    def _try_instantiate(cls: Optional[Type]):
        """
        Attempts to instantiate MemoryManager class with multiple constructor signatures.
        Returns instance if successful, None otherwise.
        """
        if not cls or not callable(cls):
            logger.debug("[Memory] Invalid MemoryManager class provided; returning None")
            return None

        app_cfg = globals().get("APP_CONFIG", {}) or {}
        attempts = []

        # inspect constructor parameters if possible
        try:
            sig = inspect.signature(cls)
            params = list(sig.parameters.keys())
        except Exception:
            params = []

        # prioritized kwargs attempts
        if "config" in params:
            attempts.append({"config": app_cfg})

        kw = {}
        if "db_name" in params:
            kw["db_name"] = app_cfg.get("memory_db_name",
                                        app_cfg.get("couchdb", {}).get("db_name", "bibliotheca_memory"))
        if "auto_heal" in params:
            kw["auto_heal"] = app_cfg.get("memory_auto_heal", True)
        if kw:
            attempts.append(kw)

        # generic kwargs attempt
        attempts.append({})

        # positional arguments attempts
        attempts_pos = [(), (app_cfg.get("memory_db_name", "bibliotheca_memory"),)]

        # Try kwargs first
        for payload in attempts:
            try:
                inst = cls(**payload)
                return inst
            except TypeError as e:
                logger.debug(f"[Memory] MemoryManager init with kwargs {payload} failed: {e}")
            except Exception as e:
                logger.debug(f"[Memory] MemoryManager init failed (kwargs={payload}): {e}", exc_info=True)

        # Then try positional args
        for args in attempts_pos:
            try:
                inst = cls(*args)
                return inst
            except Exception as e:
                logger.debug(f"[Memory] MemoryManager init with args {args} failed: {e}", exc_info=True)

        return None

    # -----------------------------
    # Initialize MEMORY singleton
    # -----------------------------
    MEMORY: Optional[object] = globals().get("MEMORY", None)
    MemoryStore: Optional[Type] = globals().get("MemoryStore", None)

    try:
        _memory_manager_cls = globals().get("_RealMemoryManager", None)

        if MEMORY is None:
            manager_inst = _try_instantiate(_memory_manager_cls)
            if manager_inst is None:
                # Final fallback
                MemoryStore = _MemoryStoreFallback
                MEMORY = _MemoryStoreFallback()
                logger.warning("[Memory] No working MemoryManager found — using in-memory fallback MemoryStore")
            else:
                MEMORY = manager_inst
                if not hasattr(MEMORY, "logger"):
                    MEMORY.logger = logger
                MemoryStore = _memory_manager_cls
                MEMORY.logger.info("[Memory] MEMORY singleton instantiated successfully ✅")

    except Exception as e:
        logger.error(f"[Memory] Failed to initialize MEMORY singleton: {e}\n{traceback.format_exc()}")
        MemoryStore = _MemoryStoreFallback
        try:
            MEMORY = _MemoryStoreFallback()
            logger.warning("[Memory] Fallback MEMORY instance created after error")
        except Exception:
            MEMORY = None

    # -----------------------------
    # Attach AIMemoryHelper safely
    # -----------------------------
    try:
        if MEMORY is not None and getattr(MEMORY, "ai_helper", None) is None:
            try:
                mod = importlib.import_module("app.utils.ai_memory_helper")
                AIMemoryHelper = getattr(mod, "AIMemoryHelper", None)
                if AIMemoryHelper:
                    MEMORY.ai_helper = AIMemoryHelper(MEMORY)
                    MEMORY.logger.info("[Memory] AI helper attached ✅")
            except Exception:
                if hasattr(MEMORY, "logger"):
                    MEMORY.logger.debug("[Memory] ai_helper attach failed", exc_info=True)
                else:
                    logger.debug("[Memory] ai_helper attach failed", exc_info=True)
    except Exception:
        pass

    # -----------------------------
    # Ensure globals for other modules
    # -----------------------------
    globals()["MEMORY"] = MEMORY
    globals()["MemoryStore"] = MemoryStore

# -----------------------------
# AutonomousAI — Beyond Perfection Edition
# -----------------------------
import os
import traceback
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger("BibliothecaAI.AutonomousAI")
logger.setLevel(logging.DEBUG)

class AutonomousAI:
    """
    Minimal autonomous AI wrapper that integrates with MEMORY singleton,
    supports safe file editing, auto-updates, and external integrations.
    Fully resilient, production-ready, and extendable.
    """
    def __init__(self, ai_instance: Optional[Any] = None):
        self.ai = ai_instance
        # Prefer the AI's memory if provided, else fallback to global MEMORY
        self.memory = getattr(ai_instance, "memory", MEMORY) if ai_instance else MEMORY
        self.task_queue = []
        self.auto_update_enabled = True
        self.external_integrations = {}
        self.logger = getattr(ai_instance, "logger", logger) if ai_instance else logger

        # Initialize integrations safely
        try:
            if hasattr(self, "init_integrations"):
                self.init_integrations()
        except Exception:
            self.logger.error("[AutonomousAI] Integration init failed:\n" + traceback.format_exc())

    # -----------------------------
    # File Editing / Auto-update
    # -----------------------------
    def edit_file(self, file_path: str, new_content: str, run_tests: bool = True, owner_token: Optional[str] = None) -> dict:
        """
        High-level wrapper to safely edit files using MEMORY's safe apply mechanism.
        Returns a result dict with success/failure info.
        """
        try:
            if not hasattr(self.memory, "apply_patch_safe"):
                raise AttributeError("Memory instance does not support apply_patch_safe")
            result = self.memory.apply_patch_safe(file_path, new_content, run_tests=run_tests, owner_token=owner_token)
            return result if isinstance(result, dict) else {"success": bool(result)}
        except Exception as e:
            self.logger.error(f"[AutonomousAI] edit_file failed for {file_path}: {e}\n{traceback.format_exc()}")
            return {"success": False, "reason": str(e)}

    def apply_code_update(
        self,
        description: str,
        file_path: str,
        generate_content_fn: Callable[[], str],
        run_tests: bool = True,
        owner_token: Optional[str] = None
    ) -> dict:
        """
        High-level wrapper for safely updating code files:
        ✅ Simulate auto-fix using MEMORY
        ✅ Safely apply generated content
        ✅ Log outcomes and prevent crashes
        """
        try:
            self.logger.info(f"[AutoUpdate] Preparing update: {description}")
            new_content = generate_content_fn()

            # Simulate plan via memory's auto_fix (apply=False)
            sim = getattr(self.memory, "auto_fix", lambda *a, **kw: {"success": True})(description, apply=False)
            if not sim.get("success", True):
                self.logger.warning("[AutoUpdate] Simulation refused to proceed.")
                return {"success": False, "reason": "simulation_failed"}

            # Apply safely
            result = self.edit_file(file_path, new_content, run_tests=run_tests, owner_token=owner_token)
            if result.get("success"):
                getattr(self.memory, "logger", self.logger).info("[AutoUpdate] Update applied successfully ✅")
            else:
                getattr(self.memory, "logger", self.logger).warning("[AutoUpdate] Update failed: " + str(result))
            return result
        except Exception as e:
            self.logger.error(f"[AutonomousAI] apply_code_update failed: {e}\n{traceback.format_exc()}")
            return {"success": False, "reason": str(e)}

    # -----------------------------
    # External Integrations
    # -----------------------------
    def init_integrations(self):
        """
        Initialize supported external integrations safely.
        Example: Telegram bot.
        """
        try:
            import telegram  # type: ignore
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            if bot_token:
                self.external_integrations["telegram"] = telegram.Bot(token=bot_token)
                self.logger.info("[Integration] Telegram initialized ✅")
            else:
                self.external_integrations["telegram"] = None
                self.logger.warning("[Integration] TELEGRAM_BOT_TOKEN not set; Telegram integration skipped")
        except Exception:
            self.external_integrations["telegram"] = None
            self.logger.warning("[Integration] Telegram init failed (non-fatal)")

    def send_telegram_message(self, chat_id: Optional[str] = None, message: str = "Hello") -> bool:
        """
        Send a message via Telegram integration, if initialized.
        Fails gracefully if bot or chat_id not available.
        """
        try:
            chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
            bot = self.external_integrations.get("telegram")
            if bot and chat_id:
                bot.send_message(chat_id=chat_id, text=message)
                self.logger.info(f"[Telegram] Message sent to {chat_id}")
                return True
            else:
                self.logger.warning("[Telegram] Bot not initialized or chat_id missing; message skipped")
                return False
        except Exception:
            self.logger.error(f"[Telegram] Failed sending message: {traceback.format_exc()}")
            return False

    # -----------------------------
    # Task Queue Management
    # -----------------------------
    def add_task(self, task_callable: Callable, *args, **kwargs):
        """
        Add a task to the AI's internal queue for autonomous execution.
        """
        try:
            self.task_queue.append((task_callable, args, kwargs))
            self.logger.debug(f"[AutonomousAI] Task added: {task_callable.__name__}")
        except Exception:
            self.logger.error(f"[AutonomousAI] Failed to add task: {traceback.format_exc()}")

    def execute_tasks(self, max_tasks: Optional[int] = None):
        """
        Execute tasks in the queue safely, up to max_tasks if specified.
        """
        executed = 0
        while self.task_queue and (max_tasks is None or executed < max_tasks):
            task, args, kwargs = self.task_queue.pop(0)
            try:
                task(*args, **kwargs)
                executed += 1
                self.logger.debug(f"[AutonomousAI] Task executed: {task.__name__}")
            except Exception:
                self.logger.error(f"[AutonomousAI] Task execution failed: {traceback.format_exc()}")
        return executed

# -----------------------------
# Default AI instance
# -----------------------------
AI = AutonomousAI(None)

# -------------------------------
# Memory.py Extension: AI & AutoSync Enhancements
# -------------------------------
import threading
import time
import logging

logger = logging.getLogger("MemoryExtension")

# -----------------------------
# MemoryStore — Beyond-Perfection Edition
# -----------------------------
import threading
import time
import json
import logging
import tempfile
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("BibliothecaAI.MemoryStore")

class MemoryStore:
    """
    Beyond-Perfection MemoryStore
    - Thread-safe memory operations
    - CouchDB persistence with auto-reconnect & fallback to local JSON
    - Local cache for ultra-fast access
    - Health checks & self-pruning expired entries
    - Telemetry hooks (optional integration)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        telemetry: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        start_autosync: bool = True,
    ):
        # Core memory state
        self.lock = threading.RLock()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.config = config or globals().get("CONFIG", {}) or {}
        self.telemetry = telemetry or globals().get("TELEMETRY", None)
        self.logger = logger or globals().get("logger", logging.getLogger("MemoryStore"))

        # CouchDB configuration
        self.db_url = self.config.get("COUCHDB_URL", "http://127.0.0.1:5984")
        self.db_name = self.config.get("COUCHDB_DB", "bibliotheca_memory")

        # Local fallback
        local_dir = Path(self.config.get("LOCAL_STORE_DIR", "data"))
        local_dir.mkdir(parents=True, exist_ok=True)
        self.local_store_file = Path(self.config.get("LOCAL_STORE_FILE", local_dir / "memory_store.json"))

        # Runtime state
        self._couch_available = False
        self._couch_server = None
        self.db = None
        self._last_sync_time: Optional[str] = None
        self._last_persist_success: Optional[bool] = None
        self._auto_sync_thread: Optional[threading.Thread] = None
        self._stop_auto_sync: Optional[threading.Event] = None

        # Initialize CouchDB connection safely
        self._init_couchdb()

        # Preload memory from persistence
        try:
            self.load_all()
        except Exception:
            self.logger.debug("[MemoryStore] load_all() failed on init", exc_info=True)

        # Start background auto-sync if requested
        if start_autosync:
            interval = int(self.config.get("MEMORY_SYNC_INTERVAL", 3600))
            try:
                start_memory_auto_sync(self, interval_sec=interval)
            except Exception:
                self.logger.debug("[MemoryStore] start_memory_auto_sync failed", exc_info=True)

        self.logger.info("MemoryStore initialized ✅ Beyond-Perfection Edition")

    # -----------------------------
    # CouchDB Connection Helpers
    # -----------------------------
    def _init_couchdb(self) -> None:
        """
        Initialize CouchDB connection:
        - Supports auth credentials via environment variables
        - Auto-create database if missing
        - Retries with exponential backoff
        - Fully logs success, warnings, and failures
        - Falls back to local JSON store on failure
        """
        self._couch_available = False
        self._couch_server = None
        self.db = None

        try:
            import couchdb  # type: ignore
        except ImportError:
            self.logger.warning("[MemoryStore] couchdb library not installed — running in local-cache-only mode")
            return

        user = os.getenv("COUCHDB_USER")
        password = os.getenv("COUCHDB_PASSWORD")
        protocol_url = self.db_url
        if user and password and "@" not in protocol_url:
            protocol_url = protocol_url.replace("://", f"://{user}:{password}@")
        self.logger.debug(f"[MemoryStore] Attempting CouchDB connection to {protocol_url}")

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self._couch_server = couchdb.Server(protocol_url)
                version = self._couch_server.version()
                self.logger.info(f"[MemoryStore] CouchDB server reachable, version: {version}")

                # Ensure DB exists
                if self.db_name in self._couch_server:
                    self.db = self._couch_server[self.db_name]
                    self.logger.info(f"[MemoryStore] Connected to existing DB '{self.db_name}' ✅")
                else:
                    self.db = self._couch_server.create(self.db_name)
                    self.logger.info(f"[MemoryStore] Created new DB '{self.db_name}' ✅")

                self._couch_available = True
                break  # success

            except couchdb.http.Unauthorized:
                self.logger.error(f"[MemoryStore] Unauthorized CouchDB access at {self.db_url}")
                break

            except couchdb.http.PreconditionFailed:
                self.logger.warning(f"[MemoryStore] CouchDB DB creation conflict on '{self.db_name}', attempting to use existing DB")
                try:
                    self.db = self._couch_server[self.db_name]
                    self._couch_available = True
                    self.logger.info(f"[MemoryStore] Using existing DB '{self.db_name}' after conflict")
                    break
                except Exception as e:
                    self.logger.error(f"[MemoryStore] Failed to fallback to existing DB: {e}", exc_info=True)
                    continue

            except Exception as e:
                self.logger.warning(f"[MemoryStore] CouchDB connection attempt {attempt} failed: {e}", exc_info=True)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    self.logger.error(f"[MemoryStore] CouchDB unavailable after {max_retries} attempts — running in local-cache-only mode")
                    self._couch_available = False
                    self._couch_server = None
                    self.db = None

    # -----------------------------
    # Local JSON atomic helpers
    # -----------------------------
    def _safe_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        """Write JSON atomically to avoid corruption (temp file + move)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            # atomic replace
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise

    def _persist_local_store(self) -> bool:
        """Persist full cache to local JSON. Returns True on success."""
        with self.lock:
            payload = {
                "metadata": {"persisted_at": datetime.utcnow().isoformat() + "Z"},
                "data": self.cache,
            }
            try:
                self._safe_write_json(self.local_store_file, payload)
                self._last_sync_time = datetime.utcnow().isoformat() + "Z"
                self._last_persist_success = True
                return True
            except Exception as e:
                self.logger.error(f"[MemoryStore] _persist_local_store failed: {e}", exc_info=True)
                self._last_persist_success = False
                # optionally report to telemetry
                if self.telemetry:
                    try:
                        self.telemetry.capture_exception(e, context="MemoryStore._persist_local_store")
                    except Exception:
                        pass
                return False

    def _load_local_store(self) -> None:
        """Load cache from local JSON fallback (if present)."""
        if not self.local_store_file.exists():
            return
        try:
            raw = self.local_store_file.read_text(encoding="utf-8")
            payload = json.loads(raw)
            data = payload.get("data", {})
            if isinstance(data, dict):
                with self.lock:
                    # Accept only dict entries that look like our schema
                    for k, v in data.items():
                        if isinstance(v, dict) and "value" in v:
                            self.cache[k] = v
            self.logger.info(f"[MemoryStore] Loaded {len(self.cache)} items from local store")
        except Exception as e:
            self.logger.error(f"[MemoryStore] _load_local_store failed: {e}", exc_info=True)
            if self.telemetry:
                try:
                    self.telemetry.capture_exception(e, context="MemoryStore._load_local_store")
                except Exception:
                    pass

    # -----------------------------
    # CRUD Operations
    # -----------------------------
    def set(self, key: str, value: Any, ttl: Optional[int] = None, meta: Optional[dict] = None) -> None:
        """
        Store an item in cache, persist locally and attempt CouchDB save.
        - ttl: seconds until expiration (optional)
        - meta: arbitrary metadata dict
        """
        with self.lock:
            now = time.time()
            ent = self.cache.get(key, {})
            version = int(ent.get("version", 0)) + 1
            entry = {
                "key": key,
                "value": value,
                "meta": meta or ent.get("meta", {}),
                "created_at": ent.get("created_at") or datetime.utcfromtimestamp(now).isoformat() + "Z",
                "updated_at": datetime.utcfromtimestamp(now).isoformat() + "Z",
                "ttl": int(ttl) if ttl is not None else None,
                "version": version,
            }
            self.cache[key] = entry

        # persist local ASAP (best-effort)
        try:
            self._persist_local_store()
        except Exception:
            # already logged inside helper
            pass

        # try to persist to CouchDB in a non-blocking but best-effort manner
        try:
            self._save_to_couchdb(key, entry)
        except Exception:
            # logged in helper
            pass

        # telemetry: optional
        if self.telemetry and hasattr(self.telemetry, "add_task"):
            try:
                self.telemetry.add_task(f"MEMORY_SET_{key}", f"Stored {key}", task_type="memory", ai_controlled=True)
            except Exception:
                pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get an item, honoring TTL. Falls back to CouchDB or local store if missing."""
        with self.lock:
            ent = self.cache.get(key)
            if ent:
                ttl = ent.get("ttl")
                if ttl and (time.time() - datetime.fromisoformat(ent["updated_at"].replace("Z", "")).timestamp()) > ttl:
                    # expired
                    try:
                        self.delete(key)
                    except Exception:
                        pass
                    return default
                return ent.get("value", default)

        # try remote DB if present
        if self.db:
            try:
                doc = None
                try:
                    doc = self.db.get(key)
                except Exception:
                    # some couchdb python versions support db[key]
                    try:
                        doc = self.db[key]
                    except Exception:
                        doc = None
                if doc:
                    # normalise doc structure into cache
                    entry = {
                        "key": key,
                        "value": doc.get("value"),
                        "meta": doc.get("meta", {}),
                        "created_at": doc.get("created_at"),
                        "updated_at": doc.get("updated_at"),
                        "ttl": doc.get("ttl"),
                        "version": doc.get("version", 1),
                    }
                    with self.lock:
                        self.cache[key] = entry
                    return entry.get("value", default)
            except Exception as e:
                self.logger.debug(f"[MemoryStore] CouchDB get failed for {key}: {e}")

        # try local JSON
        try:
            if self.local_store_file.exists():
                # lazy read (not optimal for huge files but safe)
                raw = self.local_store_file.read_text(encoding="utf-8")
                payload = json.loads(raw)
                data = payload.get("data", {})
                if key in data:
                    entry = data[key]
                    with self.lock:
                        self.cache[key] = entry
                    return entry.get("value", default)
        except Exception as e:
            self.logger.debug(f"[MemoryStore] local fallback get failed for {key}: {e}")

        return default

    def delete(self, key: str) -> bool:
        """Delete key from cache and attempt CouchDB delete. Returns True if something removed."""
        removed = False
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                removed = True
        # persist local state
        try:
            self._persist_local_store()
        except Exception:
            pass

        if self.db:
            try:
                # attempt get by id then delete doc
                doc = None
                try:
                    doc = self.db.get(key)
                except Exception:
                    try:
                        doc = self.db[key]
                    except Exception:
                        doc = None
                if doc:
                    # couchdb-python expects dict with '_id' and '_rev' to delete
                    try:
                        self.db.delete(doc)
                    except Exception:
                        # fallback: set a tombstone field if delete fails
                        doc['_deleted'] = True
                        try:
                            self.db.save(doc)
                        except Exception:
                            pass
                    removed = True
            except Exception as e:
                self.logger.debug(f"[MemoryStore] CouchDB delete failed for {key}: {e}", exc_info=True)
        return removed

    def clear(self) -> None:
        """Clear everything from cache and attempt to clear CouchDB (best-effort)."""
        with self.lock:
            self.cache.clear()
        try:
            self._persist_local_store()
        except Exception:
            pass

        if self.db:
            try:
                # iterate keys and delete
                for docid in list(self.db):
                    try:
                        doc = self.db[docid]
                        self.db.delete(doc)
                    except Exception:
                        continue
            except Exception as e:
                self.logger.warning(f"[MemoryStore] CouchDB clear failed: {e}", exc_info=True)

    # -----------------------------
    # CouchDB per-item helpers
    # -----------------------------
    def _save_to_couchdb(self, key: str, entry: Dict[str, Any]) -> None:
        """Save a single entry to CouchDB safely (best-effort)."""
        if not self.db:
            return
        try:
            # normalize doc payload
            doc = {
                "_id": key,
                "type": "memory_item",
                "value": entry.get("value"),
                "meta": entry.get("meta", {}),
                "ttl": entry.get("ttl"),
                "created_at": entry.get("created_at"),
                "updated_at": entry.get("updated_at"),
                "version": entry.get("version", 1),
            }
            existing = None
            try:
                existing = self.db.get(key)
            except Exception:
                try:
                    existing = self.db[key]
                except Exception:
                    existing = None
            if existing:
                # preserve _rev if present
                if isinstance(existing, dict) and existing.get("_rev"):
                    doc["_rev"] = existing["_rev"]
                # we merge metadata to avoid destroying other fields
                try:
                    existing.update(doc)
                    self.db.save(existing)
                except Exception:
                    # fallback to save doc (db.save may accept dict or tuple)
                    self.db.save(doc)
            else:
                self.db.save(doc)
            # note successful remote persist
            self._last_sync_time = datetime.utcnow().isoformat() + "Z"
            self._last_persist_success = True
        except Exception as e:
            self.logger.warning(f"[MemoryStore] CouchDB save failed for {key}: {e}", exc_info=True)
            self._last_persist_success = False
            if self.telemetry:
                try:
                    self.telemetry.capture_exception(e, context=f"MemoryStore._save_to_couchdb:{key}")
                except Exception:
                    pass

    def load_all(self) -> None:
        """Load entries from CouchDB if available, otherwise from local JSON."""
        loaded = 0
        if self.db:
            try:
                with self.lock:
                    for docid in self.db:
                        try:
                            doc = self.db[docid]
                            if doc and "value" in doc:
                                entry = {
                                    "key": docid,
                                    "value": doc.get("value"),
                                    "meta": doc.get("meta", {}),
                                    "created_at": doc.get("created_at"),
                                    "updated_at": doc.get("updated_at"),
                                    "ttl": doc.get("ttl"),
                                    "version": doc.get("version", 1),
                                }
                                self.cache[docid] = entry
                                loaded += 1
                        except Exception:
                            continue
                self.logger.info(f"[MemoryStore] Loaded {loaded} entries from CouchDB")
                return
            except Exception as e:
                self.logger.warning(f"[MemoryStore] load_all from CouchDB failed: {e}", exc_info=True)
        # local fallback
        try:
            self._load_local_store()
        except Exception:
            pass

    # -----------------------------
    # Utility / Metadata / Health
    # -----------------------------
    def exists(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                return True
        if self.db:
            try:
                return key in self.db
            except Exception:
                return False
        try:
            if self.local_store_file.exists():
                raw = self.local_store_file.read_text(encoding="utf-8")
                payload = json.loads(raw)
                return key in payload.get("data", {})
        except Exception:
            return False

    def query(self, prefix: str = "") -> List[Tuple[str, Any]]:
        with self.lock:
            return [(k, v.get("value")) for k, v in self.cache.items() if not prefix or k.startswith(prefix)]

    def health_check(self) -> Dict[str, Any]:
        """Return diagnostic information about the memory store state."""
        try:
            cache_size = len(self.cache)
        except Exception:
            cache_size = None
        couch_available = bool(self._couch_available)
        couch_connected = bool(self.db is not None)
        local_exists = self.local_store_file.exists()
        try:
            local_size = self.local_store_file.stat().st_size if local_exists else None
        except Exception:
            local_size = None
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cache_size": cache_size,
            "couchdb_library_available": couch_available,
            "couchdb_connected": couch_connected,
            "local_store_exists": local_exists,
            "local_store_size": local_size,
            "last_sync_time": self._last_sync_time,
            "last_persist_success": self._last_persist_success,
            "auto_sync_active": bool(self._auto_sync_thread is not None and getattr(self._auto_sync_thread, "is_alive", lambda: False)()),
        }

    def save(self, sync_remote: bool = False, max_remote_items: int = 200) -> bool:
        """
        Persist cache to local JSON and optionally to CouchDB.
        Returns True if any persistence succeeded.
        """
        any_ok = False
        try:
            # local persistence first (atomic)
            any_ok = self._persist_local_store() or any_ok
        except Exception:
            any_ok = any_ok

        if sync_remote and self.db:
            # attempt per-item remote sync (bounded)
            with self.lock:
                keys = list(self.cache.keys())[:max_remote_items]
            for k in keys:
                try:
                    self._save_to_couchdb(k, self.cache[k])
                    any_ok = True
                except Exception:
                    continue
        return bool(any_ok)

    def flush(self) -> bool:
        """Alias for save(sync_remote=True)."""
        return self.save(sync_remote=True)

    def prune_expired(self) -> int:
        """Remove expired items according to ttl; returns number removed."""
        removed = 0
        now_ts = time.time()
        keys = []
        with self.lock:
            keys = list(self.cache.keys())
        for k in keys:
            try:
                with self.lock:
                    ent = self.cache.get(k)
                    if not ent:
                        continue
                    ttl = ent.get("ttl")
                    if ttl and ent.get("updated_at"):
                        updated_ts = datetime.fromisoformat(ent["updated_at"].replace("Z", "")).timestamp()
                        if (now_ts - updated_ts) > ttl:
                            # expired
                            try:
                                self.delete(k)
                                removed += 1
                            except Exception:
                                continue
            except Exception:
                continue
        if removed:
            try:
                self._persist_local_store()
            except Exception:
                pass
        return removed

    def export_json(self) -> str:
        with self.lock:
            return json.dumps({k: v.get("value") for k, v in self.cache.items()}, indent=2, default=str)

    def import_json(self, data: str) -> None:
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    # set without ttl/meta
                    self.set(k, v)
        except Exception as e:
            self.logger.error(f"[MemoryStore] import_json failed: {e}", exc_info=True)


# -----------------------------
# Safe AI Accessor & Memory Integration — Beyond Perfection Edition
# -----------------------------
import os
import logging
import threading
import time
import atexit
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Only for type checking to avoid circular imports at runtime
    from app.utils.memory_manager import MemoryManager

# Module-level logger
logger = logging.getLogger("BibliothecaAI.Memory")
logger.setLevel(logging.DEBUG)

# -----------------------------
# AIMemoryHelper — Safe AI Memory Access
# -----------------------------
class AIMemoryHelper:
    """
    Safe accessor wrapper for MemoryStore / MemoryManager, intended for AI modules.
    Handles errors, fallback, and logging.
    """
    def __init__(self, memory_instance: "MemoryManager"):
        self._mem = memory_instance

    def get(self, key: str, default: Any = None) -> Any:
        try:
            get_fn = getattr(self._mem, "get", None)
            if callable(get_fn):
                return get_fn(key, default)
            return getattr(self._mem, "memory", {}).get(key, default)
        except Exception as e:
            getattr(self._mem, "logger", logger).warning(f"[AIMemoryHelper] get() failed for {key}: {e}", exc_info=True)
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None, meta: Optional[dict] = None) -> bool:
        try:
            set_fn = getattr(self._mem, "set", None)
            if callable(set_fn):
                return set_fn(key, value, ttl=ttl, meta=meta)
            mem_store = getattr(self._mem, "memory", None)
            if isinstance(mem_store, dict):
                mem_store[key] = {"value": value, "meta": meta, "ttl": ttl}
                return True
            return False
        except Exception as e:
            getattr(self._mem, "logger", logger).warning(f"[AIMemoryHelper] set() failed for {key}: {e}", exc_info=True)
            return False

    def delete(self, key: str) -> bool:
        try:
            del_fn = getattr(self._mem, "delete", None)
            if callable(del_fn):
                return del_fn(key)
            mem_store = getattr(self._mem, "memory", None)
            if isinstance(mem_store, dict) and key in mem_store:
                del mem_store[key]
                return True
            return False
        except Exception as e:
            getattr(self._mem, "logger", logger).warning(f"[AIMemoryHelper] delete() failed for {key}: {e}", exc_info=True)
            return False

    def exists(self, key: str) -> bool:
        try:
            exists_fn = getattr(self._mem, "exists", None)
            if callable(exists_fn):
                return bool(exists_fn(key))
            mem_store = getattr(self._mem, "memory", None)
            if isinstance(mem_store, dict):
                return key in mem_store
            return False
        except Exception:
            return False

# -----------------------------
# Background Auto-Sync / Auto-Prune
# -----------------------------
def start_memory_auto_sync(
    memory_instance: "MemoryManager",
    interval_sec: Optional[int] = None,
    prune_interval_sec: Optional[int] = None,
) -> threading.Thread:
    """
    Fully safe daemon thread to auto-save and prune memory.
    Handles errors, dynamic config updates, and graceful shutdown.
    """
    logger_local = getattr(memory_instance, "logger", logging.getLogger("MemoryManager"))

    # Determine defaults
    interval_sec = interval_sec or getattr(getattr(memory_instance, "config", {}), "MEMORY_SYNC_INTERVAL", 3600)
    prune_interval_sec = prune_interval_sec or getattr(getattr(memory_instance, "config", {}), "MEMORY_PRUNE_INTERVAL", 86400)

    # Prevent double-start
    existing_thread = getattr(memory_instance, "_auto_sync_thread", None)
    if existing_thread and getattr(existing_thread, "is_alive", lambda: False)():
        logger_local.debug("[Memory] Auto-sync already running, skipping start")
        return existing_thread

    stop_event = threading.Event()
    setattr(memory_instance, "_stop_auto_sync", stop_event)

    def _loop():
        logger_local.info("[Memory] Auto-sync thread started ✅")
        last_prune = time.time()

        try:
            while not stop_event.is_set():
                # Dynamic interval reload
                try:
                    cfg = getattr(memory_instance, "config", {})
                    interval = int(getattr(cfg, "MEMORY_SYNC_INTERVAL", interval_sec))
                    prune_interval = int(getattr(cfg, "MEMORY_PRUNE_INTERVAL", prune_interval_sec))
                except Exception:
                    interval = interval_sec
                    prune_interval = prune_interval_sec

                # Save memory
                try:
                    if hasattr(memory_instance, "save"):
                        memory_instance.save(sync_remote=True)
                    elif hasattr(memory_instance, "sync_to_db"):
                        memory_instance.sync_to_db()
                    logger_local.debug("[Memory] Memory saved successfully")
                except Exception as e:
                    logger_local.error(f"[Memory] Error saving memory: {e}", exc_info=True)

                # Prune memory
                if time.time() - last_prune >= prune_interval:
                    try:
                        if hasattr(memory_instance, "prune_expired"):
                            removed = memory_instance.prune_expired()
                            if removed:
                                logger_local.info(f"[Memory] Auto-pruned {removed} expired entries")
                        elif hasattr(memory_instance, "prune"):
                            memory_instance.prune()
                            logger_local.info("[Memory] prune() executed")
                    except Exception as e:
                        logger_local.error(f"[Memory] Error pruning memory: {e}", exc_info=True)
                    last_prune = time.time()

                # Sleep
                stop_event.wait(interval)

        except Exception as e:
            logger_local.critical(f"[Memory] Auto-sync thread crashed: {e}", exc_info=True)
        finally:
            logger_local.info("[Memory] Auto-sync thread exiting gracefully ✅")

    thread = threading.Thread(target=_loop, name="MemoryAutoSyncThread", daemon=True)
    thread.start()
    setattr(memory_instance, "_auto_sync_thread", thread)
    logger_local.info(f"[Memory] Auto-sync initialized | save every {interval_sec}s, prune every {prune_interval_sec}s")
    return thread

# -----------------------------
# Attach Auto-Sync to MemoryManager safely
# -----------------------------
def attach_auto_sync_method_to_memory_manager():
    """
    Attaches start_auto_sync to MemoryManager if not present.
    Fully robust for fallbacks, circular import prevention, and beyond-perfection usage.
    """
    try:
        from app.utils.memory import MemoryManager
    except Exception:
        logger.debug("[Memory] MemoryManager import failed; auto-sync attachment deferred")
        return

    if hasattr(MemoryManager, "start_auto_sync"):
        logger.debug("[Memory] start_auto_sync already attached")
        return

    def start_auto_sync(self, interval_sec: Optional[int] = 5, prune_interval_sec: Optional[int] = 60):
        return start_memory_auto_sync(self, interval_sec=interval_sec, prune_interval_sec=prune_interval_sec)

    MemoryManager.start_auto_sync = start_auto_sync
    logger.info("[Memory] start_auto_sync method attached to MemoryManager ✅")

# Call once after MemoryManager class definition
attach_auto_sync_method_to_memory_manager()

# -----------------------------
# MEMORY singleton integration — Beyond-Perfection Edition
# -----------------------------
MEMORY: Optional["MemoryManager"] = None
MemoryStore = None

def _instantiate_memory_singleton() -> Optional["MemoryManager"]:
    """
    Beyond-Perfection MEMORY singleton initializer.

    Responsibilities:
      - Robust MemoryManager creation
      - Fallback to MemoryStoreFallback if CouchDB/MemoryManager fails
      - Safe attachment of AIMemoryHelper
      - Auto-sync and auto-prune threads
      - Full logging and error resilience
      - Circular import safety
      - Compatibility monkeypatch for legacy interfaces
      - Exit-safe flush
    """
    global MEMORY, MemoryStore
    manager_logger = logging.getLogger("BibliothecaAI.MemoryManager")

    if MEMORY is not None:
        return MEMORY  # Already instantiated

    try:
        # ================= Delayed Imports (Circular Import Safe) =================
        # import module-level MemoryManager only when needed
        from app.utils.memory_manager import MemoryManager as MM  # type: ignore
        from app.utils.memory_store_fallback import MemoryStoreFallback  # type: ignore

        MemoryStore = MM

        # ================= Instantiate Primary MemoryManager =================
        try:
            MEMORY = MM()
            if not hasattr(MEMORY, "logger"):
                MEMORY.logger = manager_logger
            MEMORY.logger.info("[Memory] MEMORY singleton instantiated ✅")
        except Exception as primary_error:
            manager_logger.warning(f"[Memory] MemoryManager failed ({primary_error}), using fallback", exc_info=True)
            MEMORY = MemoryStoreFallback()
            if not hasattr(MEMORY, "logger"):
                MEMORY.logger = manager_logger
            MEMORY.logger.info("[Memory] Fallback MemoryStoreFallback initialized ✅")

        # ================= Attach AIMemoryHelper Safely =================
        if MEMORY and not getattr(MEMORY, "ai_helper", None):
            try:
                MEMORY.ai_helper = AIMemoryHelper(MEMORY)
                MEMORY.logger.info("[Memory] AIMemoryHelper attached ✅")
            except Exception as e:
                MEMORY.logger.warning(f"[Memory] AIMemoryHelper attach failed: {e}", exc_info=True)

        # ================= Start Auto-Sync / Auto-Prune Thread =================
        if MEMORY:
            try:
                cfg = getattr(MEMORY, "config", {}) or {}
                interval = int(getattr(cfg, "MEMORY_SYNC_INTERVAL", cfg.get("MEMORY_SYNC_INTERVAL", 3600)))
                prune_interval = int(getattr(cfg, "MEMORY_PRUNE_INTERVAL", cfg.get("MEMORY_PRUNE_INTERVAL", 86400)))
                # Use the start_memory_auto_sync defined above (supports prune_interval_sec)
                start_memory_auto_sync(MEMORY, interval_sec=interval, prune_interval_sec=prune_interval)
                MEMORY.logger.info(f"[Memory] Auto-sync thread started — interval {interval}s, prune {prune_interval}s ✅")
            except Exception as e:
                MEMORY.logger.warning(f"[Memory] Auto-sync thread failed to start: {e}", exc_info=True)

        # ================= Register Safe atexit Flush =================
        if MEMORY:
            def _flush_memory():
                try:
                    if hasattr(MEMORY, "sync_to_db"):
                        MEMORY.sync_to_db()
                        MEMORY.logger.info("[Memory] sync_to_db() completed at exit ✅")
                    if hasattr(MEMORY, "stop"):
                        try:
                            MEMORY.stop()
                            MEMORY.logger.info("[Memory] MEMORY.stop() completed at exit ✅")
                        except TypeError:
                            # older implementations may need no args
                            MEMORY.stop()
                except Exception:
                    MEMORY.logger.exception("[Memory] Error during atexit flush", exc_info=True)
            try:
                atexit.register(_flush_memory)
            except Exception:
                manager_logger.debug("[Memory] atexit registration for flush failed (non-fatal)")

    except Exception as e:
        manager_logger.exception(f"[Memory] Critical error instantiating MEMORY singleton: {e}", exc_info=True)
        MEMORY = None
        MemoryStore = None

    # ================= Health Check & Final Heartbeat =================
    if MEMORY is None:
        manager_logger.warning("[Memory] MEMORY singleton is None — limited functionality ⚠️")
    else:
        try:
            MEMORY.logger.info(f"[Memory] MEMORY singleton ready: {type(MEMORY).__name__} ✅")
        except Exception:
            manager_logger.info("[Memory] MEMORY singleton ready (logger unavailable)")

    # ================= Compatibility Monkeypatch =================
    try:
        import types as _types
        if MEMORY:
            if not hasattr(MEMORY, "logger"):
                MEMORY.logger = logging.getLogger("BibliothecaAI.MemoryManager")

            # Health check method
            if not hasattr(MEMORY, "health_check"):
                def _mm_health_check(self):
                    try:
                        return {
                            "ok": True,
                            "status": {
                                "couchdb_connected": bool(getattr(self, "couch", None)),
                                "db_name": getattr(self, "db_name", None),
                                "local_store_loaded": hasattr(self, "local_store"),
                                "persistent_state_loaded": hasattr(self, "persistent_state"),
                                "mode": getattr(self, "mode", "unknown"),
                                "last_sync": getattr(self, "last_sync_time", None),
                                "ai_helper_attached": hasattr(self, "ai_helper"),
                            },
                        }
                    except Exception as e:
                        return {"ok": False, "error": str(e)}
                MEMORY.health_check = _types.MethodType(_mm_health_check, MEMORY)

            # Memory alias for legacy compatibility
            if not hasattr(MEMORY, "memory") or MEMORY.memory is None:
                MEMORY.memory = getattr(
                    MEMORY,
                    "local_store",
                    getattr(MEMORY, "semantic_memory", getattr(MEMORY, "memory_store", {})),
                )
    except Exception:
        # Don't allow monkeypatch errors to break startup
        pass

    # ================= Assign Globals =================
    globals()["MEMORY"] = MEMORY
    globals()["MemoryStore"] = MemoryStore

    return MEMORY

# ================= Instantiate Singleton Immediately =================
MEMORY = _instantiate_memory_singleton()

# -----------------------------
# Utility helpers for patching & health checks
# -----------------------------
def _compute_source_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()

def propose_patch(original_path: str, proposed_text: str) -> dict:
    """
    Writes proposed_text to a temp file, attempts to compile, and returns status.
    """
    try:
        with open(original_path, "r", encoding="utf-8") as fh:
            orig_text = fh.read()
    except Exception as e:
        return {"ok": False, "error": f"Could not read original: {e}"}

    orig_hash = _compute_source_hash(orig_text)
    prop_hash = _compute_source_hash(proposed_text)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(tempfile.gettempdir(), f"proposed_memory_{ts}.py")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(proposed_text)
        try:
            _py_compile.compile(tmp_path, doraise=True)
            compile_ok = True
            compile_error = None
        except Exception:
            compile_ok = False
            compile_error = traceback.format_exc()
        return {
            "ok": True,
            "tmp_path": tmp_path,
            "orig_hash": orig_hash,
            "prop_hash": prop_hash,
            "compile_ok": compile_ok,
            "compile_error": compile_error,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def apply_patch_with_approval(original_path: str, tmp_path: str, approval_token: str = None) -> dict:
    """
    Apply a previously proposed patch. Requires explicit approval token to prevent accidental overwrites.
    """
    if approval_token != "APPROVE_APPLY_PATCH":
        return {"ok": False, "error": "Missing or invalid approval token. Provide approval_token='APPROVE_APPLY_PATCH' to apply."}
    try:
        _py_compile.compile(tmp_path, doraise=True)
        bak_path = original_path + ".bak_" + datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(original_path, bak_path)
        shutil.copyfile(tmp_path, original_path)
        return {"ok": True, "backup": bak_path, "applied": original_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def memory_health_snapshot() -> dict:
    """
    Return quick diagnostics about the MEMORY singleton.
    """
    try:
        if "MEMORY" in globals() and globals()["MEMORY"] is not None:
            mem = globals()["MEMORY"]
            hc = getattr(mem, "health_check", None)
            if callable(hc):
                return {"ok": True, "health": hc()}
            else:
                return {
                    "ok": True,
                    "health": {
                        "has_memory_attr": hasattr(mem, "memory"),
                        "has_local_store": hasattr(mem, "local_store"),
                        "has_couch": hasattr(mem, "couch"),
                    },
                }
        else:
            return {"ok": False, "error": "MEMORY not found in globals"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# === END APPENDED PERFECTION HELPERS ===
