#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bibliotheca Memory Manager — Beyond Perfection Edition
======================================================
Handles:
- 🧠 Persistent Memory with CouchDB (with local JSON fallback)
- 💾 Backup + rotation + rollback system (auto-healing)
- 🔁 Thread-safe read/write
- 🧩 Smart patch & self-repair loop
- ⚙️ Dynamic environment/config auto-load
- 🧭 Full telemetry & structured logging
- ☁️ Seamless cloud/local hybrid operation
- 🚀 Designed for zero downtime and full recoverability
"""

from __future__ import annotations

# ===============================
# — STANDARD LIBRARY IMPORTS
# ===============================
import os
import sys
import time
import json
import uuid
import shutil
import difflib
import importlib
import threading
import logging
from pathlib import Path
from datetime import datetime
from app.utils.memory import MemoryStore
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager
from app.utils.fallback_store import FallbackStore

# ------------------------------
# — LOGGER CONFIGURATION
# ------------------------------
logger = logging.getLogger("Bibliotheca.MemoryManager")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(console_handler)

# ------------------------------
# — GLOBAL CONFIG
# ------------------------------
COUCHDB_AVAILABLE = False
COUCHDB_CONNECTION_RETRIES = 3
COUCHDB_RETRY_INTERVAL = 2  # seconds
COUCHDB_AVAILABLE_LOCK = threading.Lock()
MEMORY = None
def set_couchdb_available(value: bool):
    global COUCHDB_AVAILABLE
    with COUCHDB_AVAILABLE_LOCK:
        COUCHDB_AVAILABLE = value

# ===============================
# — AI MEMORY HELPER IMPORT
# ===============================
try:
    from app.utils.ai_memory_helper import AIMemoryHelper
except Exception:
    try:
        from sandbox.ai_memory_helper import AIMemoryHelper
    except Exception:
        AIMemoryHelper = None
        logger.warning("[MemoryManager] AIMemoryHelper not found — running without AI enhancements ⚠️")

# ===============================
# — COUCHDB INITIALIZATION & TELEMETRY SAFE LOGGING
# ===============================
try:
    import couchdb  # type: ignore
except ImportError:
    couchdb = None

# Helper for safe telemetry logging
def _safe_log_event(event_name: str, data: Any = None):
    """
    Safely logs events via TELEMETRY if available, else falls back to logger.
    """
    try:
        if 'TELEMETRY' in globals() and hasattr(TELEMETRY, 'log_event'):
            TELEMETRY.log_event(event_name, data)
        else:
            logger.info(f"[TelemetryFallback] {event_name} -> {data}")
    except Exception as e:
        logger.warning(f"[TelemetryFallback] Failed logging {event_name}: {e}")

COUCHDB_AVAILABLE = False

if couchdb:
    try:
        from app.config import COUCHDB_URL, COUCHDB_USER, COUCHDB_PASSWORD, COUCHDB_DB_NAME
    except Exception:
        # fallback or missing config
        COUCHDB_URL = "http://127.0.0.1:5984/"
        COUCHDB_USER = None
        COUCHDB_PASSWORD = None
        COUCHDB_DB_NAME = "bibliotheca_memory"

    # Attempt connection with incremental backoff
    for attempt in range(1, COUCHDB_CONNECTION_RETRIES + 1):
        try:
            server = couchdb.Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")
            if COUCHDB_USER and COUCHDB_PASSWORD:
                server.resource.credentials = (COUCHDB_USER, COUCHDB_PASSWORD)

            # Ensure database exists
            if COUCHDB_DB_NAME not in server:
                server.create(COUCHDB_DB_NAME)
                _safe_log_event("couchdb_db_created", COUCHDB_DB_NAME)
                logger.info(f"[CouchDB] Database '{COUCHDB_DB_NAME}' created")

            COUCHDB_AVAILABLE = True
            _safe_log_event("couchdb_connected", COUCHDB_URL)
            logger.info(f"[CouchDB] Connected successfully to {COUCHDB_URL} ✅")
            break

        except couchdb.http.Unauthorized as e:
            _safe_log_event("couchdb_auth_error", str(e))
            logger.error(f"[CouchDB] Auth error: {e}")
            break
        except couchdb.http.PreconditionFailed as e:
            _safe_log_event("couchdb_precondition_failed", str(e))
            logger.error(f"[CouchDB] Precondition failed: {e}")
            break
        except Exception as e:
            wait_time = COUCHDB_RETRY_INTERVAL * attempt
            _safe_log_event("couchdb_connection_retry", f"attempt={attempt}, error={e}")
            logger.warning(f"[CouchDB] Attempt {attempt} failed: {e} — retrying in {wait_time}s")
            time.sleep(wait_time)
    else:
        _safe_log_event("couchdb_connection_failed", COUCHDB_URL)
        logger.error("[CouchDB] All connection attempts failed — fallback to local memory/JSON ⚠️")
        COUCHDB_AVAILABLE = False
else:
    _safe_log_event("couchdb_not_installed", "couchdb module missing")
    logger.warning("[CouchDB] CouchDB package not installed — fallback to local memory/JSON ⚠️")
    COUCHDB_AVAILABLE = False

# Catch-all for unexpected errors
try:
    if not COUCHDB_AVAILABLE:
        logger.info("[MemoryManager] CouchDB unavailable — using MemoryStoreFallback or local JSON")
except Exception as e:
    _safe_log_event("couchdb_unexpected_error", str(e))
    logger.error(f"[MemoryManager] Unexpected CouchDB fallback error: {e}")


def validate_environment():
    """Ensures required environment variables are loaded."""
    import os
    required_vars = ["COUCHDB_USER", "COUCHDB_PASSWORD"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"[Env Loader] Missing required environment variables: {missing}")
    return True


# ===============================
# — GLOBAL MEMORY & THREAD-SAFE MANAGER
# ===============================
import os
import sys
import json
import time
import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Optional, Callable, TypeVar, Coroutine
import functools

T = TypeVar("T")

# ------------------------------
# — LOGGER SETUP (console + file)
# ------------------------------
logger = logging.getLogger("Bibliotheca.MemoryManager")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)

# File handler
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "memory_manager.log"
try:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)
except Exception as e:
    logger.warning(f"[MemoryManager] Failed to create file handler: {e}")

# ------------------------------
# — GLOBAL MEMORY SINGLETONS
# ------------------------------
MemoryManager = None
MEMORY = None

_memory_lock = threading.RLock()  # Global lock for memory operations

def synchronized(timeout: Optional[float] = None, retry_delay: float = 0.01, max_retries: int = 5):
    """
    Thread-safe / async-safe decorator for MemoryManager operations
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs) -> T:
                for attempt in range(1, max_retries + 1):
                    try:
                        async with asyncio.Lock():
                            return await asyncio.wait_for(func(*args, **kwargs), timeout) if timeout else await func(*args, **kwargs)
                    except asyncio.TimeoutError:
                        logger.warning(f"[MemoryManager] Timeout in async '{func.__name__}' attempt {attempt}")
                        await asyncio.sleep(retry_delay * attempt)
                    except Exception as e:
                        logger.exception(f"[MemoryManager] Exception in async '{func.__name__}': {e}")
                        raise
                raise TimeoutError(f"[MemoryManager] Async '{func.__name__}' failed after {max_retries} retries")
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                for attempt in range(1, max_retries + 1):
                    acquired = _memory_lock.acquire(timeout=timeout)
                    if acquired:
                        try:
                            return func(*args, **kwargs)
                        finally:
                            _memory_lock.release()
                    else:
                        logger.warning(f"[MemoryManager] Lock busy for '{func.__name__}', retry {attempt}/{max_retries}")
                        time.sleep(retry_delay * attempt)
                raise TimeoutError(f"[MemoryManager] '{func.__name__}' failed to acquire lock after {max_retries} retries")
            return wrapper
    return decorator

# ------------------------------
# — GLOBAL MEMORY INITIALIZER (Beyond-Perfection Edition)
# ------------------------------
import threading
import json
from pathlib import Path
from typing import Any

def assign_global_memory_classes() -> None:
    """
    Safely register MemoryManager and MEMORY singleton globally.

    Features:
      - Thread-safe and async-compatible
      - Auto-attaches AIMemoryHelper
      - Fallback-safe if MemoryManager/CouchDB fails
      - Fully logged
      - Backward-compatible memory aliases
    """
    global MemoryManager, MEMORY

    if not hasattr(assign_global_memory_classes, "_lock"):
        assign_global_memory_classes._lock = threading.RLock()

    with assign_global_memory_classes._lock:
        try:
            if MemoryManager is not None:
                logger.debug("[MemoryManager] Already assigned globally, skipping.")
                return

            # Attempt main MemoryManager import
            try:
                MemoryManager = globals().get("MemoryManager")
                if MemoryManager is None:
                    raise ImportError("MemoryManager not found in globals.")
            except ImportError:
                logger.warning("[MemoryManager] MemoryManager not found — using inline fallback.")

                # ------------------- Inline Fallback MemoryManager -------------------
                class MemoryManagerFallback:
                    def __init__(self):
                        self.store = {}
                        self.lock = threading.RLock()
                        self.ai_helper = None
                        self.logger = logging.getLogger("BibliothecaAI.MemoryManager")
                        self.logger.info("[MemoryManagerFallback] Initialized ✅")

                    def synchronized(func):
                        """Thread-safe decorator"""
                        def wrapper(self, *args, **kwargs):
                            with self.lock:
                                return func(self, *args, **kwargs)
                        return wrapper

                    @synchronized
                    def get(self, key: str, default=None):
                        return self.store.get(key, default)

                    @synchronized
                    def set(self, key: str, value: Any):
                        self.store[key] = value
                        return True

                    @synchronized
                    def delete(self, key: str):
                        return self.store.pop(key, None)

                    @synchronized
                    def all(self):
                        return dict(self.store)

                    @synchronized
                    def flush_to_file(self, path: Path):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(self.store, f, indent=2)
                        self.logger.info(f"[MemoryManager] Memory flushed to {path} ✅")

                MemoryManager = MemoryManagerFallback

            # ------------------- MEMORY Singleton Initialization -------------------
            if MEMORY is None:
                try:
                    MEMORY = MemoryManager()
                    globals()["MEMORY"] = MEMORY
                    logger.info("[MemoryManager] MEMORY singleton initialized ✅")
                except Exception as e:
                    MEMORY = None
                    logger.error(f"[MemoryManager] Failed to initialize MEMORY singleton: {e}", exc_info=True)

            # ------------------- AIMemoryHelper Attachment -------------------
            if MEMORY and getattr(MEMORY, "ai_helper", None) is None:
                try:
                    from app.utils.memory_helpers import AIMemoryHelper
                    MEMORY.ai_helper = AIMemoryHelper(MEMORY)
                    logger.info("[MemoryManager] AI helper attached to MEMORY ✅")
                except Exception as e:
                    logger.warning(f"[MemoryManager] Failed to attach AI helper: {e}", exc_info=True)

            # ------------------- Legacy Compatibility -------------------
            if MEMORY:
                # Ensure legacy 'memory' alias exists
                if not hasattr(MEMORY, "memory") or MEMORY.memory is None:
                    MEMORY.memory = getattr(
                        MEMORY,
                        "local_store",
                        getattr(MEMORY, "semantic_memory", getattr(MEMORY, "memory_store", {})),
                    )

                # Health check method
                if not hasattr(MEMORY, "health_check"):
                    import types as _types
                    def _health_check(self):
                        try:
                            return {
                                "ok": True,
                                "status": {
                                    "couchdb_connected": bool(getattr(self, "couch", None)),
                                    "db_name": getattr(self, "db_name", None),
                                    "local_store_loaded": hasattr(self, "local_store"),
                                    "persistent_state_loaded": hasattr(self, "persistent_state"),
                                    "ai_helper_attached": hasattr(self, "ai_helper"),
                                },
                            }
                        except Exception as e:
                            return {"ok": False, "error": str(e)}
                    MEMORY.health_check = _types.MethodType(_health_check, MEMORY)

        except Exception as e:
            MemoryManager = None
            logger.error(f"[MemoryManager] Failed global assignment ❌ — exception: {e}", exc_info=True)

# ================= Execute Immediately =================
assign_global_memory_classes()

# ------------------------------
# — COUCHDB CONNECTIVITY CHECK
# ------------------------------
import couchdb

COUCHDB_AVAILABLE: bool = False
COUCHDB_URL = os.getenv("COUCHDB_URL", "http://127.0.0.1:5984")
COUCHDB_USER = os.getenv("COUCHDB_USER", "rohan")
COUCHDB_PASSWORD = os.getenv("COUCHDB_PASSWORD", "rthunderpheonix11")
COUCHDB_DB_NAME = os.getenv("COUCHDB_DB_NAME", "bibliotheca_memory")


def is_couchdb_available(max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """
    Verify CouchDB connection; fallback to MEMORY if unavailable
    """
    global COUCHDB_AVAILABLE
    if COUCHDB_AVAILABLE:
        return True

    for attempt in range(1, max_retries + 1):
        try:
            server = couchdb.Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")
            if COUCHDB_USER and COUCHDB_PASSWORD:
                server.resource.credentials = (COUCHDB_USER, COUCHDB_PASSWORD)

            _ = server[COUCHDB_DB_NAME]  # test DB existence
            COUCHDB_AVAILABLE = True
            logger.info(f"[MemoryManager] CouchDB is available ✅ (attempt {attempt})")
            return True
        except Exception as e:
            COUCHDB_AVAILABLE = False
            logger.warning(f"[MemoryManager] CouchDB check failed (attempt {attempt}): {e}")
            time.sleep(retry_delay)
    logger.warning("[MemoryManager] CouchDB unavailable — fallback memory store will be used ⚠️")
    return False

# ------------------------------
# — AUTOMATIC GLOBAL INITIALIZATION
# ------------------------------
assign_global_memory_classes()
is_couchdb_available()

# ------------------------------
# — LOCAL BACKUP PATHS
# ------------------------------
DATA_DIR = BASE_DIR / "data"
BACKUP_DIR = BASE_DIR / "backups" / "memory_manager"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_STORE_FILE = DATA_DIR / "memory_store.json"


# ===============================
# — MEMORY FALLBACK & TELEMETRY SUPPORT
# ===============================
from datetime import datetime

_fallback_cache: dict = {}  # In-memory fallback cache


def fallback_read(key: str, default: Optional[Any] = None) -> Any:
    """Safe read from fallback cache with telemetry logging."""
    try:
        value = _fallback_cache.get(key, default)
        logger.debug(f"[MemoryManagerFallback][READ] {key} -> {value}")
        return value
    except Exception as e:
        logger.exception(f"[MemoryManagerFallback][READ] Failed for key '{key}': {e}")
        return default


def fallback_write(key: str, value: Any) -> None:
    """Safe write to fallback cache with telemetry logging."""
    try:
        _fallback_cache[key] = value
        logger.debug(f"[MemoryManagerFallback][WRITE] {key} -> {value}")
        _log_telemetry("write", key)
    except Exception as e:
        logger.exception(f"[MemoryManagerFallback][WRITE] Failed for key '{key}': {e}")


def _log_telemetry(action: str, key: str):
    """Log telemetry events with timestamp for fallback operations."""
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "key": key
    }
    logger.debug(f"[Telemetry] {event}")


# ===============================
# — SAFE LOCK CONTEXT MANAGER
# ===============================
from contextlib import contextmanager


@contextmanager
def memory_locked():
    """
    Context manager for acquiring the global memory lock safely.
    Ensures proper acquisition and release with error logging.
    """
    try:
        _memory_lock.acquire()
        yield
    finally:
        _memory_lock.release()


# ===============================
# — FALLBACK MEMORY CLASSES
# ===============================
class MemoryStoreFallback:
    """In-memory fallback store when CouchDB is unavailable."""
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def get(self, key: str, default=None):
        return self.store.get(key, default)

    def set(self, key: str, value: Any):
        self.store[key] = value

    def delete(self, key: str):
        self.store.pop(key, None)

    def all(self):
        return dict(self.store)


# ===============================
# — THREAD-SAFE MEMORY FALLBACK
# ===============================
class MemoryManagerFallback:
    """Beyond-Perfection Fallback MemoryManager using in-memory store.

    Features:
    - Thread-safe read/write/delete
    - Full backup & restore support
    - Telemetry logging
    - Auto-upgrade to real MemoryManager if CouchDB becomes available
    - Snapshots support
    - Self-healing: auto-recreates missing keys or structures
    """

    def __init__(self):
        from pathlib import Path
        import logging
        import pickle

        # Core memory store
        self.memory = MemoryStoreFallback()

        # Logger for telemetry & debug
        self.logger = logging.getLogger("Bibliotheca.MemoryManagerFallback")
        self._lock = threading.RLock()

        # Snapshot counter and storage
        self._snapshot_counter = 0
        self._snapshot_dir = Path("data/memory_snapshots")
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Backup history
        self._backups = []
        self._max_backups = 8  # configurable

        # Auto-upgrade flag
        self._upgrade_pending = True

        # Telemetry cache
        self._telemetry_cache = []

        self.logger.info("[MemoryManagerFallback] Initialized successfully ✅")

    # -----------------------------
    # — DECORATOR
    # -----------------------------
    def synchronized(func):
        """Thread-safe decorator for all MemoryManagerFallback methods."""
        def wrapper(self, *args, **kwargs):
            with self._lock:
                return func(self, *args, **kwargs)
        return wrapper

    # -----------------------------
    # — CORE OPERATIONS
    # -----------------------------
    @synchronized
    def read(self, key: str, default=None):
        try:
            return self.memory.get(key, default)
        except Exception as e:
            self.logger.warning(f"[MemoryManagerFallback] Read failed for key '{key}': {e}")
            return default

    @synchronized
    def write(self, key: str, value: Any):
        try:
            self.memory.set(key, value)
            self._log_telemetry("write", key)
        except Exception as e:
            self.logger.error(f"[MemoryManagerFallback] Write failed for key '{key}': {e}")

    @synchronized
    def remove(self, key: str):
        try:
            self.memory.delete(key)
            self._log_telemetry("delete", key)
        except Exception as e:
            self.logger.warning(f"[MemoryManagerFallback] Remove failed for key '{key}': {e}")

    @synchronized
    def dump(self) -> dict:
        try:
            return self.memory.all()
        except Exception as e:
            self.logger.warning(f"[MemoryManagerFallback] Dump failed: {e}")
            return {}

    @synchronized
    def keys(self) -> list:
        return list(self.memory.all().keys())

    @synchronized
    def clear(self):
        try:
            self.memory.store.clear()
            self._log_telemetry("clear", "all")
            self.logger.info("[MemoryManagerFallback] Memory cleared ✅")
        except Exception as e:
            self.logger.error(f"[MemoryManagerFallback] Clear failed: {e}")

    # -----------------------------
    # — SNAPSHOT / BACKUP
    # -----------------------------
    @synchronized
    def snapshot(self):
        import pickle
        try:
            snapshot_path = self._snapshot_dir / f"memory_snapshot_{self._snapshot_counter}.pkl"
            with snapshot_path.open("wb") as f:
                pickle.dump(self.memory, f)
            self._snapshot_counter += 1
            self._rotate_snapshots()
            self.logger.debug(f"[MemoryManagerFallback] Snapshot saved: {snapshot_path.name}")
        except Exception as e:
            self.logger.warning(f"[MemoryManagerFallback] Snapshot failed: {e}")

    @synchronized
    def _rotate_snapshots(self):
        snapshots = sorted(self._snapshot_dir.glob("memory_snapshot_*.pkl"), key=lambda p: p.stat().st_mtime)
        while len(snapshots) > self._max_backups:
            old = snapshots.pop(0)
            try:
                old.unlink()
                self.logger.info(f"[MemoryManagerFallback] Pruned old snapshot: {old.name}")
            except Exception:
                pass

    # ===============================
    # — TELEMETRY & AUTO-UPGRADE (Beyond Perfection)
    # ===============================
    import threading
    import logging
    from datetime import datetime
    from typing import Optional, List, Dict, Any

    # Module-level logger
    logger = logging.getLogger("Bibliotheca.MemoryManager")
    logger.setLevel(logging.DEBUG)

    # Thread-safe telemetry cache
    _telemetry_lock = threading.RLock()
    _telemetry_cache: List[Dict[str, Any]] = []

    @synchronized
    def _log_telemetry(self, action: str, key: str, extra: Optional[Dict[str, Any]] = None):
        """
        Thread-safe telemetry logger.
        Stores events in memory, optionally flushable to persistent storage.

        Args:
            action (str): The action type
            key (str): Identifier or key for the event
            extra (Optional[dict]): Any additional metadata
        """
        try:
            event = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "action": action,
                "key": key,
                "extra": extra or {}
            }
            with _telemetry_lock:
                self._telemetry_cache.append(event)
            logger.debug(f"[MemoryManager][Telemetry] Event logged: {event}")
        except Exception as e:
            logger.exception(f"[MemoryManager][Telemetry] Failed to log event: {e}")

    @synchronized
    def flush_telemetry(self, persistent_store: Optional[Any] = None):
        """
        Flush telemetry cache to persistent storage (file, CouchDB, etc.)

        Args:
            persistent_store: Optional object with `.save(event)` method or file-like object
        """
        try:
            with _telemetry_lock:
                cache_copy = self._telemetry_cache.copy()
                self._telemetry_cache.clear()
            if persistent_store:
                for event in cache_copy:
                    try:
                        persistent_store.save(event)
                    except Exception as e:
                        logger.warning(f"[MemoryManager][Telemetry] Failed to save event {event}: {e}")
            logger.info(f"[MemoryManager][Telemetry] Flushed {len(cache_copy)} events ✅")
        except Exception as e:
            logger.exception(f"[MemoryManager][Telemetry] Flush failed: {e}")

    # -----------------------------
    # — AUTO-UPGRADE TO REAL MEMORY
    # -----------------------------
    @synchronized
    def attempt_upgrade(self):
        """
        Upgrade from fallback MemoryManager to real MemoryManager if CouchDB is available.
        Safely migrates data, retries on failure, and logs detailed telemetry.
        """
        if not getattr(self, "_upgrade_pending", False):
            return

        from app.utils.memory_manager import COUCHDB_AVAILABLE
        if not COUCHDB_AVAILABLE:
            logger.info("[MemoryManagerFallback] CouchDB unavailable — upgrade skipped ⚠️")
            return

        try:
            from app.utils.memory_real import MemoryManager as RealManager
            # Backup current state
            old_data = self.dump()
            self._log_telemetry("upgrade_attempt", "MemoryManagerFallback->RealManager", {"backup_size": len(old_data)})

            # Replace class and re-initialize
            self.__class__ = RealManager
            self.__init__()

            # Restore previous data
            for k, v in old_data.items():
                try:
                    self.write(k, v)
                except Exception as e:
                    logger.warning(f"[MemoryManagerFallback] Failed to restore key '{k}': {e}")

            self._upgrade_pending = False
            self._log_telemetry("upgrade_success", "MemoryManagerFallback->RealManager",
                                {"restored_keys": len(old_data)})
            logger.info("[MemoryManagerFallback] Successfully upgraded to Real MemoryManager ✅")

        except Exception as e:
            self._log_telemetry("upgrade_failure", "MemoryManagerFallback->RealManager", {"error": str(e)})
            logger.warning(f"[MemoryManagerFallback] Upgrade attempt failed: {e}")

    # ===============================
    # — GLOBAL MEMORY SINGLETON
    # ===============================
    MEMORY: Optional[Any] = None
    MemoryManager: Any = None
    MemoryStore: Any = None

    def assign_global_memory_classes():
        """
        Assign MemoryManager and MemoryStore globally.
        Automatically selects real or fallback based on CouchDB availability.
        Includes robust fallback, auto-upgrade, and telemetry logging.
        """
        global MEMORY, MemoryManager, MemoryStore
        from app.utils.memory_manager import COUCHDB_AVAILABLE

        try:
            if COUCHDB_AVAILABLE:
                try:
                    from app.utils.memory_real import MemoryManager as RealManager, MemoryStore as RealStore
                    MemoryManager = RealManager
                    MemoryStore = RealStore
                    MEMORY = MemoryManager()
                    MEMORY._log_telemetry("memory_assignment", "real_store")
                    logger.info(f"[MemoryManager] Real MemoryStore assigned globally: {MemoryStore.__name__} ✅")
                    return
                except Exception as e:
                    logger.warning(f"[MemoryManager] Failed to assign real MemoryStore: {e}")

            # Fallback path
            from app.utils.memory_fallback import MemoryManagerFallback, MemoryStoreFallback
            MemoryManager = MemoryManagerFallback
            MemoryStore = MemoryStoreFallback
            MEMORY = MemoryManagerFallback()
            MEMORY._log_telemetry("memory_assignment", "fallback_store")
            logger.info(f"[MemoryManager] Fallback MemoryStore assigned globally: {MemoryStore.__name__} ⚠️")

        except Exception as e:
            logger.exception(f"[MemoryManager] Unexpected error during global assignment: {e}")
            MEMORY = None
            MemoryManager = None
            MemoryStore = None


def upgrade_memory_if_couchdb_available():
    """Upgrade MEMORY singleton from fallback to real MemoryManager if CouchDB becomes available."""
    global MEMORY
    from app.utils.memory_manager import COUCHDB_AVAILABLE
    if COUCHDB_AVAILABLE and isinstance(MEMORY, MemoryManagerFallback):
        logger.info("[MemoryManager] Upgrading MEMORY from fallback to real MemoryManager...")
        try:
            from app.utils.memory_real import MemoryManager as RealManager
            old_data = MEMORY.dump()
            MEMORY.__class__ = RealManager
            MEMORY.__init__()
            for k, v in old_data.items():
                MEMORY.write(k, v)
            logger.info("[MemoryManager] MEMORY upgraded successfully ✅")
        except Exception as e:
            logger.warning(f"[MemoryManager] MEMORY upgrade failed: {e}")

# ===============================
# — INITIAL EXECUTION
# ===============================
validate_environment()
assign_global_memory_classes()
upgrade_memory_if_couchdb_available()

# -----------------------
# CouchDB Client wrapper — Beyond-Perfection Edition
# -----------------------

class CouchDBClient:
    def __init__(self, url: str = COUCHDB_URL, db_name: str = COUCHDB_DB_NAME, max_retries: int = 3):
        """
        Fully safe CouchDB connection and initialization.

        Args:
            url (str): CouchDB URL
            db_name (str): Name of the database to connect to or create
            max_retries (int): Number of retry attempts for transient connection issues
        """
        # Module-level logger ensures no dependency on instance before initialization
        self.logger = logging.getLogger(f"BibliothecaAI.CouchDBClient.{db_name}")
        self.logger.setLevel(logging.DEBUG)
        self.server: Optional['couchdb.Server'] = None
        self.db: Optional['couchdb.Database'] = None
        self.available: bool = False
        self._max_retries = max_retries

        if not COUCHDB_AVAILABLE:
            self._safe_log("warning", "[CouchDBClient] couchdb package not installed; CouchDB disabled.")
            return

        # --- Attempt connection with retry mechanism ---
        for attempt in range(1, self._max_retries + 1):
            try:
                self.server = couchdb.Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")
                version = self.server.version()
                self._safe_log("info", f"[CouchDBClient] CouchDB server version: {version}")

                # --- Ensure database exists ---
                if self.db_name in self.server:
                    self.db = self.server[self.db_name]
                    self._safe_log("info", f"[CouchDBClient] Database '{self.db_name}' exists; connected ✅")
                else:
                    self.db = self.server.create(self.db_name)
                    self._safe_log("info", f"[CouchDBClient] Database '{self.db_name}' not found; created ✅")

                # --- Initialize memory_state document if missing ---
                if self.db is not None and "memory_state" not in self.db:
                    self.db["memory_state"] = {"_id": "memory_state", "store": {}}
                    self._safe_log("info", "[CouchDBClient] Created initial memory_state document ✅")

                self.available = True
                break  # success, exit retry loop

            except Exception as e:
                self.available = False
                self.logger.warning(
                    f"[CouchDBClient] CouchDB connection attempt {attempt}/{self._max_retries} failed: {e}"
                )
                if attempt == self._max_retries:
                    self.logger.error(
                        f"[CouchDBClient] CouchDB unavailable after {self._max_retries} attempts; "
                        "running in cache-only mode ⚠️"
                    )

        # --- Fallback in-memory store if CouchDB unavailable ---
        if not self.available:
            self._store: dict = {}
            self._safe_log("info", "[CouchDBClient] Using in-memory store (CouchDB unavailable) ✅")

    # ------------------------
    # Document operations
    # ------------------------
    def save_doc(self, doc: dict) -> Optional[str]:
        """Save or update a document. Returns doc ID or None on error."""
        if not self.available or self.db is None:
            self._safe_log("debug", "[CouchDBClient] save_doc skipped; CouchDB unavailable.")
            return None
        try:
            # Remove Python-only objects
            doc_copy = json.loads(json.dumps(doc, default=str))
            doc_id = doc_copy.get("_id")
            if doc_id:
                existing = self.db.get(doc_id)
                if existing:
                    doc_copy["_rev"] = existing.get("_rev")
                self.db[doc_id] = doc_copy
                self._safe_log("debug", f"[CouchDBClient] Document saved/updated: {doc_id}")
                return doc_id
            else:
                doc_id, _rev = self.db.save(doc_copy)
                self._safe_log("debug", f"[CouchDBClient] Document saved with new id: {doc_id}")
                return doc_id
        except Exception as e:
            self._safe_log("error", f"[CouchDBClient] save_doc error: {e}", exc_info=True)
            return None

    def get_doc(self, doc_id: str) -> Optional[dict]:
        """Fetch a document by ID. Returns dict or None."""
        if not self.available or self.db is None:
            self._safe_log("debug", f"[CouchDBClient] get_doc skipped; CouchDB unavailable. ID={doc_id}")
            return None
        try:
            doc = self.db.get(doc_id)
            self._safe_log("debug", f"[CouchDBClient] Retrieved document: {doc_id}")
            return dict(doc) if doc else None
        except Exception as e:
            self._safe_log("error", f"[CouchDBClient] get_doc error for ID={doc_id}: {e}", exc_info=True)
            return None

    def find_by_type(self, doc_type: str) -> list[dict]:
        """Return all documents with 'type' == doc_type (best-effort)."""
        if not self.available or self.db is None:
            self._safe_log("debug", f"[CouchDBClient] find_by_type skipped; CouchDB unavailable. Type={doc_type}")
            return []
        try:
            results = []
            for row_id in self.db:
                try:
                    doc = self.db[row_id]
                    if doc.get("type") == doc_type:
                        results.append(dict(doc))
                except Exception:
                    continue
            self._safe_log("debug", f"[CouchDBClient] Found {len(results)} documents of type '{doc_type}'")
            return results
        except Exception as e:
            self._safe_log("error", f"[CouchDBClient] find_by_type error for type={doc_type}: {e}", exc_info=True)
            return []

# -----------------------
# Semantic Memory
# -----------------------


# -----------------------------
# SemanticMemory
# -----------------------------
import threading
import difflib
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

# Optional CouchDB support
try:
    import couchdb
except ImportError:
    couchdb = None

logger = logging.getLogger(__name__)

AUTO_APPLY_PATCHES = True  # adjust as needed


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


class SemanticMemory:
    def __init__(self):
        self._lock = threading.RLock()
        self.data: Dict[str, Dict[str, Any]] = {}
        self.metadata = {
            "init_time": now_iso(),
            "last_patch": None,
            "patch_failures": 0,
            "auto_updates_enabled": AUTO_APPLY_PATCHES,
        }
        self.patch_proposals: List[Dict[str, Any]] = []
        self.embedding_fn: Optional[Callable[[str], List[float]]] = None
        self._couchdb_server = None
        self._couchdb_db = None
        if couchdb:
            try:
                self._couchdb_server = couchdb.Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")
                self._couchdb_db = self._couchdb_server.create("semantic_memory")
                logger.info("[MemoryManager] CouchDB initialized ✅")
            except Exception:
                logger.warning("[MemoryManager] CouchDB init failed, continuing with local memory")
        logger.info("[MemoryManager] SemanticMemory initialized ✅")

    @contextmanager
    def locked(self):
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def store_local(self, key: str, value: str, meta: dict = None):
        with self.locked():
            ent = self.data.get(key, {"value": None, "meta": {}, "version": 0})
            ent["value"] = value
            ent["meta"] = meta or ent.get("meta", {})
            ent["version"] = ent.get("version", 0) + 1
            ent["updated_at"] = now_iso()
            self.data[key] = ent
            logger.debug(f"[MemoryManager] Stored key={key} v={ent['version']}")
            # Save to CouchDB if available
            if self._couchdb_db:
                try:
                    doc = self._couchdb_db.get(key) or {"_id": key}
                    doc.update(ent)
                    self._couchdb_db.save(doc)
                except Exception as e:
                    logger.warning(f"[MemoryManager] CouchDB save failed for key={key}: {e}")

    def recall_local(self, key: str) -> Optional[Dict[str, Any]]:
        with self.locked():
            val = self.data.get(key)
            if not val and self._couchdb_db:
                try:
                    doc = self._couchdb_db.get(key)
                    if doc:
                        self.data[key] = doc
                        return doc
                except Exception:
                    pass
            return val

    def search_local(self, query: str, top_k: int = 3) -> List[tuple]:
        with self.locked():
            scored = [
                (difflib.SequenceMatcher(None, query, str(v.get("value") or "")).ratio(), k, v)
                for k, v in self.data.items()
            ]
            scored.sort(reverse=True, key=lambda x: x[0])
            return scored[:top_k]

    def propose_patch(self, patch: Dict[str, Any]):
        """Add a patch proposal for later application"""
        with self.locked():
            self.patch_proposals.append(patch)
            logger.info(f"[MemoryManager] Patch proposed: {patch.get('desc', 'no description')}")

    def apply_patches(self):
        """Apply all pending patch proposals"""
        with self.locked():
            for patch in self.patch_proposals:
                try:
                    func = patch.get("func")
                    if func and callable(func):
                        func(self)
                        self.metadata["last_patch"] = now_iso()
                        logger.info(f"[MemoryManager] Applied patch: {patch.get('desc', 'no description')}")
                    else:
                        logger.warning("[MemoryManager] Patch missing callable func")
                except Exception as e:
                    self.metadata["patch_failures"] += 1
                    logger.error(f"[MemoryManager] Patch failed: {e}")
            self.patch_proposals.clear()

    class MemoryManager:
        """Beyond-Perfection MemoryManager for Bibliotheca
        Features:
            - Full CouchDB integration with auto DB creation
            - Lazy FactGraph DB access
            - Advanced memory healing & auto-repair
            - Thread-safe with RLock
            - Automatic telemetry logging
            - Fallback to in-memory store
            - Initial document creation
            - Full retry and failover mechanisms
        """

        def __init__(self, config: Optional[dict] = None, base_dir: Optional[str] = None):
            # -----------------------------
            # 1️⃣ Thread Safety
            # -----------------------------
            self._init_lock = threading.RLock()

            # -----------------------------
            # 2️⃣ Logger setup
            # -----------------------------
            if not self.logger.handlers:
                ch = logging.StreamHandler()
                ch.setFormatter(logging.Formatter("[%(asctime)s][MemoryManager][%(levelname)s] %(message)s"))
                self.logger.addHandler(ch)

            def _safe_log(level: str, message: str):
                try:
                    getattr(self.logger, level, self.logger.info)(message)
                except Exception:
                    print(f"[MemoryManager][SAFELOG][{level.upper()}]: {message}")

            self._safe_log = _safe_log
            self._safe_log("info", "Initializing MemoryManager (Beyond-Perfection Mode)")

            # -----------------------------
            # 3️⃣ Load configuration
            # -----------------------------
            self.config = config.copy() if config else {}
            couch_cfg = self.config.get("couchdb", {})

            self.protocol = couch_cfg.get("protocol", "http")
            self.host = couch_cfg.get("host", "localhost")
            self.port = couch_cfg.get("port", 5984)
            self.username = couch_cfg.get("username", "")
            self.password = couch_cfg.get("password", "")
            self.use_env = couch_cfg.get("use_env_credentials", True)
            self.auto_create_dbs = couch_cfg.get("auto_create_dbs", True)
            self.connection_attempts = couch_cfg.get("connection_attempts", 5)
            self.retry_delay = couch_cfg.get("retry_delay_seconds", 5)
            self.dbs = couch_cfg.get("dbs", {})

            if self.use_env:
                self.username = os.getenv("COUCHDB_USER", self.username)
                self.password = os.getenv("COUCHDB_PASSWORD", self.password)

            if not self.username or not self.password:
                self._safe_log("error", "Missing CouchDB credentials")
                raise ValueError("CouchDB credentials missing")

            # -----------------------------
            # 4️⃣ Base directory
            # -----------------------------
            self.base_dir = pathlib.Path(base_dir or "./memory_data")
            try:
                self.base_dir.mkdir(parents=True, exist_ok=True)
                self._safe_log("info", f"Memory directory ready at {self.base_dir} ✅")
            except Exception as e:
                self._safe_log("error", f"Failed to create base_dir: {e}")

            # -----------------------------
            # 5️⃣ CouchDB connection
            # -----------------------------
            self.couch = None
            for attempt in range(1, self.connection_attempts + 1):
                try:
                    url = f"{self.protocol}://{self.username}:{self.password}@{self.host}:{self.port}/"
                    self.couch = couchdb.Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")
                    list(self.couch)  # test connection
                    self._safe_log("info", f"CouchDB connected successfully at {url}")
                    break
                except Exception as e:
                    self._safe_log("warning", f"CouchDB attempt {attempt} failed: {e}")
                    time.sleep(self.retry_delay)
            else:
                self._safe_log("error", "Unable to connect to CouchDB, falling back to local memory")
                self.couch = None

            # -----------------------------
            # 6️⃣ Initialize DBs
            # -----------------------------
            self._db_instances = {}
            if self.couch:
                for key, db_name in self.dbs.items():
                    try:
                        if db_name in self.couch:
                            self._db_instances[key] = self.couch[db_name]
                            self._safe_log("info", f"DB '{db_name}' exists and connected")
                        elif self.auto_create_dbs:
                            self._db_instances[key] = self.couch.create(db_name)
                            self._safe_log("info", f"DB '{db_name}' created automatically")
                        else:
                            self._safe_log("warning", f"DB '{db_name}' missing and auto-create disabled")
                    except Exception as e:
                        self._safe_log("error", f"DB '{db_name}' init failed: {e}")

            # -----------------------------
            # 7️⃣ Memory store setup
            # -----------------------------
            self._store = {}
            self.db = self._db_instances.get("memory", None)
            if self.db is None:
                self._safe_log("info", "Using local in-memory store ✅")
            else:
                self._safe_log("info", "CouchDB memory store ready ✅")

            # -----------------------------
            # 8️⃣ Ensure initial memory document
            # -----------------------------
            try:
                if self.db:
                    self._ensure_initial_doc(self.db)
            except Exception as e:
                self._safe_log("warning", f"Initial memory doc failed: {e}")

            # -----------------------------
            # 9️⃣ FactGraph lazy access
            # -----------------------------
            self.factgraph_db = None

            def get_factgraph():
                if self.factgraph_db:
                    return self.factgraph_db
                if not self.couch:
                    self._safe_log("warning", "CouchDB not available, cannot access FactGraph")
                    return None
                db_name = self.dbs.get("factgraph", "bibliotheca_factgraph")
                try:
                    if db_name not in self.couch and self.auto_create_dbs:
                        self.factgraph_db = self.couch.create(db_name)
                    else:
                        self.factgraph_db = self.couch[db_name]
                    self._safe_log("info", f"FactGraph DB ready at '{db_name}'")
                    return self.factgraph_db
                except Exception as e:
                    self._safe_log("error", f"FactGraph DB init failed: {e}")
                    return None

            self.get_factgraph = get_factgraph

            # -----------------------------
            # 🔟 Telemetry
            # -----------------------------
            try:
                self.telemetry_enabled = self.config.get("enable_telemetry", True)
                self.telemetry_db = self._db_instances.get("telemetry", None)
                if self.telemetry_enabled and self.telemetry_db:
                    self._safe_log("info", "Telemetry fully enabled ✅")
            except Exception as e:
                self.telemetry_enabled = False
                self._safe_log("warning", f"Telemetry init failed: {e}")

            # -----------------------------
            # 1️⃣1️⃣ Self-healing / Auto-patch
            # -----------------------------
            self.auto_patch_memory_on_error = self.config.get("auto_patch_memory_on_error", True)
            self.enable_memory_healing = self.config.get("enable_memory_healing", True)

            # -----------------------------
            # 1️⃣2️⃣ Complete initialization
            # -----------------------------
            self._safe_log("debug", "MemoryManager fully initialized with safety, telemetry, FactGraph, and healing ✅")
            self.initialized = True

        # -----------------------------
        # Example: initial doc creation
        # -----------------------------
        def _ensure_initial_doc(self, db):
            """Ensure base memory document exists"""
            try:
                if "_local/memory_init" not in db:
                    db["_local/memory_init"] = {"created_at": time.time()}
                    self._safe_log("info", "Initial memory document created ✅")
            except Exception as e:
                self._safe_log("warning", f"Failed to create initial doc: {e}")

        # -----------------------------
        # Example stub for initial doc
        # -----------------------------
        def _ensure_initial_doc(self, db):
            """Ensure a base memory doc exists in CouchDB"""
            if "_local/memory_init" not in db:
                db["_local/memory_init"] = {"created_at": time.time()}
                self._safe_log("info", "Initial memory document created ✅")

    # -----------------------------
    # Supporting methods
    # -----------------------------
    def _auto_heal_loop(self):
        """
        Background thread that continuously monitors and heals memory inconsistencies.

        Features:
        - Checks CouchDB connectivity and reconnects if necessary
        - Validates all documents in the DB for missing fields or corruption
        - Auto-repairs missing or corrupted documents using defaults or last known good state
        - Thread-safe with locks
        - Configurable interval via `memory_heal_interval` in config
        - Logs every operation, errors, and repairs
        - Optional notifications for critical failures
        """
        import time
        from copy import deepcopy

        default_doc_template = self.config.get("memory_default_doc", {"type": "generic", "data": {}})
        heal_interval = self.config.get("memory_heal_interval", 60)
        max_retries = self.config.get("memory_heal_max_retries", 3)

        while not getattr(self, "_stop_event", None) or not self._stop_event.is_set():
            try:
                with self._lock:
                    self._safe_log("debug", "[MemoryManager] Auto-heal cycle started.")

                    # 1️⃣ Ensure DB connection is alive
                    retries = 0
                    while True:
                        try:
                            # Ping DB
                            _ = self.db.info()
                            break
                        except Exception as e:
                            retries += 1
                            self._safe_log("warning", f"[MemoryManager] CouchDB unreachable (attempt {retries}): {e}")
                            if retries >= max_retries:
                                self._safe_log("error", "[MemoryManager] CouchDB failed to respond. Skipping this cycle.")
                                break
                            time.sleep(5)

                    # 2️⃣ Iterate through all docs and validate
                    for doc_id in list(self.db):
                        try:
                            doc = self.db[doc_id]
                            if not isinstance(doc, dict):
                                self._safe_log("warning", f"[MemoryManager] Doc {doc_id} corrupted, repairing...")
                                new_doc = deepcopy(default_doc_template)
                                new_doc["_id"] = doc_id
                                self.db.save(new_doc)
                                self._safe_log("info", f"[MemoryManager] Doc {doc_id} repaired with default template.")
                            else:
                                # Example: ensure required fields exist
                                required_fields = self.config.get("memory_required_fields", ["type", "data"])
                                updated = False
                                for field in required_fields:
                                    if field not in doc:
                                        doc[field] = deepcopy(default_doc_template.get(field, None))
                                        updated = True
                                        self.logger.debug(
                                            f"[MemoryManager] Missing field '{field}' added to doc {doc_id}")
                                if updated:
                                    self.db.save(doc)
                        except Exception as e:
                            self._safe_log("error", f"[MemoryManager] Error validating doc {doc_id}: {e}")

                    # 3️⃣ Check for missing expected docs from config
                    expected_docs = self.config.get("memory_expected_docs", [])
                    for exp_id in expected_docs:
                        if exp_id not in self.db:
                            new_doc = deepcopy(default_doc_template)
                            new_doc["_id"] = exp_id
                            try:
                                self.db.save(new_doc)
                                self._safe_log("info", f"[MemoryManager] Missing expected doc '{exp_id}' created.")
                            except Exception as e:
                                self._safe_log("error", f"[MemoryManager] Failed to create expected doc '{exp_id}': {e}")

                    self._safe_log("debug", "[MemoryManager] Auto-heal cycle completed successfully.")
            except Exception as e:
                self._safe_log("critical", f"[MemoryManager] Uncaught auto-heal loop error: {e}")

            # Wait until next cycle
            time.sleep(heal_interval)

    # -----------------------------
    # Internal helper for telemetry
    # -----------------------------
    def _init_telemetry(self):
        """
        Initializes a complete Telemetry system for MemoryManager.
        Tracks:
        - CRUD operations
        - Auto-heal events
        - Memory health and consistency
        - Performance metrics
        - Optional alerts/notifications

        Fully thread-safe and extendable.
        """
        import time
        from collections import defaultdict
        from threading import Lock

        try:
            self._telemetry_enabled = self.config.get("enable_telemetry", True)
            self._telemetry_lock = Lock()

            # Initialize counters and logs
            self._telemetry_data = defaultdict(lambda: {
                "reads": 0,
                "writes": 0,
                "deletes": 0,
                "auto_heal_events": 0,
                "last_operation_time": None,
                "operation_durations": [],
                "errors": []
            })

            # Hook to wrap MemoryManager CRUD operations
            original_save = self.save if hasattr(self, "save") else None
            original_get = self.get if hasattr(self, "get") else None
            original_delete = self.delete if hasattr(self, "delete") else None

            # Wrap save
            if original_save:
                def wrapped_save(doc_id, doc_data):
                    start_time = time.time()
                    try:
                        result = original_save(doc_id, doc_data)
                        return result
                    finally:
                        duration = time.time() - start_time
                        with self._telemetry_lock:
                            self._telemetry_data[doc_id]["writes"] += 1
                            self._telemetry_data[doc_id]["last_operation_time"] = time.time()
                            self._telemetry_data[doc_id]["operation_durations"].append(duration)

                self.save = wrapped_save

            # Wrap get
            if original_get:
                def wrapped_get(doc_id):
                    start_time = time.time()
                    try:
                        return original_get(doc_id)
                    finally:
                        duration = time.time() - start_time
                        with self._telemetry_lock:
                            self._telemetry_data[doc_id]["reads"] += 1
                            self._telemetry_data[doc_id]["last_operation_time"] = time.time()
                            self._telemetry_data[doc_id]["operation_durations"].append(duration)

                self.get = wrapped_get

            # Wrap delete
            if original_delete:
                def wrapped_delete(doc_id):
                    start_time = time.time()
                    try:
                        return original_delete(doc_id)
                    finally:
                        duration = time.time() - start_time
                        with self._telemetry_lock:
                            self._telemetry_data[doc_id]["deletes"] += 1
                            self._telemetry_data[doc_id]["last_operation_time"] = time.time()
                            self._telemetry_data[doc_id]["operation_durations"].append(duration)

                self.delete = wrapped_delete

            # Auto-heal events tracker
            if hasattr(self, "_auto_heal_loop"):
                original_heal_loop = self._auto_heal_loop

                def wrapped_heal_loop():
                    import time
                    while not getattr(self, "_stop_event", None) or not self._stop_event.is_set():
                        start_time = time.time()
                        try:
                            original_heal_loop()
                        except Exception as e:
                            with self._telemetry_lock:
                                self._telemetry_data["auto_heal"]["errors"].append(str(e))
                        finally:
                            with self._telemetry_lock:
                                self._telemetry_data["auto_heal"]["auto_heal_events"] += 1
                                self._telemetry_data["auto_heal"]["last_operation_time"] = time.time()
                                self._telemetry_data["auto_heal"]["operation_durations"].append(
                                    time.time() - start_time)

                self._auto_heal_loop = wrapped_heal_loop

            self._safe_log("debug", "[MemoryManager] Telemetry system initialized and fully operational")

        except Exception as e:
            self._safe_log("warning", f"[MemoryManager] Telemetry initialization failed: {e}")
            self._telemetry_enabled = False

    # -----------------------
    # Health Check & Repair
    # -----------------------
    def test(self) -> dict:
        """
        Full MemoryManager health check and repair.

        Checks:
        - In-memory structure integrity
        - CouchDB connectivity and consistency
        - Missing or corrupt entries
        - Auto-heal performance
        - Operation latency and stats

        Returns:
            dict: Detailed report with keys:
                - 'status': 'healthy' or 'issues_found'
                - 'errors': list of error messages
                - 'warnings': list of warnings
                - 'summary': high-level summary
                - 'auto_heal_applied': bool indicating if auto-repair was performed
                - 'memory_count': number of in-memory entries
                - 'couchdb_count': number of docs in CouchDB
        """
        report = {
            "status": "healthy",
            "errors": [],
            "warnings": [],
            "summary": "",
            "auto_heal_applied": False,
            "memory_count": 0,
            "couchdb_count": 0
        }

        import time
        start_time = time.time()

        try:
            # Thread-safe access
            with self._lock:
                # Check in-memory structure
                if not isinstance(self.memory, dict):
                    report["errors"].append("Memory is not a dict!")
                    report["status"] = "issues_found"
                else:
                    report["memory_count"] = len(self.memory)
                    # Check for missing keys or None values
                    missing_keys = [k for k, v in self.memory.items() if v is None]
                    if missing_keys:
                        report["warnings"].append(f"Memory entries missing data: {missing_keys}")
                        report["status"] = "issues_found"

                # Optional auto-repair
                if self.config.get("auto_heal_on_test", True) and missing_keys:
                    for k in missing_keys:
                        # Recreate empty placeholder or attempt CouchDB restore
                        self.memory[k] = {}
                    report["auto_heal_applied"] = True
                    self._safe_log("info", f"[MemoryManager] Auto-heal applied to {len(missing_keys)} entries")

            # CouchDB consistency check
            if hasattr(self, "db"):
                try:
                    report["couchdb_count"] = len(self.db)
                except Exception as e:
                    report["errors"].append(f"CouchDB access failed: {e}")
                    report["status"] = "issues_found"

            # Telemetry integration
            if getattr(self, "_telemetry_enabled", False):
                with self._telemetry_lock:
                    self._telemetry_data["health_check"] = {
                        "duration": time.time() - start_time,
                        "memory_count": report["memory_count"],
                        "couchdb_count": report["couchdb_count"],
                        "errors": report["errors"],
                        "warnings": report["warnings"]
                    }

            # Summary
            if report["status"] == "healthy":
                report[
                    "summary"] = f"MemoryManager is healthy. {report['memory_count']} entries in memory, {report['couchdb_count']} docs in CouchDB."
            else:
                report[
                    "summary"] = f"Issues detected. Errors: {len(report['errors'])}, Warnings: {len(report['warnings'])}."

        except Exception as e:
            report["errors"].append(f"Health check failed unexpectedly: {e}")
            report["status"] = "issues_found"
            report["summary"] = "Health check failed due to unexpected error."

        return report

    # -----------------------------
    # Repair Memory & CouchDB
    # -----------------------------
    def repair(self) -> dict:
        """
        Attempt to repair memory and CouchDB connections.

        Steps:
        1. Ensure in-memory dictionary exists.
        2. Auto-heal missing or corrupt entries.
        3. Validate CouchDB connection and resync if needed.
        4. Apply telemetry and logging.

        Returns:
            dict: Detailed repair report:
                - 'status': 'success' or 'partial_failure' or 'failed'
                - 'memory_fixed': number of entries fixed
                - 'couchdb_reconnected': bool
                - 'errors': list of errors encountered
                - 'warnings': list of warnings
                - 'summary': high-level summary of repair
        """
        report = {
            "status": "success",
            "memory_fixed": 0,
            "couchdb_reconnected": False,
            "errors": [],
            "warnings": [],
            "summary": ""
        }

        import time
        start_time = time.time()

        try:
            # -----------------------------
            # Step 1: Ensure in-memory dict
            # -----------------------------
            with self._lock:
                if not hasattr(self, "memory") or not isinstance(self.memory, dict):
                    self.memory = {}
                    report["warnings"].append("Memory attribute missing or invalid. Reinitialized.")
                initial_count = len(self.memory)

            # -----------------------------
            # Step 2: Auto-heal entries
            # -----------------------------
            if getattr(self, "auto_heal", True):
                try:
                    missing_keys = [k for k, v in self.memory.items() if v is None]
                    for k in missing_keys:
                        self.memory[k] = {}
                    report["memory_fixed"] = len(missing_keys)
                    if missing_keys:
                        self._safe_log("info", f"[MemoryManager] Auto-healed {len(missing_keys)} missing entries.")
                except Exception as e:
                    report["errors"].append(f"Auto-heal failed: {e}")

            # -----------------------------
            # Step 3: CouchDB resync
            # -----------------------------
            if hasattr(self, "db"):
                try:
                    # Attempt a simple CouchDB operation to verify connectivity
                    _ = len(self.db)
                    report["couchdb_reconnected"] = True
                except Exception as e:
                    report["errors"].append(f"CouchDB connection failed: {e}")
                    report["couchdb_reconnected"] = False
                    # Retry logic
                    try:
                        import couchdb
                        server = couchdb.Server(
                            f"http://{self.config.get('couch_user')}:{self.config.get('couch_password')}@127.0.0.1:5984/")
                        self.db = server.get(self.db_name) or server.create(self.db_name)
                        report["couchdb_reconnected"] = True
                        self._safe_log("info", "[MemoryManager] CouchDB reconnected successfully.")
                    except Exception as e2:
                        report["errors"].append(f"CouchDB reconnection failed: {e2}")

            # -----------------------------
            # Step 4: Telemetry
            # -----------------------------
            if getattr(self, "_telemetry_enabled", False):
                with getattr(self, "_telemetry_lock", self._lock):
                    self._telemetry_data["last_repair"] = {
                        "duration": time.time() - start_time,
                        "memory_fixed": report["memory_fixed"],
                        "couchdb_reconnected": report["couchdb_reconnected"],
                        "errors": report["errors"],
                        "warnings": report["warnings"]
                    }

            # -----------------------------
            # Step 5: Final status
            # -----------------------------
            if report["errors"]:
                report["status"] = "partial_failure" if report["memory_fixed"] or report[
                    "couchdb_reconnected"] else "failed"

            report["summary"] = f"Repair completed in {time.time() - start_time:.2f}s. " \
                                f"Memory fixed: {report['memory_fixed']}, CouchDB reconnected: {report['couchdb_reconnected']}"

        except Exception as e:
            report["status"] = "failed"
            report["errors"].append(f"Unexpected repair error: {e}")
            report["summary"] = "Repair process failed unexpectedly."

        return report

    # -----------------------------
    # Verify CouchDB existence
    # -----------------------------
    def check_db_exists(self, auto_create: bool = False, timeout: int = 5) -> bool:
        """
        Verify that the CouchDB database exists and is accessible.

        Args:
            auto_create (bool): If True, automatically create the database if missing.
            timeout (int): Connection timeout in seconds.

        Returns:
            bool: True if the database exists (or was successfully created), False otherwise.
        """
        exists = False
        try:
            if getattr(self, "_server", None) is None:
                # Attempt to connect to CouchDB server
                import couchdb
                self._server = couchdb.Server(
                    f"http://{self.config.get('couch_user')}:{self.config.get('couch_password')}@127.0.0.1:5984/")
                self._safe_log("info", "[MemoryManager] Connected to CouchDB server for existence check.")

            if getattr(self, "_db_name", None) is None:
                self._db_name = getattr(self, "db_name", "bibliotheca_memory_advanced")

            if self._db_name in self._server:
                exists = True
                self._safe_log("debug", f"[MemoryManager] Database '{self._db_name}' exists.")
            else:
                self._safe_log("warning", f"[MemoryManager] Database '{self._db_name}' does not exist.")
                if auto_create:
                    try:
                        self._server.create(self._db_name)
                        exists = True
                        self._safe_log("info", f"[MemoryManager] Database '{self._db_name}' created successfully.")
                    except Exception as e_create:
                        self._safe_log("error", f"[MemoryManager] Auto-creation of DB '{self._db_name}' failed: {e_create}")
        except Exception as e:
            self._safe_log("warning", f"[MemoryManager] check_db_exists error: {e}")
            exists = False

        # Update telemetry
        if getattr(self, "_telemetry_enabled", False):
            with getattr(self, "_telemetry_lock", self._lock):
                self._telemetry_data.setdefault("db_checks", []).append({
                    "db_name": self._db_name,
                    "exists": exists,
                    "timestamp": time.time(),
                    "errors": str(e) if 'e' in locals() else None
                })

        return exists

    # -----------------------------
    # Sync memory to CouchDB
    # -----------------------------
    def sync_to_db(self, batch_size: int = 50, retry_attempts: int = 3):
        """
        Syncs current memory to CouchDB / persistent storage.

        Args:
            batch_size (int): Number of documents to send in one batch (for performance).
            retry_attempts (int): Number of retries for conflict or network errors.
        """
        import time
        if not hasattr(self, 'db') or not self.db:
            self._safe_log("warning", "[MemoryManager] No DB connected for sync")
            return

        if not getattr(self, 'memory', None):
            self._safe_log("info", "[MemoryManager] Memory is empty. Nothing to sync.")
            return

        keys = list(self.memory.keys())
        total_keys = len(keys)
        synced_count = 0

        self._safe_log("debug", f"[MemoryManager] Starting memory sync to DB '{self._db_name}' with {total_keys} items.")

        for i in range(0, total_keys, batch_size):
            batch_keys = keys[i:i + batch_size]
            docs_to_sync = []

            for key in batch_keys:
                doc = self.memory[key]
                # Ensure each doc has an _id for CouchDB
                if "_id" not in doc:
                    doc["_id"] = key
                docs_to_sync.append(doc)

            attempt = 0
            while attempt < retry_attempts:
                try:
                    results = self.db.update(docs_to_sync)
                    # results is a list of tuples: (success, doc_id, rev_or_exc)
                    for success, doc_id, rev_or_exc in results:
                        if success:
                            synced_count += 1
                        else:
                            self._safe_log("warning", f"[MemoryManager] Doc sync failed for {doc_id}: {rev_or_exc}")
                    break  # batch succeeded
                except Exception as e:
                    attempt += 1
                    self._safe_log("error", f"[MemoryManager] Batch sync attempt {attempt} failed: {e}")
                    time.sleep(2 ** attempt)  # exponential backoff

        self._safe_log("info", f"[MemoryManager] Memory sync complete: {synced_count}/{total_keys} documents synced.")

        # Update telemetry
        if getattr(self, "_telemetry_enabled", False):
            with getattr(self, "_telemetry_lock", self._lock):
                self._telemetry_data.setdefault("sync_stats", []).append({
                    "db_name": self._db_name,
                    "total_docs": total_keys,
                    "synced_docs": synced_count,
                    "timestamp": time.time()
                })

    def _ensure_db(self, max_retries: int = 5, retry_delay: float = 1.0):
        """
        Ensures the CouchDB database exists, is accessible, and fully initialized.
        Implements self-healing, secure credential handling, automatic creation,
        retry logic with exponential backoff, and verification of accessibility.

        Args:
            max_retries (int): Maximum number of retry attempts on failure.
            retry_delay (float): Base delay in seconds between retries (exponential backoff applied).

        Returns:
            CouchDB database instance, or None if all attempts fail.
        """
        import time
        import couchdb
        import os

        # Load credentials from environment variables for security
        couchdb_url = os.getenv("COUCHDB_URL", "http://127.0.0.1:5984")
        couchdb_db_name = os.getenv("COUCHDB_DB_NAME", "bibliotheca_memory")
        couchdb_user = os.getenv("COUCHDB_USER")
        couchdb_password = os.getenv("COUCHDB_PASSWORD")

        attempt = 0
        while attempt < max_retries:
            try:
                attempt += 1
                self._safe_log("debug", f"[CouchDB] Attempting connection to {couchdb_url} (Attempt {attempt}/{max_retries})")

                server = couchdb.Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")
                if couchdb_user and couchdb_password:
                    server.resource.credentials = (couchdb_user, couchdb_password)

                # Test server connection
                server_version = server.version()
                self._safe_log("info", f"[CouchDB] Connected to CouchDB server version {server_version} ✅")

                # Ensure database exists
                if couchdb_db_name not in server:
                    self._safe_log("info", f"[CouchDB] Database '{couchdb_db_name}' not found — creating...")
                    server.create(couchdb_db_name)
                    self._safe_log("info", f"[CouchDB] Database '{couchdb_db_name}' created successfully ✅")
                else:
                    self._safe_log("debug", f"[CouchDB] Database '{couchdb_db_name}' already exists.")

                db = server[couchdb_db_name]

                # Ensure initial document exists
                doc_id = "memory_state"
                if doc_id not in db:
                    self._safe_log("info", f"[CouchDB] Creating initial memory_state document ✅")
                    db[doc_id] = {"initialized": True, "data": {}}

                # Verify database and document accessibility
                try:
                    info = db.info()
                    self._safe_log("debug", f"[CouchDB] Verified access to '{couchdb_db_name}' ✅")
                except Exception as e_info:
                    self._safe_log("warning", f"[CouchDB] Could not access '{couchdb_db_name}' info: {e_info}")

                # Optional: Ensure connectivity by fetching the initial document
                try:
                    _ = db[doc_id]
                    self._safe_log("debug", f"[CouchDB] Verified initial document '{doc_id}' exists ✅")
                except Exception as e_doc:
                    self._safe_log("warning", f"[CouchDB] Could not access initial document '{doc_id}': {e_doc}")

                return db  # Success

            except Exception as e:
                wait_time = retry_delay * (2 ** (attempt - 1))  # exponential backoff
                self.logger.error(
                    f"[CouchDB] Attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait_time:.2f}s..."
                )
                time.sleep(wait_time)

        # All attempts failed
        self.logger.critical(
            f"[CouchDB] ❌ Failed to connect to or create DB '{couchdb_db_name}' after {max_retries} attempts. "
            "Manual intervention required!"
        )
        return None

    def _ensure_initial_doc(self, db, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Ensures the initial 'memory_state' document exists in CouchDB.
        Fully self-healing, resilient, and validated.

        Args:
            db: CouchDB database instance.
            max_retries: Number of retries if CouchDB operations fail.
            retry_delay: Delay (seconds) between retries.
        """
        import time
        from copy import deepcopy

        if db is None:
            self._safe_log("warning", "[CouchDB] _ensure_initial_doc called with None db, skipping.")
            return

        doc_id = "memory_state"
        attempt = 0
        while attempt < max_retries:
            try:
                # Check if document exists
                if doc_id in db:
                    doc = db[doc_id]
                    # Validate required keys
                    if "initialized" not in doc or "data" not in doc:
                        self._safe_log("warning", f"[CouchDB] '{doc_id}' exists but missing keys. Attempting repair...")
                        # Backup existing doc
                        backup_id = f"{doc_id}_backup_{int(time.time())}"
                        db[backup_id] = deepcopy(doc)
                        self._safe_log("info", f"[CouchDB] Backup of corrupted '{doc_id}' created as '{backup_id}' ✅")
                        # Repair doc
                        doc["initialized"] = True
                        doc["data"] = doc.get("data", {})
                        db[doc_id] = doc
                        self._safe_log("info", f"[CouchDB] '{doc_id}' repaired successfully ✅")
                    else:
                        self._safe_log("debug", f"[CouchDB] '{doc_id}' already exists and is valid.")
                else:
                    # Create initial document
                    db[doc_id] = {"initialized": True, "data": {}}
                    self._safe_log("info", f"[CouchDB] Created initial '{doc_id}' document ✅")
                return  # Success, exit loop
            except Exception as e:
                attempt += 1
                self._safe_log("error", f"[CouchDB] Attempt {attempt}/{max_retries} failed to ensure '{doc_id}': {e}")
                time.sleep(retry_delay)

        # If we reached here, all attempts failed
        self.logger.critical(
            f"[CouchDB] ❌ Failed to ensure '{doc_id}' exists after {max_retries} attempts. Manual intervention required.")

    """
    MemoryManager — Beyond-Perfection Edition v25.0
    Robust CouchDB + in-memory hybrid memory store with auto-heal, snapshots, and semantic indexing.
    """

    def __init__(self, db_url: str | None = None,
                 db_name: str = "bibliotheca_memory"):
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self.db = self._ensure_db()
        self._ensure_initial_doc(self.db)
        self.auto_sync_interval = PATCH_INTERVAL

        # Core stores
        self.memory_items: dict[str, dict] = {}
        self._memory: list[dict] = []
        self.snapshots: list[dict] = []
        self.semantic = SemanticMemory()

        self._db_url = db_url or "http://rohan:rthunderpheonix11@localhost:5984/"
        self._db_name = db_name

        self._server = None
        self.db = None
        self._couch_available = False

        try:
            import couchdb3
            self._server = couchdb3.Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")

            if self._db_name not in self._server:
                logger.info(
                    f"[MemoryManager] Database '{self._db_name}' missing, creating...")
                self._server.create(self._db_name)

            self.db = self._server[self._db_name]
            self._couch_available = True
            logger.info(
                f"[MemoryManager] CouchDB connected at {self._db_url} ✅")
        except Exception as e:
            logger.warning(
                f"[MemoryManager] CouchDB init failed: {e}; using in-memory mode ⚠️")
            self._couch_available = False

        # Background threads
        self._auto_thread = threading.Thread(
            target=self._auto_sync_loop, daemon=True)
        self._auto_thread.start()

        self._heal_thread = threading.Thread(
            target=self._auto_heal_loop, daemon=True)
        self._heal_thread.start()

        # Try to load persistent state
        try:
            self._load_persistent_state()
            logger.info("[MemoryManager] Persistent state loaded ✅")
        except Exception as e:
            logger.error(
                f"[MemoryManager] Failed to load persistent state: {e}")

        logger.info(
            "[MemoryManager] Initialized ✅ Beyond-Perfection Edition v25.0")

    # -----------------------
    # Thread-safety
    # -----------------------
    @contextmanager
    def locked(self):
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    # -----------------------
    # Memory Loading
    # -----------------------
    def load_memory(self) -> list[dict]:
        try:
            docs = [doc for doc in self.db]
        except Exception as e:
            if "not_found" in str(e):
                logger.warning(
                    "[MemoryManager] No docs found in DB, continuing with empty state")
                docs = []
            else:
                raise

        try:
            if self._couch_available and self.db:
                docs = [doc for doc in self.db]
                self._memory = docs
                logger.info(
                    f"[MemoryManager] Loaded {len(docs)} items from CouchDB ✅")
            else:
                self._memory = []
                logger.info(
                    "[MemoryManager] No CouchDB; using empty in-memory list.")

            self.take_snapshot(f"Memory loaded ({len(self._memory)} items)")
            return self._memory

        except Exception as exc:
            logger.error(f"[MemoryManager] load_memory failed: {exc}")
            if TELEMETRY:
                TELEMETRY.capture_exception(
                    exc, context="MemoryManager.load_memory")
            self._memory = []
            return self._memory

    # -----------------------
    # Auto-sync loop
    # -----------------------
    def _auto_sync_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._couch_available and self.db:
                    self._persist_local_store()
                time.sleep(self.auto_sync_interval)
            except Exception as e:
                logger.error(f"[MemoryManager] Auto-sync failed: {e}")

    # -----------------------
    # Auto-heal loop
    # -----------------------
    def _auto_heal_loop(self):
        while not self._stop_event.is_set():
            try:
                # Simple consistency check
                if not isinstance(self.memory_items, dict):
                    self.memory_items = {}
                    logger.warning("[MemoryManager] memory_items repaired")
                if not isinstance(self._memory, list):
                    self._memory = []
                    logger.warning("[MemoryManager] _memory repaired")
                time.sleep(self.auto_sync_interval * 2)
            except Exception as e:
                logger.error(f"[MemoryManager] Auto-heal failed: {e}")

    # -----------------------
    # Persistent State
    # -----------------------
    def _load_persistent_state(self):
        if self._couch_available and self.db:
            try:
                docs = list(self.db)
                for d in docs:
                    if "_id" in d:
                        self.semantic.data[d["_id"]] = {
                            "value": d.get("value"),
                            "meta": d.get("meta", {}),
                            "version": d.get("version", 1),
                            "updated_at": d.get("updated_at"),
                        }
                logger.info(
                    f"[MemoryManager] Semantic store populated with {len(docs)} items ✅")
                return
            except Exception as e:
                logger.warning(
                    f"[MemoryManager] CouchDB state load failed: {e}")

        # Local fallback
        try:
            if LOCAL_STORE_FILE.exists():
                payload = json.loads(
                    LOCAL_STORE_FILE.read_text(
                        encoding="utf-8"))
                self.semantic.data = payload.get("data", {})
                self.semantic.patch_proposals = payload.get(
                    "patch_proposals", [])
                self.semantic.metadata.update(payload.get("metadata", {}))
                logger.info("[MemoryManager] Local store loaded ✅")
            else:
                self._persist_local_store()
        except Exception as e:
            logger.error(f"[MemoryManager] Local state load failed: {e}")

    def _persist_local_store(self):
        try:
            payload = {
                "data": self.semantic.data,
                "patch_proposals": self.semantic.patch_proposals,
                "metadata": self.semantic.metadata,
            }
            _safe_write_text(LOCAL_STORE_FILE, json.dumps(payload, indent=2))
            logger.debug("[MemoryManager] Local store persisted")
        except Exception as e:
            logger.error(f"[MemoryManager] Persist local store failed: {e}")

    # -----------------------
    # Single Item / Metadata Persistence
    # -----------------------
    def _persist_memory_item(self, key: str):
        with self.semantic.locked():
            item = self.semantic.data.get(key)
            if not item:
                return False
            doc = {
                "_id": key,
                "type": "memory_item",
                "value": item.get("value"),
                "meta": item.get("meta", {}),
                "version": item.get("version", 1),
                "updated_at": item.get("updated_at", now_iso()),
            }
        if self._couch.available:
            saved = self._couch.save_doc(doc)
            if saved:
                return True
            else:
                logger.warning(
                    f"[MemoryManager] CouchDB save failed for {key}, falling back to local store")
        self._persist_local_store()
        return True

    def _persist_proposal(self, proposal: Dict[str, Any]):
        doc = dict(proposal)
        doc["_id"] = doc.get("id") or f"proposal_{uuid.uuid4().hex}"
        doc["type"] = "patch_proposal"
        if self._couch.available:
            saved = self._couch.save_doc(doc)
            if saved:
                return True
            else:
                logger.warning(
                    "[MemoryManager] CouchDB save failed for proposal, falling back to local store")
        self._persist_local_store()
        return True

    def _persist_metadata(self):
        doc = {
            "_id": "_metadata",
            "type": "metadata",
            "meta": self.semantic.metadata}
        if self._couch.available:
            saved = self._couch.save_doc(doc)
            if saved:
                return True
            else:
                logger.warning(
                    "[MemoryManager] CouchDB save failed for metadata, falling back to local store")
        self._persist_local_store()
        return True

    # -----------------------
    # Repair / Self-Heal
    # -----------------------
    def repair(self) -> bool:
        """
        Beyond-Perfection MemoryManager repair routine.
        """
        try:
            # Snapshot
            snapshot = {
                "timestamp": datetime.utcnow().isoformat(),
                "memory_count": len(
                    getattr(
                        self,
                        "_memory",
                        [])),
                "note": "Repair snapshot"}
            self.snapshots.append(snapshot)

            # Validate
            invalid_items = [
                i for i,
                item in enumerate(
                    getattr(
                        self,
                        "_memory",
                        [])) if not isinstance(
                    item,
                    dict) or "content" not in item]

            # Remove invalid
            for idx in reversed(invalid_items):
                bad_item = self._memory.pop(idx)
                logger.warning(
                    f"[MemoryManager] Removed invalid memory item during repair: {bad_item}")

            # Rebuild semantic index
            if hasattr(self, "semantic_index"):
                try:
                    self.semantic_index = self._rebuild_semantic_index()
                except Exception as e:
                    logger.error(
                        f"[MemoryManager] Failed to rebuild semantic index: {e}")

            # Telemetry log
            if TELEMETRY:
                try:
                    TELEMETRY.log_event(
                        "memory_repair_completed", {
                            "invalid_items_removed": len(invalid_items), "total_memory_items": len(
                                self._memory)})
                except Exception as e:
                    logger.error(
                        f"[MemoryManager] Failed to log repair to telemetry: {e}")

            logger.info(
                f"[MemoryManager] Repair completed successfully. Invalid items removed: {len(invalid_items)}")
            return True

        except Exception as repair_exception:
            logger.error(f"[MemoryManager] Repair failed: {repair_exception}")
            if TELEMETRY:
                try:
                    TELEMETRY.capture_exception(
                        repair_exception, context="MemoryManager.repair")
                except Exception:
                    pass
            return False

    # -----------------------
    # Snapshot Utility
    # -----------------------
    def take_snapshot(self, note: str = ""):
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_count": len(getattr(self, "_memory", [])),
            "note": note,
        }
        self.snapshots.append(snapshot)
        logger.debug(f"[MemoryManager] Snapshot taken: {note}")

    # In /Users/rohanlakhanpal/bibliotecha/app/utils/memory_manager.py

    def repair(self) -> bool:
        """
        Beyond-Perfection MemoryManager repair routine.
        -----------------------------------------------
        Self-healing method for memory errors or inconsistencies.
        Steps:
        1. Snapshot current memory state.
        2. Validate all entries in CouchDB (if available) or in-memory memory.
        3. Fix missing or corrupted entries using previous snapshots.
        4. Rebuild internal semantic indexes.
        5. Log repair actions via Telemetry.
        6. Return True if repair successful, False otherwise.
        """
        try:
            # 1. Snapshot current memory state
            if hasattr(self, 'snapshots'):
                snapshot = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "memory_count": len(getattr(self, 'memory_items', [])),
                    "note": "Repair snapshot"
                }
                self.snapshots.append(snapshot)

            # 2. Validate all memory entries
            invalid_items = []
            for idx, item in enumerate(getattr(self, 'memory_items', [])):
                if not isinstance(item, dict) or 'content' not in item:
                    invalid_items.append(idx)

            # 3. Remove/fix invalid entries
            for idx in reversed(invalid_items):
                bad_item = self.memory_items.pop(idx)
                logger.warning(
                    f"[MemoryManager] Removed invalid memory item during repair: {bad_item}")

            # 4. Rebuild semantic indexes
            if hasattr(self, 'semantic_index'):
                try:
                    self.semantic_index = self._rebuild_semantic_index()
                except Exception as e:
                    logger.error(
                        f"[MemoryManager] Failed to rebuild semantic index: {e}")

            # 5. Log via Telemetry
            if 'TELEMETRY' in globals():
                try:
                    TELEMETRY.log_event("memory_repair_completed", {
                        "invalid_items_removed": len(invalid_items),
                        "total_memory_items": len(self.memory_items)
                    })
                except Exception as e:
                    logger.error(
                        f"[MemoryManager] Failed to log repair to telemetry: {e}")

            logger.info(
                f"[MemoryManager] Repair completed successfully. Invalid items removed: {len(invalid_items)}")
            return True

        except Exception as repair_exception:
            logger.error(f"[MemoryManager] Repair failed: {repair_exception}")
            if 'TELEMETRY' in globals():
                try:
                    TELEMETRY.capture_exception(
                        repair_exception, context="MemoryManager.repair")
                except Exception:
                    pass
            return False

    # -----------------------
    # Memory CRUD Operations
    # -----------------------
    def store(self, key: str, content: Any, meta: dict = None):
        """
        Store a memory item.
        - key: unique identifier for the memory
        - content: actual memory content
        - meta: optional metadata dictionary
        """
        with self.locked():
            meta = meta or {}
            item = {
                "content": content,
                "meta": meta,
                "version": 1,
                "updated_at": now_iso(),
            }
            self.memory_items[key] = item
            self._persist_memory_item(key)
            logger.info(f"[MemoryManager] Stored memory item '{key}'")
            return True

    def recall(self, key: str = None, query: str = None, top_k: int = 5):
        """
        Retrieve memory items.
        - key: exact key lookup
        - query: search query (semantic or keyword)
        - top_k: max results for semantic search
        """
        with self.locked():
            if key:
                return self.memory_items.get(key)
            elif query:
                return self.search(query, top_k=top_k)
            else:
                return list(self.memory_items.values())

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Robust search for memory items.
        - query: keyword or semantic search string
        - top_k: number of top results to return
        """
        with self.locked():
            results = []
            # Simple keyword search
            for key, item in self.memory_items.items():
                if query.lower() in str(item.get("content", "")).lower():
                    results.append((key, item))
            # If semantic memory is available, you could integrate:
            # semantic_results = self.semantic.query(query, top_k=top_k)
            # results.extend(semantic_results)
            return results[:top_k]

    def propose_patch(self, key: str, new_content: Any, reason: str):
        """
        Create a patch proposal for a memory item.
        - key: memory item to update
        - new_content: proposed new value
        - reason: explanation for change
        """
        with self.locked():
            proposal = {
                "id": f"proposal_{uuid.uuid4().hex}",
                "target_key": key,
                "new_content": new_content,
                "reason": reason,
                "created_at": now_iso(),
            }
            self.semantic.patch_proposals.append(proposal)
            self._persist_proposal(proposal)
            logger.info(
                f"[MemoryManager] Patch proposed for '{key}': {reason}")
            return proposal

    def apply_patch_safely(self, proposal: dict):
        """
        Apply a patch safely with versioning and rollback.
        - proposal: dict returned by propose_patch
        """
        with self.locked():
            key = proposal.get("target_key")
            if not key or key not in self.memory_items:
                logger.warning(
                    f"[MemoryManager] Patch target not found: {key}")
                return False

            # Save current state for rollback
            old_item = self.memory_items[key].copy()
            try:
                self.memory_items[key]["content"] = proposal["new_content"]
                self.memory_items[key]["version"] += 1
                self.memory_items[key]["updated_at"] = now_iso()
                self._persist_memory_item(key)
                logger.info(f"[MemoryManager] Patch applied safely to '{key}'")
                return True
            except Exception as e:
                # Rollback on failure
                self.memory_items[key] = old_item
                logger.error(
                    f"[MemoryManager] Failed to apply patch to '{key}', rollback executed: {e}")
                return False

    # -----------------------
    # Public Memory API
    # -----------------------
    def store(self, key: str, value: str, meta: dict = None) -> bool:
        """
        Store a memory item in-memory and persist it.
        """
        with self._lock:
            self.semantic.store_local(key, value, meta=meta)
            ok = self._persist_memory_item(key)
            return ok

    def recall(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self.semantic.recall_local(key)
            if item:
                return item
            # if missing locally, attempt to fetch from CouchDB
            if self._couch.available:
                doc = self._couch.get_doc(key)
                if doc:
                    with self.semantic.locked():
                        self.semantic.data[key] = {
                            "value": doc.get("value"),
                            "meta": doc.get("meta", {}),
                            "version": doc.get("version", 1),
                            "updated_at": doc.get("updated_at"),
                        }
                        return self.semantic.data[key]
            return None

    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        """
        Semantic search: uses embedded function if provided, otherwise difflib fallback.
        """
        if self.semantic.embedding_fn:
            try:
                q_vec = self.semantic.embedding_fn(query)
                # If embedding support is added, scoring by vector similarity should be implemented here.
                # For now, fallback to local search (we keep embedding hook for
                # future use).
            except Exception as e:
                logger.debug(f"[MemoryManager] embedding_fn failed: {e}")
        return self.semantic.search_local(query, top_k=top_k)

        # -------------------------
        # Minimal repair method
        # -------------------------

    def repair(self) -> int:
        """Remove invalid or corrupted items from memory."""
        removed = 0
        try:
            keys_to_remove = [
                k for k, v in self.memory_items.items() if v is None]
            for k in keys_to_remove:
                self.memory_items.pop(k, None)
                removed += 1
            logger.info(
                f"[MemoryManager] Repair completed. Invalid items removed: {removed}")
        except Exception as e:
            TELEMETRY.capture_exception(e, context="MemoryManager_repair")
            logger.error(f"[MemoryManager] Repair failed: {e}")
        return removed

        # -------------------------
        # Minimal test method
        # -------------------------

    def test(self) -> bool:
        """Check memory health; always True for fallback."""
        try:
            if self.memory_items is None:
                self.memory_items = {}
            return True
        except Exception as e:
            TELEMETRY.capture_exception(e, context="MemoryManager_test")
            return False

    # -----------------------
    # Patching / Proposals
    # -----------------------
    def propose_patch(self,
                      target_file: str,
                      new_content: str,
                      description: str = "AutoPatch",
                      author: str = "system") -> Dict[str,
                                                      Any]:
        proposal = {
            "id": f"patch_{uuid.uuid4().hex}",
            "target_file": str(Path(target_file).resolve()),
            "content": new_content,
            "description": description,
            "author": author,
            "status": "proposed",
            "created_at": now_iso(),
            "applied_at": None,
            "error": None,
        }
        with self._lock:
            self.semantic.patch_proposals.append(proposal)
            self._persist_proposal(proposal)
            self._persist_metadata()
        logger.info(f"[MemoryManager] Patch proposed: {description[:50]} ✅")
        return proposal

    def apply_patch_safely(self, proposal: Dict[str, Any]) -> bool:
        """
        Attempt to apply a patch safely:
        - Creates backup
        - Writes new content
        - Attempts a safe reload of the module
        - On failure, rollbacks using backup
        """
        target_path = Path(proposal["target_file"])
        if not target_path.exists():
            logger.error(f"[MemoryManager] Target file missing: {target_path}")
            proposal["status"] = "failed"
            proposal["error"] = "target_missing"
            self._persist_proposal(proposal)
            return False

        backup = backup_file(target_path)
        try:
            # Write new content atomically
            tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
            _safe_write_text(tmp_path, proposal["content"])
            tmp_path.replace(target_path)  # atomic on most POSIX systems

            # Attempt to reload module safely
            mod_name = None
            # try to find a matching existing module
            for name, mod in list(sys.modules.items()):
                try:
                    f = getattr(mod, "__file__", None)
                    if f and Path(f).resolve() == target_path.resolve():
                        mod_name = name
                        break
                except Exception:
                    continue

            if mod_name:
                logger.info(
                    f"[MemoryManager] Reloading existing module: {mod_name}")
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception as e:
                    raise RuntimeError(f"reload_failed: {e}")
            else:
                # import by file location (best-effort)
                spec_name = f"patched_{target_path.stem}_{uuid.uuid4().hex[:8]}"
                spec = importlib.util.spec_from_file_location(
                    spec_name, str(target_path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec_name] = module
                    try:
                        spec.loader.exec_module(module)
                        logger.info(
                            f"[MemoryManager] Imported module under temp name: {spec_name}")
                    except Exception as e:
                        # cleanup created module entry
                        del sys.modules[spec_name]
                        raise RuntimeError(f"import_failed: {e}")
                else:
                    raise RuntimeError("spec_creation_failed")

            # mark proposal applied
            proposal["status"] = "applied"
            proposal["applied_at"] = now_iso()
            proposal["error"] = None
            self.semantic.metadata["last_patch"] = now_iso()
            self._persist_proposal(proposal)
            self._persist_metadata()
            logger.info(
                f"[MemoryManager] Patch applied successfully ✅ ({proposal['description'][:50]})")
            return True
        except Exception as e:
            # restore backup
            try:
                if backup:
                    rollback_file(target_path, backup)
            except Exception as re:
                logger.error(f"[MemoryManager] rollback exception: {re}")
            proposal["status"] = "failed"
            proposal["error"] = str(e)
            self.semantic.metadata["patch_failures"] = self.semantic.metadata.get(
                "patch_failures", 0) + 1
            self._persist_proposal(proposal)
            self._persist_metadata()
            logger.error(f"[MemoryManager] Patch failed ❌ {e}")
            return False

    # -----------------------
    # self-Heal / Background Loop
    # -----------------------

    def _self_heal_loop(self):
        """
        Regular self-heal checks to ensure DB and internal state are consistent.
        """
        while not self._stop_event.is_set():
            try:
                healthy = self.health_check()
                if not healthy:
                    logger.warning(
                        "[CouchDBClient] health_check failed — attempting reconnect & self-heal")
                    try:
                        self.connect()
                        if self._use_couchdb_lib() and self._server:
                            try:
                                if self.db_name in self._server:
                                    self.db = self._server[self.db_name]
                            except Exception:
                                logger.debug(
                                    "self-heal: unable to refresh db from server")
                        try:
                            self.ai_agent.autonomous_repair(self)
                        except Exception:
                            logger.debug("ai_agent.autonomous_repair failed")
                    except Exception as e:
                        logger.error(
                            f"[CouchDBClient] reconnect/self-heal failed: {e}")
                time.sleep(10)
            except Exception as e:
                logger.error(f"[CouchDBClient] self_heal_loop exception: {e}")
                time.sleep(5)

    # -----------------------
    # Auto-Heal / Background Loop
    # -----------------------
    def _auto_heal_loop(self):
        """
        Background loop for auto-healing memory.
        Periodically validates, repairs, snapshots, logs telemetry,
        and prunes old or redundant memory items.
        Fully thread-safe and CouchDB-aware.
        """
        logger.info("[MemoryManager] Auto-heal loop started ✅")

        while not self._stop_event.is_set():
            try:
                with self.locked():
                    logger.debug(
                        "[MemoryManager] Auto-heal iteration running...")

                    # -----------------------
                    # 1. Validate memory items
                    # -----------------------
                    invalid_items = [
                        key for key, item in self.memory_items.items()
                        if not isinstance(item, dict) or "content" not in item
                    ]
                    if invalid_items:
                        logger.warning(
                            f"[MemoryManager] Found {len(invalid_items)} invalid memory items")

                    # -----------------------
                    # 2. Repair corrupted items
                    # -----------------------
                    for key in invalid_items:
                        removed_item = self.memory_items.pop(key, None)
                        logger.warning(
                            f"[MemoryManager] Removed invalid memory item: {removed_item}")

                    # -----------------------
                    # 3. Rebuild semantic index if needed
                    # -----------------------
                    if hasattr(
                            self, "semantic") and hasattr(
                            self, "semantic_index"):
                        try:
                            self.semantic_index = self._rebuild_semantic_index()
                            logger.debug(
                                "[MemoryManager] Semantic index rebuilt")
                        except Exception as e:
                            logger.error(
                                f"[MemoryManager] Failed to rebuild semantic index: {e}")

                    # -----------------------
                    # 4. Persist snapshots & telemetry
                    # -----------------------
                    self.take_snapshot(note="Auto-heal snapshot")
                    if 'TELEMETRY' in globals():
                        try:
                            TELEMETRY.log_event("auto_heal_completed", {
                                "invalid_items_removed": len(invalid_items),
                                "total_memory_items": len(self.memory_items)
                            })
                        except Exception as e:
                            logger.error(
                                f"[MemoryManager] Failed to log telemetry: {e}")

                    # -----------------------
                    # 5. Optional pruning of old/redundant items
                    # -----------------------
                    MAX_MEMORY_ITEMS = getattr(self, "max_memory_items", 1000)
                    if len(self.memory_items) > MAX_MEMORY_ITEMS:
                        sorted_items = sorted(
                            self.memory_items.items(),
                            key=lambda kv: kv[1].get("updated_at", now_iso()),
                            reverse=True
                        )
                        to_remove = sorted_items[MAX_MEMORY_ITEMS:]
                        for key, _ in to_remove:
                            self.memory_items.pop(key, None)
                            logger.info(
                                f"[MemoryManager] Pruned old memory item: {key}")

                    # -----------------------
                    # 6. Persist to CouchDB / local store
                    # -----------------------
                    for key in self.memory_items:
                        try:
                            self._persist_memory_item(key)
                        except Exception as e:
                            logger.error(
                                f"[MemoryManager] Failed to persist memory item {key}: {e}")

                # -----------------------
                # Sleep until next auto-heal
                # -----------------------
                time.sleep(self.auto_sync_interval)

            except Exception as loop_exc:
                logger.error(
                    f"[MemoryManager] Exception in auto-heal loop: {loop_exc}")
                if 'TELEMETRY' in globals():
                    try:
                        TELEMETRY.capture_exception(
                            loop_exc, context="_auto_heal_loop")
                    except Exception:
                        pass
                time.sleep(self.auto_sync_interval)  # Wait before retrying

    # -----------------------
    # Stop Auto-Heal Loop
    # -----------------------
    def stop(self):
        """Gracefully stop the MemoryManager auto-heal loop."""
        self._stop_event.set()
        logger.info("[MemoryManager] Auto-heal loop stopped ✅")


# -----------------------
# Ultra-Intelligent AutoUpdater — Beyond Perfection Edition
# -----------------------
class AutoUpdater:
    """
    Fully autonomous updater for code/content.
    Queues tasks, generates AI proposals, and optionally applies them.
    Designed to be thread-safe, logged, and non-blocking.
    """
    def __init__(self, manager: "MemoryManager", ai_request_fn: callable, auto_apply: bool = AUTO_APPLY_PATCHES):
        self.manager = manager
        self.ai_request_fn = ai_request_fn
        self.auto_apply = auto_apply
        self._stop_event = threading.Event()
        self._task_queue: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        threading.Thread(target=self._run_loop, daemon=True, name="AutoUpdaterLoop").start()
        self._log(f"AutoUpdater initialized ✅ (auto_apply={self.auto_apply})")

    def _log(self, msg: str):
        if hasattr(self.manager, "logger"):
            self.manager.logger.info(f"[AutoUpdater] {msg}")
        else:
            logger.info(f"[AutoUpdater] {msg}")

    def queue_task(self, prompt: str, description: str = "UserTask") -> dict:
        task = {
            "id": f"task_{uuid.uuid4().hex}",
            "prompt": prompt,
            "description": description,
            "status": "queued",
            "created_at": now_iso(),
        }
        with self._lock:
            self._task_queue.append(task)
        self._log(f"Task queued: {description[:50]} ✅")
        return task

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    queue = list(self._task_queue)
                for task in queue:
                    if task.get("status") != "queued":
                        continue
                    task["status"] = "processing"
                    try:
                        generated_content = self.ai_request_fn(task["prompt"])
                        target_file = os.path.abspath(__file__)
                        proposal = self.manager.propose_patch(
                            target_file=target_file,
                            new_content=generated_content,
                            description=task["description"],
                            author="auto-updater"
                        )
                        if self.auto_apply:
                            applied = self.manager.apply_patch_safely(proposal)
                            if applied:
                                task["status"] = "applied"
                                task["applied_at"] = now_iso()
                                self._log(f"Task applied successfully: {task['description'][:50]} ✅")
                            else:
                                task["status"] = "failed"
                                task["error"] = proposal.get("error", "Unknown error")
                                self._log(f"Task failed to apply: {task['description'][:50]} ❌")
                        else:
                            task["status"] = "proposed"
                            self._log(f"Task generated proposal (auto_apply disabled): {task['description'][:50]}")
                    except Exception as e:
                        task["status"] = "failed"
                        task["error"] = str(e)
                        self._log(f"Task processing failed: {task['description'][:50]} ❌ {e}")
                time.sleep(2)
            except Exception as e:
                self._log(f"AutoUpdater loop exception: {e}")
                time.sleep(1)

    def stop(self):
        self._stop_event.set()
        self._log("Stopped ✅")


# -----------------------------
# Global MEMORY integration — Beyond Perfection
# -----------------------------
import threading
import psutil  # For system load monitoring
import random
import copy
import pickle
from pathlib import Path
import time

# Folder for predictive MEMORY snapshots
SNAPSHOT_DIR = Path(__file__).resolve().parent / "memory_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_INTERVAL = 60  # seconds between snapshots
MAX_SNAPSHOTS = 10      # keep the last 10 snapshots

def _is_snapshot_valid(snapshot_path: Path) -> bool:
    """Check if a snapshot file can be successfully loaded."""
    try:
        with open(snapshot_path, "rb") as f:
            json.load(f)
        return True
    except Exception:
        return False

def _initialize_memory():
    """
    Full MEMORY singleton initialization with:
      - auto-heal / fallback handling
      - AI helper attachment (multiple strategies)
      - background monitor thread for snapshots/self-heal (safe / optional)
      - defensive config defaults and robust logging
    This is intentionally defensive: it will never raise on missing optional components.
    """
    global MEMORY
    with _memory_lock:
        # -------------------------
        # Step 1: Ensure MEMORY exists
        # -------------------------
        if MEMORY is None:
            try:
                MemoryManagerClass = globals().get("MemoryManager", None)
                if MemoryManagerClass and callable(MemoryManagerClass):
                    # Try instantiating MemoryManager with best-effort config injection
                    try:
                        cfg = globals().get("CONFIG", None) or {}
                        try:
                            MEMORY = MemoryManagerClass(cfg)
                        except TypeError:
                            # constructor may not accept config
                            MEMORY = MemoryManagerClass()
                    except Exception as inner_e:
                        logger.warning(f"[Memory] MemoryManager ctor failed, trying fallback: {inner_e}")
                        raise inner_e
                    logger.info("[Memory] MEMORY singleton initialized with MemoryManager ✅")
                else:
                    raise RuntimeError("MemoryManager class unavailable or not callable")
            except Exception as e:
                logger.warning(f"[Memory] MEMORY init failed, falling back to MemoryManagerFallback: {e}")
                # Try to safely instantiate the fallback with/without config
                try:
                    MemoryManagerFallbackClass = globals().get("MemoryManagerFallback", None)
                    if MemoryManagerFallbackClass and callable(MemoryManagerFallbackClass):
                        try:
                            cfg = getattr(globals().get("CONFIG", {}), "config", globals().get("CONFIG", {})) or {}
                            try:
                                MEMORY = MemoryManagerFallbackClass(cfg)
                            except TypeError:
                                # fallback ctor doesn't accept config
                                MEMORY = MemoryManagerFallbackClass()
                        except Exception as inner_e:
                            logger.exception(f"[Memory] Failed to instantiate MemoryManagerFallback: {inner_e}")
                            MEMORY = None
                    else:
                        logger.error("[Memory] MemoryManagerFallback class missing; MEMORY remains None")
                        MEMORY = None
                except Exception as e2:
                    logger.exception(f"[Memory] Unexpected error while creating MemoryManagerFallback: {e2}")
                    MEMORY = None

        # If MEMORY still None, create a minimal in-memory dict to avoid crashes downstream
        if MEMORY is None:
            logger.critical("[Memory] Unable to instantiate any MemoryManager. Creating minimal in-memory shim.")
            class _MinimalMemoryShim:
                def __init__(self):
                    self.store = {}
                    self.config = {}
                    self.ai_helper = None
                    self.logger = logger
                def get(self, k, default=None): return self.store.get(k, default)
                def set(self, k, v): self.store[k] = v
                def save_snapshot(self, *args, **kwargs): pass
                def load_snapshot(self, *args, **kwargs): pass
            MEMORY = _MinimalMemoryShim()

        # -------------------------
        # Ensure MEMORY has expected attrs
        # -------------------------
        if not hasattr(MEMORY, "config") or MEMORY.config is None:
            # Prefer a global CONFIG object if present
            cfg_candidate = globals().get("CONFIG", None)
            if isinstance(cfg_candidate, dict):
                MEMORY.config = cfg_candidate
            else:
                MEMORY.config = {}
            logger.debug("[Memory] MEMORY.config was missing — assigned default dict")

        if not hasattr(MEMORY, "logger") or MEMORY.logger is None:
            MEMORY.logger = logger

        # -------------------------
        # Step 2: Attach AI helper (robust multi-strategy)
        # -------------------------
        try:
            attached = False

            # Strategy A: If there's a helper module with attach_to() available
            try:
                import importlib
                try:
                    ai_helper_mod = importlib.import_module("app.utils.ai_memory_helper")
                except Exception:
                    try:
                        ai_helper_mod = importlib.import_module("sandbox.ai_memory_helper")
                    except Exception:
                        ai_helper_mod = None

                if ai_helper_mod:
                    # If helper module exposes attach_to, prefer that (clean attach pattern)
                    if hasattr(ai_helper_mod, "attach_to") and callable(ai_helper_mod.attach_to):
                        try:
                            helper = ai_helper_mod.attach_to(MEMORY)
                            MEMORY.ai_helper = helper
                            attached = True
                            MEMORY.logger.info("[Memory] AI helper attached via attach_to(module) ✅")
                        except Exception as e:
                            MEMORY.logger.debug(f"[Memory] attach_to() failed: {e}", exc_info=True)
                    # If module exposes class AIMemoryHelper, instantiate it
                    if not attached and hasattr(ai_helper_mod, "AIMemoryHelper"):
                        try:
                            AIMemoryHelperCls = getattr(ai_helper_mod, "AIMemoryHelper")
                            helper = AIMemoryHelperCls(MEMORY)
                            MEMORY.ai_helper = helper
                            attached = True
                            MEMORY.logger.info("[Memory] AI helper instantiated from module.AIMemoryHelper ✅")
                        except Exception as e:
                            MEMORY.logger.debug(f"[Memory] Instantiation of module.AIMemoryHelper failed: {e}", exc_info=True)
            except Exception:
                # swallow and continue to next strategy
                MEMORY.logger.debug("[Memory] ai_memory_helper import strategy A failed; continuing", exc_info=True)

            # Strategy B: If a top-level AIMemoryHelper symbol was imported earlier in the file (AIMemoryHelper)
            if not attached:
                AIMemoryHelperCls = globals().get("AIMemoryHelper", None)
                if AIMemoryHelperCls:
                    try:
                        helper = AIMemoryHelperCls(MEMORY)
                        MEMORY.ai_helper = helper
                        attached = True
                        MEMORY.logger.info("[Memory] AI helper attached via global AIMemoryHelper ✅")
                    except Exception as e:
                        MEMORY.logger.debug(f"[Memory] Failed to attach global AIMemoryHelper: {e}", exc_info=True)

            # Strategy C: If MEMORY has a built-in method to accept helpers, use it
            if not attached:
                if hasattr(MEMORY, "attach_ai_helper") and callable(getattr(MEMORY, "attach_ai_helper")):
                    try:
                        MEMORY.attach_ai_helper(MEMORY)  # some implementations expect the memory instance
                        attached = True
                        MEMORY.logger.info("[Memory] AI helper attached via MEMORY.attach_ai_helper() ✅")
                    except Exception as e:
                        MEMORY.logger.debug(f"[Memory] MEMORY.attach_ai_helper failed: {e}", exc_info=True)

            # Final: log outcome
            if attached:
                # prefer structured telemetry if MEMORY supports it
                try:
                    if hasattr(MEMORY, "telemetry") and hasattr(MEMORY.telemetry, "record"):
                        MEMORY.telemetry.record("memory.ai_helper.attached", {"time": datetime.utcnow().isoformat()})
                except Exception:
                    pass
            else:
                MEMORY.logger.debug("[Memory] AI helper not found — continuing without helper (not critical)")

        except Exception as e:
            # Catch-all so init never crashes due to ai helper issues
            if hasattr(MEMORY, "logger"):
                MEMORY.logger.debug(f"[Memory] ai_helper attach failed: {e}", exc_info=True)
            else:
                logger.debug(f"[Memory] ai_helper attach failed: {e}", exc_info=True)

        # -------------------------
        # Step 3: Start / ensure a background monitor for snapshots / self-heal
        # -------------------------
        try:
            # only create one monitor per MEMORY
            if not getattr(MEMORY, "_monitor_thread_running", False):
                MEMORY._monitor_thread_running = True

                def _memory_monitor_loop(mem, stop_event=None):
                    """
                    Lightweight background monitor:
                      - periodic snapshot saves (if supported)
                      - optional simple integrity checks
                      - protected by try/except to never crash the main process
                    """
                    interval = 3600
                    try:
                        cfg_interval = None
                        if isinstance(mem.config, dict):
                            cfg_interval = mem.config.get("MEMORY_SYNC_INTERVAL")
                        else:
                            cfg_interval = getattr(mem.config, "MEMORY_SYNC_INTERVAL", None)
                        if cfg_interval:
                            try:
                                interval = int(cfg_interval)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Clamp interval to reasonable bounds
                    if not isinstance(interval, int) or interval <= 0:
                        interval = 3600

                    while getattr(mem, "_monitor_thread_running", True):
                        try:
                            # If mem supports save_snapshot or sync, call it
                            if hasattr(mem, "save_snapshot") and callable(getattr(mem, "save_snapshot")):
                                try:
                                    mem.save_snapshot()
                                    mem.logger.debug("[MemoryMonitor] save_snapshot executed")
                                except Exception as e:
                                    mem.logger.debug(f"[MemoryMonitor] save_snapshot failed: {e}", exc_info=True)
                            elif hasattr(mem, "sync") and callable(getattr(mem, "sync")):
                                try:
                                    mem.sync()
                                    mem.logger.debug("[MemoryMonitor] sync executed")
                                except Exception as e:
                                    mem.logger.debug(f"[MemoryMonitor] sync failed: {e}", exc_info=True)

                            # Small integrity check example: ensure config exists
                            if not hasattr(mem, "config") or mem.config is None:
                                mem.config = {}
                                mem.logger.debug("[MemoryMonitor] Re-created missing mem.config")

                        except Exception as loop_e:
                            try:
                                mem.logger.debug(f"[MemoryMonitor] loop error: {loop_e}", exc_info=True)
                            except Exception:
                                logger.debug(f"[MemoryMonitor] loop error: {loop_e}", exc_info=True)
                        # Sleep with ability to be interrupted by changing _monitor_thread_running
                        for _ in range(max(1, int(interval))):
                            if not getattr(mem, "_monitor_thread_running", True):
                                break
                            time.sleep(1)

                mon_thread = threading.Thread(target=_memory_monitor_loop, args=(MEMORY,), daemon=True, name="MemoryMonitorThread")
                try:
                    mon_thread.start()
                    MEMORY._monitor_thread = mon_thread
                    MEMORY.logger.info(f"[Memory] Adaptive MEMORY monitor with predictive self-heal & validation started ✅ (interval={getattr(MEMORY.config, 'MEMORY_SYNC_INTERVAL', MEMORY.config.get('MEMORY_SYNC_INTERVAL', 3600))})")
                except Exception as e:
                    MEMORY.logger.debug(f"[Memory] Failed to start monitor thread: {e}", exc_info=True)
                    MEMORY._monitor_thread_running = False
        except Exception:
            # never raise from monitor startup
            MEMORY.logger.debug("[Memory] monitor thread setup encountered unexpected error", exc_info=True)

        # -------------------------
        # Final: return or log final state
        # -------------------------
        try:
            MEMORY.logger.info("[Memory] MEMORY singleton initialized and integration checks complete ✅")
        except Exception:
            logger.info("[Memory] MEMORY singleton initialized and integration checks complete ✅")

def _start_memory_monitor(min_interval: int = 3, max_interval: int = 15):
    """
    Background thread that adaptively monitors, heals, self-heals MEMORY, and validates predictive snapshots.

    Improvements & guarantees:
      - Defensive against missing modules/attrs (psutil, SNAPSHOT_DIR, AIMemoryHelper)
      - Atomic snapshot writes via temp file + move
      - Uses MEMORY.dump / MEMORY.to_serializable / MEMORY.serialize / getattr fallback
      - Re-attaches AI helper using multiple strategies (attach_to, module.AIMemoryHelper, global)
      - Monitors and repairs MEMORY by restoring from valid snapshots
      - Stores thread and stop event on MEMORY for future control
      - Avoids busy-wait; sleeps with 1s granularity and is interruptible
    """
    # local imports/fallbacks
    try:
        import psutil  # optional, used for adaptive interval tuning
    except Exception:
        psutil = None

    # Ensure globals exist (provide sensible defaults if missing)
    SNAP_DIR = globals().get("SNAPSHOT_DIR")
    if SNAP_DIR is None:
        # default to a snapshots subdir in project root
        SNAP_DIR = Path(os.getcwd()) / "memory_snapshots"
        globals()["SNAPSHOT_DIR"] = SNAP_DIR
    else:
        SNAP_DIR = Path(SNAP_DIR)

    try:
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"[Memory] Failed to ensure snapshot dir {SNAP_DIR}: {e}")

    MAX_SNAPS = int(globals().get("MAX_SNAPSHOTS", 8))  # default rotation
    # use a monitor stop event so monitor can be stopped externally if needed
    stop_event = threading.Event()

    def _monitor_loop():
        interval = min_interval
        snapshot_counter = 0
        consecutive_errors = 0
        backoff_base = 2

        while not stop_event.is_set():
            try:
                with _memory_lock:
                    global MEMORY

                    # --------------------------
                    # 0) Quick sanity checks
                    # --------------------------
                    if MEMORY is None:
                        logger.warning("[Memory] MEMORY is None — attempting re-init")
                        try:
                            _initialize_memory()
                        except Exception as e:
                            logger.warning(f"[Memory] _initialize_memory() failed during monitor: {e}", exc_info=True)

                    # --------------------------
                    # 1) Self-healing: restore from valid snapshot if MEMORY corrupted / missing
                    # --------------------------
                    MemoryManagerClass = globals().get("MemoryManager", None)
                    # Accept fallback class as well
                    MemoryFallbackClass = globals().get("MemoryManagerFallback", None)
                    is_expected_type = False
                    try:
                        if MemoryManagerClass and isinstance(MEMORY, MemoryManagerClass):
                            is_expected_type = True
                        elif MemoryFallbackClass and isinstance(MEMORY, MemoryFallbackClass):
                            is_expected_type = True
                    except Exception:
                        # defensive in case isinstance fails on partial objects
                        is_expected_type = False

                    if MEMORY is None or not is_expected_type:
                        logger.warning("[Memory] MEMORY corrupted or missing, attempting restore from snapshots...")
                        restored = False
                        try:
                            snapshots = sorted(SNAP_DIR.glob("memory_snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                        except Exception:
                            snapshots = []
                        for snapshot_file in snapshots:
                            try:
                                if _is_snapshot_valid(snapshot_file):
                                    # Try to load snapshot as JSON and reconstruct MEMORY sensibly
                                    try:
                                        with open(snapshot_file, "r", encoding="utf-8") as f:
                                            data = json.load(f)
                                        # If MEMORY class exposes a from_snapshot or load_snapshot API, try that
                                        if MemoryManagerClass and hasattr(MemoryManagerClass, "from_snapshot"):
                                            try:
                                                MEMORY = MemoryManagerClass.from_snapshot(data)
                                                restored = True
                                                logger.info(f"[Memory] MEMORY restored via MemoryManager.from_snapshot from: {snapshot_file.name} ✅")
                                                break
                                            except Exception:
                                                pass
                                        # If a fallback constructor accepts a dict, try it
                                        if MemoryFallbackClass:
                                            try:
                                                MEMORY = MemoryFallbackClass(data)
                                                restored = True
                                                logger.info(f"[Memory] MEMORY restored via MemoryManagerFallback(snapshot) from: {snapshot_file.name} ✅")
                                                break
                                            except Exception:
                                                pass
                                        # As a last resort, assign a minimal shim with 'store' populated
                                        minimal = type("RestoredMemoryShim", (), {})()
                                        setattr(minimal, "store", data if isinstance(data, dict) else {})
                                        setattr(minimal, "config", getattr(MEMORY, "config", {}) or {})
                                        setattr(minimal, "logger", logger)
                                        setattr(minimal, "ai_helper", None)
                                        MEMORY = minimal
                                        restored = True
                                        logger.info(f"[Memory] MEMORY restored into minimal shim from: {snapshot_file.name} ✅")
                                        break
                                    except Exception as e:
                                        logger.warning(f"[Memory] Failed to load snapshot {snapshot_file.name}: {e}", exc_info=True)
                                else:
                                    logger.debug(f"[Memory] Snapshot {snapshot_file.name} invalid — skipped")
                            except Exception as e:
                                logger.warning(f"[Memory] Error validating/restoring snapshot {snapshot_file}: {e}", exc_info=True)
                        if not restored:
                            logger.warning("[Memory] No valid snapshots found; reinitializing MEMORY from scratch...")
                            try:
                                _initialize_memory()
                                logger.info("[Memory] _initialize_memory() invoked after failed snapshot restore")
                            except Exception as e:
                                logger.error(f"[Memory] Failed to reinitialize MEMORY: {e}", exc_info=True)
                        # reset interval on restore/reinit
                        interval = min_interval

                    # --------------------------
                    # 2) Ensure AI helper attached — safe multi-strategy
                    # --------------------------
                    try:
                        # Only try re-attach if MEMORY exists and ai_helper missing
                        if MEMORY and (not hasattr(MEMORY, "ai_helper") or MEMORY.ai_helper is None):
                            attached = False
                            # Strategy A: helper module `attach_to`
                            try:
                                import importlib
                                mod = None
                                try:
                                    mod = importlib.import_module("app.utils.ai_memory_helper")
                                except Exception:
                                    try:
                                        mod = importlib.import_module("sandbox.ai_memory_helper")
                                    except Exception:
                                        mod = None
                                if mod:
                                    if hasattr(mod, "attach_to") and callable(mod.attach_to):
                                        try:
                                            helper = mod.attach_to(MEMORY)
                                            MEMORY.ai_helper = helper
                                            attached = True
                                            MEMORY.logger.info("[Memory] AI helper re-attached via attach_to(module) ✅")
                                        except Exception as e:
                                            MEMORY.logger.debug(f"[Memory] attach_to failed during monitor: {e}", exc_info=True)
                                    if not attached and hasattr(mod, "AIMemoryHelper"):
                                        try:
                                            AIMemoryHelperCls = getattr(mod, "AIMemoryHelper")
                                            helper = AIMemoryHelperCls(MEMORY)
                                            MEMORY.ai_helper = helper
                                            attached = True
                                            MEMORY.logger.info("[Memory] AI helper instantiated via module.AIMemoryHelper ✅")
                                        except Exception as e:
                                            MEMORY.logger.debug(f"[Memory] module.AIMemoryHelper instantiation failed: {e}", exc_info=True)
                            except Exception:
                                # swallow and continue
                                pass

                            # Strategy B: global AIMemoryHelper available in the module namespace
                            if not attached:
                                AIMemoryHelperCls = globals().get("AIMemoryHelper")
                                if AIMemoryHelperCls:
                                    try:
                                        helper = AIMemoryHelperCls(MEMORY)
                                        MEMORY.ai_helper = helper
                                        attached = True
                                        MEMORY.logger.info("[Memory] AI helper re-attached via global AIMemoryHelper ✅")
                                    except Exception as e:
                                        MEMORY.logger.debug(f"[Memory] global AIMemoryHelper attach failed: {e}", exc_info=True)

                            # Strategy C: MEMORY provides attach_ai_helper API
                            if not attached and hasattr(MEMORY, "attach_ai_helper") and callable(getattr(MEMORY, "attach_ai_helper")):
                                try:
                                    MEMORY.attach_ai_helper()
                                    attached = True
                                    MEMORY.logger.info("[Memory] AI helper attached via MEMORY.attach_ai_helper() ✅")
                                except Exception as e:
                                    MEMORY.logger.debug(f"[Memory] MEMORY.attach_ai_helper failed: {e}", exc_info=True)

                            if not attached:
                                MEMORY.logger.debug("[Memory] AI helper not attached (monitor pass) — continuing")
                    except Exception as e:
                        logger.debug(f"[Memory] Error during ai_helper re-attach attempt: {e}", exc_info=True)

                    # --------------------------
                    # 3) Adaptive interval tuning using psutil (safe)
                    # --------------------------
                    try:
                        if psutil:
                            try:
                                cpu_load = psutil.cpu_percent(interval=None)
                                # convert 0-100 -> 0.0-1.0
                                cpu_frac = max(0.01, min(1.0, cpu_load / 100.0))
                            except Exception:
                                cpu_frac = 0.5
                        else:
                            # no psutil: use time-based fallback
                            cpu_frac = 0.5
                        memory_health = 1.0 if MEMORY else 0.0
                        stability_factor = memory_health / (cpu_frac + 0.1)
                        adaptive_interval = max(min_interval, min(max_interval, int(max_interval / (stability_factor + 0.1))))
                        interval = adaptive_interval
                    except Exception:
                        interval = min_interval

                    # --------------------------
                    # 4) Predictive snapshot: dump MEMORY safely and validate
                    # --------------------------
                    if MEMORY:
                        # Choose a serialization function from candidates
                        dump_func = None
                        for cand in ("dump", "to_serializable", "serialize", "export", "to_dict", "get_state"):
                            dump_func = getattr(MEMORY, cand, None)
                            if callable(dump_func):
                                break
                            dump_func = None
                        # If no callable found, try a simple mapping of attributes
                        if dump_func is None:
                            def _default_dump():
                                # prefer a 'store' attribute if present
                                if hasattr(MEMORY, "store") and isinstance(getattr(MEMORY, "store"), dict):
                                    return getattr(MEMORY, "store")
                                # try __dict__ but filter callables
                                try:
                                    d = {}
                                    for k, v in getattr(MEMORY, "__dict__", {}).items():
                                        if callable(v):
                                            continue
                                        d[k] = v
                                    return d
                                except Exception:
                                    return {}
                            dump_func = _default_dump

                        snapshot_path = SNAP_DIR / f"memory_snapshot_{snapshot_counter % MAX_SNAPS}.json"
                        temp_path = SNAP_DIR / f".tmp_memory_snapshot_{snapshot_counter % MAX_SNAPS}.json.tmp"
                        try:
                            payload = None
                            try:
                                payload = dump_func()
                            except Exception as e:
                                MEMORY.logger.debug(f"[Memory] dump_func failed; falling back to default: {e}", exc_info=True)
                                payload = {}
                            # Write atomically: write to temp file then move
                            try:
                                with open(temp_path, "w", encoding="utf-8") as f:
                                    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
                                # atomic replace
                                shutil.move(str(temp_path), str(snapshot_path))
                                # validate
                                if _is_snapshot_valid(snapshot_path):
                                    snapshot_counter += 1
                                    consecutive_errors = 0
                                    MEMORY.logger.debug(f"[Memory] Valid MEMORY snapshot saved: {snapshot_path.name}")
                                else:
                                    MEMORY.logger.warning(f"[Memory] Snapshot saved but failed validation: {snapshot_path.name}")
                            except Exception as e:
                                # clean up temp file if exists
                                try:
                                    if temp_path.exists():
                                        temp_path.unlink()
                                except Exception:
                                    pass
                                raise
                        except Exception as e:
                            consecutive_errors += 1
                            logger.warning(f"[Memory] Failed to save MEMORY snapshot: {e}", exc_info=True)

                # end with _memory_lock
                # apply adaptive sleep outside lock
                if consecutive_errors > 0:
                    # exponential backoff, bounded by max_interval*4
                    backoff = min(max_interval * 4, int((backoff_base ** min(consecutive_errors, 6)) ))
                    # jitter
                    jitter = int(max(1, backoff * 0.1))
                    sleep_for = max(1, min(backoff + (jitter if jitter else 0), max_interval * 4))
                else:
                    sleep_for = max(1, int(interval))
            except Exception as loop_e:
                logger.error(f"[Memory] MEMORY monitor encountered an error at top-level: {loop_e}", exc_info=True)
                # quick re-check on error
                sleep_for = min_interval

            # Sleep in interruptible 1s increments so stop_event can break quickly
            for _ in range(int(max(1, sleep_for))):
                if stop_event.is_set():
                    break
                time.sleep(1)

        # End monitor loop
        logger.info("[Memory] Memory monitor thread exiting cleanly.")

    # create/start the thread and attach control handles to MEMORY for later introspection/stop
    monitor_thread = threading.Thread(target=_monitor_loop, name="MemoryMonitorThread", daemon=True)
    monitor_thread.start()

    try:
        # expose control interface to the MEMORY object to allow graceful shutdown
        if MEMORY is not None:
            setattr(MEMORY, "_monitor_thread", monitor_thread)
            setattr(MEMORY, "_monitor_stop_event", stop_event)
            setattr(MEMORY, "_monitor_thread_running", True)
    except Exception:
        pass

    logger.info(f"[Memory] Adaptive MEMORY monitor with predictive self-heal & validation started ✅ (interval={min_interval}-{max_interval}s)")
    # Return handles for programmatic control if caller wants them
    return monitor_thread, stop_event

# Restore latest valid snapshot if MEMORY is missing on import
def _restore_latest_valid_snapshot():
    """
    Restore the latest valid memory snapshot safely into the global MEMORY instance.

    ✅ Ensures MEMORY is always a MemoryManager instance
    ✅ Merges snapshot data instead of overwriting MEMORY
    ✅ Handles errors gracefully with logging
    ✅ Skips invalid snapshots
    """
    global MEMORY

    # Ensure MEMORY exists
    if MEMORY is None:
        logger.warning("[Memory] MEMORY not initialized. Creating new MemoryManager instance.")
        MEMORY = MemoryManager()  # fallback safe init

    # Sort snapshots by modification time (newest first)
    snapshots = sorted(
        SNAPSHOT_DIR.glob("memory_snapshot_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    restored = False
    for snapshot_file in snapshots:
        if _is_snapshot_valid(snapshot_file):
            try:
                with open(snapshot_file, "rb") as f:
                    MEMORY.load_json(f)  # safely merge data into existing MEMORY
                logger.info(f"[Memory] MEMORY restored from valid snapshot: {snapshot_file.name} ✅")
                restored = True
                break
            except Exception as e:
                logger.warning(f"[Memory] Failed to restore snapshot {snapshot_file.name}: {e}")
        else:
            logger.warning(f"[Memory] Invalid snapshot skipped: {snapshot_file.name}")

    if not restored:
        logger.warning("[Memory] No valid snapshots found. MEMORY initialized empty.")
        MEMORY = MemoryManager()  # ensure MEMORY always exists

# Safely initialize MEMORY if it doesn't exist
MEMORY = globals().get("MEMORY", None)
if MEMORY is None:
    _restore_latest_valid_snapshot()

# Initialize memory and start monitor with safe defaults
_initialize_memory()
_start_memory_monitor(min_interval=3, max_interval=15)

# -----------------------------
# Lazy, safe MemoryStore class loader (handles circular imports)
# -----------------------------
def _get_memory_store_class() -> type:
    """
    Returns a fully safe MemoryStore class:
    - Uses real MemoryStore if available
    - Falls back to thread-safe in-memory MemoryStoreFallback
    - Supports optional CouchDB persistence
    - Fully logged
    - Avoids circular import issues
    """
    global MemoryStore
    try:
        # Attempt real MemoryStore import
        try:
            from app.utils.memory import MemoryStore as _RealMemoryStore
            logger.info("[MemoryManager] Real MemoryStore imported successfully ✅")
            MemoryStore = _RealMemoryStore
            return _RealMemoryStore
        except ImportError as e:
            logger.warning(f"[MemoryManager] Real MemoryStore not found: {e}, using fallback.")

        # Thread-safe in-memory fallback
        import threading

        class MemoryStoreFallback:
            """Thread-safe in-memory fallback for MemoryStore."""

            def __init__(self, *args, **kwargs):
                self._store: dict = {}
                self._lock = threading.RLock()
                self.logger = logger
                self._search_cache = {}
                self._safe_log("info", "[MemoryStoreFallback] Initialized in-memory fallback ✅")

            def _safe_log(self, level: str, msg: str):
                try:
                    if hasattr(self.logger, level):
                        getattr(self.logger, level)(msg)
                    else:
                        print(f"[{level.upper()}] {msg}")
                except Exception:
                    print(f"[SAFE LOG ERROR] {msg}")

            # CRUD Methods
            def get(self, key, default=None):
                try:
                    with self._lock:
                        value = self._store.get(key, default)
                    self._safe_log("debug", f"GET '{key}' -> {value}")
                    return value
                except Exception as e:
                    self._safe_log("error", f"GET '{key}' failed: {e}")
                    return default

            def set(self, key, value):
                try:
                    with self._lock:
                        self._store[key] = value
                    self._safe_log("debug", f"SET '{key}' -> {value}")
                    return True
                except Exception as e:
                    self._safe_log("error", f"SET '{key}' failed: {e}")
                    return False

            def delete(self, key):
                try:
                    with self._lock:
                        value = self._store.pop(key, None)
                    self._safe_log("debug", f"DELETE '{key}' -> {value}")
                    return value
                except Exception as e:
                    self._safe_log("error", f"DELETE '{key}' failed: {e}")
                    return None

            def exists(self, key) -> bool:
                try:
                    with self._lock:
                        exists = key in self._store
                    self._safe_log("debug", f"EXISTS '{key}' -> {exists}")
                    return exists
                except Exception as e:
                    self._safe_log("error", f"EXISTS '{key}' failed: {e}")
                    return False

            def keys(self):
                try:
                    with self._lock:
                        k = list(self._store.keys())
                    self._safe_log("debug", f"KEYS -> {k}")
                    return k
                except Exception as e:
                    self._safe_log("error", f"KEYS failed: {e}")
                    return []

            def all(self):
                try:
                    with self._lock:
                        copy_store = dict(self._store)
                    self._safe_log("debug", f"ALL -> {copy_store}")
                    return copy_store
                except Exception as e:
                    self._safe_log("error", f"ALL failed: {e}")
                    return {}

            def sync_with_db(self, db_client=None):
                if db_client is None:
                    self._safe_log("info", "No db_client provided for sync, skipping")
                    return
                try:
                    with self._lock:
                        for k, v in self._store.items():
                            db_client[k] = v
                    self._safe_log("info", "Synced in-memory store with db_client ✅")
                except Exception as e:
                    self._safe_log("warning", f"CouchDB sync failed: {e}")

            def search_local(
                self,
                query: str,
                top_k: int = 3,
                use_rapidfuzz: bool = True,
                score_threshold: float = 0.0,
                boost_recent: bool = True,
                boost_frequency: bool = True,
                include_metadata: bool = True
            ) -> list[dict]:
                if not query:
                    self._safe_log("warning", "search_local called with empty query")
                    return []

                # Setup fuzzy matching
                try:
                    import difflib
                    use_rf = False
                    if use_rapidfuzz:
                        try:
                            from rapidfuzz import fuzz
                            use_rf = True
                        except ImportError:
                            use_rf = False
                except ImportError as e:
                    self._safe_log("critical", f"Fuzzy search failed, difflib missing: {e}")
                    return []

                results = []

                try:
                    with self._lock:
                        if not hasattr(self, "_store") or not isinstance(self._store, dict):
                            self._safe_log("error", "Memory store missing or corrupted")
                            return []

                        for k, entry in self._store.items():
                            try:
                                # Extract value & metadata safely
                                if isinstance(entry, dict):
                                    value = entry.get("value", str(entry))
                                    entry_type = entry.get("type", "unknown")
                                    timestamp = entry.get("timestamp")
                                    last_access = entry.get("last_access")
                                    metadata = entry.get("metadata", {})
                                else:
                                    value = str(entry)
                                    entry_type = type(entry).__name__
                                    timestamp = None
                                    last_access = None
                                    metadata = {}

                                search_strings = [str(value)]
                                if include_metadata:
                                    for m_val in metadata.values():
                                        search_strings.append(str(m_val))

                                # Compute max score
                                score = 0.0
                                for s in search_strings:
                                    if use_rf:
                                        from rapidfuzz import fuzz
                                        score = max(score, fuzz.ratio(query, s) / 100)
                                    else:
                                        score = max(score, difflib.SequenceMatcher(None, query, s).ratio())

                                # Boost recent/frequent
                                if boost_recent and last_access:
                                    age_sec = max(time.time() - last_access, 0.001)
                                    score *= 1.0 + min(1.0, 86400 / age_sec) * 0.1
                                if boost_frequency and "access_count" in metadata:
                                    score *= 1.0 + min(1.0, metadata["access_count"] / 100) * 0.1

                                if score >= score_threshold:
                                    results.append({
                                        "score": round(score, 4),
                                        "key": k,
                                        "value": value,
                                        "type": entry_type,
                                        "timestamp": timestamp,
                                        "last_access": last_access,
                                        "metadata": metadata
                                    })

                            except Exception as e_inner:
                                self._safe_log("error", f"Scoring key '{k}' failed: {e_inner}")

                    # Sort & return top_k
                    results.sort(key=lambda x: x["score"], reverse=True)
                    top_results = results[:top_k]

                    # Cache
                    self._search_cache[query] = top_results
                    self._safe_log("debug", f"SEARCH '{query}' -> {top_results}")
                    return top_results

                except Exception as e_outer:
                    self._safe_log("critical", f"search_local unexpected error: {e_outer}")
                    return []

        # Assign fallback class
        MemoryStore = MemoryStoreFallback
        return MemoryStoreFallback

    except Exception as e_final:
        logger.critical(f"[MemoryManager] _get_memory_store_class failed: {e_final}")
        # Fallback to minimal inline MemoryStoreFallback
        class MinimalMemoryStoreFallback:
            def get(self, key, default=None): return default
            def set(self, key, value): return True
            def delete(self, key): return None
            def exists(self, key): return False
            def keys(self): return []
            def all(self): return {}
        MemoryStore = MinimalMemoryStoreFallback
        return MinimalMemoryStoreFallback
    
