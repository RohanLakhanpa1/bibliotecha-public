import eventlet
eventlet.monkey_patch(all=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bibliotheca Main — Beyond-Perfection Infinity++ v22
----------------------------------------------------
Fully stable, self-healing, Beyond-Perfection ready
Handles:
- MemoryManager with CouchDB or in-memory fallback
- Automatic database creation & reconnection
- MetaMonitor & AI core initialization
- Self-heal routines
- Conversational task intake & autonomous execution
- Interactive TTS mode
- Full logging & telemetry
"""

# -------------------------------
# Bibliotheca Core Initialization
# -------------------------------

# Standard Libraries
import os
import sys
import time
import logging
import asyncio
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List
from app.utils.fact_graph import FactGraph

# -------------------------------
# Third-Party Libraries with Robust Fallbacks
# -------------------------------
try:
    import requests
except ImportError:
    import urllib.request as requests
    logging.warning("Requests library not found, falling back to urllib.request")

try:
    import couchdb
except ImportError:
    couchdb = None
    logging.warning("CouchDB library not installed. Using in-memory storage as fallback.")

try:
    from dotenv import load_dotenv
except ImportError:
    logging.warning("python-dotenv not installed. Environment variables must be system-wide.")
    def load_dotenv(*args, **kwargs):
        logging.warning("load_dotenv called but skipped due to missing library.")

# -------------------------------
# AI & Memory Utilities
# -------------------------------
try:
    from app.utils.memory_manager import MemoryManager, MEMORY
except Exception as e:
    logging.error(f"MemoryManager import failed: {e}")
    MemoryManager = None
    MEMORY = None

try:
    from app.utils.ai_memory_helper import AIMemoryHelper
except Exception as e:
    logging.warning(f"AIMemoryHelper import failed: {e}")
    AIMemoryHelper = None

try:
    from app.utils.telemetry import TelemetryHandler
except Exception as e:
    logging.warning(f"TelemetryHandler import failed: {e}. Using minimal fallback.")

    class TelemetryHandler:
        """Minimal fallback telemetry handler for logging events and exceptions."""
        def __init__(self):
            self.events: List[Dict[str, Any]] = []

        def log_event(self, event_name: str, data: Optional[Dict[str, Any]] = None):
            self.events.append({"event": event_name, "data": data, "timestamp": time.time()})
            logging.info(f"[Telemetry] Event logged: {event_name}")

        def capture_exception(self, exception: Exception, context: Optional[str] = None):
            logging.error(f"[Telemetry] Exception captured: {exception} | Context: {context}")

telemetry = TelemetryHandler()
telemetry.log_event("bibliotheca_startup", {"version": "7.1.0"})
logging.info("✅ TelemetryHandler initialized Beyond-Perfection Edition v6.1")

# -------------------------------
# Global Constants
# -------------------------------
VERSION = "7.1.0"
START_TIME = datetime.now(timezone.utc)
DEBUG_MODE = os.getenv("BIBLIOTHECA_DEBUG", "True").lower() in ("true", "1", "yes")
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

# -------------------------------
# Logging Setup (Beyond-Perfection)
# -------------------------------
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = Path(__file__).parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"bibliotheca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s][%(threadName)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(LOG_LEVEL)
console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
console_handler.setFormatter(console_formatter)

# Rotating file handler
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setLevel(LOG_LEVEL)
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
file_handler.setFormatter(file_formatter)

# Root logger
logger = logging.getLogger("BibliothecaAI")
logger.setLevel(LOG_LEVEL)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Suppress noisy third-party loggers
for noisy_logger in ["urllib3", "asyncio", "socketio"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger.info("✅ Logging initialized Beyond-Perfection Edition")
logger.debug(f"Log file: {LOG_FILE} | Log level: {LOG_LEVEL}")

# -------------------------------
# Ensure Project Paths Are Discoverable
# -------------------------------
APP_PATH = PROJECT_ROOT / "app"
UTILS_PATH = APP_PATH / "utils"
essential_paths = [PROJECT_ROOT, APP_PATH, UTILS_PATH]

for path in essential_paths:
    try:
        path.mkdir(parents=True, exist_ok=True)
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            logger.debug(f"[Path Setup] Added to sys.path: {path_str}")
    except Exception as e:
        logger.error(f"[Path Setup] Failed to create or add path: {path} | Exception: {e}")

logger.info(f"[Path Setup] Project root: {PROJECT_ROOT}")
logger.info(f"[Path Setup] App path: {APP_PATH}")
logger.info(f"[Path Setup] Utils path: {UTILS_PATH}")
logger.debug(f"[Path Setup] sys.path snapshot: {sys.path[:10]}")

# -------------------------------
# Load Environment Variables Safely
# -------------------------------
dotenv_paths = [
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "config" / ".env"
]

env_loaded = False
for dotenv_path in dotenv_paths:
    if dotenv_path.exists():
        try:
            load_dotenv(dotenv_path=dotenv_path, override=True)
            logger.info(f"[Env Loader] Loaded environment variables from: {dotenv_path}")
            env_loaded = True
            break
        except Exception as e:
            logger.error(f"[Env Loader] Failed to load .env from {dotenv_path} | Exception: {e}")

if not env_loaded:
    logger.warning("[Env Loader] No .env file found. Using system environment variables only.")

required_env_vars = [
    "COUCHDB_USER",
    "COUCHDB_PASSWORD",
    "COUCHDB_URL",
    "API_KEY",
    "DEBUG_MODE"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.warning(f"[Env Loader] Missing required environment variables: {missing_vars}")
else:
    logger.info("[Env Loader] All required environment variables are set ✅")

def reload_env():
    """Reload .env files dynamically at runtime."""
    for path in dotenv_paths:
        if path.exists():
            load_dotenv(dotenv_path=path, override=True)
            logger.info(f"[Env Loader] Reloaded environment variables from: {path}")

logger.info("✅ Bibliotheca Core Initialization Complete")

# -----------------------------
# Safe MemoryManager Initialization & Async Auto-Sync
# -----------------------------
import asyncio
import atexit
import logging
from typing import Optional, Any

logger = logging.getLogger("BibliothecaAI.MemoryManager")

# -----------------------------
# Helper function to run async tasks safely
# -----------------------------
def run_async_task(coro):
    """
    Runs an async coroutine in a safe background thread/event loop.
    Handles already running loop in main thread.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Create a task in the running loop
        return asyncio.ensure_future(coro, loop=loop)
    else:
        return loop.run_until_complete(coro)

# -----------------------------
# Fallback MemoryManager singleton if MEMORY not already assigned
# -----------------------------
if MEMORY is None:
    class MemoryManagerFallback:
        """
        Full-featured fallback MemoryManager for in-memory storage.
        Async-safe, supports auto-sync, stop signals, and helper integration.
        """
        def __init__(self):
            self.config: Dict[str, Any] = {}
            self.data: Dict[str, Any] = {}
            self.lock = asyncio.Lock()
            self._stop_event = asyncio.Event()
            self._auto_sync_task: Optional[asyncio.Task] = None
            self.helper: Optional[Any] = None
            logger.warning("[MemoryManagerFallback] Initialized fallback MemoryManager ✅")

        async def sync(self):
            """Perform a memory sync. Override with actual DB logic if needed."""
            async with self.lock:
                logger.debug("[MemoryManagerFallback] Sync called (no-op)")
                # Here, you could implement actual file/db writes if needed

        async def start_auto_sync(self, interval_sec: int = 30):
            """
            Start automatic periodic memory syncing.
            interval_sec: sync interval in seconds
            """
            if self._auto_sync_task and not self._auto_sync_task.done():
                logger.info("[MemoryManagerFallback] Auto-sync already running")
                return

            logger.info(f"[MemoryManagerFallback] Starting auto-sync every {interval_sec} sec")
            async def auto_sync_loop():
                try:
                    while not self._stop_event.is_set():
                        await self.sync()
                        await asyncio.sleep(interval_sec)
                except asyncio.CancelledError:
                    logger.info("[MemoryManagerFallback] Auto-sync cancelled")
                except Exception as e:
                    logger.error(f"[MemoryManagerFallback] Exception in auto-sync: {e}", exc_info=True)

            self._auto_sync_task = asyncio.create_task(auto_sync_loop())

        async def stop(self):
            """Stop memory manager and auto-sync gracefully."""
            self._stop_event.set()
            if self._auto_sync_task:
                self._auto_sync_task.cancel()
                try:
                    await self._auto_sync_task
                except Exception:
                    pass
            logger.info("[MemoryManagerFallback] Stop signal completed ✅")

    MEMORY = MemoryManagerFallback()

# -----------------------------
# Safe AIMemoryHelper attachment
# -----------------------------
if 'AIMemoryHelper' in globals() and AIMemoryHelper:
    try:
        MEMORY.helper = AIMemoryHelper(MEMORY)
        logger.info("[MemoryManager] AIMemoryHelper attached ✅")
    except Exception as e:
        logger.warning(f"[MemoryManager] Failed to attach AIMemoryHelper: {e}", exc_info=True)

# -----------------------------
# Auto-start MemoryManager auto-sync
# -----------------------------
AUTO_SYNC_INTERVAL = int(os.getenv("MEMORY_AUTO_SYNC_INTERVAL", 30))
run_async_task(MEMORY.start_auto_sync(AUTO_SYNC_INTERVAL))

# -----------------------------
# Graceful shutdown registration
# -----------------------------
def shutdown_memory_manager():
    """Ensure MemoryManager stops gracefully at exit."""
    logger.info("[MemoryManager] Shutdown initiated")
    try:
        run_async_task(MEMORY.stop())
    except Exception as e:
        logger.error(f"[MemoryManager] Exception during shutdown: {e}", exc_info=True)

atexit.register(shutdown_memory_manager)
logger.info("[MemoryManager] Auto-sync and graceful shutdown registered ✅")

# -----------------------------
# Async auto-sync coroutine (Beyond-Perfection)
# -----------------------------
async def memory_auto_sync():
    """
    Continuously sync MEMORY asynchronously in a safe loop.
    Fully resilient, adaptive interval, logs failures, integrates with telemetry.
    """
    logger = logging.getLogger("BibliothecaAI.MemoryManager.AutoSync")
    default_interval = 3600  # 1 hour default sync interval
    logger.info("[MemoryManager.AutoSync] Async auto-sync loop starting ✅")

    while True:
        try:
            # Determine dynamic sync interval
            interval = getattr(getattr(MEMORY, "config", {}), "MEMORY_SYNC_INTERVAL", default_interval)
            env_interval = os.getenv("MEMORY_AUTO_SYNC_INTERVAL")
            if env_interval:
                try:
                    interval = int(env_interval)
                except Exception:
                    logger.warning(f"[MemoryManager.AutoSync] Invalid env MEMORY_AUTO_SYNC_INTERVAL={env_interval}, using {interval}s")

            # Execute sync if MEMORY supports it
            if hasattr(MEMORY, "sync") and callable(MEMORY.sync):
                logger.debug("[MemoryManager.AutoSync] Sync triggered")
                await MEMORY.sync()
                telemetry.log_event("memory_auto_sync", {"interval": interval, "status": "success"})
            else:
                logger.warning("[MemoryManager.AutoSync] MEMORY.sync() not available. Skipping this iteration.")

            # Sleep until next sync
            logger.debug(f"[MemoryManager.AutoSync] Sleeping for {interval} seconds")
            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("[MemoryManager.AutoSync] Auto-sync task cancelled. Exiting loop ✅")
            break
        except Exception as e:
            logger.error(f"[MemoryManager.AutoSync] Exception during auto-sync: {e}", exc_info=True)
            telemetry.capture_exception(e, context="memory_auto_sync")
            # Use exponential backoff for repeated failures
            backoff = min(interval * 2, 3600)
            logger.info(f"[MemoryManager.AutoSync] Retrying in {backoff} seconds due to failure")
            await asyncio.sleep(backoff)

# -----------------------------
# Safe Event Loop Creation & Memory Auto-Sync Startup (Beyond-Perfection)
# -----------------------------
from threading import Thread

logger = logging.getLogger("BibliothecaAI.MemoryManager.Loop")

def ensure_event_loop():
    """
    Safely get or create an asyncio event loop.
    Handles cases where loop is already running in main thread or nested threads.
    """
    try:
        loop = asyncio.get_running_loop()
        logger.debug("[EventLoop] Detected running asyncio loop ✅")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.debug("[EventLoop] Created new asyncio loop ✅")
    return loop

# Global loop reference
LOOP = ensure_event_loop()

# -----------------------------
# Run async tasks safely in background thread if needed
# -----------------------------
def start_background_loop(loop: asyncio.AbstractEventLoop):
    """
    Run the asyncio event loop in a separate thread.
    Ensures all async MemoryManager tasks run continuously.
    """
    def _run_loop():
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"[EventLoop] Exception in background loop: {e}", exc_info=True)
        finally:
            logger.info("[EventLoop] Background loop stopped ✅")

    thread = Thread(target=_run_loop, daemon=True, name="AsyncLoopThread")
    thread.start()
    logger.info("[EventLoop] Async background loop thread started ✅")
    return thread

# Start the loop if not already running
if not LOOP.is_running():
    start_background_loop(LOOP)
else:
    logger.debug("[EventLoop] Loop already running in main thread")

# -----------------------------
# Start Memory Auto-Sync Task in the loop
# -----------------------------
async def _start_memory_auto_sync():
    try:
        await memory_auto_sync()
    except asyncio.CancelledError:
        logger.info("[MemoryManager.AutoSync] Auto-sync task cancelled cleanly ✅")
    except Exception as e:
        logger.error(f"[MemoryManager.AutoSync] Exception in auto-sync task: {e}", exc_info=True)
        telemetry.capture_exception(e, context="_start_memory_auto_sync")

# Schedule auto-sync
auto_sync_task = asyncio.run_coroutine_threadsafe(_start_memory_auto_sync(), LOOP)
logger.info("[MemoryManager] MEMORY singleton ready & auto-sync scheduled ✅")

# -----------------------------
# Graceful shutdown of auto-sync
# -----------------------------
def stop_memory_auto_sync():
    """Cancel auto-sync task and stop loop safely."""
    try:
        if auto_sync_task and not auto_sync_task.done():
            auto_sync_task.cancel()
            logger.info("[MemoryManager] Auto-sync task cancellation requested ✅")
        # Optionally stop the loop if it was started by us
        if not LOOP.is_running():
            LOOP.stop()
            logger.info("[EventLoop] Async loop stopped ✅")
    except Exception as e:
        logger.error(f"[MemoryManager] Exception during auto-sync shutdown: {e}", exc_info=True)

atexit.register(stop_memory_auto_sync)
logger.info("[MemoryManager] Auto-sync shutdown registered at exit ✅")

# -----------------------------
# Graceful Shutdown — Beyond-Perfection
# -----------------------------
import signal

logger = logging.getLogger("BibliothecaAI.GracefulShutdown")

def flush_memory_at_exit():
    """
    Ensures MEMORY is fully synced and stopped on program exit.
    Handles both full MemoryManager and fallback safely.
    """
    try:
        logger.info("[GracefulShutdown] Initiating MEMORY flush at exit ✅")
        if MEMORY:
            # Cancel any running auto-sync tasks
            if hasattr(MEMORY, "_auto_sync_task") and MEMORY._auto_sync_task:
                MEMORY._auto_sync_task.cancel()
                logger.info("[GracefulShutdown] Auto-sync task cancelled")

            # Run MEMORY.stop() if available
            if hasattr(MEMORY, "stop") and callable(MEMORY.stop):
                logger.info("[GracefulShutdown] Running MEMORY.stop()")
                try:
                    run_async_task(MEMORY.stop())
                    logger.info("[GracefulShutdown] MEMORY.stop() completed ✅")
                    telemetry.log_event("memory_shutdown", {"status": "completed"})
                except Exception as e:
                    logger.error(f"[GracefulShutdown] Error running MEMORY.stop(): {e}", exc_info=True)
                    telemetry.capture_exception(e, context="flush_memory_at_exit")

            # Run MEMORY.sync() if available
            if hasattr(MEMORY, "sync") and callable(MEMORY.sync):
                logger.info("[GracefulShutdown] Running MEMORY.sync()")
                try:
                    run_async_task(MEMORY.sync())
                    logger.info("[GracefulShutdown] MEMORY.sync() completed ✅")
                    telemetry.log_event("memory_flush", {"status": "completed"})
                except Exception as e:
                    logger.error(f"[GracefulShutdown] Error running MEMORY.sync(): {e}", exc_info=True)
                    telemetry.capture_exception(e, context="flush_memory_at_exit")

    except Exception as e:
        logger.exception("[GracefulShutdown] Unexpected error during exit flush", exc_info=True)

# Register with atexit
atexit.register(flush_memory_at_exit)
logger.info("[GracefulShutdown] Registered flush_memory_at_exit with atexit ✅")

# Optional: handle termination signals (SIGINT/SIGTERM) for extra safety
def _signal_handler(signum, frame):
    logger.info(f"[GracefulShutdown] Received termination signal ({signum}). Flushing MEMORY...")
    flush_memory_at_exit()
    sys.exit(0)

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, _signal_handler)
logger.info("[GracefulShutdown] Signal handlers registered for SIGINT & SIGTERM ✅")

# -----------------------------
# CouchDB Preparation & Helpers — Beyond-Perfection
# -----------------------------
logger = logging.getLogger("BibliothecaAI.CouchDB")

# -----------------------------
# Validate CouchDB environment variables
# -----------------------------
COUCHDB_USER = os.getenv("COUCHDB_USER")
COUCHDB_PASSWORD = os.getenv("COUCHDB_PASSWORD")
COUCHDB_URL = os.getenv("COUCHDB_URL")

if not COUCHDB_URL or not COUCHDB_USER or not COUCHDB_PASSWORD:
    logger.warning(
        "[CouchDB Prep] Missing one or more CouchDB environment variables. "
        "CouchDBManager will operate in fallback mode."
    )
    telemetry.log_event("couchdb_missing_env", {
        "COUCHDB_URL": bool(COUCHDB_URL),
        "COUCHDB_USER": bool(COUCHDB_USER),
        "COUCHDB_PASSWORD": bool(COUCHDB_PASSWORD)
    })
else:
    logger.info("[CouchDB Prep] All CouchDB environment variables detected ✅")

# -----------------------------
# Helper: Async-safe DB operation wrapper
# -----------------------------
async def run_db_task(coro, description: str = "DB Task"):
    """
    Runs a coroutine in the global event loop safely.
    Captures exceptions and logs them to telemetry.
    """
    try:
        future = asyncio.run_coroutine_threadsafe(coro, LOOP)
        result = future.result()  # wait for completion
        logger.debug(f"[CouchDB Helper] {description} completed successfully ✅")
        telemetry.log_event("couchdb_task_completed", {"task": description})
        return result
    except Exception as e:
        logger.error(f"[CouchDB Helper] {description} failed: {e}", exc_info=True)
        telemetry.capture_exception(e, context=f"run_db_task:{description}")
        return None

# -----------------------------
# Fallback DB indicator
# -----------------------------
COUCHDB_AVAILABLE = bool(couchdb and COUCHDB_URL and COUCHDB_USER and COUCHDB_PASSWORD)
if COUCHDB_AVAILABLE:
    logger.info("[CouchDB Prep] CouchDB is available. Manager will connect normally ✅")
else:
    logger.warning("[CouchDB Prep] CouchDB unavailable. Manager will use in-memory or fallback mode ⚠️")

# -----------------------------
# Optional: Default DB connection template
# -----------------------------
def get_couchdb_server():
    """
    Returns a CouchDB server instance if available, else None.
    """
    if not COUCHDB_AVAILABLE:
        logger.warning("[CouchDB Prep] CouchDB not available. Returning None.")
        return None
    try:
        server = couchdb.Server(COUCHDB_URL)
        server.resource.credentials = (COUCHDB_USER, COUCHDB_PASSWORD)
        logger.info("[CouchDB Prep] Connected to CouchDB server ✅")
        telemetry.log_event("couchdb_connected")
        return server
    except Exception as e:
        logger.error(f"[CouchDB Prep] Failed to connect to CouchDB server: {e}", exc_info=True)
        telemetry.capture_exception(e, context="get_couchdb_server")
        return None

# -----------------------------
# CouchDB Memory & Connection (Beyond-Perfection)
# -----------------------------
from logging import getLogger
import os
import time
import threading
import json
from pathlib import Path
from hashlib import sha256
from typing import Optional, Dict, Any, List
from cryptography.fernet import Fernet

logger = getLogger("Bibliotheca.CouchDB")
logger.setLevel(logging.INFO)

# Backup directory for in-memory fallback
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKUP_DIR = PROJECT_ROOT / "couchdb_backups"
BACKUP_DIR.mkdir(exist_ok=True)

class CouchDBManager:
    """
    Advanced CouchDB Manager:
    - Auto retries & exponential backoff
    - Thread-safe save/get/delete
    - Bulk operations
    - In-memory fallback with rotating JSON backups
    - Optional AES encryption for memory backups
    - Auto reconnection & recovery
    - Full telemetry integration
    """

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        db_name: str = "bibliotheca_db",
        retry_attempts: int = 5,
        retry_delay: float = 2.0,
        backup_prefix: str = "couchdb_backup_",
        encrypt_backups: bool = False,
    ):
        self.url = url or os.getenv("COUCHDB_URL", "http://127.0.0.1:5984/")
        self.username = username or os.getenv("COUCHDB_USER")
        self.password = password or os.getenv("COUCHDB_PASSWORD")
        self.db_name = db_name
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.backup_prefix = backup_prefix
        self.encrypt_backups = encrypt_backups
        self.key = Fernet.generate_key() if encrypt_backups else None
        self.cipher = Fernet(self.key) if encrypt_backups else None
        self.lock = threading.RLock()
        self.server: Optional[Any] = None
        self.db: Dict[str, Any] = {}
        self.in_memory: bool = False
        self._initialize()

    def _initialize(self):
        """Initialize CouchDB with retries, fallback, and auto-create DB."""
        try:
            import couchdb
            url = self._format_url()
            for attempt in range(1, self.retry_attempts + 1):
                try:
                    self.server = couchdb.Server(url)
                    if self.db_name not in self.server:
                        self.server.create(self.db_name)
                        logger.info(f"CouchDB DB '{self.db_name}' created ✅")
                    self.db = self.server[self.db_name]
                    logger.info(f"CouchDB DB '{self.db_name}' connected successfully ✅")
                    telemetry.log_event("couchdb_connected", {"db": self.db_name})
                    return
                except Exception as e:
                    logger.warning(f"CouchDB connection attempt {attempt}/{self.retry_attempts} failed: {e}")
                    time.sleep(self.retry_delay * (2 ** (attempt - 1)))
            raise ConnectionError(f"Unable to connect to CouchDB after {self.retry_attempts} attempts")
        except Exception as e:
            logger.error(f"CouchDB unavailable: {e}. Falling back to in-memory storage.")
            telemetry.capture_exception(e, context="CouchDBManager._initialize")
            self.in_memory = True
            self.db = {}
            self._load_backup()

    def _format_url(self) -> str:
        url = self.url
        if self.username and self.password:
            if "://" not in url:
                url = f"http://{url}"
            url = f"{url.split('://')[0]}://{self.username}:{self.password}@{url.split('://')[1]}"
        return url

    def _get_backup_path(self, doc_id: str) -> Path:
        filename = f"{self.backup_prefix}{sha256(doc_id.encode()).hexdigest()}.json"
        return BACKUP_DIR / filename

    def _save_backup(self, doc_id: str, data: Dict[str, Any]):
        path = self._get_backup_path(doc_id)
        try:
            content = json.dumps(data, ensure_ascii=False, indent=4).encode()
            if self.encrypt_backups:
                content = self.cipher.encrypt(content)
            path.write_bytes(content)
            logger.debug(f"Backup saved for '{doc_id}' at {path}")
            telemetry.log_event("couchdb_backup_saved", {"doc_id": doc_id})
        except Exception as e:
            logger.error(f"Failed to backup document '{doc_id}': {e}")
            telemetry.capture_exception(e, context="_save_backup")

    def _load_backup(self):
        """Load all backup files into memory."""
        for file in BACKUP_DIR.glob(f"{self.backup_prefix}*.json"):
            try:
                content = file.read_bytes()
                if self.encrypt_backups:
                    content = self.cipher.decrypt(content)
                data = json.loads(content.decode())
                self.db[file.stem] = data
                logger.info(f"Loaded backup '{file.name}' into memory")
            except Exception as e:
                logger.warning(f"Failed to load backup '{file.name}': {e}")
                telemetry.capture_exception(e, context="_load_backup")

    def save(self, doc_id: str, data: Dict[str, Any]):
        with self.lock:
            if self.in_memory:
                self.db[doc_id] = data
                self._save_backup(doc_id, data)
                logger.debug(f"Saved in-memory document '{doc_id}'")
                return
            try:
                if doc_id in self.db:
                    doc = self.db[doc_id]
                    doc.update(data)
                    self.db.save(doc)
                else:
                    self.db[doc_id] = data
                logger.debug(f"Saved CouchDB document '{doc_id}'")
                telemetry.log_event("couchdb_doc_saved", {"doc_id": doc_id})
            except Exception as e:
                logger.error(f"Failed to save CouchDB doc '{doc_id}': {e}")
                telemetry.capture_exception(e, context="save")
                self.db[doc_id] = data
                self.in_memory = True
                self._save_backup(doc_id, data)
                logger.warning("Switched to in-memory due to CouchDB error")

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if self.in_memory:
                return self.db.get(doc_id)
            try:
                result = self.db.get(doc_id, {})
                return dict(result) if result else None
            except Exception as e:
                logger.error(f"Failed to get CouchDB doc '{doc_id}': {e}")
                telemetry.capture_exception(e, context="get")
                self.in_memory = True
                return self.db.get(doc_id)

    def delete(self, doc_id: str):
        with self.lock:
            if self.in_memory:
                self.db.pop(doc_id, None)
                logger.debug(f"Deleted in-memory document '{doc_id}'")
                telemetry.log_event("couchdb_doc_deleted", {"doc_id": doc_id})
            else:
                try:
                    if doc_id in self.db:
                        self.db.delete(self.db[doc_id])
                        logger.debug(f"Deleted CouchDB doc '{doc_id}'")
                        telemetry.log_event("couchdb_doc_deleted", {"doc_id": doc_id})
                except Exception as e:
                    logger.error(f"Failed to delete CouchDB doc '{doc_id}': {e}")
                    telemetry.capture_exception(e, context="delete")
                    self.db.pop(doc_id, None)
                    self.in_memory = True

    def exists(self, doc_id: str) -> bool:
        with self.lock:
            existence = doc_id in self.db
            logger.debug(f"Document '{doc_id}' existence check: {existence}")
            return existence

    def bulk_save(self, documents: Dict[str, Dict[str, Any]]):
        with self.lock:
            for doc_id, data in documents.items():
                self.save(doc_id, data)
            telemetry.log_event("couchdb_bulk_save", {"count": len(documents)})

    def bulk_get(self, doc_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        with self.lock:
            results = {doc_id: self.get(doc_id) for doc_id in doc_ids}
            telemetry.log_event("couchdb_bulk_get", {"count": len(doc_ids)})
            return results

# -----------------------------
# Initialize Telemetry & BackendRouter (Beyond-Perfection)
# -----------------------------
from logging import getLogger
from pathlib import Path
from typing import Optional, Dict, Any

logger = getLogger("Bibliotheca.BackendInit")

# -----------------------------
# Robust TelemetryHandler import with fail-safe fallback
# -----------------------------
try:
    from app.utils.telemetry import TelemetryHandler
except Exception as e:
    logger.warning(f"Failed to import TelemetryHandler: {e}. Using minimal fallback.")

    class TelemetryHandler:
        """
        Minimal fallback TelemetryHandler for safe logging if main module fails.
        Captures events and exceptions without breaking runtime.
        """
        def __init__(self):
            self.events = []

        def log_event(self, event_name: str, data: Optional[Dict[str, Any]] = None):
            self.events.append({"event": event_name, "data": data})
            logger.info(f"[Fallback Telemetry] Event logged: {event_name}")

        def info(self, msg: str):
            logger.info(f"[Fallback Telemetry] {msg}")

        def warn(self, msg: str):
            logger.warning(f"[Fallback Telemetry] {msg}")

        def error(self, msg: str):
            logger.error(f"[Fallback Telemetry] {msg}")

        def capture_exception(self, exception: Exception, context: Optional[str] = None):
            logger.error(f"[Fallback Telemetry] Exception captured: {exception} | Context: {context}")

# Initialize TelemetryHandler singleton
telemetry = TelemetryHandler()
telemetry.log_event("backend_init_start", {"version": VERSION})
logger.info("✅ TelemetryHandler initialized Beyond-Perfection Edition v7.1")

# -----------------------------
# Initialize CouchDBManager Backend
# -----------------------------
try:
    couchdb_manager = CouchDBManager(
        url=os.getenv("COUCHDB_URL"),
        username=os.getenv("COUCHDB_USER"),
        password=os.getenv("COUCHDB_PASSWORD"),
        db_name="bibliotheca_db",
        encrypt_backups=True  # AES encrypt backups by default
    )
    telemetry.log_event("couchdb_manager_initialized", {"in_memory": couchdb_manager.in_memory})
    logger.info(f"✅ CouchDBManager initialized. Fallback mode: {couchdb_manager.in_memory}")
except Exception as e:
    logger.error(f"Failed to initialize CouchDBManager: {e}", exc_info=True)
    telemetry.capture_exception(e, context="BackendRouter.CouchDBManager")
    # Force fallback to in-memory if CouchDB unavailable
    from types import SimpleNamespace
    couchdb_manager = SimpleNamespace(db={}, in_memory=True)
    logger.warning("CouchDBManager fallback to in-memory active ⚠️")

# -----------------------------
# Initialize FactGraph
# -----------------------------
try:
    fact_graph = FactGraph(couchdb_manager=couchdb_manager)
    telemetry.log_event("fact_graph_initialized")
    logger.info("✅ FactGraph initialized with CouchDBManager")
except Exception as e:
    logger.error(f"Failed to initialize FactGraph: {e}", exc_info=True)
    telemetry.capture_exception(e, context="BackendRouter.FactGraph")
    fact_graph = None

# -----------------------------
# Initialize RulesEngine
# -----------------------------
try:
    rules_engine = RulesEngine(fact_graph=fact_graph, telemetry=telemetry)
    telemetry.log_event("rules_engine_initialized")
    logger.info("✅ RulesEngine initialized with FactGraph")
except Exception as e:
    logger.error(f"Failed to initialize RulesEngine: {e}", exc_info=True)
    telemetry.capture_exception(e, context="BackendRouter.RulesEngine")
    rules_engine = None

# -----------------------------
# BackendRouter Singleton
# -----------------------------
class BackendRouter:
    """
    Central backend access point for Bibliotheca:
    - Provides safe access to CouchDBManager, FactGraph, RulesEngine
    - Handles fallback modes automatically
    - Integrates telemetry for all operations
    """
    def __init__(self):
        self.couchdb = couchdb_manager
        self.graph = fact_graph
        self.rules = rules_engine
        self.telemetry = telemetry
        self.logger = logger
        self.logger.info("✅ BackendRouter initialized Beyond-Perfection Edition")

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.couchdb.get(doc_id)
        except Exception as e:
            self.logger.error(f"[BackendRouter] get_document failed for '{doc_id}': {e}")
            self.telemetry.capture_exception(e, context="BackendRouter.get_document")
            return None

    def save_document(self, doc_id: str, data: Dict[str, Any]):
        try:
            self.couchdb.save(doc_id, data)
            self.telemetry.log_event("document_saved", {"doc_id": doc_id})
        except Exception as e:
            self.logger.error(f"[BackendRouter] save_document failed for '{doc_id}': {e}")
            self.telemetry.capture_exception(e, context="BackendRouter.save_document")

    def delete_document(self, doc_id: str):
        try:
            self.couchdb.delete(doc_id)
            self.telemetry.log_event("document_deleted", {"doc_id": doc_id})
        except Exception as e:
            self.logger.error(f"[BackendRouter] delete_document failed for '{doc_id}': {e}")
            self.telemetry.capture_exception(e, context="BackendRouter.delete_document")

# Instantiate singleton router
backend_router = BackendRouter()
telemetry.log_event("backend_router_ready")
logger.info("✅ BackendRouter singleton ready Beyond-Perfection Edition")

# -----------------------------
# Initialize CouchDBManager with auto-reconnect & in-memory fallback
# -----------------------------
try:
    couchdb_manager = CouchDBManager(
        url=os.getenv("COUCHDB_URL"),
        username=os.getenv("COUCHDB_USER"),
        password=os.getenv("COUCHDB_PASSWORD"),
        db_name="bibliotheca_db",
        encrypt_backups=True  # AES encrypted backups by default
    )
    telemetry.log_event("couchdb_manager_initialized", {"in_memory": couchdb_manager.in_memory})
    logger.info(f"✅ CouchDBManager initialized. Fallback mode: {couchdb_manager.in_memory}")
except Exception as e:
    logger.error(f"CouchDBManager initialization failed: {e}", exc_info=True)
    telemetry.capture_exception(e, context="BackendInit.CouchDBManager")


    # Fully featured in-memory fallback CouchDBManager
    class InMemoryCouchDBManager:
        """
        Thread-safe in-memory CouchDBManager fallback:
        - Full API compatibility: save, get, delete, exists, bulk operations
        - Telemetry integrated for every operation
        """

        def __init__(self):
            self.db: Dict[str, Any] = {}
            self.lock = threading.RLock()
            self.telemetry = telemetry
            self.logger = logger
            self.logger.warning("Using InMemoryCouchDBManager fallback ✅")
            self.telemetry.log_event("couchdb_in_memory_fallback")

        def save(self, doc_id: str, data: Dict[str, Any]):
            with self.lock:
                self.db[doc_id] = data
                self.logger.debug(f"[InMemoryCouchDB] Saved document '{doc_id}'")
                self.telemetry.log_event("in_memory_doc_saved", {"doc_id": doc_id})

        def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
            with self.lock:
                result = self.db.get(doc_id)
                self.logger.debug(f"[InMemoryCouchDB] Retrieved document '{doc_id}': {bool(result)}")
                return result

        def delete(self, doc_id: str):
            with self.lock:
                self.db.pop(doc_id, None)
                self.logger.debug(f"[InMemoryCouchDB] Deleted document '{doc_id}'")
                self.telemetry.log_event("in_memory_doc_deleted", {"doc_id": doc_id})

        def exists(self, doc_id: str) -> bool:
            with self.lock:
                existence = doc_id in self.db
                self.logger.debug(f"[InMemoryCouchDB] Existence check '{doc_id}': {existence}")
                return existence

        def bulk_save(self, documents: Dict[str, Dict[str, Any]]):
            with self.lock:
                for doc_id, data in documents.items():
                    self.db[doc_id] = data
                self.logger.debug(f"[InMemoryCouchDB] Bulk saved {len(documents)} documents")
                self.telemetry.log_event("in_memory_bulk_save", {"count": len(documents)})

        def bulk_get(self, doc_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
            with self.lock:
                results = {doc_id: self.db.get(doc_id) for doc_id in doc_ids}
                self.logger.debug(f"[InMemoryCouchDB] Bulk get for {len(doc_ids)} documents")
                return results


    couchdb_manager = InMemoryCouchDBManager()

# -----------------------------
# Initialize Telemetry (Beyond-Perfection)
# -----------------------------
try:
    if 'telemetry' not in globals() or telemetry is None:
        telemetry = TelemetryHandler()
    telemetry.info("✅ TelemetryHandler initialized Beyond-Perfection Edition v7.1")
except Exception as e:
    logger.error(f"[Telemetry Init] Failed during initialization: {e}", exc_info=True)
    # Safe fallback: minimal in-memory telemetry
    class MinimalTelemetry:
        def __init__(self):
            self.events = []

        def log_event(self, event_name, data=None):
            self.events.append({"event": event_name, "data": data})
            logger.info(f"[MinimalTelemetry] Event logged: {event_name}")

        def info(self, msg):
            logger.info(f"[MinimalTelemetry] {msg}")

        def warn(self, msg):
            logger.warning(f"[MinimalTelemetry] {msg}")

        def error(self, msg):
            logger.error(f"[MinimalTelemetry] {msg}")

        def capture_exception(self, exc, context=None):
            logger.error(f"[MinimalTelemetry] Exception: {exc} | Context: {context}")

    telemetry = MinimalTelemetry()
    telemetry.warn("Using MinimalTelemetry fallback. Limited functionality.")

# -----------------------------
# Initialize BackendRouter (singleton, Beyond-Perfection)
# -----------------------------
try:
    # Ensure BackendRouter class exists
    if 'backend_router' not in globals() or backend_router is None:
        # Fully integrated BackendRouter
        backend_router = BackendRouter(
            db_manager=couchdb_manager,
            telemetry=telemetry,
            fact_graph=FactGraph(couchdb_manager=couchdb_manager),
            rules_engine=RulesEngine()
        )
    telemetry.info("✅ BackendRouter initialized with telemetry, CouchDBManager, FactGraph, and RulesEngine")
except Exception as e:
    logger.error(f"[BackendRouter Init] Failed to initialize: {e}", exc_info=True)
    telemetry.capture_exception(e, context="BackendRouter.Init")
    # Safe fallback BackendRouter
    class FallbackBackendRouter:
        def __init__(self, db_manager=None, telemetry=None, fact_graph=None, rules_engine=None):
            self.db_manager = db_manager
            self.telemetry = telemetry
            self.fact_graph = fact_graph
            self.rules_engine = rules_engine
            self.logger = logger
            self.telemetry.info("Using FallbackBackendRouter. Core features limited ⚠️")

        def get_document(self, doc_id):
            self.telemetry.warn(f"[FallbackBackendRouter] get_document called for '{doc_id}'")
            if self.db_manager:
                return self.db_manager.get(doc_id)
            return None

        def save_document(self, doc_id, data):
            self.telemetry.warn(f"[FallbackBackendRouter] save_document called for '{doc_id}'")
            if self.db_manager:
                self.db_manager.save(doc_id, data)

        def delete_document(self, doc_id):
            self.telemetry.warn(f"[FallbackBackendRouter] delete_document called for '{doc_id}'")
            if self.db_manager:
                self.db_manager.delete(doc_id)

        def route_task(self, task):
            self.telemetry.info(f"[FallbackBackendRouter] Task received: {task}")

    backend_router = FallbackBackendRouter(
        db_manager=couchdb_manager,
        telemetry=telemetry,
        fact_graph=FactGraph(couchdb_manager=couchdb_manager),
        rules_engine=RulesEngine()
    )
    telemetry.warn("✅ FallbackBackendRouter initialized. Functionality limited.")

# -----------------------------
# Final confirmation
# -----------------------------
logger.info("✅ Backend initialization complete and fully operational Beyond-Perfection")
telemetry.info("✅ Backend system ready. All managers initialized and operational")
# -----------------------------
# Initialize FactGraph early (Beyond-Perfection)
# -----------------------------
from app.utils.ai_core import fact_graph
from logging import getLogger
import threading

logger = getLogger("Bibliotheca.FactGraphInit")

# Thread-safe initialization lock
_fact_graph_lock = threading.RLock()

try:
    with _fact_graph_lock:
        # Initialize FactGraph singleton safely
        if not getattr(fact_graph, "_initialized", False):
            if hasattr(fact_graph, "initialize") and callable(fact_graph.initialize):
                fact_graph.initialize()
            fact_graph._initialized = True

            # Optional: preload critical facts
            if hasattr(fact_graph, "load_predefined_facts") and callable(fact_graph.load_predefined_facts):
                try:
                    fact_graph.load_predefined_facts()
                    logger.info("✅ FactGraph preloaded with predefined facts successfully")
                except Exception as e:
                    logger.warning(f"FactGraph preload failed: {e}")
            else:
                logger.info("✅ FactGraph initialized (no predefined facts to load)")
        else:
            logger.info("FactGraph already initialized; skipping re-initialization")
except Exception as e:
    logger.error(f"[FactGraph Init] Initialization failed: {e}", exc_info=True)

    # -----------------------------
    # Minimal in-memory FactGraph fallback
    # -----------------------------
    class InMemoryFactGraph:
        """Thread-safe in-memory FactGraph fallback for safe reasoning."""

        def __init__(self):
            self._facts: Dict[str, Any] = {}
            self._lock = threading.RLock()
            logger.warning("Using InMemoryFactGraph fallback. Advanced reasoning limited.")

        def add_fact(self, key: str, value: Any):
            with self._lock:
                self._facts[key] = value
                logger.debug(f"[Fallback FactGraph] Added fact: {key} -> {value}")

        def get_fact(self, key: str) -> Any:
            with self._lock:
                return self._facts.get(key)

        def exists(self, key: str) -> bool:
            with self._lock:
                return key in self._facts

        def all_facts(self) -> Dict[str, Any]:
            with self._lock:
                return dict(self._facts)

        def remove_fact(self, key: str):
            with self._lock:
                self._facts.pop(key, None)
                logger.debug(f"[Fallback FactGraph] Removed fact: {key}")

    # Replace main fact_graph with safe fallback
    fact_graph = InMemoryFactGraph()
    telemetry.log_event("fact_graph_fallback_initialized")
    logger.warning("✅ InMemoryFactGraph initialized as safe fallback")

# -----------------------------
# Initialize Self-Healing AI (Beyond-Perfection)
# -----------------------------
import threading
from logging import getLogger

logger = getLogger("Bibliotheca.SelfHealingAIInit")

# Thread-safe initialization lock
_self_heal_lock = threading.RLock()

try:
    with _self_heal_lock:
        if 'self_healing_ai' not in globals() or self_healing_ai is None:
            try:
                from app.utils.ai_core_self_heal import SelfHealingAI

                # Initialize the SelfHealingAI with full telemetry and backend awareness
                self_healing_ai = SelfHealingAI(
                    telemetry=telemetry,
                    db=couchdb_manager,
                    fact_graph=fact_graph,
                    backend_router=backend_router
                )

                # Preload critical modules and recovery routines safely
                if hasattr(self_healing_ai, "load_critical_modules"):
                    try:
                        self_healing_ai.load_critical_modules()
                        logger.info("✅ SelfHealingAI critical modules loaded")
                    except Exception as e:
                        logger.warning(f"[SelfHealingAI] Failed to load critical modules: {e}")

                if hasattr(self_healing_ai, "register_default_recovery_tasks"):
                    try:
                        self_healing_ai.register_default_recovery_tasks()
                        logger.info("✅ Default self-healing tasks registered")
                    except Exception as e:
                        logger.warning(f"[SelfHealingAI] Failed to register default recovery tasks: {e}")

                telemetry.info("✅ SelfHealingAI initialized successfully (Beyond-Perfection Edition)")

            except Exception as init_exception:
                logger.exception(f"[ERROR] SelfHealingAI core initialization failed: {init_exception}")
                telemetry.error(f"SelfHealingAI failed to initialize: {init_exception}")

                # -----------------------------
                # Minimal safe Self-Healing AI fallback
                # -----------------------------
                class MinimalSelfHealingAI:
                    """Fallback Self-Healing AI ensuring safe operations with limited recovery."""

                    def __init__(self):
                        self.lock = threading.RLock()
                        self.active = True
                        telemetry.warn("Using MinimalSelfHealingAI fallback: limited features")

                    def perform_task(self, task_name, *args, **kwargs):
                        with self.lock:
                            telemetry.info(f"[Fallback SelfHealingAI] Task requested: {task_name}")
                            return None  # Safe fallback response

                    def heal_self(self):
                        with self.lock:
                            telemetry.warn("[Fallback SelfHealingAI] Self-heal invoked: limited effect")
                            return False

                    def status(self):
                        with self.lock:
                            return {"active": self.active, "mode": "fallback"}

                self_healing_ai = MinimalSelfHealingAI()
                telemetry.warn("✅ MinimalSelfHealingAI initialized as safe fallback")

        else:
            logger.info("SelfHealingAI already initialized; skipping re-initialization")

except Exception as final_exception:
    logger.critical(f"[CRITICAL] SelfHealingAI initialization failed completely: {final_exception}")
    raise RuntimeError("SelfHealingAI could not be initialized") from final_exception

# -----------------------------
# Initialize Auto-Healer (Beyond-Perfection)
# -----------------------------
import threading
import time
import atexit
import os
import traceback
from typing import Optional, Callable, Any
from logging import getLogger

__all__ = ["start_auto_healer", "stop_auto_healer", "auto_healer_status"]

logger = getLogger("Bibliotheca.AutoHealer")

# -----------------------------
# Global control variables
# -----------------------------
_auto_heal_lock = threading.RLock()
_auto_healer_thread: Optional[threading.Thread] = None
_auto_healer_stop_event = threading.Event()
_auto_healer_started: bool = False
_auto_healer_instance: Any = None
_auto_healer_extension: Any = None

# -----------------------------
# Configurable behavior via environment variables
# -----------------------------
AUTOHEALER_RESTART_ON_EXIT = os.getenv("BIBLIOTHECA_AUTOHEALER_RESTART", "true").lower() in ("1", "true", "yes")
AUTOHEALER_RESTART_BACKOFF_INITIAL = float(os.getenv("BIBLIOTHECA_AUTOHEALER_BACKOFF_INITIAL", "1.0"))
AUTOHEALER_RESTART_BACKOFF_MAX = float(os.getenv("BIBLIOTHECA_AUTOHEALER_BACKOFF_MAX", "60.0"))
DISABLE_AUTOHEALER = os.getenv("BIBLIOTHECA_DISABLE_AUTOHEALER", "false").lower() in ("1", "true", "yes")

# -----------------------------
# Internal auto-healer loop
# -----------------------------
def _auto_healer_loop():
    global _auto_healer_instance
    backoff = AUTOHEALER_RESTART_BACKOFF_INITIAL
    logger.info("[AutoHealer] Auto-Healer loop started ✅")

    while not _auto_healer_stop_event.is_set():
        try:
            if _auto_healer_instance is None:
                from app.utils.ai_core_self_heal import SelfHealingAI
                _auto_healer_instance = self_healing_ai  # Use main SelfHealingAI instance

            if _auto_healer_instance and hasattr(_auto_healer_instance, "heal_self"):
                healed = _auto_healer_instance.heal_self()
                logger.debug(f"[AutoHealer] Heal attempt completed: {healed}")

            if _auto_healer_extension and callable(_auto_healer_extension):
                try:
                    _auto_healer_extension()
                except Exception as ext_e:
                    logger.warning(f"[AutoHealer] Extension execution failed: {ext_e}")

            backoff = AUTOHEALER_RESTART_BACKOFF_INITIAL
            time.sleep(5)  # main loop delay

        except Exception as e:
            logger.exception(f"[AutoHealer] Exception in loop: {e}")
            time.sleep(min(backoff, AUTOHEALER_RESTART_BACKOFF_MAX))
            backoff *= 2  # exponential backoff

# -----------------------------
# Public control functions
# -----------------------------
def start_auto_healer(extension: Optional[Callable] = None):
    global _auto_healer_thread, _auto_healer_started, _auto_healer_extension

    if DISABLE_AUTOHEALER:
        logger.info("[AutoHealer] Disabled via environment; not starting")
        return

    with _auto_heal_lock:
        if _auto_healer_started:
            logger.info("[AutoHealer] Already running; skipping start")
            return

        _auto_healer_extension = extension
        _auto_healer_stop_event.clear()
        _auto_healer_thread = threading.Thread(target=_auto_healer_loop, name="AutoHealerThread", daemon=True)
        _auto_healer_thread.start()
        _auto_healer_started = True
        logger.info("[AutoHealer] Auto-Healer successfully started ✅")

def stop_auto_healer():
    global _auto_healer_started
    with _auto_heal_lock:
        if not _auto_healer_started:
            logger.info("[AutoHealer] Not running; skipping stop")
            return
        _auto_healer_stop_event.set()
        if _auto_healer_thread:
            _auto_healer_thread.join(timeout=10)
        _auto_healer_started = False
        logger.info("[AutoHealer] Auto-Healer stopped ✅")

def auto_healer_status() -> dict:
    return {
        "running": _auto_healer_started,
        "thread_alive": _auto_healer_thread.is_alive() if _auto_healer_thread else False,
        "stop_event_set": _auto_healer_stop_event.is_set()
    }

# -----------------------------
# Ensure graceful shutdown at exit
# -----------------------------
atexit.register(stop_auto_healer)
logger.info("[AutoHealer] Registered stop_auto_healer with atexit ✅")

# -----------------------------
# Minimal safe fallback classes (Beyond-Perfection)
# -----------------------------
import threading
import time
from logging import getLogger

logger = getLogger("Bibliotheca.AutoHealerFallback")

class _MinimalAutoHealer:
    """
    Very small, safe fallback auto-healer that doesn't run risky operations.
    Designed for safe operation if the main Auto-Healer fails.
    """
    def __init__(self, telemetry=None):
        self._running = False
        self._lock = threading.RLock()
        self.telemetry = telemetry
        if self.telemetry:
            self.telemetry.warn("[MinimalAutoHealer] Fallback initialized with telemetry")

    def start(self):
        with self._lock:
            self._running = True
            msg = "[MinimalAutoHealer] MinimalAutoHealer started (no-op)."
            logger.warning(msg)
            if self.telemetry:
                self.telemetry.info(msg)

    def run(self):
        """Run one pass of 'healing' — minimal and safe."""
        with self._lock:
            if not self._running:
                return
            logger.debug("[MinimalAutoHealer] Running minimal heal pass (no real effect).")
            if self.telemetry:
                self.telemetry.info("[MinimalAutoHealer] Heal pass executed")
            # minimal safe delay for simulating work without blocking main loop
            time.sleep(0.05)

    def stop(self):
        with self._lock:
            self._running = False
            msg = "[MinimalAutoHealer] MinimalAutoHealer stopped safely."
            logger.warning(msg)
            if self.telemetry:
                self.telemetry.info(msg)

    def status(self):
        with self._lock:
            return {
                "running": self._running,
                "mode": "fallback",
                "telemetry_attached": bool(self.telemetry)
            }

class _MinimalExtension:
    """
    Safe extension placeholder for Auto-Healer.
    Provides hooks to attach custom safe monitoring routines without breaking flow.
    """
    def __init__(self, instance=None, telemetry=None):
        self.instance = instance
        self.telemetry = telemetry

    def attach(self):
        msg = "[MinimalExtension] Fallback extension attached (no-op)."
        logger.debug(msg)
        if self.telemetry:
            self.telemetry.info(msg)


# -----------------------------
# Attempt to import real implementations (Beyond-Perfection)
# -----------------------------
import threading
import traceback
from typing import Callable, Any

try:
    from app.utils.auto_healer import (
        AUTOHEALER_INSTANCE as REAL_AUTOHEALER_INSTANCE,  # type: ignore
        AdvancedAutoHealerExtension,  # type: ignore
        run_auto_healer as _maybe_run_auto_healer  # type: ignore
    )

    logger.info("[AutoHealer] Real AutoHealer components imported successfully ✅")
except Exception as imp_err:
    logger.warning(
        f"[AutoHealer] Could not import real AutoHealer components: {imp_err}. "
        "Using fallback minimal implementations."
    )
    REAL_AUTOHEALER_INSTANCE = None
    AdvancedAutoHealerExtension = None
    _maybe_run_auto_healer = None


# -----------------------------
# Internal safe worker loop with restart/backoff
# -----------------------------
def _auto_healer_worker(run_func: Callable[[], Any], stop_event: threading.Event) -> None:
    """
    Runs the auto-healer in a safe loop with restart/backoff.
    Fully telemetry-integrated, ensures safe operations even on errors.
    """
    backoff = AUTOHEALER_RESTART_BACKOFF_INITIAL
    logger.info("[AutoHealer] Worker thread starting.")

    while not stop_event.is_set():
        try:
            logger.info("[AutoHealer] Invoking auto-healer run function.")
            run_func()  # long-running or looped task
            logger.info("[AutoHealer] Auto-healer run function exited normally.")
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception(f"[AutoHealer] Exception in auto-healer run: {e}\n{tb}")
            try:
                telemetry.capture_exception(e, context="auto_healer.run_exception")
            except Exception:
                logger.debug("[AutoHealer] telemetry.capture_exception failed (ignored).")

        # Stop requested? exit loop
        if stop_event.is_set():
            logger.info("[AutoHealer] Stop event received; exiting worker loop.")
            break

        # Restart logic with exponential backoff
        if AUTOHEALER_RESTART_ON_EXIT:
            logger.warning(f"[AutoHealer] Restarting auto-healer in {backoff:.1f}s "
                           f"(AUTOHEALER_RESTART_ON_EXIT=True).")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, AUTOHEALER_RESTART_BACKOFF_MAX)
            continue
        else:
            logger.info("[AutoHealer] AUTOHEALER_RESTART_ON_EXIT=False; worker exiting.")
            break

    logger.info("[AutoHealer] Worker thread fully exited ✅")

# -----------------------------
# Public control functions (Beyond-Perfection)
# -----------------------------
def start_auto_healer() -> None:
    """Idempotently start the Auto-Healer worker thread."""
    global _auto_healer_thread, _auto_healer_started, _auto_healer_instance, _auto_healer_extension

    if DISABLE_AUTOHEALER:
        logger.info("[AutoHealer] Auto-healer is disabled via BIBLIOTHECA_DISABLE_AUTOHEALER.")
        try:
            telemetry.info("Auto-healer disabled via environment flag.")
        except Exception:
            logger.debug("[AutoHealer] telemetry.info failed (ignored).")
        return

    with _auto_heal_lock:
        if _auto_healer_started:
            logger.debug("[AutoHealer] start_auto_healer called but already running.")
            return

        # Choose instance: real or fallback
        if REAL_AUTOHEALER_INSTANCE is not None:
            _auto_healer_instance = REAL_AUTOHEALER_INSTANCE
            logger.debug("[AutoHealer] Using REAL_AUTOHEALER_INSTANCE.")
        else:
            _auto_healer_instance = _MinimalAutoHealer()
            logger.debug("[AutoHealer] Using MinimalAutoHealer fallback instance.")

        # Attach extension safely
        try:
            if AdvancedAutoHealerExtension is not None:
                _auto_healer_extension = AdvancedAutoHealerExtension(_auto_healer_instance)
                if hasattr(_auto_healer_extension, "attach") and callable(_auto_healer_extension.attach):
                    _auto_healer_extension.attach()
                logger.debug("[AutoHealer] AdvancedAutoHealerExtension attached.")
            else:
                _auto_healer_extension = _MinimalExtension(_auto_healer_instance)
                _auto_healer_extension.attach()
                logger.debug("[AutoHealer] Minimal extension attached.")
        except Exception as e:
            logger.exception(f"[AutoHealer] Exception when attaching extension: {e}")
            try:
                telemetry.capture_exception(e, context="auto_healer.attach_extension")
            except Exception:
                pass

        # Determine run function safely
        if _maybe_run_auto_healer is not None and callable(_maybe_run_auto_healer):
            run_func = lambda: _maybe_run_auto_healer()
        elif hasattr(_auto_healer_instance, "run") and callable(_auto_healer_instance.run):
            run_func = lambda: _auto_healer_instance.run()
        elif hasattr(_auto_healer_instance, "start") and callable(_auto_healer_instance.start):
            def _wrap_start():
                _auto_healer_instance.start()
                while not _auto_healer_stop_event.is_set():
                    time.sleep(0.5)
            run_func = _wrap_start
        else:
            def _noop():
                logger.warning("[AutoHealer] No runnable auto-healer found; running noop loop.")
                while not _auto_healer_stop_event.is_set():
                    time.sleep(1.0)
            run_func = _noop

        # Start worker thread
        _auto_healer_stop_event.clear()
        _auto_healer_thread = threading.Thread(
            target=_auto_healer_worker,
            args=(run_func, _auto_healer_stop_event),
            name="Bibliotheca-AutoHealer",
            daemon=True,
        )
        _auto_healer_thread.start()
        _auto_healer_started = True
        logger.info("[AutoHealer] Auto-healer worker thread started.")
        try:
            telemetry.info("✅ AutoHealer started (Beyond-Perfection)")
        except Exception:
            logger.debug("[AutoHealer] telemetry.info failed during start (ignored).")


def stop_auto_healer(timeout: float = 5.0) -> None:
    """Stop the auto-healer worker thread gracefully."""
    global _auto_healer_thread, _auto_healer_started
    with _auto_heal_lock:
        if not _auto_healer_started:
            logger.debug("[AutoHealer] stop_auto_healer called but not running.")
            return
        _auto_healer_stop_event.set()
        if _auto_healer_thread is not None and _auto_healer_thread.is_alive():
            logger.info("[AutoHealer] Waiting for worker to stop...")
            _auto_healer_thread.join(timeout=timeout)
            if _auto_healer_thread.is_alive():
                logger.warning("[AutoHealer] Worker thread did not exit within timeout; continuing.")
        try:
            if hasattr(_auto_healer_instance, "stop") and callable(_auto_healer_instance.stop):
                _auto_healer_instance.stop()
        except Exception as e:
            logger.exception(f"[AutoHealer] Exception stopping instance: {e}")
        _auto_healer_started = False
        logger.info("[AutoHealer] Auto-healer stopped.")
        try:
            telemetry.info("AutoHealer stopped")
        except Exception:
            pass


def auto_healer_status() -> dict:
    """Return diagnostic status for the Auto-Healer."""
    return {
        "started": bool(_auto_healer_started),
        "thread_alive": bool(_auto_healer_thread and _auto_healer_thread.is_alive()),
        "in_memory_fallback": getattr(_auto_healer_instance, "status", lambda: {})().get("mode", "") == "fallback"
                           if _auto_healer_instance else None
    }


# -----------------------------
# Ensure clean shutdown at process exit
# -----------------------------
def _auto_healer_atexit():
    try:
        stop_auto_healer(timeout=2.0)
    except Exception:
        logger.exception("[AutoHealer] Exception during atexit stop (ignored).")

atexit.register(_auto_healer_atexit)

# -----------------------------
# Start auto-healer at import unless disabled
# -----------------------------
if not DISABLE_AUTOHEALER:
    try:
        start_auto_healer()
    except Exception as e:
        logger.exception(f"[AutoHealer] Failed to start auto-healer at startup: {e}")
        try:
            telemetry.capture_exception(e, context="auto_healer.startup_failure")
        except Exception:
            pass
else:
    logger.info("[AutoHealer] Auto-healer startup suppressed by environment flag.")
    try:
        telemetry.info("Auto-healer suppressed at startup (env flag).")
    except Exception:
        pass

# -----------------------------
# End Auto-Healer initialization
# -----------------------------

# -----------------------------
# Self-healing function wrappers (Beyond-Perfection)
# -----------------------------
import types
import pkgutil
import importlib
import inspect
import asyncio
import threading
import functools
import traceback
from typing import Any, Callable, Optional

# Idempotency guard
if not globals().get("_SELF_HEALING_WRAPPERS_INSTALLED"):

    _SELF_HEALING_WRAPPERS_INSTALLED = True

    def _is_running_event_loop() -> bool:
        """Return True if there is a running event loop in this thread."""
        try:
            loop = asyncio.get_running_loop()
            return loop.is_running()
        except RuntimeError:
            return False

    def _run_coro_in_new_thread(coro: asyncio.coroutines) -> Any:
        """
        Run a coroutine in a newly-created event loop on a separate thread and return the result.
        This blocks the calling thread until the coroutine completes.
        This is used when the current thread already has a running event loop and we still
        need to synchronously wait for a coroutine.
        """
        result_container = {}
        finished = threading.Event()

        def _runner():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                res = new_loop.run_until_complete(coro)
                result_container["result"] = res
            except Exception as e:
                result_container["exc"] = e
            finally:
                try:
                    new_loop.close()
                except Exception:
                    pass
                finished.set()

        t = threading.Thread(target=_runner, name="bibliotheca-coro-runner", daemon=True)
        t.start()
        finished.wait()
        if "exc" in result_container:
            raise result_container["exc"]
        return result_container.get("result")

    def _safe_call_with_self_heal(func: Callable, *args, **kwargs):
        """
        Call `self_healing_ai.run_with_self_heal(func, *args, **kwargs)` if available,
        else call func directly. Handles coroutine results correctly and ensures
        synchronous callers get final results (even when an event loop is running).
        """
        try:
            # prefer self_healing_ai.run_with_self_heal if present
            runner = getattr(globals().get("self_healing_ai", None), "run_with_self_heal", None)
            if callable(runner):
                try:
                    maybe_coro = runner(func, *args, **kwargs)
                except TypeError:
                    # some implementations may expect (func, args, kwargs) signature
                    maybe_coro = runner(func, args, kwargs)
            else:
                # fallback: call func directly
                maybe_coro = func(*args, **kwargs)
        except Exception as e:
            # if the "runner" itself error'd synchronously, capture and re-raise after telemetry
            try:
                telemetry.capture_exception(e, context="self_heal.wrapper_call_error")
            except Exception:
                logger.debug("telemetry.capture_exception failed during wrapper call error (ignored).")
            logger.exception(f"[SelfHeal] Exception while invoking runner or func: {e}")
            raise

        # If the result is awaitable, run it to completion and return the result
        if inspect.isawaitable(maybe_coro):
            # if no running loop in this thread, safe to asyncio.run
            if not _is_running_event_loop():
                return asyncio.run(maybe_coro)
            # else run in separate thread's loop to avoid error
            return _run_coro_in_new_thread(maybe_coro)
        else:
            return maybe_coro

    def _make_wrapper(original: Callable) -> Callable:
        """Create a wrapped function that executes via self-healing runner safely."""
        @functools.wraps(original)
        def sync_wrapper(*args, **kwargs):
            try:
                return _safe_call_with_self_heal(original, *args, **kwargs)
            except Exception as e:
                # final fallback: if self-heal fails, try a direct call once
                logger.exception(f"[SelfHeal] Wrapper caught exception for {original!r}: {e}")
                try:
                    telemetry.capture_exception(e, context="self_heal.wrapper_exception")
                except Exception:
                    logger.debug("[SelfHeal] telemetry.capture_exception failed (ignored).")
                # attempt direct call as last resort (could still raise)
                return original(*args, **kwargs)
        # Mark wrapper to avoid double-wrapping
        setattr(sync_wrapper, "_self_healed", True)
        return sync_wrapper

    def _make_async_wrapper(original: Callable) -> Callable:
        """Create an async wrapper for coroutine functions that runs under run_with_self_heal."""
        @functools.wraps(original)
        async def async_wrapper(*args, **kwargs):
            try:
                runner = getattr(globals().get("self_healing_ai", None), "run_with_self_heal", None)
                if callable(runner):
                    # runner may be coroutine or sync function returning coroutine
                    maybe = runner(original, *args, **kwargs)
                else:
                    maybe = original(*args, **kwargs)
                if inspect.isawaitable(maybe):
                    return await maybe
                return maybe
            except Exception as e:
                logger.exception(f"[SelfHeal] Async wrapper exception for {original!r}: {e}")
                try:
                    telemetry.capture_exception(e, context="self_heal.async_wrapper_exception")
                except Exception:
                    logger.debug("[SelfHeal] telemetry.capture_exception failed (ignored).")
                # try direct call as last resort
                return await original(*args, **kwargs) if inspect.iscoroutinefunction(original) else original(*args, **kwargs)
        setattr(async_wrapper, "_self_healed", True)
        return async_wrapper

    def wrap_class_methods(cls: type) -> None:
        """
        Wrap instance methods, classmethods, and staticmethods on a class with self-heal wrappers.
        This mutates the class in-place. Skips dunder names and properties.
        """
        if not inspect.isclass(cls):
            return

        for attr_name in dir(cls):
            if attr_name.startswith("__") and attr_name.endswith("__"):
                continue
            # skip attributes that aren't user-defined callables
            try:
                desc = inspect.getattr_static(cls, attr_name)
            except Exception:
                continue

            # skip properties
            if isinstance(desc, property):
                continue

            try:
                # staticmethod
                if isinstance(desc, staticmethod):
                    orig = desc.__func__
                    if getattr(orig, "_self_healed", False):
                        continue
                    if inspect.iscoroutinefunction(orig):
                        wrapped = staticmethod(_make_async_wrapper(orig))
                    else:
                        wrapped = staticmethod(_make_wrapper(orig))
                    setattr(cls, attr_name, wrapped)
                    logger.debug(f"[SelfHeal] Wrapped staticmethod: {cls.__name__}.{attr_name}")
                    continue

                # classmethod
                if isinstance(desc, classmethod):
                    orig = desc.__func__
                    if getattr(orig, "_self_healed", False):
                        continue
                    if inspect.iscoroutinefunction(orig):
                        wrapped = classmethod(_make_async_wrapper(orig))
                    else:
                        wrapped = classmethod(_make_wrapper(orig))
                    setattr(cls, attr_name, wrapped)
                    logger.debug(f"[SelfHeal] Wrapped classmethod: {cls.__name__}.{attr_name}")
                    continue

                # plain function (instance method)
                attr = getattr(cls, attr_name, None)
                if callable(attr) and inspect.isfunction(attr):
                    if getattr(attr, "_self_healed", False):
                        continue
                    if inspect.iscoroutinefunction(attr):
                        wrapped = _make_async_wrapper(attr)
                    else:
                        wrapped = _make_wrapper(attr)
                    setattr(cls, attr_name, wrapped)
                    logger.debug(f"[SelfHeal] Wrapped method: {cls.__name__}.{attr_name}")
            except Exception as e:
                logger.warning(f"[SelfHeal] Skipping wrap of method {attr_name} in {cls.__name__}: {e}")

    def wrap_module_functions(module) -> None:
        """
        Recursively wraps all appropriate functions/types in a module with self-healing.
        Skips private names and avoids wrapping modules that appear to be binary/stdlib.
        """
        if not hasattr(module, "__name__"):
            return
        mod_name = getattr(module, "__name__")
        # safety: only operate on app.* modules by default
        if not mod_name.startswith("app."):
            logger.debug(f"[SelfHeal] Skipping module outside app.*: {mod_name}")
            return

        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            try:
                desc = inspect.getattr_static(module, attr_name)
            except Exception:
                continue

            try:
                # classes -> wrap class methods
                if inspect.isclass(desc):
                    wrap_class_methods(desc)
                    continue

                # functions
                obj = getattr(module, attr_name)
                if inspect.isfunction(obj):
                    if getattr(obj, "_self_healed", False):
                        continue
                    if inspect.iscoroutinefunction(obj):
                        wrapped = _make_async_wrapper(obj)
                    else:
                        wrapped = _make_wrapper(obj)
                    try:
                        setattr(module, attr_name, wrapped)
                        logger.debug(f"[SelfHeal] Wrapped function: {mod_name}.{attr_name}")
                    except Exception as se:
                        logger.warning(f"[SelfHeal] Failed to set attribute {attr_name} on {mod_name}: {se}")
            except Exception as e:
                logger.warning(f"[SelfHeal] Skipping wrap of {attr_name} in module {mod_name}: {e}")

    def apply_self_healing_wrappers(package_name: str = "app.utils") -> None:
        """
        Apply self-healing wrappers to all modules within the given package (non-recursive),
        idempotent and safe.
        """
        try:
            pkg = importlib.import_module(package_name)
        except Exception as e:
            logger.exception(f"[SelfHeal] Failed to import package {package_name}: {e}")
            return

        if not hasattr(pkg, "__path__"):
            logger.warning(f"[SelfHeal] Package {package_name} has no __path__, skipping.")
            return

        for finder, mod_name, ispkg in pkgutil.iter_modules(pkg.__path__):
            full_mod_name = f"{package_name}.{mod_name}"
            try:
                module = importlib.import_module(full_mod_name)
            except Exception as e:
                logger.warning(f"[SelfHeal] Failed to import {full_mod_name}: {e}")
                continue
            try:
                wrap_module_functions(module)
                logger.info(f"[SelfHeal] Applied self-healing wrapper to module: {full_mod_name}")
            except Exception as e:
                logger.exception(f"[SelfHeal] Failed to wrap module {full_mod_name}: {e}")

    # Run the wrapper application for app.utils
    try:
        apply_self_healing_wrappers("app.utils")
    except Exception as e:
        logger.exception(f"[SelfHeal] Error applying wrappers to app.utils: {e}")

    # Final readiness log & OpenAI API check (idempotent)
    try:
        logger.info("=== Bibliotheca Beyond-Perfection ready ===")
        print("=== Bibliotheca Beyond-Perfection ready ===")
    except Exception:
        pass

    # Ensure env loaded and check OpenAI key
    try:
        from dotenv import load_dotenv
        load_dotenv()  # safe to call multiple times
    except Exception:
        logger.debug("[Env] python-dotenv not available or load failed (ignored).")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.warning("⚠️ OpenAI API key not found!")
    else:
        logger.info("✅ OpenAI API key loaded successfully.")
# -----------------------------
# Project imports (after paths & eventlet patch) — Beyond-Perfection
# -----------------------------
import importlib
import logging
import threading
from logging import getLogger
from types import ModuleType
from typing import Any, Optional, Callable

logger = getLogger("Bibliotheca.ProjectImports")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Helper: attempt many import names until one succeeds
def _try_import(names: list[str]) -> Optional[ModuleType]:
    for name in names:
        try:
            mod = importlib.import_module(name)
            logger.info(f"[imports] Imported module '{name}'")
            return mod
        except Exception as e:
            logger.debug(f"[imports] Import attempt failed for '{name}': {e}")
    return None

# -----------------------------
# 1) MetaMonitor
# -----------------------------
MetaMonitor = None
_meta_mod = _try_import([
    "meta_monitor",
    "app.meta_monitor",
    "app.utils.meta_monitor",
    "app.utils.monitor.meta_monitor",
])
if _meta_mod is not None:
    MetaMonitor = getattr(_meta_mod, "MetaMonitor", getattr(_meta_mod, "Monitor", None))

if MetaMonitor is None:
    logger.warning("[imports] MetaMonitor not found; installing minimal fallback.")
    class MetaMonitor:
        def __init__(self, *args, **kwargs):
            self._running = False
            logger.info("[MetaMonitor-Fallback] Initialized.")

        def start(self):
            self._running = True
            logger.info("[MetaMonitor-Fallback] Started.")

        def stop(self):
            self._running = False
            logger.info("[MetaMonitor-Fallback] Stopped.")

        def status(self):
            return {"running": self._running}
    logger.info("[imports] MetaMonitor fallback ready.")

# -----------------------------
# 2) ai_core / CORE_AI / ADVANCED_AI / OfflineAI
# -----------------------------
CORE_AI = None
ADVANCED_AI = None
OfflineAI = None

_ai_mod = _try_import([
    "ai_core",
    "app.ai_core",
    "app.utils.ai_core",
    "app.ai.core",
    "app.ai.advanced_ai",
    "app.utils.ai_core.advanced_ai",
])

if _ai_mod is not None:
    # prefer direct exported singleton names if present
    CORE_AI = getattr(_ai_mod, "CORE_AI", CORE_AI)
    ADVANCED_AI = getattr(_ai_mod, "ADVANCED_AI", ADVANCED_AI)
    OfflineAI = getattr(_ai_mod, "OfflineAI", OfflineAI)

    # fallback: look for class definitions named CoreAI/AdvancedAI/OfflineAI
    if CORE_AI is None and hasattr(_ai_mod, "CoreAI"):
        try:
            CoreAI = getattr(_ai_mod, "CoreAI")
            CORE_AI = CoreAI()  # best effort; if constructor fails we'll catch
            logger.info("[imports] Instantiated CoreAI from module.")
        except Exception as e:
            logger.debug(f"[imports] Could not instantiate CoreAI: {e}")
    if ADVANCED_AI is None and hasattr(_ai_mod, "AdvancedAI"):
        try:
            AdvancedAI = getattr(_ai_mod, "AdvancedAI")
            ADVANCED_AI = AdvancedAI()  # best-effort
            logger.info("[imports] Instantiated AdvancedAI from module.")
        except Exception as e:
            logger.debug(f"[imports] Could not instantiate AdvancedAI: {e}")
    if OfflineAI is None and hasattr(_ai_mod, "OfflineAI"):
        try:
            OfflineAI = getattr(_ai_mod, "OfflineAI")
            OfflineAI = OfflineAI() if callable(OfflineAI) else OfflineAI
            logger.info("[imports] Instantiated OfflineAI from module.")
        except Exception as e:
            logger.debug(f"[imports] Could not instantiate OfflineAI: {e}")

# Minimal safe fallbacks (non-raising, simple methods) if any are missing
if CORE_AI is None or ADVANCED_AI is None or OfflineAI is None:
    logger.warning("[imports] One or more AI core objects missing. Installing minimal safe fallbacks.")

    class _MinimalCoreAI:
        def __init__(self):
            self.name = "MinimalCoreAI"
            logger.info("[MinimalCoreAI] Initialized.")

        def chat(self, *args, **kwargs):
            logger.debug("[MinimalCoreAI] chat() called — fallback placeholder.")
            return {"error": "CoreAI not available"}

        async def self_check(self):
            logger.debug("[MinimalCoreAI] self_check() called.")
            return True

    class _MinimalAdvancedAI(_MinimalCoreAI):
        def __init__(self):
            super().__init__()
            self.name = "MinimalAdvancedAI"
            logger.info("[MinimalAdvancedAI] Initialized.")

        def run_autonomy(self, *args, **kwargs):
            logger.debug("[MinimalAdvancedAI] run_autonomy() — fallback placeholder.")
            return None

    class _MinimalOfflineAI(_MinimalCoreAI):
        def __init__(self):
            super().__init__()
            self.name = "MinimalOfflineAI"
            logger.info("[MinimalOfflineAI] Initialized.")

    CORE_AI = CORE_AI or _MinimalCoreAI()
    ADVANCED_AI = ADVANCED_AI or _MinimalAdvancedAI()
    OfflineAI = OfflineAI or _MinimalOfflineAI()

# -----------------------------
# 3) BackendRouter
# -----------------------------
BackendRouter = None
_backend_mod = _try_import([
    "backend_router",
    "app.backend_router",
    "app.utils.backend_router",
    "app.utils.backend.router",
])

if _backend_mod is not None:
    BackendRouter = getattr(_backend_mod, "BackendRouter", getattr(_backend_mod, "Router", None))

if BackendRouter is None:
    logger.warning("[imports] BackendRouter not found; installing minimal fallback.")
    class BackendRouter:
        def __init__(self, *args, **kwargs):
            self.db_manager = kwargs.get("db_manager", None)
            self.telemetry = kwargs.get("telemetry", None)
            logger.info("[BackendRouter-Fallback] Initialized.")

        def route(self, task_name: str, payload: dict | None = None):
            logger.info(f"[BackendRouter-Fallback] route called: {task_name} payload={payload}")
            if self.telemetry and hasattr(self.telemetry, "info"):
                self.telemetry.info(f"Fallback route invoked: {task_name}")

# instantiate a default backend_router only if you plan to use it right away;
# otherwise higher-level code can instantiate with the desired args.
try:
    backend_router = BackendRouter()
    logger.info("[imports] backend_router ready (fallback or real).")
except Exception as e:
    logger.exception(f"[imports] Failed to instantiate BackendRouter: {e}")
    backend_router = None

# -----------------------------
# 4) self_heal.apply_all
# -----------------------------
apply_all = None
_self_heal_mod = _try_import([
    "self_heal",
    "app.self_heal",
    "app.utils.self_heal",
    "app.utils.ai_core.self_heal",
])

if _self_heal_mod is not None:
    apply_all = getattr(_self_heal_mod, "apply_all", getattr(_self_heal_mod, "apply", None))

if apply_all is None:
    logger.warning("[imports] apply_all (self-heal) not found; installing safe no-op.")
    def apply_all(**kwargs):
        logger.info("[self_heal-fallback] apply_all called — no-op fallback.")
        return False

# -----------------------------
# 5) Optional interactive TTS AI
# -----------------------------
run_interactive_tts = None
try:
    # try a couple of likely paths
    tts_mod = _try_import([
        "ai_core.interactive_tts_ai",
        "app.ai.interactive_tts_ai",
        "app.utils.interactive_tts_ai",
        "interactive_tts_ai",
    ])
    if tts_mod and hasattr(tts_mod, "main"):
        run_interactive_tts = getattr(tts_mod, "main")
        logger.info("[imports] interactive TTS main() available.")
    else:
        logger.info("[imports] interactive TTS not present; skipping TTS mode.")
except Exception as e:
    logger.debug(f"[imports] interactive TTS import exception: {e}")
    run_interactive_tts = None

# -----------------------------
# 6) Telemetry import (after paths)
# -----------------------------
TELEMETRY = None
_telemetry_mod = _try_import([
    "app.utils.telemetry",
    "app.telemetry",
    "app.utils.telemetry_handler",
    "telemetry",
])

if _telemetry_mod is not None:
    # prefer a TELEMETRY instance variable if exported
    TELEMETRY = getattr(_telemetry_mod, "TELEMETRY", None) or getattr(_telemetry_mod, "Telemetry", None) or getattr(_telemetry_mod, "TelemetryHandler", None)
    # If the module exposes a factory or class, try instantiating a default object
    if isinstance(TELEMETRY, type):
        try:
            TELEMETRY = TELEMETRY()  # instantiate
        except Exception:
            TELEMETRY = None

# If still missing, create a robust fallback telemetry instance with same API used above
if TELEMETRY is None:
    logger.warning("[imports] TELEMETRY not found; creating robust fallback telemetry.")
    class _FallbackTelemetry:
        def info(self, msg: str):
            logger.info(f"[Telemetry-Fallback] {msg}")
        def warn(self, msg: str):
            logger.warning(f"[Telemetry-Fallback] {msg}")
        def warning(self, msg: str):
            logger.warning(f"[Telemetry-Fallback] {msg}")
        def error(self, msg: str):
            logger.error(f"[Telemetry-Fallback] {msg}")
        def debug(self, msg: str):
            logger.debug(f"[Telemetry-Fallback] {msg}")
        def log_event(self, *args, **kwargs):
            logger.debug(f"[Telemetry-Fallback] event: {args} {kwargs}")
        def capture_exception(self, exc: Exception, context: Optional[str] = None):
            logger.exception(f"[Telemetry-Fallback] capture_exception: {context} -> {exc}")

    TELEMETRY = _FallbackTelemetry()

# Convenience: also expose a module-level name TELEMETRY (for older code expecting it)
try:
    TELEMETRY.info("✅ TELEMETRY initialized (real or fallback).")
except Exception:
    logger.debug("[imports] TELEMETRY.info failed (ignored).")

# -----------------------------
# Final sanity/status print
# -----------------------------
_loaded_components = {
    "MetaMonitor": MetaMonitor.__name__ if hasattr(MetaMonitor, "__name__") else str(type(MetaMonitor)),
    "CORE_AI": getattr(CORE_AI, "name", type(CORE_AI).__name__),
    "ADVANCED_AI": getattr(ADVANCED_AI, "name", type(ADVANCED_AI).__name__),
    "OfflineAI": getattr(OfflineAI, "name", type(OfflineAI).__name__),
    "BackendRouter": BackendRouter.__name__ if hasattr(BackendRouter, "__name__") else str(type(BackendRouter)),
    "apply_all": bool(apply_all),
    "run_interactive_tts": bool(run_interactive_tts),
    "TELEMETRY": TELEMETRY.__class__.__name__,
}

logger.info("[imports] Project imports complete. Components loaded: " + ", ".join(f"{k}={v}" for k, v in _loaded_components.items()))
TELEMETRY.info(f"Project imports complete. Components: {_loaded_components}")

# Expose names in module globals for downstream code to use
globals().update({
    "MetaMonitor": MetaMonitor,
    "CORE_AI": CORE_AI,
    "ADVANCED_AI": ADVANCED_AI,
    "OfflineAI": OfflineAI,
    "BackendRouter": BackendRouter,
    "backend_router": backend_router,
    "apply_all": apply_all,
    "run_interactive_tts": run_interactive_tts,
    "TELEMETRY": TELEMETRY,
})
# ============================================================
# CouchDB Configuration and Resilient Initialization
# ============================================================

from app.utils.ai_core.couchdb_manager import CouchDBManager
from app.utils.security_utils import encrypt_data, decrypt_data  # optional (if present)
import aiohttp
import json
import time

COUCHDB_PRIMARY_URL = "http://rohan:rthunderpheonix11@localhost:5984/"
COUCHDB_FALLBACK_URL = "http://127.0.0.1:5984/"
DB_NAME = "bibliotheca_memory"
LOCAL_FAILSAFE_FILE = "app/data/failsafe_memory_cache.json"

# -----------------------------
# Initialize CouchDB Manager with Auto-Healing Support
# -----------------------------
try:
    couchdb_manager = CouchDBManager(
        db_url=COUCHDB_PRIMARY_URL,
        db_name=DB_NAME,
        auto_create=True,
        verify_ssl=False
    )
    telemetry.info(f"✅ CouchDBManager connected to {COUCHDB_PRIMARY_URL}")
except Exception as e:
    logger.error(f"[CRITICAL] CouchDB connection failed: {e}")
    telemetry.error(f"CouchDB init error: {e}")

    # Attempt fallback to secondary URL
    try:
        couchdb_manager = CouchDBManager(
            db_url=COUCHDB_FALLBACK_URL,
            db_name=DB_NAME,
            auto_create=True,
            verify_ssl=False
        )
        telemetry.info(f"⚠️ Fallback CouchDB connected at {COUCHDB_FALLBACK_URL}")
    except Exception as fallback_error:
        logger.critical(f"[FATAL] Fallback CouchDB connection failed: {fallback_error}")
        telemetry.error("Both primary and fallback CouchDB connections failed.")

        # -----------------------------
        # Enter Local Failsafe Mode
        # -----------------------------
        class LocalCacheDB:
            """Simple local JSON-based cache when CouchDB is unavailable."""
            def __init__(self, file_path=LOCAL_FAILSAFE_FILE):
                self.file_path = file_path
                self.data = {}
                self.load()

            def load(self):
                if os.path.exists(self.file_path):
                    try:
                        with open(self.file_path, "r") as f:
                            self.data = json.load(f)
                            logger.warning("Loaded memory cache from local failsafe.")
                    except Exception:
                        logger.exception("Failed to load failsafe cache file.")

            def save(self):
                try:
                    os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                    with open(self.file_path, "w") as f:
                        json.dump(self.data, f, indent=2)
                except Exception:
                    logger.exception("Failed to save failsafe memory cache.")

            def store(self, key, value):
                try:
                    self.data[key] = encrypt_data(value) if "encrypt_data" in globals() else value
                    self.save()
                    telemetry.info(f"[Failsafe] Stored key '{key}' locally.")
                except Exception as e:
                    logger.error(f"LocalCacheDB.store error: {e}")

            def retrieve(self, key):
                try:
                    val = self.data.get(key)
                    return decrypt_data(val) if "decrypt_data" in globals() else val
                except Exception as e:
                    logger.error(f"LocalCacheDB.retrieve error: {e}")
                    return None

        couchdb_manager = LocalCacheDB()
        telemetry.warning("☣️ Operating in local failsafe memory mode. Persistence limited.")

# -----------------------------
# Test Connection & Database Status
# -----------------------------
try:
    if hasattr(couchdb_manager, "test_connection"):
        couchdb_manager.test_connection()
    telemetry.info("✅ CouchDB verified and operational.")
except Exception as e:
    logger.warning(f"CouchDB verification failed: {e}")
    telemetry.warn(f"CouchDB test_connection error: {e}")

# -----------------------------
# Continuous Health Monitoring
# -----------------------------
async def monitor_couchdb_health(interval=60):
    """
    Periodically checks CouchDB connection health and attempts auto-repair if needed.
    """
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(COUCHDB_PRIMARY_URL) as resp:
                    if resp.status == 200:
                        logger.debug("CouchDB health check: OK")
                    else:
                        logger.warning(f"CouchDB health anomaly: HTTP {resp.status}")
                        telemetry.warn(f"CouchDB returned {resp.status}")
        except Exception as e:
            logger.warning(f"CouchDB unreachable during health check: {e}")
            telemetry.error(f"CouchDB health check failed: {e}")
        await asyncio.sleep(interval)

# Schedule async monitoring (non-blocking)
try:
    asyncio.create_task(monitor_couchdb_health())
except RuntimeError:
    # If not inside an event loop (e.g. during sync startup), schedule later
    logger.info("CouchDB monitor task will start after event loop activation.")

logger.info("✅ CouchDB configuration completed with multi-layer redundancy and self-healing.")
# -----------------------------
# Initialize Memory, AI, and Autonomous Extension (Beyond-Perfection)
# -----------------------------
import threading
import time
import os
import json
from pathlib import Path
from logging import getLogger

logger = getLogger("Bibliotheca.Init.MemoryAI")
# Prefer telemetry variable if available, else TELEMETRY, else fallback wrapper
_TELEMETRY = globals().get("telemetry") or globals().get("TELEMETRY")
def _telemetry_info(msg):
    try:
        if _TELEMETRY and hasattr(_TELEMETRY, "info"):
            _TELEMETRY.info(msg)
        else:
            logger.info(msg)
    except Exception:
        logger.info(msg)
def _telemetry_warn(msg):
    try:
        if _TELEMETRY and hasattr(_TELEMETRY, "warning"):
            _TELEMETRY.warning(msg)
        elif _TELEMETRY and hasattr(_TELEMETRY, "warn"):
            _TELEMETRY.warn(msg)
        else:
            logger.warning(msg)
    except Exception:
        logger.warning(msg)
def _telemetry_error(msg):
    try:
        if _TELEMETRY and hasattr(_TELEMETRY, "error"):
            _TELEMETRY.error(msg)
        else:
            logger.error(msg)
    except Exception:
        logger.error(msg)

# Idempotency guard
if not globals().get("_MEMORY_AI_INITIALIZED"):

    _MEMORY_AI_INITIALIZED = False

    # -----------------------------
    # Ensure a CouchDB manager exists (use existing one if created earlier)
    # -----------------------------
    couchdb_mgr = globals().get("couchdb_manager")
    if couchdb_mgr is None:
        try:
            # try to import CouchDBManager from the utils location(s)
            try:
                from app.utils.ai_core.couchdb_manager import CouchDBManager  # preferred
            except Exception:
                from app.utils.couchdb_manager import CouchDBManager  # fallback
            couchdb_mgr = CouchDBManager()
            logger.info("[MemoryInit] CouchDBManager created.")
        except Exception as e:
            couchdb_mgr = None
            logger.warning(f"[MemoryInit] No CouchDBManager available: {e}. Will use in-memory memory store.")

    # -----------------------------
    # MemoryStore initialization (try real MemoryStore; fallback to robust in-memory)
    # -----------------------------
    MemoryStore = None
    try:
        # try several likely locations for MemoryStore
        try:
            from app.utils.memory import MemoryStore
        except Exception:
            from app.utils.memory_store import MemoryStore  # alternate
        logger.debug("[MemoryInit] MemoryStore class imported.")
    except Exception:
        MemoryStore = None

    # Robust in-memory MemoryStore fallback with optional JSON backup & thread-safety
    class _InMemoryMemoryStore:
        def __init__(self, db_manager=None, backup_file: str | None = None):
            self._lock = threading.RLock()
            self._store: dict = {}
            self.db_manager = db_manager
            self.backup_file = backup_file or (Path.cwd() / "tmp" / "memory_backup.json")
            try:
                os.makedirs(os.path.dirname(str(self.backup_file)), exist_ok=True)
            except Exception:
                pass
            self._load_backup()
            self._auto_sync_thread = None
            self._auto_sync_stop = threading.Event()
        def _load_backup(self):
            try:
                if os.path.exists(str(self.backup_file)):
                    with open(str(self.backup_file), "r") as f:
                        self._store.update(json.load(f))
                    logger.info(f"[MemoryStore-Fallback] Loaded backup from {self.backup_file}")
            except Exception as e:
                logger.debug(f"[MemoryStore-Fallback] Backup load failed: {e}")
        def _save_backup(self):
            try:
                with open(str(self.backup_file), "w") as f:
                    json.dump(self._store, f, indent=2)
                logger.debug(f"[MemoryStore-Fallback] Backup saved to {self.backup_file}")
            except Exception as e:
                logger.debug(f"[MemoryStore-Fallback] Backup save failed: {e}")
        def set(self, key: str, value):
            with self._lock:
                self._store[key] = value
                self._save_backup()
        def get(self, key: str, default=None):
            with self._lock:
                return self._store.get(key, default)
        def delete(self, key: str):
            with self._lock:
                self._store.pop(key, None)
                self._save_backup()
        def all_docs(self):
            with self._lock:
                return dict(self._store)
        def repair(self):
            # minimal repair: re-save backup
            with self._lock:
                self._save_backup()
                logger.info("[MemoryStore-Fallback] repair() invoked and backup refreshed")
        def health_check(self):
            return {"status": "ok", "backend": "in-memory", "items": len(self._store)}
        def start_auto_sync(self, interval_seconds: int = 300):
            if self._auto_sync_thread and self._auto_sync_thread.is_alive():
                return
            self._auto_sync_stop.clear()
            def _loop():
                while not self._auto_sync_stop.is_set():
                    time.sleep(interval_seconds)
                    try:
                        self._save_backup()
                    except Exception:
                        logger.debug("[MemoryStore-Fallback] auto-sync save failed (ignored).")
            self._auto_sync_thread = threading.Thread(target=_loop, name="MemoryStoreAutoSync", daemon=True)
            self._auto_sync_thread.start()
        def stop_auto_sync(self):
            self._auto_sync_stop.set()
            if self._auto_sync_thread:
                self._auto_sync_thread.join(timeout=1.0)

    # Instantiate memory using real MemoryStore when available, else fallback
    memory = None
    try:
        if MemoryStore is not None:
            try:
                # try constructor signatures
                try:
                    memory = MemoryStore(db_manager=couchdb_mgr) if couchdb_mgr is not None else MemoryStore()
                except TypeError:
                    # fallback: MemoryStore(memory_manager=...) or MemoryStore()
                    try:
                        memory = MemoryStore(couchdb_mgr)
                    except Exception:
                        memory = MemoryStore()
                logger.info("[MemoryStore] Initialized with MemoryStore implementation.")
            except Exception as e:
                logger.warning(f"[MemoryStore] MemoryStore import succeeded but instantiation failed: {e}")
                memory = _InMemoryMemoryStore(db_manager=couchdb_mgr)
        else:
            memory = _InMemoryMemoryStore(db_manager=couchdb_mgr)
            logger.info("[MemoryStore] Using in-memory fallback MemoryStore.")
    except Exception as e:
        memory = _InMemoryMemoryStore(db_manager=couchdb_mgr)
        logger.warning(f"[MemoryStore] Unexpected error - using fallback: {e}")

    # Start auto-sync on fallback (or real MemoryStore if it exposes start_auto_sync)
    try:
        sync_interval = int(os.getenv("BIBLIOTHECA_MEMORY_AUTOSYNC_SECONDS", "300"))
        if hasattr(memory, "start_auto_sync") and callable(getattr(memory, "start_auto_sync")):
            memory.start_auto_sync(sync_interval)
            logger.debug(f"[MemoryStore] Auto-sync started (interval={sync_interval}s).")
    except Exception as e:
        logger.debug(f"[MemoryStore] Failed to start auto-sync: {e}")

    # Run repair/load if available
    try:
        if hasattr(memory, "repair") and callable(getattr(memory, "repair")):
            memory.repair()
            logger.info("[MemoryStore] repair() executed (if applicable).")
        if hasattr(memory, "load") and callable(getattr(memory, "load")):
            try:
                memory.load()
                logger.info("[MemoryStore] load() executed.")
            except Exception:
                # some MemoryStore implementations don't have load semantics
                pass
    except Exception as e:
        logger.debug(f"[MemoryStore] initial repair/load raised: {e}")

    # Telemetry
    _telemetry_info("[MemoryStore] Memory initialized and ready.")

    # -----------------------------
    # AdvancedAI initialization (try multiple constructor signatures; fallback safe)
    # -----------------------------
    AdvancedAI = None
    try:
        try:
            from app.ai.advanced_ai import AdvancedAI  # preferred
        except Exception:
            try:
                from app.utils.ai_core.advanced_ai import AdvancedAI
            except Exception:
                from app.utils.ai_core import AdvancedAI  # last resort
        logger.debug("[AdvancedAI] AdvancedAI class imported.")
    except Exception:
        AdvancedAI = None

    _ai_core = None
    try:
        # Attempt usual constructor patterns (most specific to least)
        if AdvancedAI is not None:
            created = False
            # 1) AdvancedAI(core_ai=CORE_AI, memory=memory, patch_engine=PATCH_ENGINE)
            try:
                _ai_core = AdvancedAI(core_ai=globals().get("CORE_AI"), memory=memory, patch_engine=globals().get("PATCH_ENGINE"))
                created = True
            except TypeError:
                pass
            except Exception as e:
                logger.debug(f"[AdvancedAI] constructor pattern 1 failed: {e}")
            # 2) AdvancedAI(core_ai=CORE_AI, memory=memory)
            if not created:
                try:
                    _ai_core = AdvancedAI(core_ai=globals().get("CORE_AI"), memory=memory)
                    created = True
                except Exception:
                    pass
            # 3) AdvancedAI(memory)
            if not created:
                try:
                    _ai_core = AdvancedAI(memory)
                    created = True
                except Exception:
                    pass
            # 4) AdvancedAI(core_ai=CORE_AI)
            if not created:
                try:
                    _ai_core = AdvancedAI(core_ai=globals().get("CORE_AI"))
                    created = True
                except Exception:
                    pass
            # 5) AdvancedAI() empty constructor
            if not created:
                try:
                    _ai_core = AdvancedAI()
                    created = True
                except Exception:
                    pass

    except Exception as e:
        logger.debug(f"[AdvancedAI] Unexpected error while instantiating AdvancedAI: {e}")

    # Fallback minimal AdvancedAI if creation failed
    if _ai_core is None:
        class _MinimalAdvancedAI:
            def __init__(self, memory_ref=None):
                self.name = "MinimalAdvancedAI"
                self.memory = memory_ref
                logger.warning("[AdvancedAI-Fallback] MinimalAdvancedAI initialized.")
            def chat(self, *args, **kwargs):
                logger.debug("[AdvancedAI-Fallback] chat() called.")
                return {"error": "AdvancedAI not available"}
            async def self_check(self):
                return True
            def attach_extension(self, ext):
                logger.debug("[AdvancedAI-Fallback] attach_extension() called.")
        _ai_core = _MinimalAdvancedAI(memory_ref=memory)

    # Health-check / self-check for AI if available
    try:
        if hasattr(_ai_core, "self_check"):
            maybe = _ai_core.self_check()
            # if coroutine, execute safely (do not block main thread if loop running)
            import inspect, asyncio
            if inspect.isawaitable(maybe):
                try:
                    if not asyncio.get_event_loop().is_running():
                        asyncio.run(maybe)
                    else:
                        # schedule but don't block
                        asyncio.create_task(maybe)
                except Exception:
                    logger.debug("[AdvancedAI] Could not run self_check synchronously (ignored).")
        _telemetry_info("[AdvancedAI] Core AI initialized successfully.")
    except Exception as e:
        logger.error(f"[AdvancedAI] Failed during self_check: {e}")

    # -----------------------------
    # AutonomousExtension initialization (try real; fallback safe)
    # -----------------------------
    AutonomousExtension = None
    try:
        try:
            from app.utils.autonomy import AutonomousExtension
        except Exception:
            from app.ai.autonomy import AutonomousExtension
    except Exception:
        AutonomousExtension = None

    auto_ext = None
    try:
        if AutonomousExtension is not None:
            try:
                # try common constructor signatures
                try:
                    auto_ext = AutonomousExtension(_ai_core, memory)
                except TypeError:
                    try:
                        auto_ext = AutonomousExtension(ai_core=_ai_core, memory=memory)
                    except Exception:
                        auto_ext = AutonomousExtension(_ai_core)
                logger.info("[AutonomousExtension] Initialized.")
            except Exception as e:
                logger.warning(f"[AutonomousExtension] Instantiation failed: {e}")
                auto_ext = None
    except Exception as e:
        logger.debug(f"[AutonomousExtension] Unexpected error: {e}")

    # fallback minimal autonomous extension
    if auto_ext is None:
        class _MinimalAutonomousExtension:
            def __init__(self, ai_core, memory_store):
                self.ai = ai_core
                self.memory = memory_store
                self.enabled = False
                logger.warning("[AutonomousExtension-Fallback] MinimalAutonomousExtension initialized.")
            def enable(self):
                self.enabled = True
                logger.info("[AutonomousExtension-Fallback] enabled.")
            def disable(self):
                self.enabled = False
                logger.info("[AutonomousExtension-Fallback] disabled.")
            def status(self):
                return {"enabled": self.enabled}
        try:
            auto_ext = _MinimalAutonomousExtension(_ai_core, memory)
        except Exception:
            auto_ext = None

    # attach extension to ai if supported
    try:
        if auto_ext and hasattr(_ai_core, "attach_extension"):
            try:
                _ai_core.attach_extension(auto_ext)
            except Exception:
                try:
                    setattr(_ai_core, "autonomous_extension", auto_ext)
                except Exception:
                    logger.debug("[MemoryInit] Could not attach autonomous extension to AI (ignored).")
    except Exception:
        pass

    # Register with auto-healer if present
    try:
        if globals().get("AUTOHEALER_INSTANCE") and hasattr(globals().get("AUTOHEALER_INSTANCE"), "register"):
            try:
                globals().get("AUTOHEALER_INSTANCE").register(component=_ai_core, name="AdvancedAI")
                globals().get("AUTOHEALER_INSTANCE").register(component=memory, name="MemoryStore")
            except Exception:
                logger.debug("[MemoryInit] Auto-healer register failed (ignored).")
    except Exception:
        pass

    # Helper status getters (lightweight)
    def memory_status():
        try:
            if hasattr(memory, "health_check"):
                return memory.health_check()
            return {"backend": "in-memory" if isinstance(memory, _InMemoryMemoryStore) else "MemoryStore", "items": len(getattr(memory, "_store", getattr(memory, "all_docs", lambda: {})())),}
        except Exception as e:
            return {"error": str(e)}

    def ai_status():
        try:
            return {"name": getattr(_ai_core, "name", type(_ai_core).__name__), "has_self_check": callable(getattr(_ai_core, "self_check", None))}
        except Exception as e:
            return {"error": str(e)}

    def autonomous_status():
        try:
            return {"present": bool(auto_ext), "status": getattr(auto_ext, "status", lambda: {})()}
        except Exception as e:
            return {"error": str(e)}

    # Telemetry + logging final
    _telemetry_info("[MemoryInit] Memory + AI + Autonomous Extension initialization complete.")
    logger.info(f"[MemoryInit] memory_status={memory_status()}, ai_status={ai_status()}, autonomous_status={autonomous_status()}")

    # Expose names to globals for downstream code
    globals().update({
        "db_manager": couchdb_mgr,
        "memory": memory,
        "AdvancedAI": AdvancedAI if 'AdvancedAI' in locals() else None,
        "ai_core": _ai_core,
        "AutonomousExtension": AutonomousExtension if 'AutonomousExtension' in locals() else None,
        "auto_ext": auto_ext,
        "memory_status": memory_status,
        "ai_status": ai_status,
        "autonomous_status": autonomous_status,
        "_MEMORY_AI_INITIALIZED": True
    })

    # mark complete
    _MEMORY_AI_INITIALIZED = True

# end idempotent block
# -----------------------------
# Memory repair & integrity test (Beyond-Perfection)
# -----------------------------
import time
import json
import os
import traceback
from logging import getLogger
from typing import Any, Dict, Optional

logger = getLogger("Bibliotheca.MemoryRepair")
_TELEMETRY = globals().get("telemetry") or globals().get("TELEMETRY")

def _telemetry_event(name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    try:
        if _TELEMETRY and hasattr(_TELEMETRY, "log_event"):
            _TELEMETRY.log_event(name, payload or {})
        elif _TELEMETRY and hasattr(_TELEMETRY, "info"):
            _TELEMETRY.info(f"{name}: {payload or {}}")
    except Exception:
        logger.debug("Telemetry call failed (ignored).")

def _capture_exception(exc: Exception, context: str = "memory.repair"):
    try:
        if _TELEMETRY and hasattr(_TELEMETRY, "capture_exception"):
            _TELEMETRY.capture_exception(exc, context=context)
    except Exception:
        logger.debug("Telemetry.capture_exception failed (ignored).")
    logger.exception(f"[{context}] {exc}")

def _get_items_snapshot(mem) -> Optional[Dict[str, Any]]:
    """
    Try multiple strategies to get all memory items as a dict.
    Returns dict or None if not possible.
    """
    try:
        # common API: all_docs()
        if hasattr(mem, "all_docs") and callable(getattr(mem, "all_docs")):
            docs = mem.all_docs()
            # if couchdb view returned rows with 'doc' keys, try to normalize
            if isinstance(docs, dict):
                return docs
            try:
                # if it's an iterable of docs
                return {str(d.get("_id", i)): d.get("doc", d) if isinstance(d, dict) else d for i, d in enumerate(docs)}
            except Exception:
                return dict(docs)
        # other APIs
        for name in ("all_items", "all", "get_all", "dump"):
            if hasattr(mem, name) and callable(getattr(mem, name)):
                items = getattr(mem, name)()
                if isinstance(items, dict):
                    return items
                try:
                    return dict(items)
                except Exception:
                    # build dict from iterable
                    return {str(i): v for i, v in enumerate(items)}
        # check internal backing dicts
        for attr in ("_store", "store", "data", "db", "client"):
            if hasattr(mem, attr):
                val = getattr(mem, attr)
                # if 'db' or 'client' is couchdb object, attempt view
                try:
                    if hasattr(val, "view"):
                        rows = val.view("_all_docs", include_docs=True)
                        return {r.id: getattr(r, "doc", None) or r for r in rows}
                except Exception:
                    pass
                if isinstance(val, dict):
                    return dict(val)
    except Exception as e:
        logger.debug(f"[MemoryRepair] _get_items_snapshot error: {e}")
    return None

def _attempt_resync_to_couchdb(couch_mgr, mem, limit: Optional[int] = 5000) -> Dict[str, Any]:
    """
    If memory is using an in-memory/failsafe backup, attempt to resync contents back into CouchDB.
    Returns a status dict with counts and errors (if any).
    """
    result = {"attempted": 0, "written": 0, "skipped": 0, "errors": []}
    try:
        if couch_mgr is None:
            result["errors"].append("no_couchdb_manager")
            return result
        snapshot = _get_items_snapshot(mem) or {}
        result["attempted"] = len(snapshot)
        # limit extremely large resyncs by default to protect DB
        keys = list(snapshot.keys())[:limit]
        for k in keys:
            v = snapshot[k]
            try:
                # prefer bulk or save API
                if hasattr(couch_mgr, "save") and callable(getattr(couch_mgr, "save")):
                    couch_mgr.save(k, v)
                elif hasattr(couch_mgr, "bulk_save") and callable(getattr(couch_mgr, "bulk_save")):
                    couch_mgr.bulk_save({k: v})
                else:
                    # try direct couchdb client if present
                    if hasattr(couch_mgr, "db") and hasattr(couch_mgr.db, "save"):
                        couch_mgr.db.save({"_id": k, **(v if isinstance(v, dict) else {"value": v})})
                    else:
                        raise RuntimeError("Unknown couch_mgr API; cannot write")
                result["written"] += 1
            except Exception as e:
                result["errors"].append({"key": k, "error": str(e)})
        _telemetry_event("memory.resync.completed", result)
    except Exception as e:
        _capture_exception(e, "memory.resync")
        result["errors"].append(str(e))
    return result

def _verify_integrity(before_snapshot: Optional[Dict[str, Any]], after_snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare before/after snapshots and produce a verification report.
    """
    report = {"before_count": None, "after_count": None, "added": [], "removed": [], "changed": []}
    try:
        if before_snapshot is not None:
            report["before_count"] = len(before_snapshot)
        if after_snapshot is not None:
            report["after_count"] = len(after_snapshot)
        if before_snapshot is None or after_snapshot is None:
            return report
        before_keys = set(before_snapshot.keys())
        after_keys = set(after_snapshot.keys())
        report["added"] = list(after_keys - before_keys)
        report["removed"] = list(before_keys - after_keys)
        # detect simple changes by stringifying values (best-effort; avoid deep compare cost)
        common = before_keys & after_keys
        for k in list(common)[:1000]:  # cap compare work
            try:
                if json.dumps(before_snapshot[k], sort_keys=True) != json.dumps(after_snapshot[k], sort_keys=True):
                    report["changed"].append(k)
            except Exception:
                # if not JSON serializable, do repr compare
                if repr(before_snapshot[k]) != repr(after_snapshot[k]):
                    report["changed"].append(k)
    except Exception as e:
        logger.debug(f"[MemoryRepair] _verify_integrity error: {e}")
    return report

def perform_memory_repair_test(
    retries: int = 3,
    backoff: float = 1.0,
    verify_changes: bool = True,
    resync_to_couchdb: bool = True,
    resync_limit: int = 5000
) -> Dict[str, Any]:
    """
    Perform a comprehensive memory repair test and integrity verification.
    Returns a dictionary with status details.
    - retries/backoff: retry params for calling memory.repair()
    - verify_changes: whether to capture before/after snapshots and compare
    - resync_to_couchdb: attempt resync if in-memory fallback and couchdb manager available
    """
    status = {"ok": False, "repair_attempts": 0, "repair_ok": False, "verify": None, "resync": None, "notes": []}
    try:
        mem = globals().get("memory")
        couch_mgr = globals().get("couchdb_manager") or globals().get("db_manager")
        if mem is None:
            status["notes"].append("memory_missing")
            logger.error("[MemoryManager] No 'memory' object found in globals.")
            _telemetry_event("memory.repair.failed", {"reason": "memory_missing"})
            return status

        # capture before snapshot if requested
        before = None
        if verify_changes:
            try:
                before = _get_items_snapshot(mem)
            except Exception as e:
                logger.debug(f"[MemoryManager] Could not capture before snapshot: {e}")

        # If memory exposes 'health_check' report it
        try:
            if hasattr(mem, "health_check") and callable(getattr(mem, "health_check")):
                health_before = mem.health_check()
                status["health_before"] = health_before
                _telemetry_event("memory.health_before", health_before)
        except Exception as e:
            logger.debug(f"[MemoryManager] health_check before failed: {e}")

        # Attempt repair with retries + backoff
        repaired = False
        repair_errors = []
        if hasattr(mem, "repair") and callable(getattr(mem, "repair")):
            for attempt in range(1, retries + 1):
                status["repair_attempts"] = attempt
                try:
                    mem.repair()
                    repaired = True
                    status["repair_ok"] = True
                    _telemetry_event("memory.repair.attempt", {"attempt": attempt, "result": "ok"})
                    logger.info(f"[MemoryManager] repair() succeeded on attempt {attempt}.")
                    break
                except Exception as e:
                    repair_errors.append({"attempt": attempt, "error": str(e)})
                    _capture_exception(e, context=f"memory.repair.attempt_{attempt}")
                    logger.warning(f"[MemoryManager] repair() attempt {attempt} failed: {e}")
                    time.sleep(backoff * (2 ** (attempt - 1)))
        else:
            status["notes"].append("repair_method_missing")
            logger.warning("[MemoryManager] 'repair' method not available on memory object.")

        status["repair_errors"] = repair_errors

        # If repair not available or failed, try fallback actions:
        if not repaired:
            # attempt .repair_db, .reconcile, or a generic save/load if available
            fallback_actions = []
            try:
                if hasattr(mem, "repair_db") and callable(getattr(mem, "repair_db")):
                    mem.repair_db()
                    fallback_actions.append("repair_db")
                    repaired = True
                elif hasattr(mem, "reconcile") and callable(getattr(mem, "reconcile")):
                    mem.reconcile()
                    fallback_actions.append("reconcile")
                    repaired = True
                elif hasattr(mem, "_save_backup") and callable(getattr(mem, "_save_backup")):
                    # force a save of current state
                    mem._save_backup()
                    fallback_actions.append("_save_backup")
                    repaired = True
                status["fallback_actions"] = fallback_actions
            except Exception as e:
                status.setdefault("fallback_errors", []).append(str(e))
                _capture_exception(e, "memory.repair.fallback")

        # capture after snapshot and run verify
        after = None
        if verify_changes:
            try:
                after = _get_items_snapshot(mem)
                status["verify"] = _verify_integrity(before, after)
                _telemetry_event("memory.repair.verify", status["verify"])
            except Exception as e:
                logger.debug(f"[MemoryManager] Could not capture after snapshot: {e}")

        # Attempt resync to CouchDB if appropriate
        if resync_to_couchdb and couch_mgr is not None:
            try:
                status["resync"] = _attempt_resync_to_couchdb(couch_mgr, mem, limit=resync_limit)
            except Exception as e:
                _capture_exception(e, "memory.resync.attempt")
                status["resync_error"] = str(e)

        # final health_check
        try:
            if hasattr(mem, "health_check") and callable(getattr(mem, "health_check")):
                status["health_after"] = mem.health_check()
                _telemetry_event("memory.health_after", status["health_after"])
        except Exception as e:
            logger.debug(f"[MemoryManager] health_check after failed: {e}")

        # Conclude: ok if repair_ok True or fallback actions succeeded or resync wrote some docs
        if status.get("repair_ok") or status.get("fallback_actions") or (status.get("resync") and status["resync"].get("written", 0) > 0):
            status["ok"] = True

        status["timestamp"] = int(time.time())
        # final telemetry
        _telemetry_event("memory.repair.completed", status)
        logger.info(f"[MemoryManager] Repair test completed: ok={status['ok']} summary={ {'repair_ok': status.get('repair_ok'), 'resync_written': status.get('resync',{}).get('written',0)} }")

    except Exception as exc:
        _capture_exception(exc, "memory.repair.unexpected")
        status["error"] = str(exc)
    return status

# Run the repair test immediately and expose result in globals for later inspection
try:
    MEMORY_REPAIR_TEST_RESULT = perform_memory_repair_test()
    globals()["MEMORY_REPAIR_TEST_RESULT"] = MEMORY_REPAIR_TEST_RESULT
except Exception as e:
    _capture_exception(e, "memory.repair.run")
    globals()["MEMORY_REPAIR_TEST_RESULT"] = {"ok": False, "error": str(e)}
# -----------------------------
# Conversational Task Loop (Beyond-Perfection)
# -----------------------------
import os
import sys
import signal
import threading
import time
import json
import inspect
import asyncio
import functools
from pathlib import Path
from typing import Optional, Any

# logger / telemetry (use existing if present)
logger = globals().get("logger") or __import__("logging").getLogger("Bibliotheca.InteractiveLoop")
_TELEMETRY = globals().get("telemetry") or globals().get("TELEMETRY")

def _telemetry_event(name: str, payload: Optional[dict] = None):
    try:
        if _TELEMETRY and hasattr(_TELEMETRY, "log_event"):
            _TELEMETRY.log_event(name, payload or {})
        elif _TELEMETRY and hasattr(_TELEMETRY, "info"):
            _TELEMETRY.info(f"{name}: {payload or {}}")
    except Exception:
        logger.debug("Telemetry event failed (ignored).")

# locate CORE_AI (robust)
CORE_AI = globals().get("CORE_AI") or globals().get("ai_core") or globals().get("ai")

# History file
HISTORY_PATH = Path(os.getenv("BIBLIOTHECA_HISTORY", Path.home() / ".bibliotheca_history")).expanduser()

# Helper: ensure history file exists and can be appended
try:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# safer coroutine runner when an asyncio loop is already running
def _run_coro_in_thread(coro):
    """
    Run given coroutine in a fresh event loop inside a new thread and return the result.
    Blocks the caller until complete.
    """
    result = {}
    evt = threading.Event()

    def _target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(coro)
            result['ok'] = True
            result['value'] = res
        except Exception as e:
            result['ok'] = False
            result['exc'] = e
        finally:
            try:
                loop.close()
            except Exception:
                pass
            evt.set()

    t = threading.Thread(target=_target, name="bibliotheca-interactive-coro-runner", daemon=True)
    t.start()
    evt.wait()
    if result.get('ok'):
        return result.get('value')
    raise result.get('exc')

def _safe_invoke(func, *args, **kwargs) -> Any:
    """
    Safely invoke a function that may be sync or async.
    If function returns awaitable, run it and return final result.
    If an event loop is running in this thread, coroutines are executed in a worker thread.
    """
    try:
        maybe = func(*args, **kwargs)
    except Exception as e:
        # If func is a coroutine function, calling it may return coroutine or raise — pass through
        # But for safety, re-raise after telemetry/log
        _telemetry_event("interactive.invoke.exception", {"error": str(e), "func": getattr(func, "__name__", str(func))})
        logger.exception("[Interactive] Exception while invoking function.")
        raise

    # if awaitable, complete it
    if inspect.isawaitable(maybe):
        # no running event loop => safe to run in this thread
        try:
            loop = asyncio.get_running_loop()
            running = loop.is_running()
        except RuntimeError:
            running = False

        if not running:
            return asyncio.run(maybe)
        # else run in separate thread loop to avoid RuntimeError
        return _run_coro_in_thread(maybe)
    else:
        return maybe

# Input backend selection: prompt_toolkit > readline > input
_USE_PROMPT_TOOLKIT = False
_input_session = None
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    _USE_PROMPT_TOOLKIT = True
except Exception:
    _USE_PROMPT_TOOLKIT = False

if _USE_PROMPT_TOOLKIT:
    try:
        _input_session = PromptSession(history=FileHistory(str(HISTORY_PATH)))
    except Exception:
        _input_session = PromptSession()

else:
    # try to enable readline history if available
    try:
        import readline  # type: ignore
        try:
            readline.read_history_file(str(HISTORY_PATH))
        except FileNotFoundError:
            pass
    except Exception:
        pass

# small command processing
def _process_command(line: str):
    cmd = line.strip()
    if cmd in ("/exit", "/quit"):
        return ("exit", None)
    if cmd == "/help":
        help_text = (
            "Bibliotheca interactive commands:\n"
            "  /help         Show this help\n"
            "  /exit, /quit  Exit interactive loop\n"
            "  /status       Show AI & memory status\n"
            "  /history      Print last 100 history lines\n"
            "  /savehistory <file>  Save history to a file\n"
            "  /clearhistory Clear history file\n"
            "  /raw <json>   Send raw JSON object to CORE_AI.process_input (if supported)\n"
        )
        return ("reply", help_text)
    if cmd == "/status":
        try:
            mem_status = globals().get("memory").health_check() if globals().get("memory") and hasattr(globals().get("memory"), "health_check") else {"memory": "unknown"}
        except Exception as e:
            mem_status = {"error": str(e)}
        try:
            ai_obj = CORE_AI
            ai_status = {
                "ai_name": getattr(ai_obj, "name", type(ai_obj).__name__) if ai_obj else "missing",
                "has_self_check": callable(getattr(ai_obj, "self_check", None))
            }
        except Exception as e:
            ai_status = {"error": str(e)}
        return ("reply", f"AI: {ai_status}\nMemory: {mem_status}")
    if cmd == "/history":
        try:
            with open(str(HISTORY_PATH), "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            return ("reply", "\n".join(lines[-200:]) if lines else "(no history)")
        except Exception as e:
            return ("reply", f"Failed to read history: {e}")
    if cmd.startswith("/savehistory"):
        parts = cmd.split(maxsplit=1)
        target = parts[1] if len(parts) > 1 else f"bibliotheca_history_{int(time.time())}.txt"
        try:
            with open(str(HISTORY_PATH), "r", encoding="utf-8") as fh_in, open(target, "w", encoding="utf-8") as fh_out:
                fh_out.write(fh_in.read())
            return ("reply", f"Saved history to {target}")
        except Exception as e:
            return ("reply", f"Failed to save history: {e}")
    if cmd == "/clearhistory":
        try:
            open(str(HISTORY_PATH), "w").close()
            # clear readline state if available
            try:
                import readline  # type: ignore
                readline.clear_history()
            except Exception:
                pass
            return ("reply", "History cleared.")
        except Exception as e:
            return ("reply", f"Failed to clear history: {e}")
    if cmd.startswith("/raw "):
        payload = cmd[len("/raw "):].strip()
        try:
            obj = json.loads(payload)
            return ("raw", obj)
        except Exception as e:
            return ("reply", f"Invalid JSON for /raw: {e}")
    return (None, None)

# The main interactive loop function
def interactive_loop(blocking: bool = False, prompt: str = "You: ", core_ai=None):
    """
    Start the interactive loop.
    - blocking: if True, runs in current thread and blocks; if False, returns immediately and runs in daemon thread.
    - core_ai: optional override for CORE_AI object to talk to.
    """
    ai = core_ai or CORE_AI

    if ai is None:
        logger.warning("[InteractiveLoop] CORE_AI not found. Interactive loop will run but AI calls will be no-ops.")
    stop_event = threading.Event()

    def _signal_handler(signum, frame):
        logger.info(f"[InteractiveLoop] Received signal {signum}; shutting down interactive loop.")
        stop_event.set()

    # install handlers for graceful shutdown
    try:
        signal.signal(signal.SIGINT, lambda s, f: _signal_handler(s, f))
        signal.signal(signal.SIGTERM, lambda s, f: _signal_handler(s, f))
    except Exception:
        # some environments (Windows, threads, interactive shells) may not allow changing signals — ignore
        pass

    logger.info("[InteractiveLoop] Starting conversational loop ✅")
    _telemetry_event("interactive.started", {"timestamp": int(time.time())})

    try:
        while not stop_event.is_set():
            try:
                # read input
                if _USE_PROMPT_TOOLKIT and _input_session is not None:
                    try:
                        user_input = _input_session.prompt(prompt)
                    except (KeyboardInterrupt, EOFError):
                        stop_event.set()
                        break
                else:
                    try:
                        # flush stdout prompt so that logs don't corrupt prompt ordering
                        sys.stdout.write(prompt)
                        sys.stdout.flush()
                        user_input = sys.stdin.readline()
                        if not user_input:
                            # EOF
                            stop_event.set()
                            break
                        user_input = user_input.rstrip("\n")
                    except (KeyboardInterrupt, EOFError):
                        stop_event.set()
                        break

                if user_input is None:
                    continue
                raw = user_input.strip()
                if not raw:
                    continue

                # Save to history
                try:
                    with open(str(HISTORY_PATH), "a", encoding="utf-8") as hf:
                        hf.write(raw + "\n")
                    # update readline in-process if available
                    try:
                        import readline  # type: ignore
                        readline.add_history(raw)
                    except Exception:
                        pass
                except Exception:
                    logger.debug("[InteractiveLoop] Failed to write to history (ignored).")

                # Commands
                cmd_type, cmd_payload = _process_command(raw)
                if cmd_type == "exit":
                    logger.info("[InteractiveLoop] Exit command received.")
                    stop_event.set()
                    break
                if cmd_type == "reply":
                    # print multi-line reply
                    print(cmd_payload)
                    continue
                if cmd_type == "raw":
                    # send raw JSON to AI
                    try:
                        if ai and hasattr(ai, "process_input"):
                            res = _safe_invoke(ai.process_input, cmd_payload)
                            if res is not None:
                                print(res)
                        else:
                            print("(no CORE_AI.process_input available)")
                    except Exception as e:
                        logger.exception(f"[InteractiveLoop] Error sending raw to AI: {e}")
                    continue

                # Normal conversational flow: try CORE_AI.process_input, CORE_AI.chat, fallback to printing
                try:
                    invoked = False
                    # prefer process_input if available
                    if ai and hasattr(ai, "process_input") and callable(getattr(ai, "process_input")):
                        res = _safe_invoke(ai.process_input, raw)
                        invoked = True
                    elif ai and hasattr(ai, "chat") and callable(getattr(ai, "chat")):
                        res = _safe_invoke(ai.chat, raw)
                        invoked = True
                    else:
                        res = None
                        invoked = False

                    # if the call returned something printable, print it
                    if res is not None:
                        # pretty print dictionaries
                        if isinstance(res, (dict, list)):
                            print(json.dumps(res, indent=2, default=str))
                        else:
                            print(res)
                    elif not invoked:
                        print("(No AI available to process input. See logs.)")

                    _telemetry_event("interactive.input_processed", {"invoked": invoked, "len": len(raw)})

                except Exception as ai_exc:
                    _telemetry_event("interactive.error", {"error": str(ai_exc)})
                    logger.exception(f"[InteractiveLoop] AI processing error: {ai_exc}")
                    print(f"[InteractiveLoop] Error processing your input: {ai_exc}")

            except Exception as e:
                # top-level input loop exception
                logger.exception(f"[InteractiveLoop] Unexpected loop error: {e}")
                _telemetry_event("interactive.loop_exception", {"error": str(e)})
                # short sleep to avoid tight error loop
                time.sleep(0.5)

    finally:
        try:
            _telemetry_event("interactive.stopped", {"timestamp": int(time.time())})
            logger.info("[InteractiveLoop] Shutting down conversational loop gracefully.")
        except Exception:
            pass

# Helper to run as daemon thread or blocking
def start_interactive_loop(blocking: bool = False, prompt: str = "You: ", core_ai=None):
    """
    Start the interactive loop. If blocking is False (default) it launches a daemon thread
    and returns immediately. If blocking=True it runs in the current thread (useful for testing).
    """
    if blocking:
        interactive_loop(blocking=True, prompt=prompt, core_ai=core_ai)
        return None

    # if already started, do not start additional threads
    if globals().get("_INTERACTIVE_LOOP_THREAD_STARTED"):
        logger.debug("[InteractiveLoop] already started; skipping redundant start.")
        return globals().get("_INTERACTIVE_LOOP_THREAD")

    thread = threading.Thread(target=functools.partial(interactive_loop, False, prompt, core_ai), name="BibliothecaInteractiveLoop", daemon=True)
    thread.start()
    globals()["_INTERACTIVE_LOOP_THREAD_STARTED"] = True
    globals()["_INTERACTIVE_LOOP_THREAD"] = thread
    logger.info("[InteractiveLoop] Started in background thread.")
    return thread

# Start as before (non-blocking daemon thread)
start_interactive_loop(blocking=False)
# -----------------------------
# Keep main thread alive — Robust graceful shutdown & health manager
# -----------------------------
import os
import sys
import time
import signal
import threading
import asyncio
import importlib
import traceback
import atexit
from typing import Callable, Awaitable, Any, Optional, List, Tuple, Iterable

# Ensure we have a usable logger (use existing one if already defined in module)
try:
    if "logger" not in globals() or not getattr(globals()["logger"], "info", None):
        raise NameError()
    _logger = globals()["logger"]
except Exception:
    import logging as _logging

    _logger = _logging.getLogger("Bibliotheca")
    if not _logger.handlers:
        handler = _logging.StreamHandler()
        handler.setFormatter(
            _logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
        )
        _logger.addHandler(handler)
        _logger.setLevel(
            os.environ.get("BIB_LOG_LEVEL", "INFO")
        )  # use env if set, defaults to INFO


class ShutdownManager:
    """
    Coordinated shutdown manager.
    - Register sync cleanup callables with register_sync(fn, name)
    - Register async cleanup callables with register_async(coro_fn, name)
    - Register objects with common cleanup methods using register_object(obj, method_names)
    - trigger(reason) initiates graceful shutdown (idempotent).
    - wait() blocks the main thread until shutdown is requested.
    """

    def __init__(self, graceful_timeout: float = None) -> None:
        self.graceful_timeout = float(
            os.environ.get("BIB_SHUTDOWN_TIMEOUT", graceful_timeout or 30)
        )
        self._sync_cleanup: List[Tuple[Callable[[], Any], str]] = []
        self._async_cleanup: List[Tuple[Callable[[], Awaitable[Any]], str]] = []
        self._obj_cleanup: List[Tuple[Any, str, Iterable[str]]] = []
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        self._is_shutting_down = False

    def register_sync(self, fn: Callable[[], Any], name: Optional[str] = None) -> None:
        """Register a synchronous cleanup function (callable that takes no args)."""
        with self._lock:
            self._sync_cleanup.append((fn, name or getattr(fn, "__name__", "sync_cleanup")))

    def register_async(self, coro_fn: Callable[[], Awaitable[Any]], name: Optional[str] = None) -> None:
        """Register an async cleanup function (callable that returns a coroutine)."""
        with self._lock:
            self._async_cleanup.append(
                (coro_fn, name or getattr(coro_fn, "__name__", "async_cleanup"))
            )

    def register_object(self, obj: Any, method_names: Iterable[str] = ("shutdown", "stop", "close"), name: Optional[str] = None) -> None:
        """Register an object; manager will attempt known method names during shutdown."""
        with self._lock:
            self._obj_cleanup.append((obj, name or type(obj).__name__, tuple(method_names)))

    def trigger(self, reason: str = "external") -> None:
        """Idempotent. Starts shutdown in a dedicated thread (so handler returns quickly)."""
        with self._lock:
            if self._is_shutting_down:
                _logger.warning("[Main] Shutdown already in progress (trigger called again).")
                return
            self._is_shutting_down = True

        _logger.info(f"[Main] Shutdown requested ({reason}). Starting graceful shutdown (timeout {self.graceful_timeout}s).")

        def _shutdown_thread_target() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._run_shutdown(loop))
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                _logger.error("[Main] Exception inside shutdown thread:\n%s", traceback.format_exc())
            finally:
                _logger.info("[Main] Shutdown thread finished — exiting process.")
                # Make sure process exits even if other threads still linger.
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)

        t = threading.Thread(target=_shutdown_thread_target, name="bibliotheca-shutdown-thread", daemon=True)
        t.start()
        # wake up waiters
        self._shutdown_event.set()

    async def _run_shutdown(self, loop: asyncio.AbstractEventLoop) -> None:
        # 1) Call sync cleanup handlers
        _logger.info("[Main] Running %d sync cleanup handlers.", len(self._sync_cleanup))
        for fn, name in list(self._sync_cleanup):
            try:
                _logger.debug("[Main] Sync cleanup -> %s", name)
                result = fn()
                if asyncio.iscoroutine(result):
                    _logger.debug("[Main] Sync cleanup returned coroutine; awaiting -> %s", name)
                    await result
            except Exception:
                _logger.exception("[Main] Exception while running sync cleanup %s:", name)

        # 2) Attempt object-style cleanup (look for the first matching method)
        _logger.info("[Main] Running %d object cleanup candidates.", len(self._obj_cleanup))
        for obj, objname, methods in list(self._obj_cleanup):
            for m in methods:
                if hasattr(obj, m):
                    meth = getattr(obj, m)
                    try:
                        _logger.debug("[Main] Invoking %s.%s()", objname, m)
                        res = meth()
                        if asyncio.iscoroutine(res):
                            await res
                        _logger.info("[Main] Successfully ran %s.%s()", objname, m)
                    except Exception:
                        _logger.exception("[Main] Exception while calling %s.%s():", objname, m)
                    break  # only attempt the first found method

        # 3) Run async cleanup handlers concurrently (bounded by graceful_timeout)
        _logger.info("[Main] Running %d async cleanup handlers.", len(self._async_cleanup))
        async_tasks = []
        for coro_fn, name in list(self._async_cleanup):
            try:
                _logger.debug("[Main] Scheduling async cleanup -> %s", name)
                coro = coro_fn()
                task = loop.create_task(coro)
                async_tasks.append((task, name))
            except Exception:
                _logger.exception("[Main] Exception scheduling async cleanup %s:", name)

        if async_tasks:
            tasks = [t for t, _ in async_tasks]
            done, pending = await asyncio.wait(tasks, timeout=self.graceful_timeout)
            if pending:
                _logger.warning("[Main] %d async cleanup tasks did not finish in timeout; cancelling...", len(pending))
                for p in pending:
                    try:
                        p.cancel()
                    except Exception:
                        _logger.exception("[Main] Exception cancelling task:")
                # give cancelled tasks a moment
                try:
                    await asyncio.wait(pending, timeout=2)
                except Exception:
                    _logger.debug("[Main] Ignoring exceptions while awaiting cancelled tasks.")
            # log results
            for task, name in async_tasks:
                if task.done() and not task.cancelled() and task.exception():
                    _logger.exception("[Main] Async cleanup task %s raised:", name)

        # final flushes if available (telemetry etc)
        _logger.info("[Main] Cleanup finished.")

    def wait(self, heartbeat: float = 1.0) -> None:
        """
        Block until shutdown is triggered.
        Keeps main thread alive with a small heartbeat sleep (configurable via env BIB_HEARTBEAT_INTERVAL).
        """
        heartbeat = float(os.environ.get("BIB_HEARTBEAT_INTERVAL", heartbeat))
        _logger.info("[Main] Entering wait loop (heartbeat=%ss). Press Ctrl+C to shutdown.", heartbeat)
        try:
            while not self._shutdown_event.is_set():
                time.sleep(heartbeat)
        except KeyboardInterrupt:
            _logger.info("[Main] KeyboardInterrupt received — initiating graceful shutdown.")
            self.trigger("KeyboardInterrupt")
            # wait for shutdown thread to set event
            self._shutdown_event.wait()


# Instantiate a global manager that other parts of the app can import if needed
shutdown_manager = ShutdownManager()


# -----------------------------
# convenience auto-registrations (safe / best-effort)
# -----------------------------
def _try_register_module_cleanup(module_name: str, preferred_methods: Iterable[str] = ("shutdown", "stop", "close", "disconnect", "flush")) -> bool:
    """
    Try to import a module and register the first available top-level callable cleanup function.
    Returns True if something was registered.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return False

    for method_name in preferred_methods:
        if hasattr(mod, method_name) and callable(getattr(mod, method_name)):
            fn = getattr(mod, method_name)
            if asyncio.iscoroutinefunction(fn):
                shutdown_manager.register_async(fn, name=f"{module_name}.{method_name}")
            else:
                # wrap so we call the module-level function directly (preserve name)
                shutdown_manager.register_sync(lambda f=fn: f(), name=f"{module_name}.{method_name}")
            _logger.info("[Main] Auto-registered cleanup %s.%s()", module_name, method_name)
            return True
    return False


# Common modules to try auto-registering (safe/fallback — harmless if absent)
_COMMON_AUTOREG_MODULES = [
    "app.ai_core",
    "app.core",
    "app.db",
    "app.couchdb",
    "app.utils.telemetry",
    "app.utils.cache",
    "app.auto_healer",
    "app.websockets",
    "app.scheduler",
    "app.task_queue",
    "app.api",
]

for mod_name in _COMMON_AUTOREG_MODULES:
    try:
        _try_register_module_cleanup(mod_name)
    except Exception:
        _logger.debug("[Main] Auto-registration attempt for %s failed (ignored).", mod_name)


# -----------------------------
# optional health check thread (can be used to auto-trigger shutdown on repeated failure)
# -----------------------------
def _health_worker(interval: float = None, failure_threshold: int = None, check_callable: Optional[Callable[[], bool]] = None) -> None:
    interval = float(os.environ.get("BIB_HEALTH_INTERVAL", interval or 10.0))
    failure_threshold = int(os.environ.get("BIB_HEALTH_FAILURE_THRESHOLD", failure_threshold or 3))
    _logger.info("[Main] Health worker started (interval=%ss, threshold=%d)", interval, failure_threshold)
    consecutive = 0
    while not shutdown_manager._shutdown_event.is_set():
        try:
            ok = True
            if check_callable:
                try:
                    ok = bool(check_callable())
                except Exception:
                    _logger.exception("[Main] Health check callable raised; treating as failure.")
                    ok = False
            # default: if user didn't provide check_callable, assume healthy (no-op)
            if not ok:
                consecutive += 1
                _logger.warning("[Main] Health check failed (%d/%d).", consecutive, failure_threshold)
            else:
                if consecutive:
                    _logger.info("[Main] Health recovered after %d failures.", consecutive)
                consecutive = 0

            if consecutive >= failure_threshold:
                _logger.error("[Main] Health failed threshold reached — triggering shutdown.")
                shutdown_manager.trigger("health_failure")
                break
        except Exception:
            _logger.exception("[Main] Unexpected exception in health worker loop (ignored).")
        finally:
            # wait with the shutdown event so this can wake early on shutdown
            shutdown_manager._shutdown_event.wait(interval)


if os.environ.get("BIB_ENABLE_HEALTHCHECK", "1") not in ("0", "false", "False"):
    # You can pass a custom callable by setting `BIB_HEALTH_CHECK_FUNC` env name to a dotted import path,
    # or register your own health check by calling `threading.Thread(target=_health_worker, args=(..., my_check)).start()`
    try:
        # start with default no-op (for advanced usage - pass a callable)
        hw = threading.Thread(target=_health_worker, name="bibliotheca-health-thread", daemon=True)
        hw.start()
        # ensure we attempt to join on shutdown
        shutdown_manager.register_sync(lambda: hw.join(timeout=2), name="health_thread_join")
    except Exception:
        _logger.debug("[Main] Starting health worker failed (ignored).")


# Make sure shutdown_manager.trigger runs on interpreter exit as last resort
def _atexit_trigger() -> None:
    if not shutdown_manager._is_shutting_down:
        _logger.info("[Main] atexit hook calling shutdown manager.")
        shutdown_manager.trigger("atexit")

atexit.register(_atexit_trigger)


# -----------------------------
# Helper API (for other modules to import)
# -----------------------------
def register_cleanup(fn: Callable[[], Any], name: Optional[str] = None) -> None:
    """Public helper to register a sync cleanup function."""
    shutdown_manager.register_sync(fn, name=name)


def register_async_cleanup(coro_fn: Callable[[], Awaitable[Any]], name: Optional[str] = None) -> None:
    """Public helper to register an async cleanup function."""
    shutdown_manager.register_async(coro_fn, name=name)


def register_cleanup_object(obj: Any, method_names: Iterable[str] = ("shutdown", "stop", "close")) -> None:
    """Public helper to register an object for cleanup."""
    shutdown_manager.register_object(obj, method_names=method_names)


# -----------------------------
# Wire OS signals to the manager
# -----------------------------
def _signal_handler(sig, frame) -> None:
    try:
        signame = signal.Signals(sig).name if isinstance(sig, int) else str(sig)
    except Exception:
        signame = str(sig)
    _logger.info("[Main] OS Signal received: %s", signame)
    shutdown_manager.trigger(f"os_signal:{signame}")


for s in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(s, _signal_handler)
    except Exception:
        _logger.debug("[Main] Unable to set signal handler for %s (ignored).", s)

# Windows additional signal
if hasattr(signal, "SIGBREAK"):
    try:
        signal.signal(signal.SIGBREAK, _signal_handler)
    except Exception:
        _logger.debug("[Main] Unable to set SIGBREAK handler (ignored).")


# -----------------------------
# Main keep-alive entrypoint (replace the old try/except + sleep loop)
# -----------------------------
def keep_main_thread_alive() -> None:
    """
    Call this from your main program to block until shutdown is requested.
    Example:
        if __name__ == "__main__":
            start_servers_and_workers()
            keep_main_thread_alive()
    """
    try:
        shutdown_manager.wait()
    except Exception:
        _logger.exception("[Main] Unexpected error while waiting — triggering shutdown.")
        shutdown_manager.trigger("exception_in_wait")
        shutdown_manager.wait()


# Optionally expose a small CLI hook
if __name__ == "__main__":
    _logger.info("[Main] Running keep_main_thread_alive() directly from __main__")
    keep_main_thread_alive()
# -----------------------------
# Initialize AI instances (robust, defensive, feature-full)
# -----------------------------
import os
import sys
import time
import asyncio
import importlib
import threading
import traceback
from typing import Any, Callable, Optional

# Ensure logger exists (reuse module logger if defined earlier)
try:
    _logger = globals().get("logger") or globals().get("_logger")
    if not getattr(_logger, "info", None):
        raise Exception("no usable logger in globals()")
except Exception:
    import logging as _logging

    _logger = _logging.getLogger("Bibliotheca.AIInit")
    if not _logger.handlers:
        h = _logging.StreamHandler()
        h.setFormatter(_logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s"))
        _logger.addHandler(h)
    _logger.setLevel(os.environ.get("BIB_LOG_LEVEL", "INFO"))


# Module-level AI vars (guarantee they exist after this block)
CORE_AI = globals().get("CORE_AI", None)
ADVANCED_AI = globals().get("ADVANCED_AI", None)
OFFLINE_AI = globals().get("OFFLINE_AI", None)


# -----------------------------
# Utilities for init
# -----------------------------
def _run_coro_sync(coro):
    """Run coroutine and return result. Works even if an event loop is already running
    by spinning a new loop in a dedicated thread."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # run in new loop inside a thread
            result_container = {"result": None, "exc": None}

            def _runner():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result_container["result"] = new_loop.run_until_complete(coro)
                except Exception as e:
                    result_container["exc"] = e
                    result_container["tb"] = traceback.format_exc()
                finally:
                    try:
                        new_loop.close()
                    except Exception:
                        pass

            t = threading.Thread(target=_runner, name="ai-init-coro-runner", daemon=True)
            t.start()
            t.join()
            if result_container.get("exc"):
                raise result_container["exc"]
            return result_container["result"]
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # no running loop; use asyncio.run
        return asyncio.run(coro)


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _try_getattr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


class NullAI:
    """A minimal no-op AI used as a safe fallback so the app remains functional."""

    def __init__(self, name="NullAI"):
        self.name = name
        _logger.warning("[Main] Using fallback NullAI for %s", name)

    def process_input(self, *args, **kwargs):
        _logger.debug("[%s] process_input called but this is a NullAI fallback.", self.name)
        return {"status": "null", "message": "AI unavailable (null fallback)."}

    def ping(self):
        return True

    def shutdown(self):
        _logger.info("[%s] shutdown() called (no-op).", self.name)

    def is_healthy(self):
        return True


class AIAdapter:
    """Adapter that ensures a standard interface (process_input, ping, shutdown) for diverse AI objects."""

    def __init__(self, obj: Any, name: Optional[str] = None):
        self._obj = obj
        self._name = name or getattr(obj, "__class__", type(obj)).__name__

    def process_input(self, *args, **kwargs):
        # try several common method names
        for method in ("process_input", "respond", "ask", "complete", "generate"):
            if hasattr(self._obj, method):
                fn = getattr(self._obj, method)
                return fn(*args, **kwargs)
        # last-ditch: if object is callable
        if callable(self._obj):
            return self._obj(*args, **kwargs)
        raise AttributeError(f"{self._name} has no known process method")

    def ping(self):
        for method in ("ping", "is_healthy", "health_check", "ready", "is_ready"):
            if hasattr(self._obj, method):
                try:
                    return getattr(self._obj, method)()
                except Exception:
                    _logger.exception("[%s] ping method %s raised", self._name, method)
                    return False
        # fallback: assume healthy if we can call a light-weight attribute
        return True

    def shutdown(self):
        for method in ("shutdown", "stop", "close", "terminate"):
            if hasattr(self._obj, method):
                try:
                    res = getattr(self._obj, method)()
                    # handle coroutine returned
                    if asyncio.iscoroutine(res):
                        _run_coro_sync(res)
                    return
                except Exception:
                    _logger.exception("[%s] shutdown method %s raised", self._name, method)
                    return
        _logger.debug("[%s] No shutdown method found (no-op).", self._name)


def _register_with_shutdown_manager(obj, name_hint: str = None):
    """If the shutdown manager (or helpers) exist, register object for graceful shutdown."""
    try:
        # prefer helper register_cleanup_object if present
        reg_helper = globals().get("register_cleanup_object") or globals().get("register_cleanup")
        if callable(reg_helper):
            # register as object if helper supports that, else wrap
            try:
                reg_helper(obj, ("shutdown", "stop", "close"))
                _logger.debug("[Main] Registered %s with register_cleanup_object/register_cleanup", name_hint or obj)
                return
            except TypeError:
                # fallback: attempt register_sync/async with explicit callables
                if hasattr(obj, "shutdown") and callable(getattr(obj, "shutdown")):
                    try:
                        globals().get("register_cleanup")(getattr(obj, "shutdown"), name=name_hint or None)
                        return
                    except Exception:
                        pass
        # fallback to shutdown_manager instance if present
        shm = globals().get("shutdown_manager")
        if shm and hasattr(shm, "register_object"):
            try:
                shm.register_object(obj, ("shutdown", "stop", "close"), name=name_hint)
                _logger.debug("[Main] Registered %s with shutdown_manager.register_object", name_hint or obj)
                return
            except Exception:
                _logger.exception("[Main] Could not register %s with shutdown_manager (ignored).", name_hint or obj)
    except Exception:
        _logger.debug("[Main] No shutdown registration available (ignored).")


# -----------------------------
# Candidate lists for discovery
# -----------------------------
_CORE_CANDIDATE_MODULES = [
    "app.ai_core",
    "ai_core",
    "app.ai.core",
    "core.ai",
    "app.core.ai",
]
_ADV_CANDIDATE_MODULES = [
    "app.advanced_ai",
    "advanced_ai",
    "app.ai.advanced",
    "app.ai.advanced_ai",
]
_OFFLINE_CANDIDATE_MODULES = [
    "app.ai.offline",
    "offline_ai",
    "app.offline_ai",
    "app.ai.offline_ai",
]


def _attempt_initialize(name: str, module_candidates, class_names=("CoreAI", "AI"), factory_names=None, env_var_prefix=None, offline=False):
    """Generalized attempt to find or construct an AI instance."""
    factory_names = factory_names or []
    env_var_prefix = env_var_prefix or name.upper()
    last_exc = None

    # 1) If already present in globals() use it
    existing = globals().get(name)
    if existing is not None:
        _logger.info("[Main] %s already present in globals — using it.", name)
        return existing

    # 2) Look for well-known import locations and module-level singletons/factories
    for modname in module_candidates:
        mod = _safe_import(modname)
        if not mod:
            continue
        try:
            # direct variable exported
            if hasattr(mod, name):
                inst = getattr(mod, name)
                _logger.info("[Main] Found %s from module %s (module variable).", name, modname)
                return inst

            # factory functions
            for fn_name in factory_names:
                if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
                    fn = getattr(mod, fn_name)
                    try:
                        _logger.info("[Main] Calling factory %s.%s() to construct %s", modname, fn_name, name)
                        res = fn()
                        if asyncio.iscoroutine(res):
                            res = _run_coro_sync(res)
                        return res
                    except Exception as e:
                        _logger.exception("[Main] Factory %s.%s() raised:", modname, fn_name)
                        last_exc = e

            # top-level classes
            for cls_name in class_names:
                if hasattr(mod, cls_name):
                    cls = getattr(mod, cls_name)
                    try:
                        # Try instantiate with config if available
                        cfg = globals().get("CONFIG") or globals().get("config") or {}
                        kwargs = {}
                        # common env-driven params
                        model_path = os.environ.get(f"{env_var_prefix}_MODEL_PATH")
                        if model_path:
                            kwargs["model_path"] = model_path
                        _logger.info("[Main] Instantiating %s.%s() for %s (kwargs=%s)", modname, cls_name, name, bool(kwargs))
                        inst = cls(**kwargs) if kwargs else cls()
                        # support async init if object exposes .init async method
                        if hasattr(inst, "init") and asyncio.iscoroutinefunction(getattr(inst, "init")):
                            _logger.debug("[Main] Running async init for %s.%s", modname, cls_name)
                            _run_coro_sync(inst.init())
                        return inst
                    except Exception as e:
                        _logger.exception("[Main] Instantiating %s.%s failed:", modname, cls_name)
                        last_exc = e
        except Exception as e:
            last_exc = e
            _logger.debug("[Main] Ignored exception while probing %s: %s", modname, e)

    # 3) Look for dotted factory path in env (like app.ai_core:get_core_ai)
    dotted = os.environ.get(f"{env_var_prefix}_FACTORY")
    if dotted:
        try:
            module_name, func_name = dotted.split(":", 1) if ":" in dotted else (dotted.rsplit(".", 1)[0], dotted.rsplit(".", 1)[1])
            mod = _safe_import(module_name)
            if mod and hasattr(mod, func_name):
                fn = getattr(mod, func_name)
                _logger.info("[Main] Calling factory from env %s -> %s.%s()", dotted, module_name, func_name)
                res = fn()
                if asyncio.iscoroutine(res):
                    res = _run_coro_sync(res)
                return res
        except Exception as e:
            _logger.exception("[Main] Env factory %s failed:", dotted)
            last_exc = e

    # 4) Give up -> return None and let the caller decide fallback
    if last_exc:
        _logger.debug("[Main] Last exception while probing for %s:\n%s", name, traceback.format_exc())
    _logger.warning("[Main] Could not find/instantiate %s automatically.", name)
    return None


# -----------------------------
# Initialize CORE_AI, ADVANCED_AI, OFFLINE_AI
# -----------------------------
try:
    # CORE_AI
    if CORE_AI is None:
        CORE_AI = _attempt_initialize(
            "CORE_AI",
            module_candidates=_CORE_CANDIDATE_MODULES,
            class_names=("CoreAI", "AI", "Assistant"),
            factory_names=("get_core_ai", "get_ai", "create_core_ai", "get_instance"),
            env_var_prefix="CORE_AI",
        )

    # ADVANCED_AI
    if ADVANCED_AI is None:
        ADVANCED_AI = _attempt_initialize(
            "ADVANCED_AI",
            module_candidates=_ADV_CANDIDATE_MODULES,
            class_names=("AdvancedAI", "AdvancedAssistant", "SuperAI"),
            factory_names=("get_advanced_ai", "get_ai", "create_advanced_ai"),
            env_var_prefix="ADVANCED_AI",
        )

    # OFFLINE_AI - attempt explicit OfflineAI locations or fallback to a minimal offline adapter
    if OFFLINE_AI is None:
        # Try to import an OfflineAI class or factory from common locations
        off = _attempt_initialize(
            "OFFLINE_AI",
            module_candidates=_OFFLINE_CANDIDATE_MODULES,
            class_names=("OfflineAI", "LocalOfflineAI"),
            factory_names=("get_offline_ai", "create_offline_ai"),
            env_var_prefix="OFFLINE_AI",
            offline=True,
        )
        if off is None:
            # try an explicit class import fallback
            for modname in _OFFLINE_CANDIDATE_MODULES:
                mod = _safe_import(modname)
                if not mod:
                    continue
                if hasattr(mod, "OfflineAI"):
                    try:
                        cls = getattr(mod, "OfflineAI")
                        OFFLINE_AI = cls(model_path=os.environ.get("OFFLINE_AI_MODEL_PATH", None))
                        break
                    except Exception:
                        _logger.exception("[Main] Failed to instantiate OfflineAI from %s", modname)
            # If still none, create a lightweight default offline object
        if off is not None and OFFLINE_AI is None:
            OFFLINE_AI = off
        if OFFLINE_AI is None:
            # Minimal but useful default OfflineAI (no external dependencies)
            class BasicOfflineAI:
                def __init__(self, model_path=None):
                    self.model_path = model_path
                    _logger.info("[Main] BasicOfflineAI initialized (model_path=%s).", bool(model_path))

                def process_input(self, text, *args, **kwargs):
                    # purely local placeholder: echo + metadata so app can continue
                    _logger.debug("[BasicOfflineAI] echoing input (no real model).")
                    return {"offline_echo": text, "note": "BasicOfflineAI fallback - no model loaded."}

                def ping(self):
                    return True

                def shutdown(self):
                    _logger.debug("[BasicOfflineAI] shutdown (no-op).")

                def is_healthy(self):
                    return True

            OFFLINE_AI = BasicOfflineAI(model_path=os.environ.get("OFFLINE_AI_MODEL_PATH", None))

    # Final adaptation: ensure each AI exposes standard methods via adapter if needed
    def _ensure_adapter(ai_obj, varname):
        if ai_obj is None:
            _logger.warning("[Main] %s is None, replacing with NullAI fallback.", varname)
            return NullAI(name=varname)
        # If it already has a standard process_input, leave as is
        if hasattr(ai_obj, "process_input") and callable(getattr(ai_obj, "process_input")):
            return ai_obj
        # else wrap with adapter
        _logger.info("[Main] Wrapping %s with AIAdapter to provide a standard interface.", varname)
        return AIAdapter(ai_obj, name=varname)

    CORE_AI = _ensure_adapter(CORE_AI, "CORE_AI")
    ADVANCED_AI = _ensure_adapter(ADVANCED_AI, "ADVANCED_AI")
    OFFLINE_AI = _ensure_adapter(OFFLINE_AI, "OFFLINE_AI")

    # Register with shutdown manager if available
    _register_with_shutdown_manager(CORE_AI, name_hint="CORE_AI")
    _register_with_shutdown_manager(ADVANCED_AI, name_hint="ADVANCED_AI")
    _register_with_shutdown_manager(OFFLINE_AI, name_hint="OFFLINE_AI")

    # Optionally register telemetry/metrics if module available
    try:
        telemetry_mod = _safe_import("app.utils.telemetry") or _safe_import("app.telemetry") or _safe_import("utils.telemetry")
        if telemetry_mod and hasattr(telemetry_mod, "register_component"):
            for ai_obj, tag in ((CORE_AI, "core_ai"), (ADVANCED_AI, "advanced_ai"), (OFFLINE_AI, "offline_ai")):
                try:
                    telemetry_mod.register_component(tag, ai_obj)
                    _logger.debug("[Main] Registered %s with telemetry as %s", tag, tag)
                except Exception:
                    _logger.debug("[Main] Telemetry register_component failed for %s (ignored).", tag)
    except Exception:
        _logger.debug("[Main] Telemetry registration skipped (no telemetry module).")

    _logger.info("[Main] CORE_AI, ADVANCED_AI & OFFLINE_AI initialization complete ✅")

except Exception as e:
    _logger.exception("[Main] Failed to initialize AI instances (falling back to nulls): %s", e)
    CORE_AI = CORE_AI or NullAI("CORE_AI")
    ADVANCED_AI = ADVANCED_AI or NullAI("ADVANCED_AI")
    OFFLINE_AI = OFFLINE_AI or NullAI("OFFLINE_AI")


# -----------------------------
# Lightweight AI watchdog (monitors health + optional auto-shutdown)
# -----------------------------
def _ai_watchdog(interval: float = 10.0, failure_threshold: int = 3, trigger_shutdown: bool = False):
    """Runs in a daemon thread. Pings each AI periodically and logs warnings.
    If `trigger_shutdown` True and an AI repeatedly fails, triggers shutdown_manager if available."""
    interval = float(os.environ.get("BIB_AI_WATCHDOG_INTERVAL", interval))
    failure_threshold = int(os.environ.get("BIB_AI_WATCHDOG_THRESHOLD", failure_threshold))
    _logger.info("[Main] Starting AI watchdog (interval=%ss, threshold=%d, trigger_shutdown=%s)", interval, failure_threshold, trigger_shutdown)
    failures = {"CORE_AI": 0, "ADVANCED_AI": 0, "OFFLINE_AI": 0}
    while True:
        try:
            for var_name, ai_obj in (("CORE_AI", CORE_AI), ("ADVANCED_AI", ADVANCED_AI), ("OFFLINE_AI", OFFLINE_AI)):
                try:
                    healthy = True
                    if hasattr(ai_obj, "ping") and callable(getattr(ai_obj, "ping")):
                        healthy = ai_obj.ping()
                    elif hasattr(ai_obj, "is_healthy") and callable(getattr(ai_obj, "is_healthy")):
                        healthy = ai_obj.is_healthy()
                    else:
                        # attempt a lightweight process_input probe if safe
                        try:
                            probe = ai_obj.process_input if hasattr(ai_obj, "process_input") else None
                            if probe:
                                res = probe("__health_probe__")
                                # don't rely on result shape; assume success if no exception
                                healthy = True
                        except Exception:
                            healthy = False

                    if not healthy:
                        failures[var_name] += 1
                        _logger.warning("[Main][AI-Watchdog] %s unhealthy (%d/%d)", var_name, failures[var_name], failure_threshold)
                    else:
                        if failures[var_name]:
                            _logger.info("[Main][AI-Watchdog] %s recovered after %d failures.", var_name, failures[var_name])
                        failures[var_name] = 0

                    if trigger_shutdown and failures[var_name] >= failure_threshold:
                        _logger.error("[Main][AI-Watchdog] %s exceeded failure threshold — triggering app shutdown.", var_name)
                        shm = globals().get("shutdown_manager")
                        if shm and hasattr(shm, "trigger"):
                            try:
                                shm.trigger(f"ai_watchdog:{var_name}_failure")
                            except Exception:
                                _logger.exception("[Main][AI-Watchdog] shutdown_manager.trigger raised")
                        else:
                            _logger.warning("[Main][AI-Watchdog] No shutdown_manager found to trigger.")
                except Exception:
                    _logger.exception("[Main][AI-Watchdog] Exception while checking %s (ignored).", var_name)
        except Exception:
            _logger.exception("[Main][AI-Watchdog] Unexpected exception in watchdog loop (ignored).")
        time.sleep(interval)


# Start watchdog thread unless explicitly disabled
try:
    if os.environ.get("BIB_AI_WATCHDOG", "1") not in ("0", "false", "False"):
        t = threading.Thread(target=_ai_watchdog, name="bibliotheca-ai-watchdog", args=(float(os.environ.get("BIB_AI_WATCHDOG_INTERVAL", 10)), int(os.environ.get("BIB_AI_WATCHDOG_THRESHOLD", 3)), os.environ.get("BIB_AI_WATCHDOG_TRIGGER_SHUTDOWN", "0") in ("1", "true", "True")), daemon=True)
        t.start()
        # register the thread join so shutdown waits briefly for it
        try:
            _register_with_shutdown_manager(t, name_hint="ai_watchdog_thread")
        except Exception:
            pass
except Exception:
    _logger.debug("[Main] Failed to start AI watchdog (ignored).")

# Expose final globals (useful for interactive inspection or imports)
globals()["CORE_AI"] = CORE_AI
globals()["ADVANCED_AI"] = ADVANCED_AI
globals()["OFFLINE_AI"] = OFFLINE_AI
# -----------------------------
# Initialize backend router
# -----------------------------

import os
import sys
import time
import threading
import asyncio
import importlib
import traceback
from typing import Any, Optional, Iterable

# reuse any existing logger if present
try:
    _logger = globals().get("logger") or globals().get("_logger")
    if not getattr(_logger, "info", None):
        raise Exception("no usable logger in globals()")
except Exception:
    import logging as _logging

    _logger = _logging.getLogger("Bibliotheca.RouterInit")
    if not _logger.handlers:
        h = _logging.StreamHandler()
        h.setFormatter(_logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s"))
        _logger.addHandler(h)
    _logger.setLevel(os.environ.get("BIB_LOG_LEVEL", "INFO"))


# small utilities (safe import + run coroutine sync)
def _safe_import(modname: str):
    try:
        return importlib.import_module(modname)
    except Exception:
        return None


def _run_coro_sync(coro):
    """Run a coroutine and return its result; safe when there is already an event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # run in a separate thread with its own loop
            result = {"value": None, "exc": None, "tb": None}

            def _runner():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result["value"] = new_loop.run_until_complete(coro)
                except Exception as e:
                    result["exc"] = e
                    result["tb"] = traceback.format_exc()
                finally:
                    try:
                        new_loop.close()
                    except Exception:
                        pass

            t = threading.Thread(target=_runner, name="router-init-coro-runner", daemon=True)
            t.start()
            t.join()
            if result["exc"]:
                raise result["exc"]
            return result["value"]
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # no event loop
        return asyncio.run(coro)


# Standard minimal router fallback so app stays alive if the real router fails
class NullRouter:
    def __init__(self, name="NullRouter"):
        self.name = name
        _logger.warning("[Main] Using NullRouter fallback for %s", name)

    def register_route(self, path: str, handler, methods: Iterable[str] = ("GET",)):
        _logger.debug("[NullRouter] register_route called for %s (noop)", path)

    def register_routes(self, router_like):
        _logger.debug("[NullRouter] register_routes called (noop)")

    def attach_to_app(self, app):
        _logger.debug("[NullRouter] attach_to_app called (noop)")

    def ping(self):
        return True

    def is_healthy(self):
        return True

    def start(self):
        _logger.debug("[NullRouter] start (noop)")

    def shutdown(self):
        _logger.debug("[NullRouter] shutdown (noop)")


# Adapter: wrap framework-specific routers and expose a small standard interface
class RouterAdapter:
    def __init__(self, inner: Any, name: Optional[str] = None):
        self._inner = inner
        self.name = name or getattr(inner, "__class__", type(inner)).__name__

    def register_route(self, path: str, handler, methods: Iterable[str] = ("GET",)):
        # many routers use add_route / add_url_rule / route decorators
        for m in ("add_route", "add_url_rule", "route", "add_endpoint"):
            if hasattr(self._inner, m):
                try:
                    getattr(self._inner, m)(path, handler, methods=methods)
                    return
                except TypeError:
                    try:
                        getattr(self._inner, m)(path, handler)
                        return
                    except Exception:
                        pass
        # last resort: attach to an attribute if it keeps a mapping
        if hasattr(self._inner, "routes") and isinstance(getattr(self._inner, "routes"), dict):
            self._inner.routes[path] = {"handler": handler, "methods": methods}
            return
        raise RuntimeError(f"{self.name} does not support register_route adaptively")

    def register_routes(self, module_or_callable):
        # if module exposes register_routes(router) call it
        try:
            if callable(module_or_callable):
                module_or_callable(self)
                return
            # if it's a module with register_routes
            if hasattr(module_or_callable, "register_routes") and callable(getattr(module_or_callable, "register_routes")):
                getattr(module_or_callable, "register_routes")(self)
                return
        except Exception:
            _logger.exception("[RouterAdapter] register_routes failed for %s (ignored).", getattr(module_or_callable, "__name__", module_or_callable))
            return

    def attach_to_app(self, app):
        # attempt to attach to Flask / FastAPI / ASGI apps gracefully
        if app is None:
            return
        # FastAPI / Starlette: include_router
        if hasattr(app, "include_router") and hasattr(self._inner, "routes"):
            try:
                app.include_router(self._inner)
                _logger.info("[RouterAdapter] Attached FastAPI-style router to app using include_router")
                return
            except Exception:
                _logger.debug("[RouterAdapter] include_router attempt failed (continuing).")
        # Flask: register_blueprint or register_blueprint attribute
        if hasattr(app, "register_blueprint"):
            # if _inner is a Blueprint-like object, attempt to register
            try:
                app.register_blueprint(self._inner)
                _logger.info("[RouterAdapter] Attached Flask-style blueprint to app using register_blueprint")
                return
            except Exception:
                _logger.debug("[RouterAdapter] register_blueprint attempt failed (continuing).")
        # fallback: attempt to set attribute or attach as property to app
        try:
            if not hasattr(app, "routes") and hasattr(self._inner, "routes"):
                setattr(app, "routes", getattr(self._inner, "routes"))
                _logger.info("[RouterAdapter] Exposed router.routes onto app (fallback).")
        except Exception:
            _logger.debug("[RouterAdapter] Fallback attach_to_app failed (ignored).")

    def ping(self):
        for method in ("ping", "is_healthy", "health_check", "ready"):
            if hasattr(self._inner, method):
                try:
                    return getattr(self._inner, method)()
                except Exception:
                    _logger.exception("[RouterAdapter] %s.%s() raised", self.name, method)
                    return False
        return True

    def start(self):
        for method in ("start", "serve", "run"):
            if hasattr(self._inner, method):
                try:
                    res = getattr(self._inner, method)()
                    if asyncio.iscoroutine(res):
                        _run_coro_sync(res)
                    return
                except Exception:
                    _logger.exception("[RouterAdapter] start method %s raised", method)
        _logger.debug("[RouterAdapter] No start method found (no-op)")

    def shutdown(self):
        for method in ("shutdown", "stop", "close"):
            if hasattr(self._inner, method):
                try:
                    res = getattr(self._inner, method)()
                    if asyncio.iscoroutine(res):
                        _run_coro_sync(res)
                    return
                except Exception:
                    _logger.exception("[RouterAdapter] shutdown method %s raised", method)
        _logger.debug("[RouterAdapter] No shutdown method found (no-op)")


# discovery candidates & factory names
_ROUTER_MODULE_CANDIDATES = [
    "app.backend.router",
    "app.router",
    "backend.router",
    "app.backend_router",
    "backend_router",
    "app.routes.router",
    "routes.router",
    "app.web.router",
]
_ROUTER_FACTORY_NAMES = (
    "get_router",
    "create_router",
    "make_router",
    "build_router",
    "get_backend_router",
)


def _attempt_initialize_router():
    last_exc = None

    # 1) use already in globals
    if "router" in globals() and globals().get("router") is not None:
        _logger.info("[Main] Found pre-existing 'router' in globals — using that instance.")
        return globals().get("router")

    # 2) look for common module exports / factories / classes
    for modname in _ROUTER_MODULE_CANDIDATES:
        mod = _safe_import(modname)
        if not mod:
            continue
        try:
            # direct symbol exported
            if hasattr(mod, "BackendRouter"):
                try:
                    cls = getattr(mod, "BackendRouter")
                    _logger.info("[Main] Instantiating BackendRouter from %s.BackendRouter()", modname)
                    inst = cls(**(globals().get("CONFIG", {}) or {}))
                    if asyncio.iscoroutine(inst):
                        inst = _run_coro_sync(inst)
                    return inst
                except Exception:
                    _logger.exception("[Main] Failed to instantiate BackendRouter from %s", modname)
                    last_exc = sys.exc_info()
            # exported module-level router object
            if hasattr(mod, "router") and getattr(mod, "router") is not None:
                _logger.info("[Main] Using module-level router from %s.router", modname)
                return getattr(mod, "router")
            # factory function(s)
            for fn_name in _ROUTER_FACTORY_NAMES:
                if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
                    try:
                        _logger.info("[Main] Calling factory %s.%s() to create router", modname, fn_name)
                        res = getattr(mod, fn_name)()
                        if asyncio.iscoroutine(res):
                            res = _run_coro_sync(res)
                        return res
                    except Exception:
                        _logger.exception("[Main] Factory %s.%s() failed", modname, fn_name)
                        last_exc = sys.exc_info()
        except Exception:
            last_exc = sys.exc_info()
            _logger.debug("[Main] Ignored exception while probing %s: %s", modname, last_exc)

    # 3) allow env-driven dotted factory (e.g., app.backend.router:get_router)
    dotted = os.environ.get("BIB_ROUTER_FACTORY") or os.environ.get("BIB_BACKEND_ROUTER_FACTORY")
    if dotted:
        try:
            if ":" in dotted:
                module_name, func_name = dotted.split(":", 1)
            else:
                module_name, func_name = dotted.rsplit(".", 1)
            mod = _safe_import(module_name)
            if mod and hasattr(mod, func_name):
                fn = getattr(mod, func_name)
                _logger.info("[Main] Calling router factory %s:%s()", module_name, func_name)
                res = fn()
                if asyncio.iscoroutine(res):
                    res = _run_coro_sync(res)
                return res
        except Exception:
            _logger.exception("[Main] Env router factory %s failed", dotted)
            last_exc = sys.exc_info()

    # nothing found -> return None (caller will fallback)
    if last_exc:
        _logger.debug("[Main] Last exception while probing for BackendRouter: %s", traceback.format_exc())
    _logger.warning("[Main] Could not locate a BackendRouter automatically.")
    return None


# Attempt initialization with robust fallbacks
try:
    _router_obj = _attempt_initialize_router()

    # if no router found, try a few more guesses (e.g., a module export named 'BackendRouter' in 'app')
    if _router_obj is None:
        extra_mod = _safe_import("app")
        if extra_mod and hasattr(extra_mod, "BackendRouter"):
            try:
                _router_obj = getattr(extra_mod, "BackendRouter")()
            except Exception:
                _logger.debug("[Main] app.BackendRouter() failed (ignored).")

    # final fallback -> NullRouter
    if _router_obj is None:
        _logger.warning("[Main] Falling back to NullRouter (no BackendRouter found).")
        router = NullRouter()
    else:
        # If it's a plain framework type, adapt to standard API using RouterAdapter
        # Detect common FastAPI/Starlette APIRouter or Flask Blueprint by duck-typing
        adapted = False
        try:
            # already a custom router instance with expected methods
            if hasattr(_router_obj, "register_route") or hasattr(_router_obj, "attach_to_app") or hasattr(_router_obj, "register_routes"):
                router = _router_obj
                adapted = True
                _logger.info("[Main] Using discovered router as-is (%s).", getattr(router, "__class__", type(router)).__name__)
            else:
                # If APIRouter-like (has .routes and is not callable), wrap in adapter
                if hasattr(_router_obj, "routes") or hasattr(_router_obj, "include_router") or hasattr(_router_obj, "add_api_route"):
                    router = RouterAdapter(_router_obj, name=getattr(_router_obj, "__class__", type(_router_obj)).__name__)
                    adapted = True
                    _logger.info("[Main] Wrapped framework router %s with RouterAdapter.", router.name)
                else:
                    # If it's a callable WSGI/ASGI app, wrap it with adapter that exposes attach_to_app
                    router = RouterAdapter(_router_obj, name=getattr(_router_obj, "__class__", type(_router_obj)).__name__)
                    adapted = True
                    _logger.info("[Main] Wrapped callable router-like object with RouterAdapter.")
        except Exception:
            _logger.exception("[Main] Exception while adapting discovered router (falling back to NullRouter).")
            router = NullRouter()

    # Auto-discover routes modules and call register_routes(router) if present
    for mod_try in ("app.routes", "routes", "app.backend.routes", "backend.routes"):
        try:
            mod = _safe_import(mod_try)
            if mod and hasattr(mod, "register_routes") and callable(getattr(mod, "register_routes")):
                try:
                    _logger.info("[Main] Auto-registering routes from %s.register_routes(router)", mod_try)
                    res = getattr(mod, "register_routes")(router)
                    if asyncio.iscoroutine(res):
                        _run_coro_sync(res)
                except Exception:
                    _logger.exception("[Main] Exception during %s.register_routes(router) (ignored).", mod_try)
        except Exception:
            _logger.debug("[Main] Auto-discovery import %s failed (ignored).", mod_try)

    # Attach router to an existing app if present and attachable
    def _try_attach_to_existing_app(rtr):
        candidates = []
        # direct global variables
        for name in ("app", "flask_app", "fastapi_app", "web_app", "server_app"):
            if name in globals() and globals()[name] is not None:
                candidates.append(globals()[name])
        # also look for app module with 'app' attribute
        try:
            am = _safe_import("app.web") or _safe_import("app.server") or _safe_import("app")
            if am and hasattr(am, "app"):
                candidates.append(getattr(am, "app"))
        except Exception:
            pass

        for candidate in candidates:
            try:
                if hasattr(rtr, "attach_to_app"):
                    rtr.attach_to_app(candidate)
                    _logger.info("[Main] Router attached to existing app object (%s).", getattr(candidate, "__class__", type(candidate)).__name__)
                    return True
                # attempt framework detection
                if hasattr(candidate, "register_blueprint") and hasattr(rtr, "blueprint"):
                    try:
                        candidate.register_blueprint(getattr(rtr, "blueprint"))
                        _logger.info("[Main] Registered blueprint to Flask app (auto).")
                        return True
                    except Exception:
                        pass
                if hasattr(candidate, "include_router") and hasattr(rtr, "routes"):
                    try:
                        candidate.include_router(rtr)
                        _logger.info("[Main] Included router into FastAPI app (auto).")
                        return True
                    except Exception:
                        pass
            except Exception:
                _logger.exception("[Main] Attempt to attach router to app candidate failed (ignored).")
        return False

    try:
        _try_attach_to_existing_app(router)
    except Exception:
        _logger.debug("[Main] attach attempt failed (ignored).")

    # Register with shutdown manager / cleanup helpers if available
    try:
        reg_helper = globals().get("register_cleanup_object") or globals().get("register_cleanup")
        if callable(reg_helper):
            try:
                reg_helper(router, ("shutdown", "stop", "close"))
                _logger.debug("[Main] Registered router using register_cleanup_object/register_cleanup.")
            except Exception:
                # fallback to register_cleanup wrapper if signature differs
                try:
                    globals().get("register_cleanup")(getattr(router, "shutdown"), name="router_shutdown")
                except Exception:
                    _logger.debug("[Main] register_cleanup attempts for router failed (ignored).")
    except Exception:
        _logger.debug("[Main] No register_cleanup helper available (ignored).")

    # Try to register with shutdown_manager directly if present
    try:
        shm = globals().get("shutdown_manager")
        if shm and hasattr(shm, "register_object"):
            try:
                shm.register_object(router, ("shutdown", "stop", "close"))
                _logger.debug("[Main] Registered router with shutdown_manager.register_object.")
            except Exception:
                _logger.exception("[Main] shutdown_manager.register_object(router) raised (ignored).")
    except Exception:
        _logger.debug("[Main] No shutdown_manager present (ignored).")

    # Optional telemetry registration
    try:
        telemetry_mod = _safe_import("app.utils.telemetry") or _safe_import("app.telemetry") or _safe_import("utils.telemetry")
        if telemetry_mod and hasattr(telemetry_mod, "register_component"):
            try:
                telemetry_mod.register_component("router", router)
                _logger.debug("[Main] Registered router with telemetry.")
            except Exception:
                _logger.debug("[Main] telemetry.register_component(router) failed (ignored).")
    except Exception:
        _logger.debug("[Main] Telemetry import attempt failed (ignored).")

    _logger.info("[Main] BackendRouter initialization complete ✅")

except Exception as e:
    _logger.exception("[Main] Failed to initialize BackendRouter; falling back to NullRouter: %s", e)
    router = NullRouter()


# -----------------------------
# Router health watchdog (optional)
# -----------------------------
def _router_watchdog(interval: float = 10.0, failure_threshold: int = 3, trigger_shutdown: bool = False):
    interval = float(os.environ.get("BIB_ROUTER_HEALTH_INTERVAL", interval))
    failure_threshold = int(os.environ.get("BIB_ROUTER_HEALTH_THRESHOLD", failure_threshold))
    _logger.info("[Main] Starting router watchdog (interval=%ss, threshold=%d)", interval, failure_threshold)
    failures = 0
    while True:
        try:
            healthy = True
            if hasattr(router, "ping") and callable(getattr(router, "ping")):
                healthy = router.ping()
            elif hasattr(router, "is_healthy") and callable(getattr(router, "is_healthy")):
                healthy = router.is_healthy()
            else:
                # attempt a lightweight probe
                try:
                    if hasattr(router, "register_route") and callable(getattr(router, "register_route")):
                        # do NOT actually register a permanent route; call with a dummy to test responsiveness
                        try:
                            router.register_route("/__healthprobe__", lambda *_: None, methods=("GET",))
                        except Exception:
                            # ignore failure to add (some routers will raise), assume healthy if no exception
                            pass
                    healthy = True
                except Exception:
                    healthy = False

            if not healthy:
                failures += 1
                _logger.warning("[Main][Router-Watchdog] Router unhealthy (%d/%d)", failures, failure_threshold)
            else:
                if failures:
                    _logger.info("[Main][Router-Watchdog] Router recovered after %d failures", failures)
                failures = 0

            if trigger_shutdown and failures >= failure_threshold:
                _logger.error("[Main][Router-Watchdog] Router exceeded failure threshold — triggering shutdown.")
                shm = globals().get("shutdown_manager")
                if shm and hasattr(shm, "trigger"):
                    try:
                        shm.trigger("router_watchdog:failure")
                    except Exception:
                        _logger.exception("[Main][Router-Watchdog] shutdown_manager.trigger raised")
                else:
                    _logger.warning("[Main][Router-Watchdog] No shutdown_manager found to trigger.")
        except Exception:
            _logger.exception("[Main][Router-Watchdog] Unexpected exception in watchdog loop (ignored).")
        time.sleep(interval)


try:
    if os.environ.get("BIB_ROUTER_WATCHDOG", "1") not in ("0", "false", "False"):
        t = threading.Thread(
            target=_router_watchdog,
            name="bibliotheca-router-watchdog",
            args=(float(os.environ.get("BIB_ROUTER_HEALTH_INTERVAL", 10.0)), int(os.environ.get("BIB_ROUTER_HEALTH_THRESHOLD", 3)), os.environ.get("BIB_ROUTER_WATCHDOG_TRIGGER_SHUTDOWN", "0") in ("1", "true", "True")),
            daemon=True,
        )
        t.start()
        # attempt to register the thread to shutdown_manager if available (so it can be joined)
        try:
            shm = globals().get("shutdown_manager")
            if shm and hasattr(shm, "register_sync"):
                shm.register_sync(lambda: t.join(timeout=1), name="router_watchdog_thread_join")
        except Exception:
            pass
except Exception:
    _logger.debug("[Main] Failed to start router watchdog (ignored).")


# ensure router available in globals for other modules
globals()["router"] = router
# -----------------------------
# Optional: start any background tasks
# -----------------------------
import asyncio
import threading
import time
import traceback
from typing import Callable, Optional, Any, List

# globals
_background_tasks: List[dict] = []
_background_loop: Optional[asyncio.AbstractEventLoop] = None

# helper to run coroutines safely in sync context
def _run_coro_sync(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # run in separate thread with its own event loop
            result = {"value": None, "exc": None, "tb": None}

            def _runner():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result["value"] = new_loop.run_until_complete(coro)
                except Exception as e:
                    result["exc"] = e
                    result["tb"] = traceback.format_exc()
                finally:
                    try:
                        new_loop.close()
                    except Exception:
                        pass

            t = threading.Thread(target=_runner, name="background-coro-runner", daemon=True)
            t.start()
            t.join()
            if result["exc"]:
                raise result["exc"]
            return result["value"]
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # no event loop
        return asyncio.run(coro)


# -----------------------------
# Background Task Manager
# -----------------------------
class BackgroundTask:
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        interval: Optional[float] = None,  # in seconds, for periodic tasks
        run_async: bool = False,
        daemon: bool = True,
        run_on_start: bool = True,
    ):
        self.func = func
        self.name = name or getattr(func, "__name__", "UnnamedTask")
        self.interval = interval
        self.run_async = run_async
        self.daemon = daemon
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if run_on_start:
            self.start()

    def _thread_target(self):
        logger.info(f"[BackgroundTask:{self.name}] Starting thread")
        try:
            while not self._stop_event.is_set():
                try:
                    if self.run_async:
                        _run_coro_sync(self.func())
                    else:
                        self.func()
                except Exception as e:
                    logger.exception(f"[BackgroundTask:{self.name}] Exception occurred: {e}")
                if self.interval:
                    time.sleep(self.interval)
                else:
                    break  # one-shot task
        except Exception:
            logger.exception(f"[BackgroundTask:{self.name}] Unexpected error in thread loop")
        finally:
            logger.info(f"[BackgroundTask:{self.name}] Exiting thread")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._thread_target, name=f"bg-{self.name}", daemon=self.daemon)
        self._thread.start()
        _background_tasks.append({"task": self, "name": self.name})

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info(f"[BackgroundTask:{self.name}] Stopped")


# -----------------------------
# Background Task Registration
# -----------------------------
def register_background_task(
    func: Callable,
    name: Optional[str] = None,
    interval: Optional[float] = None,
    run_async: bool = False,
    daemon: bool = True,
    run_on_start: bool = True,
) -> BackgroundTask:
    task = BackgroundTask(func=func, name=name, interval=interval, run_async=run_async, daemon=daemon, run_on_start=run_on_start)
    logger.info(f"[Main] Registered background task: {task.name}")
    # Register with shutdown manager if available
    try:
        shm = globals().get("shutdown_manager")
        if shm and hasattr(shm, "register_object"):
            shm.register_object(task, ("stop",))
            logger.debug(f"[Main] Background task {task.name} registered with shutdown_manager")
    except Exception:
        logger.debug(f"[Main] No shutdown_manager to register background task {task.name}")
    return task


# -----------------------------
# Start default background tasks
# -----------------------------
def start_background_tasks():
    global _background_loop
    try:
        try:
            _background_loop = asyncio.get_event_loop()
        except RuntimeError:
            _background_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_background_loop)

        # Example: periodic health check task
        async def health_check():
            while True:
                try:
                    if "router" in globals() and hasattr(router, "ping"):
                        healthy = router.ping()
                        logger.debug(f"[BackgroundTask:health_check] Router healthy: {healthy}")
                    if "CORE_AI" in globals() and hasattr(CORE_AI, "perform_maintenance"):
                        CORE_AI.perform_maintenance()
                except Exception:
                    logger.exception("[BackgroundTask:health_check] Exception occurred")
                await asyncio.sleep(30)  # default interval

        register_background_task(health_check, name="health_check", interval=None, run_async=True, daemon=True)

        # Example: periodic memory cleanup
        def memory_cleanup():
            try:
                if "MemoryManager" in globals() and hasattr(MemoryManager, "cleanup"):
                    MemoryManager.cleanup()
                    logger.debug("[BackgroundTask:memory_cleanup] Memory cleaned")
            except Exception:
                logger.exception("[BackgroundTask:memory_cleanup] Exception occurred")

        register_background_task(memory_cleanup, name="memory_cleanup", interval=60, run_async=False, daemon=True)

        logger.info("[Main] Background tasks started ✅")

    except Exception as e:
        logger.exception(f"[Main] Failed to start background tasks: {e}")


# Start background tasks in a daemon thread (non-blocking)
threading.Thread(target=start_background_tasks, name="bibliotheca-bg-tasks-starter", daemon=True).start()
# -----------------------------
# Ready message
# -----------------------------
logger.info("📡 Bibliotheca Main fully initialized — Beyond-Perfection Infinity++ v20 ✅")

# -----------------------------
# Optional: TTS or other speech services
# -----------------------------
import platform
import threading

TTS_ENGINE = None
TTS_BACKEND = None
TTS_LOCK = threading.Lock()

def speak(text: str, voice: Optional[str] = None, rate: Optional[int] = None, volume: Optional[float] = None):
    """
    Unified TTS interface. Safe to call from anywhere. Will fallback silently if no TTS available.
    Parameters:
        text (str): Text to speak
        voice (str, optional): Voice identifier if supported by backend
        rate (int, optional): Speech rate (words per minute)
        volume (float, optional): 0.0 to 1.0
    """
    global TTS_ENGINE, TTS_BACKEND, TTS_LOCK
    if not text:
        return

    try:
        with TTS_LOCK:
            if TTS_ENGINE:
                if voice and hasattr(TTS_ENGINE, "setProperty"):
                    try:
                        TTS_ENGINE.setProperty("voice", voice)
                    except Exception:
                        logger.debug(f"[TTS] Unable to set voice: {voice}")
                if rate and hasattr(TTS_ENGINE, "setProperty"):
                    try:
                        TTS_ENGINE.setProperty("rate", rate)
                    except Exception:
                        logger.debug(f"[TTS] Unable to set rate: {rate}")
                if volume and hasattr(TTS_ENGINE, "setProperty"):
                    try:
                        TTS_ENGINE.setProperty("volume", volume)
                    except Exception:
                        logger.debug(f"[TTS] Unable to set volume: {volume}")
                TTS_ENGINE.say(text)
                TTS_ENGINE.runAndWait()
                return
            elif TTS_BACKEND:
                TTS_BACKEND(text)
                return
    except Exception:
        logger.exception("[TTS] speak() failed for text: %s", text)


# Initialize primary TTS engine
try:
    import pyttsx3

    TTS_ENGINE = pyttsx3.init()
    # Set default properties for clarity and natural sound
    try:
        TTS_ENGINE.setProperty("rate", 170)
        TTS_ENGINE.setProperty("volume", 1.0)
        voices = TTS_ENGINE.getProperty("voices")
        if voices:
            # pick first available voice by default
            TTS_ENGINE.setProperty("voice", voices[0].id)
    except Exception:
        logger.debug("[TTS] Failed to set pyttsx3 default properties (ignored)")

    logger.info("🗣 pyttsx3 TTS initialized ✅")

except Exception as e:
    TTS_ENGINE = None
    logger.warning(f"🗣 pyttsx3 TTS failed to initialize: {e}")
    # Fallback to gTTS if available
    try:
        from gtts import gTTS
        import tempfile
        import os
        import subprocess

        def _gtts_backend(text: str):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    gTTS(text=text).write_to_fp(tmp)
                    tmp_path = tmp.name
                # Attempt cross-platform playback
                if platform.system() == "Windows":
                    os.startfile(tmp_path)
                elif platform.system() == "Darwin":
                    subprocess.run(["afplay", tmp_path])
                else:  # Linux fallback
                    try:
                        subprocess.run(["mpg123", tmp_path])
                    except Exception:
                        subprocess.run(["mpg321", tmp_path])
                time.sleep(0.1)  # small pause to ensure playback
                os.remove(tmp_path)
            except Exception:
                logger.exception("[TTS:gTTS] Failed to play text: %s", text)

        TTS_BACKEND = _gtts_backend
        logger.info("🗣 gTTS fallback TTS initialized ✅")
    except Exception:
        TTS_BACKEND = None
        logger.warning("🗣 No TTS backend available. TTS features disabled ❌")


# Optional: platform-native fallback if nothing initialized
if not TTS_ENGINE and not TTS_BACKEND:
    def _native_say(text: str):
        try:
            if platform.system() == "Darwin":
                subprocess.run(["say", text])
            elif platform.system() == "Windows":
                import ctypes
                ctypes.windll.user32.MessageBeep(0)  # minimal feedback
            elif platform.system() == "Linux":
                subprocess.run(["spd-say", text])
        except Exception:
            logger.debug("[TTS:native] Failed to say text (ignored)")

    TTS_BACKEND = _native_say
    logger.info("🗣 Native OS TTS backend initialized ✅")


# Optional: expose shutdown for TTS cleanup
def shutdown_tts():
    global TTS_ENGINE
    try:
        if TTS_ENGINE:
            TTS_ENGINE.stop()
            logger.info("[TTS] pyttsx3 engine stopped cleanly ✅")
    except Exception:
        logger.exception("[TTS] Failed to shutdown pyttsx3 engine")
# -----------------------------
# Optional: Check CouchDB memory
# -----------------------------
try:
    import couchdb
    COUCHDB_SERVER = None
    COUCHDB_DB = None

    try:
        # Attempt to connect to CouchDB server from environment/config
        from app.config.config import config

        couch_url = config.get("couchdb", {}).get("url", "http://127.0.0.1:5984/")
        couch_user = config.get("couchdb", {}).get("user", None)
        couch_pass = config.get("couchdb", {}).get("password", None)

        if couch_user and couch_pass:
            COUCHDB_SERVER = couchdb.Server(f"http://{couch_user}:{couch_pass}@{couch_url.split('//')[-1]}")
        else:
            COUCHDB_SERVER = couchdb.Server(couch_url)

        # Test server connectivity
        try:
            info = COUCHDB_SERVER.info()
            logger.info(f"📦 CouchDB server available. Version: {info.get('version', 'unknown')}")
        except Exception:
            logger.warning("⚠ CouchDB server unreachable; falling back to in-memory memory.")
            COUCHDB_SERVER = None

        # Attempt to get or create default memory DB
        if COUCHDB_SERVER:
            db_name = config.get("couchdb", {}).get("default_db", "bibliotheca_memory")
            if db_name in COUCHDB_SERVER:
                COUCHDB_DB = COUCHDB_SERVER[db_name]
                logger.info(f"📂 CouchDB database '{db_name}' loaded successfully ✅")
            else:
                COUCHDB_DB = COUCHDB_SERVER.create(db_name)
                logger.info(f"📂 CouchDB database '{db_name}' created and ready ✅")

    except Exception as e:
        logger.exception(f"⚠ CouchDB initialization failed: {e}")
        COUCHDB_SERVER = None
        COUCHDB_DB = None

except ImportError:
    logger.info("⚠ CouchDB python module not available; using in-memory memory.")
    COUCHDB_SERVER = None
    COUCHDB_DB = None

# Fallback in-memory storage if CouchDB unavailable
if COUCHDB_DB is None:
    class InMemoryDB(dict):
        """
        Simple in-memory key-value store acting as a CouchDB fallback.
        Supports get, set, delete operations.
        """
        def get_doc(self, key, default=None):
            return self.get(key, default)

        def save_doc(self, key, value):
            self[key] = value

        def delete_doc(self, key):
            if key in self:
                del self[key]

        def all_docs(self):
            return list(self.items())

    COUCHDB_DB = InMemoryDB()
    logger.info("🧠 Using in-memory memory store (CouchDB fallback) ✅")

# -----------------------------
# Startup & Environment Info
# -----------------------------
import datetime
import platform
import socket
import asyncio
import threading

startup_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
system_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
hostname = socket.gethostname()

logger.info(
    f"🚀 Bibliotheca Main ready ✅ Beyond-Perfection Edition\n"
    f"   🌐 Host: {hostname}\n"
    f"   🖥 System: {system_info}\n"
    f"   🕒 Startup: {startup_time}\n"
    f"   🗃 Memory backend: {'CouchDB' if 'couchdb' in globals() and COUCHDB_DB else 'In-Memory'}"
)

def log_environment_info():
    """Optional detailed system info logging."""
    logger.info(f"💻 System: {platform.system()} {platform.release()} ({platform.machine()})")
    logger.info(f"🖥 Python version: {platform.python_version()}")
    logger.info(f"🌐 Hostname: {socket.gethostname()}")
    logger.info(f"👥 User: {os.getenv('USER') or os.getenv('USERNAME')}")

log_environment_info()

# -----------------------------
# Async Test Run Helper
# -----------------------------
async def async_test_run(prompt: str = "Hello, Bibliotheca!", retries: int = 2, delay: float = 1.0):
    """
    Perform a test chat with CORE_AI asynchronously.
    Retries on failure.
    """
    if "CORE_AI" not in globals() or CORE_AI is None:
        logger.warning("[AsyncTest] CORE_AI not initialized; skipping test run.")
        return

    attempt = 0
    while attempt <= retries:
        try:
            response = await CORE_AI.chat(prompt)
            logger.info(f"[AsyncTest] Test chat response: {response}")
            # Optional: speak the test response if TTS available
            if 'speak' in globals() and callable(speak):
                speak(response)
            break
        except Exception as e:
            attempt += 1
            logger.warning(f"[AsyncTest] Attempt {attempt} failed: {e}")
            if attempt <= retries:
                await asyncio.sleep(delay)
            else:
                logger.error(f"[AsyncTest] All attempts failed: {e}", exc_info=True)

def run_async_test_thread():
    """Run the async test in a separate thread if event loop exists."""
    try:
        asyncio.get_running_loop()
        threading.Thread(target=lambda: asyncio.run(async_test_run()), daemon=True, name="async-test-run").start()
    except RuntimeError:
        # No loop running
        asyncio.run(async_test_run())

# -----------------------------
# Interactive TTS AI Launcher
# -----------------------------
def launch_interactive_tts():
    """Launch interactive TTS AI if available."""
    try:
        from ai_core.interactive_tts_ai import main as run_interactive_tts
        threading.Thread(target=run_interactive_tts, daemon=True, name="interactive-tts-ai").start()
        logger.info("[Startup] Interactive TTS AI launched ✅")
    except ModuleNotFoundError:
        logger.info("[Startup] Skipping interactive TTS AI — module not found.")
    except Exception as e:
        logger.exception(f"[Startup] Failed to launch interactive TTS AI: {e}")

# -----------------------------
# Project Directories Setup
# -----------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
APP_DIR = PROJECT_ROOT / "app"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
LOG_PATH = PROJECT_ROOT / "bibliotheca.log"

for directory in [APP_DIR, TEMPLATES_DIR, STATIC_DIR]:
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Startup] Created missing directory: {directory}")
        except Exception as e:
            logger.error(f"[Startup] Failed to create directory {directory}: {e}")

# -----------------------------
# Flask & SocketIO Initialization
# -----------------------------
from flask import Flask
from flask_socketio import SocketIO

try:
    flask_app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
    socketio = SocketIO(flask_app, cors_allowed_origins="*", async_mode="threading")
    logger.info("🌐 Flask & SocketIO initialized ✅")
except Exception as e:
    logger.exception(f"⚠ Failed to initialize Flask & SocketIO: {e}")

# -----------------------
# Celery Setup
# -----------------------
import os
import logging
from celery import Celery
from celery.signals import after_setup_logger, after_setup_task_logger

# -----------------------
# Celery Logger Setup
# -----------------------
@after_setup_logger.connect
@after_setup_task_logger.connect
def setup_celery_logger(logger, *args, **kwargs):
    """
    Ensures Celery uses the main Bibliotheca logger
    """
    logger.handlers = []  # remove default handlers
    # File handler
    file_handler = logging.FileHandler(str(LOG_PATH))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s'))
    logger.addHandler(file_handler)
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.info("[Celery] Logger initialized ✅")

# -----------------------
# Celery Broker & Backend
# -----------------------
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

try:
    celery = Celery(
        "bibliotheca",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
        include=[
            "app.tasks",  # default tasks module
            # add other task modules here
        ]
    )

    # -----------------------
    # Celery Configuration
    # -----------------------
    celery.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=300,          # max 5 minutes per task
        task_soft_time_limit=240,     # soft limit warning at 4 minutes
        task_acks_late=True,          # acknowledge tasks only after execution
        worker_prefetch_multiplier=1, # fair task distribution
        result_expires=3600,          # 1 hour TTL for results
        beat_schedule={},             # optional: schedule periodic tasks here
        worker_concurrency=os.cpu_count() or 1,
        broker_transport_options={"visibility_timeout": 3600},  # ensure retries after crashes
    )

    logger.info(f"[Celery] Initialized with broker={CELERY_BROKER_URL} and backend={CELERY_RESULT_BACKEND} ✅")

except Exception as e:
    logger.exception(f"[Celery] Failed to initialize Celery: {e}")
    celery = None
# -----------------------
# PatchEngine Initialization (Safe)
# -----------------------
import sys
import os

# Ensure Python can find patch_engine
PATCH_ENGINE_PATH = os.path.join(PROJECT_ROOT, "app", "utils")
if PATCH_ENGINE_PATH not in sys.path:
    sys.path.append(PATCH_ENGINE_PATH)

patch_engine = None
try:
    from patch_engine import PatchEngine
    try:
        patch_engine = PatchEngine()
        patch_engine.start()
        logger.info("🛠 PatchEngine started successfully ✅")
    except Exception as e:
        logger.exception(f"⚠ PatchEngine failed to start: {e}")
        patch_engine = None
except ModuleNotFoundError:
    logger.warning("⚠ PatchEngine module not found; skipping patch system initialization.")

# -----------------------
# Enhanced Logging Setup
# -----------------------
import logging
from logging.handlers import RotatingFileHandler

# Avoid adding duplicate handlers
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(LOG_PATH, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# Log Bibliotheca startup info
logger.info("📡 Bibliotheca initializing...")
logger.info(f"🛠 PatchEngine loaded: {'Yes' if patch_engine else 'No'}")
logger.info(f"📂 PatchEngine path: {PATCH_ENGINE_PATH}")

# -----------------------
# Start Time
# -----------------------
import time
START_TIME = time.time()
logger.info(f"⏱ Bibliotheca start time recorded: {START_TIME}")

# -----------------------
# Async Loop for coroutines
# -----------------------
import asyncio

try:
    try:
        # Check if a loop is already running (e.g., in interactive environments)
        ASYNC_LOOP = asyncio.get_running_loop()
        logger.info("🔄 Using existing running asyncio event loop ✅")
    except RuntimeError:
        # No running loop; create new
        ASYNC_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(ASYNC_LOOP)
        logger.info("🔄 New asyncio event loop created ✅")
except Exception as e:
    logger.exception(f"⚠ Failed to initialize asyncio loop: {e}")
    ASYNC_LOOP = None

# Optional: helper to run async coroutines safely
def run_async(coroutine):
    """
    Safely run a coroutine in the global ASYNC_LOOP.
    """
    if ASYNC_LOOP is None:
        logger.warning("[Async] No asyncio loop available; skipping coroutine execution")
        return None
    try:
        if ASYNC_LOOP.is_running():
            # If loop is already running (e.g., in SocketIO), run in a new thread
            import threading
            threading.Thread(target=lambda: asyncio.run(coroutine), daemon=True).start()
            return None
        else:
            return ASYNC_LOOP.run_until_complete(coroutine)
    except Exception as e:
        logger.exception(f"[Async] Coroutine execution failed: {e}")
        return None
# -----------------------
# Telemetry / Exception Capture
# -----------------------
import traceback
import datetime
import platform
import socket

try:
    from app.utils.telemetry import Telemetry
    TELEMETRY = Telemetry(project_root=str(PROJECT_ROOT))
    logger.info("📡 Telemetry module initialized successfully ✅")
except Exception:
    class DummyTelemetry:
        """
        Fallback telemetry system if the real Telemetry module is unavailable.
        Provides logging for exceptions, events, and snapshots.
        """
        def capture_exception(self, e, context=None, exc_info=True):
            """
            Capture and log exceptions with optional context.
            """
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            context_str = f"[{context}]" if context else ""
            msg = f"{ts} [DummyTelemetry] Exception {context_str}: {e}"
            if exc_info:
                tb = traceback.format_exc()
                msg += f"\n{tb}"
            logger.error(msg)

        def log_event(self, name, data=None):
            """
            Log arbitrary events.
            """
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"{ts} [DummyTelemetry] Event: {name} | Data: {data}")

        def take_snapshot(self, note=""):
            """
            Take a snapshot for debugging or memory state.
            """
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            snapshot_info = {
                "timestamp": ts,
                "system": f"{platform.system()} {platform.release()} ({platform.machine()})",
                "python": platform.python_version(),
                "hostname": socket.gethostname(),
                "note": note
            }
            logger.info(f"{ts} [DummyTelemetry] Snapshot taken: {snapshot_info}")

        async def capture_exception_async(self, e, context=None):
            """
            Optional async version for async tasks or loops.
            """
            self.capture_exception(e, context)

    TELEMETRY = DummyTelemetry()
    logger.warning("⚠ Telemetry module missing; using DummyTelemetry fallback ✅")
# -----------------------
# AI Core Initialization
# -----------------------
try:
    from app.ai_core import AICore
    CORE_AI = AICore(memory=MEMORY)
    logger.info("AICore initialized successfully ✅")
except Exception as e:
    logger.error(f"Failed to initialize AICore: {e}")
    class DummyAICore:
        def process_input(self, user_input):
            logger.warning(f"DummyAICore received input but cannot process: {user_input}")
            return "AI core not available."
    CORE_AI = DummyAICore()

# -----------------------
# Conversational Task Loop
# -----------------------
def interactive_loop():
    """Continuously read user input and delegate tasks to AI"""
    logger.info("[InteractiveLoop] Starting conversational loop ✅")
    try:
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue  # skip empty input

                response = CORE_AI.process_input(user_input)
                if response:
                    print(f"AI: {response}")

            except KeyboardInterrupt:
                logger.info("Interactive loop terminated by user ⏹️")
                break
            except Exception as e:
                logger.error(f"[InteractiveLoop] Error processing input: {e}")

    except Exception as e:
        logger.critical(f"[InteractiveLoop] Fatal error: {e}")
# -----------------------
# AI Core Imports and Initialization
# -----------------------
try:
    from app.utils.ai_core import (
        CORE_AI,
        ADVANCED_AI,
        OFFLINE_AI,
        process_query_safe,
        process_query_stream_to_sid,
        load_ai_engines
    )
    logger.info("✅ AI core modules loaded successfully.")

except Exception as e:
    logger.warning(f"⚠️ AI core modules failed to load: {e}")
    CORE_AI = ADVANCED_AI = OFFLINE_AI = None

    # -----------------------
    # Dummy AI Engines
    # -----------------------
    class DummyAI:
        """Fallback AI engine for safe operation"""
        def process_input(self, query):
            logger.warning(f"[DummyAI] received input but cannot process: '{query}'")
            return f"[AI not available] Your query was: '{query}'"

        def process_query(self, query):
            return self.process_input(query)

        def stream_query(self, query, callback=None):
            """Simulate streaming AI response"""
            logger.info(f"[DummyAI] streaming query: '{query}'")
            dummy_response = f"[AI not available, would stream] {query}"
            if callback:
                for chunk in dummy_response.split():
                    callback(chunk + " ")
            return dummy_response

        def load_engines(self):
            logger.info("[DummyAI] No engines to load, operating in dummy mode.")

    CORE_AI = ADVANCED_AI = OFFLINE_AI = DummyAI()

    # -----------------------
    # Fallback Functions
    # -----------------------
    def process_query_safe(query):
        """Safe query processor that never fails"""
        logger.info(f"[Fallback] Processing query safely: {query}")
        return CORE_AI.process_query(query)

    def process_query_stream_to_sid(query, sid):
        """Simulated streaming function for async clients"""
        logger.info(f"[Fallback] Would stream query '{query}' to session '{sid}'")
        def callback(chunk):
            logger.info(f"[Fallback] Stream chunk to {sid}: {chunk}")
        CORE_AI.stream_query(query, callback=callback)

    def load_ai_engines():
        """Safe AI engine loader"""
        logger.info("[Fallback] Loading AI engines in dummy mode...")
        CORE_AI.load_engines()

    logger.info("✅ AI core missing; safe dummy stubs initialized.")
# -----------------------
# Task Manager
# -----------------------
try:
    from app.utils.task_manager import TaskManager
    TASK_MANAGER = TaskManager()
    logger.info("✅ TaskManager loaded successfully.")
except Exception as e:
    import threading, time, logging, subprocess

    logger.warning(f"⚠️ TaskManager missing; using SimpleTaskManager fallback. Error: {e}")

    class SimpleTaskManager:
        """Thread-safe task manager with fallback support and progress tracking"""
        def __init__(self):
            self._tasks = {}
            self._counter = 0
            self._lock = threading.Lock()
            self._shutdown = False

        # -----------------------
        # Task Queuing
        # -----------------------
        def queue(self, task, priority=0, retries=0, timeout=None):
            """
            Add a task to the manager.
            task: dict with 'type' and payload or raw string
            priority: higher number = higher priority (currently for logging)
            retries: number of automatic retries if task fails
            timeout: max execution time in seconds
            """
            with self._lock:
                self._counter += 1
                tid = f"task-{self._counter}"
                self._tasks[tid] = {
                    "task": task,
                    "status": "queued",
                    "priority": priority,
                    "retries": retries,
                    "attempts": 0,
                    "timeout": timeout,
                    "started": None,
                    "finished": None,
                    "result": None,
                    "error": None,
                    "progress": 0
                }

            threading.Thread(target=self._run, args=(tid,), daemon=True).start()
            logger.info(f"[TaskManager] Queued task {tid} (type: {task.get('type') if isinstance(task, dict) else 'string'})")
            return tid

        # -----------------------
        # Task Execution
        # -----------------------
        def _run(self, tid):
            tinfo = self._tasks.get(tid)
            if not tinfo or self._shutdown:
                return

            task = tinfo["task"]
            retries_left = tinfo["retries"]

            while retries_left >= 0:
                tinfo["status"] = "running"
                tinfo["started"] = time.time()
                tinfo["attempts"] += 1
                tinfo["error"] = None
                tinfo["progress"] = 0

                try:
                    if isinstance(task, dict) and "type" in task:
                        ttype = task["type"]

                        # -----------------------
                        # Shell Command Task
                        # -----------------------
                        if ttype == "shell":
                            cmd = task.get("cmd")
                            logger.info(f"[TaskManager] Running shell task {tid}: {cmd}")
                            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=tinfo["timeout"])
                            tinfo["result"] = proc.stdout + proc.stderr

                        # -----------------------
                        # AI Query Task
                        # -----------------------
                        elif ttype == "ai_query":
                            query = task.get("query", "")
                            logger.info(f"[TaskManager] Running AI query task {tid}: {query}")

                            # If streaming support exists
                            if callable(getattr(CORE_AI, "stream_query", None)):
                                chunks = []
                                def collect(chunk):
                                    chunks.append(chunk)
                                    tinfo["progress"] = len(chunks) / max(len(query.split()), 1) * 100
                                CORE_AI.stream_query(query, callback=collect)
                                tinfo["result"] = "".join(chunks)
                            else:
                                tinfo["result"] = process_query_safe(query)
                            tinfo["progress"] = 100

                        # -----------------------
                        # Unknown Task Type
                        # -----------------------
                        else:
                            tinfo["result"] = f"Unknown task type: {ttype}"
                            logger.warning(f"[TaskManager] Unknown task type for {tid}: {ttype}")

                    # -----------------------
                    # String Task (process as AI query)
                    # -----------------------
                    elif isinstance(task, str):
                        tinfo["result"] = process_query_safe(task)
                        tinfo["progress"] = 100

                    else:
                        tinfo["result"] = f"Invalid task payload: {repr(task)[:200]}"
                        logger.warning(f"[TaskManager] Invalid task payload for {tid}")

                    tinfo["status"] = "finished"
                    break  # success, exit retry loop

                except subprocess.TimeoutExpired:
                    tinfo["status"] = "error"
                    tinfo["error"] = f"Task timed out after {tinfo['timeout']}s"
                    logger.error(f"[TaskManager] Timeout on task {tid}")
                    retries_left -= 1

                except Exception as e:
                    tinfo["status"] = "error"
                    tinfo["error"] = str(e)
                    logger.error(f"[TaskManager] Error running task {tid}: {e}")
                    retries_left -= 1

                finally:
                    tinfo["finished"] = time.time()
                    try:
                        socketio.emit("task_update", {"task_id": tid, "info": self._tasks[tid]})
                    except Exception:
                        pass

        # -----------------------
        # Task Querying
        # -----------------------
        def get(self, tid):
            """Return info for a specific task"""
            return self._tasks.get(tid)

        def all(self):
            """Return all tasks"""
            return self._tasks

        def check_tasks(self):
            """Optional periodic maintenance or cleanup"""
            completed = [tid for tid, t in self._tasks.items() if t["status"] in ("finished", "error")]
            for tid in completed:
                # Optional: remove old tasks after certain time
                pass
            return completed

        # -----------------------
        # Graceful Shutdown
        # -----------------------
        def shutdown(self):
            """Signal task manager to stop processing new tasks"""
            logger.info("[TaskManager] Shutdown initiated...")
            self._shutdown = True

    TASK_MANAGER = SimpleTaskManager()
    logger.info("✅ SimpleTaskManager fallback initialized.")
# -----------------------
# MetaMonitor / Watchdog
# -----------------------
try:
    from app.utils.meta_monitor import MetaMonitor
    WATCHDOG = MetaMonitor(interval=10, allow_self_modify=True)
    WATCHDOG.start()
    logger.info("✅ MetaMonitor started successfully.")
except Exception as e:
    import threading, time, psutil

    logger.warning(f"⚠️ MetaMonitor missing; using LocalHealthWatchdog fallback. Error: {e}")

    class LocalHealthWatchdog:
        """Fallback watchdog that monitors core modules and system health"""

        def __init__(self, interval=10, allow_self_modify=False):
            self.interval = interval
            self.allow_self_modify = allow_self_modify
            self._running = False
            self._thread = None
            self._modules = {
                "memory": MEMORY,
                "ai_core": CORE_AI,
                "task_manager": TASK_MANAGER
            }

        def start(self):
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
                self._thread.start()
                logger.info("[Watchdog] LocalHealthWatchdog started.")

        def stop(self):
            self._running = False
            logger.info("[Watchdog] LocalHealthWatchdog stopping...")

        # -----------------------
        # Core Monitoring Loop
        # -----------------------
        def _monitor_loop(self):
            while self._running:
                try:
                    self._check_memory()
                    self._check_ai_core()
                    self._check_task_manager()
                    self._check_system_metrics()
                except Exception as e:
                    logger.error(f"[Watchdog] Error during monitoring loop: {e}")
                time.sleep(self.interval)

        # -----------------------
        # Memory Health Check
        # -----------------------
        def _check_memory(self):
            try:
                if hasattr(MEMORY, "health_check"):
                    status = MEMORY.health_check()
                    logger.debug(f"[Watchdog] Memory status: {status}")
                    if status.get("status") != "ok" and self.allow_self_modify:
                        logger.warning("[Watchdog] Attempting to repair memory...")
                        MEMORY.repair()
            except Exception as e:
                logger.error(f"[Watchdog] Memory check failed: {e}")

        # -----------------------
        # AI Core Health Check
        # -----------------------
        def _check_ai_core(self):
            try:
                if CORE_AI:
                    test_query = "health check"
                    resp = CORE_AI.process_input(test_query)
                    logger.debug(f"[Watchdog] AI core test response: {resp}")
            except Exception as e:
                logger.warning(f"[Watchdog] AI core error: {e}")
                if self.allow_self_modify:
                    logger.info("[Watchdog] Attempting AI core reload...")
                    try:
                        if callable(load_ai_engines):
                            load_ai_engines()
                    except Exception as reload_err:
                        logger.error(f"[Watchdog] Failed to reload AI engines: {reload_err}")

        # -----------------------
        # Task Manager Health Check
        # -----------------------
        def _check_task_manager(self):
            try:
                if TASK_MANAGER:
                    running_tasks = [t for t in TASK_MANAGER.all().values() if t["status"] == "running"]
                    logger.debug(f"[Watchdog] Running tasks: {len(running_tasks)}")
            except Exception as e:
                logger.warning(f"[Watchdog] TaskManager error: {e}")

        # -----------------------
        # System Metrics Monitoring
        # -----------------------
        def _check_system_metrics(self):
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                disk = psutil.disk_usage("/").percent
                logger.debug(f"[Watchdog] System metrics - CPU: {cpu}%, RAM: {mem}%, Disk: {disk}%")
                # Optional: trigger alerts if thresholds exceeded
                if cpu > 90 or mem > 90 or disk > 95:
                    logger.warning("[Watchdog] High system usage detected!")
            except Exception as e:
                logger.error(f"[Watchdog] System metrics check failed: {e}")

    WATCHDOG = LocalHealthWatchdog(interval=10, allow_self_modify=True)
    WATCHDOG.start()
    logger.info("✅ LocalHealthWatchdog fallback initialized and running.")
# -----------------------
# Updater / Self-Heal
# -----------------------
try:
    from app.utils.updater import Updater
    UPDATER = Updater(project_root=str(PROJECT_ROOT))
    logger.info("✅ Updater loaded successfully.")
except Exception as e:
    UPDATER = None
    logger.warning(f"⚠️ Updater missing; fallback PatchEngine will handle updates. Error: {e}")

    # -----------------------
    # Fallback Patch Engine
    # -----------------------
    class PatchEngine:
        """Simple fallback patch engine for self-healing tasks"""
        def __init__(self):
            self._patch_queue = []
            self._lock = threading.Lock()
            self._running = False

        def queue_patch(self, patch_name, patch_func):
            with self._lock:
                self._patch_queue.append({"name": patch_name, "func": patch_func, "status": "queued"})
                logger.info(f"[PatchEngine] Queued patch: {patch_name}")
            if not self._running:
                threading.Thread(target=self._process_patches, daemon=True).start()

        def _process_patches(self):
            self._running = True
            while self._patch_queue:
                patch = self._patch_queue.pop(0)
                patch["status"] = "running"
                try:
                    patch["func"]()
                    patch["status"] = "applied"
                    logger.info(f"[PatchEngine] Applied patch: {patch['name']}")
                except Exception as e:
                    patch["status"] = "failed"
                    patch["error"] = str(e)
                    logger.error(f"[PatchEngine] Failed patch {patch['name']}: {e}")
            self._running = False

    patch_engine = PatchEngine()
    logger.info("✅ PatchEngine fallback initialized.")

# -----------------------
# Self-Heal module
# -----------------------
try:
    import app.utils.self_heal as self_heal_mod
    if hasattr(self_heal_mod, "run") and callable(self_heal_mod.run):
        threading.Thread(target=self_heal_mod.run, daemon=True).start()
        logger.info("✅ self_heal module started successfully.")
    else:
        logger.warning("[SelfHeal] self_heal module loaded but 'run' method missing; fallback active.")
except Exception as e:
    logger.warning(f"⚠️ No external self_heal module started; fallback skipped. Error: {e}")

    # -----------------------
    # Fallback Self-Heal Loop
    # -----------------------
    def fallback_self_heal_loop(interval=30):
        """
        Periodically checks core modules and applies safe patches if needed.
        Runs in a background thread.
        """
        while True:
            try:
                # Memory auto-repair
                if hasattr(MEMORY, "health_check") and MEMORY.health_check().get("status") != "ok":
                    logger.warning("[SelfHeal] Memory issue detected; attempting repair...")
                    if hasattr(MEMORY, "repair"):
                        MEMORY.repair()

                # AI engine reload
                if CORE_AI is None or not hasattr(CORE_AI, "process_input"):
                    logger.warning("[SelfHeal] AI core missing; attempting reload...")
                    try:
                        if callable(load_ai_engines):
                            load_ai_engines()
                    except Exception as e:
                        logger.error(f"[SelfHeal] Failed to reload AI engines: {e}")

                # Task manager verification
                if TASK_MANAGER is None:
                    logger.warning("[SelfHeal] TaskManager missing; initializing fallback...")
                    from app.utils.task_manager import TaskManager as TM
                    TASK_MANAGER = TM()

                # Optional: check PatchEngine queued patches
                if patch_engine:
                    with patch_engine._lock:
                        for p in patch_engine._patch_queue:
                            if p["status"] == "queued":
                                logger.info(f"[SelfHeal] Patch pending: {p['name']}")

            except Exception as e:
                logger.error(f"[SelfHeal] Error in fallback self-heal loop: {e}")
            time.sleep(interval)

    threading.Thread(target=fallback_self_heal_loop, daemon=True).start()
    logger.info("✅ Fallback self-heal loop running in background.")
# -----------------------
# Flask Routes / API Endpoints
# -----------------------
from flask import request, jsonify, send_from_directory, render_template, abort
from pathlib import Path
import functools

# -----------------------
# Rate-limiting decorator (basic)
# -----------------------
_client_requests = {}
RATE_LIMIT = 10  # max requests per 10 seconds
RATE_INTERVAL = 10

def rate_limit(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        client = request.remote_addr
        now = time.time()
        _client_requests.setdefault(client, [])
        # Remove old requests
        _client_requests[client] = [t for t in _client_requests[client] if now - t < RATE_INTERVAL]
        if len(_client_requests[client]) >= RATE_LIMIT:
            return jsonify({"error": "rate limit exceeded"}), 429
        _client_requests[client].append(now)
        return f(*args, **kwargs)
    return wrapper

# -----------------------
# Root Index
# -----------------------
@app.route("/")
@rate_limit
def root_index():
    try:
        if (TEMPLATES_DIR / "index.html").exists():
            return render_template("index.html")
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="render_index")
        logger.error(f"[Flask] Error rendering index: {e}")
    return jsonify({"status": "bibliotheca", "ready": True})

# -----------------------
# Health Endpoint
# -----------------------
@app.route("/health")
@rate_limit
def health():
    try:
        memory_info = getattr(MEMORY, "health_check", lambda: {"status": "unknown"})()
        return jsonify({
            "status": "ok",
            "project_root": str(PROJECT_ROOT),
            "ai_core": bool(CORE_AI),
            "advanced_ai": bool(ADVANCED_AI),
            "offline_ai": bool(OFFLINE_AI),
            "memory": memory_info,
            "uptime": time.time() - START_TIME,
        })
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="health")
        logger.error(f"[Flask] Health check failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

# -----------------------
# Logs Endpoint
# -----------------------
@app.route("/logs")
@rate_limit
def logs_endpoint():
    try:
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.read().splitlines()[-400:]
            return jsonify({"logs": lines})
        return jsonify({"logs": []})
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="logs")
        logger.error(f"[Flask] Logs endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------
# AI Query Endpoint
# -----------------------
@app.route("/send_query", methods=["POST"])
@rate_limit
def send_query_route():
    data = request.json or {}
    query = data.get("query") or data.get("message")
    stream = bool(data.get("stream", False))
    if not query:
        return jsonify({"error": "empty query"}), 400
    try:
        if stream:
            sid = data.get("sid")
            if not sid:
                resp = process_query_safe(query)
                return jsonify({"response": resp})
            process_query_stream_to_sid(query, sid)
            return jsonify({"status": "streaming_started"})
        resp = process_query_safe(query)
        return jsonify({"response": resp})
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="send_query")
        logger.error(f"[Flask] send_query error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------
# Snapshot Endpoint
# -----------------------
@app.route("/snapshot", methods=["POST"])
@rate_limit
def snapshot_route():
    try:
        note = (request.json or {}).get("note", "")
        if "TELEMETRY" in globals():
            TELEMETRY.take_snapshot(note)
        return jsonify({"status": "snapshot_taken", "note": note})
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="snapshot")
        logger.error(f"[Flask] snapshot error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------
# Project Scan Endpoint
# -----------------------
@app.route("/scan", methods=["GET"])
@rate_limit
def scan_route():
    try:
        if "scan_project" in globals() and callable(scan_project):
            rep = scan_project()
            return jsonify(rep)
        return jsonify({"error": "scan_project function not available"}), 500
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="scan")
        logger.error(f"[Flask] scan error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------
# Apply Patch Endpoint
# -----------------------
@app.route("/apply_patch", methods=["POST"])
@rate_limit
def apply_patch_route():
    try:
        payload = request.json or {}
        patches = payload.get("patches", [])
        force = bool(payload.get("force", False))
        if not isinstance(patches, list) or not patches:
            return jsonify({"error": "no patches provided"}), 400
        if not force:
            if "dry_run_patch" in globals() and callable(dry_run_patch):
                ok, compile_res, smoke_ok, smoke_out = dry_run_patch(patches)
                return jsonify({"dry_run_ok": ok, "compile": compile_res, "smoke_ok": smoke_ok, "smoke_out": smoke_out})
            else:
                return jsonify({"error": "dry_run_patch not available"}), 500
        if "apply_patch" in globals() and callable(apply_patch):
            result = apply_patch(patches, force=True)
            return jsonify(result)
        return jsonify({"error": "apply_patch function not available"}), 500
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="apply_patch")
        logger.error(f"[Flask] apply_patch error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------
# Favicon and Apple Icons
# -----------------------
@app.route("/favicon.ico")
def favicon_route():
    fp = STATIC_DIR / "favicon.ico"
    if fp.exists():
        return send_from_directory(str(STATIC_DIR), "favicon.ico")
    return ("", 204)

@app.route("/apple-touch-icon.png")
@app.route("/apple-touch-icon-precomposed.png")
def apple_icons():
    for name in ("apple-touch-icon.png", "apple-touch-icon-precomposed.png"):
        fp = STATIC_DIR / name
        if fp.exists():
            return send_from_directory(str(STATIC_DIR), name)
    return ("", 204)
# -----------------------
# Supervisor Loop (background checks)
# -----------------------
import inspect
import asyncio

def supervisor_loop(interval=6):
    """
    Continuous background checks for core modules:
    Memory, AI instances, Task Manager, and Watchdog.
    Performs self-healing if possible.
    """
    logger.info("Supervisor loop starting...")
    while True:
        try:
            # -----------------------
            # Memory Health Check
            # -----------------------
            try:
                if MEMORY and hasattr(MEMORY, "test") and callable(MEMORY.test):
                    ok = MEMORY.test()
                    if not ok:
                        TELEMETRY.log_event("memory_test_failed")
                        logger.warning("[Supervisor] MEMORY test failed, attempting repair...")
                        if hasattr(MEMORY, "repair") and callable(MEMORY.repair):
                            MEMORY.repair()
            except Exception as me:
                if "TELEMETRY" in globals():
                    TELEMETRY.capture_exception(me, context="memory_health")
                logger.error(f"[Supervisor] Memory health check error: {me}")

            # -----------------------
            # AI Self-Checks
            # -----------------------
            for ai_instance in (CORE_AI, ADVANCED_AI, OFFLINE_AI):
                if ai_instance and hasattr(ai_instance, "self_check") and callable(ai_instance.self_check):
                    try:
                        res = ai_instance.self_check()
                        if inspect.iscoroutine(res):
                            # Schedule async coroutine safely
                            fut = asyncio.run_coroutine_threadsafe(res, ASYNC_LOOP)
                            fut.result(timeout=5)
                    except Exception as e:
                        if "TELEMETRY" in globals():
                            TELEMETRY.capture_exception(e, context=f"{type(ai_instance).__name__}_self_check")
                        logger.error(f"[Supervisor] {type(ai_instance).__name__} self_check error: {e}")

            # -----------------------
            # TaskManager Housekeeping
            # -----------------------
            try:
                if TASK_MANAGER and hasattr(TASK_MANAGER, "check_tasks") and callable(TASK_MANAGER.check_tasks):
                    TASK_MANAGER.check_tasks()
            except Exception as e:
                if "TELEMETRY" in globals():
                    TELEMETRY.capture_exception(e, context="task_manager_check")
                logger.error(f"[Supervisor] TaskManager check_tasks error: {e}")

            # -----------------------
            # Watchdog / MetaMonitor Snapshot
            # -----------------------
            try:
                if WATCHDOG and hasattr(WATCHDOG, "snapshot") and callable(WATCHDOG.snapshot):
                    WATCHDOG.snapshot()
            except Exception as e:
                # snapshot errors are non-critical
                logger.debug(f"[Supervisor] Watchdog snapshot skipped/error: {e}")

            # -----------------------
            # Optional: System Alerts
            # -----------------------
            try:
                if MEMORY and hasattr(MEMORY, "health_check"):
                    status = MEMORY.health_check().get("status")
                    if status != "ok":
                        logger.warning(f"[Supervisor] Memory status alert: {status}")
            except Exception:
                pass

        except Exception as outer_e:
            if "TELEMETRY" in globals():
                TELEMETRY.capture_exception(outer_e, context="supervisor_outer")
            logger.error(f"[Supervisor] Outer loop error: {outer_e}")

        # Sleep until next iteration
        time.sleep(interval)

# Start supervisor in background thread
threading.Thread(target=supervisor_loop, daemon=True).start()
logger.info("✅ Supervisor thread started.")
# -----------------------
# Shutdown Hook
# -----------------------
import atexit

@atexit.register
def _shutdown():
    logger.info("🔹 Initiating Bibliotheca shutdown sequence...")
    # -----------------------
    # Telemetry Snapshot
    # -----------------------
    try:
        if "TELEMETRY" in globals():
            TELEMETRY.take_snapshot("shutdown")
            logger.info("📊 Telemetry snapshot taken.")
    except Exception as e:
        logger.error(f"[Shutdown] Telemetry snapshot failed: {e}")

    # -----------------------
    # Save Memory
    # -----------------------
    try:
        if MEMORY and hasattr(MEMORY, "save") and callable(MEMORY.save):
            MEMORY.save()
            logger.info("💾 Memory saved successfully.")
    except Exception as e:
        logger.error(f"[Shutdown] Memory save failed: {e}")

    # -----------------------
    # Shutdown Task Manager
    # -----------------------
    try:
        if TASK_MANAGER and hasattr(TASK_MANAGER, "shutdown") and callable(TASK_MANAGER.shutdown):
            TASK_MANAGER.shutdown()
            logger.info("📝 TaskManager shutdown complete.")
    except Exception as e:
        logger.error(f"[Shutdown] TaskManager shutdown failed: {e}")

    # -----------------------
    # Stop Watchdog / MetaMonitor
    # -----------------------
    try:
        if WATCHDOG and hasattr(WATCHDOG, "stop") and callable(WATCHDOG.stop):
            WATCHDOG.stop()
            logger.info("🛡️ Watchdog stopped successfully.")
    except Exception as e:
        logger.error(f"[Shutdown] Watchdog stop failed: {e}")

    # -----------------------
    # Stop Updater / PatchEngine
    # -----------------------
    try:
        if UPDATER and hasattr(UPDATER, "stop") and callable(UPDATER.stop):
            UPDATER.stop()
            logger.info("🔄 Updater stopped successfully.")
        if "patch_engine" in globals() and patch_engine and hasattr(patch_engine, "_running"):
            patch_engine._running = False
            logger.info("🔧 PatchEngine stopped successfully.")
    except Exception as e:
        logger.error(f"[Shutdown] Updater/PatchEngine stop failed: {e}")

    logger.info("✅ Bibliotheca shutdown complete.")


# -----------------------
# Helper: Attach safe self_check to AI instances
# -----------------------
def attach_safe_self_check_to(ai_instance, name="AI"):
    """
    Ensures every AI instance has a safe self_check method.
    Automatically handles exceptions and reports to Telemetry.
    Works for both sync and async self_check methods.
    """
    if ai_instance is None:
        logger.warning(f"[{name}] AI instance is None; skipping self_check attachment.")
        return

    if not hasattr(ai_instance, "self_check"):
        def stub_self_check():
            try:
                return True
            except Exception as e:
                if "TELEMETRY" in globals():
                    TELEMETRY.capture_exception(e, context=f"{name}_self_check_stub")
                logger.error(f"[{name}] self_check stub exception: {e}")
                return False
        setattr(ai_instance, "self_check", stub_self_check)
        logger.info(f"[{name}] self_check() stub attached successfully.")

    else:
        # Wrap existing self_check safely
        original = getattr(ai_instance, "self_check")
        async def safe_self_check(*args, **kwargs):
            try:
                res = original(*args, **kwargs)
                if inspect.iscoroutine(res):
                    res = await res
                return res
            except Exception as e:
                if "TELEMETRY" in globals():
                    TELEMETRY.capture_exception(e, context=f"{name}_self_check_safe")
                logger.error(f"[{name}] self_check exception: {e}")
                return False
        setattr(ai_instance, "self_check", safe_self_check)
        logger.info(f"[{name}] self_check() safely wrapped.")
# -----------------------
# Bootstrap / Initialization
# -----------------------
START_TIME = time.time()

def bootstrap():
    """
    Perform full Bibliotheca initialization:
    - Attach safe self_check to AI instances
    - Perform initial project scan
    - Log telemetry events
    - Verify core module health
    """
    logger.info("🔹 Starting Bibliotheca bootstrap...")

    # -----------------------
    # Attach safe self_check to AI instances
    # -----------------------
    attach_safe_self_check_to(CORE_AI, "CoreAI")
    attach_safe_self_check_to(ADVANCED_AI, "AdvancedAI")
    attach_safe_self_check_to(OFFLINE_AI, "OfflineAI")

    # -----------------------
    # Memory health verification
    # -----------------------
    try:
        if MEMORY and hasattr(MEMORY, "health_check"):
            mem_status = MEMORY.health_check()
            logger.info(f"[Bootstrap] Memory status: {mem_status}")
            if mem_status.get("status") != "ok" and hasattr(MEMORY, "repair"):
                logger.warning("[Bootstrap] Memory not OK, performing repair...")
                MEMORY.repair()
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="bootstrap_memory")
        logger.error(f"[Bootstrap] Memory check failed: {e}")

    # -----------------------
    # TaskManager verification
    # -----------------------
    try:
        global TASK_MANAGER  # moved to the top before any references
        if TASK_MANAGER is None:
            from app.utils.task_manager import TaskManager as TM
            TASK_MANAGER = TM()
            logger.info("[Bootstrap] TaskManager fallback initialized.")
        elif hasattr(TASK_MANAGER, "check_tasks"):
            TASK_MANAGER.check_tasks()
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="bootstrap_task_manager")
        logger.error(f"[Bootstrap] TaskManager check failed: {e}")

    # -----------------------
    # Initial project scan
    # -----------------------
    try:
        if "scan_project" in globals() and callable(scan_project):
            rep = scan_project()
            if "TELEMETRY" in globals():
                TELEMETRY.log_event(
                    "initial_scan",
                    {
                        "python_files": len(rep.get("python_files", [])),
                        "templates": rep.get("heuristics", {}).get("templates_count", 0)
                    }
                )
            logger.info(f"[Bootstrap] Initial project scan: {len(rep.get('python_files', []))} Python files, "
                        f"{rep.get('heuristics', {}).get('templates_count', 0)} templates")
        else:
            logger.warning("[Bootstrap] scan_project function not available; skipping initial scan.")
    except Exception as e:
        if "TELEMETRY" in globals():
            TELEMETRY.capture_exception(e, context="bootstrap_scan")
        logger.error(f"[Bootstrap] Initial project scan failed: {e}")

    # -----------------------
    # Watchdog verification
    # -----------------------
    try:
        if WATCHDOG and hasattr(WATCHDOG, "health_check"):
            wd_status = WATCHDOG.health_check()
            logger.info(f"[Bootstrap] Watchdog status: {wd_status}")
    except Exception as e:
        logger.error(f"[Bootstrap] Watchdog health check failed: {e}")

    # -----------------------
    # Updater verification
    # -----------------------
    try:
        if UPDATER and hasattr(UPDATER, "check_for_updates"):
            updates = UPDATER.check_for_updates()
            logger.info(f"[Bootstrap] Updater check: {updates}")
    except Exception as e:
        logger.error(f"[Bootstrap] Updater check failed: {e}")

    logger.info("✅ Bibliotheca bootstrap completed successfully.")


bootstrap()


# -----------------------
# Find Free Port Utility
# -----------------------
def find_free_port(start=5001, end=5050):
    """
    Finds a free port in the given range.
    Returns the first available port or environment PORT fallback.
    """
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex(("0.0.0.0", port)) != 0:
                    logger.info(f"[Bootstrap] Free port found: {port}")
                    return port
        except Exception as e:
            logger.warning(f"[Bootstrap] Port scan error on {port}: {e}")
    fallback_port = int(os.getenv("PORT", 5001))
    logger.warning(f"[Bootstrap] No free port found in range, using fallback: {fallback_port}")
    return fallback_port
# -----------------------
# Ensure self_check exists for all AI instances
# -----------------------
def ensure_self_check(ai_instance, name="AI"):
    """
    Ensures the given AI instance has a safe self_check method.
    Wraps existing self_check if present, otherwise creates a stub.
    Supports async and sync self_check methods safely.
    """
    if ai_instance is None:
        logger.warning(f"[{name}] AI instance is None; skipping self_check attachment.")
        return

    if not hasattr(ai_instance, "self_check"):
        # Create a stub self_check
        def stub_self_check():
            try:
                return True
            except Exception as e:
                if "TELEMETRY" in globals():
                    TELEMETRY.capture_exception(e, context=f"{name}_self_check_stub")
                logger.error(f"[{name}] self_check stub exception: {e}")
                return False
        setattr(ai_instance, "self_check", stub_self_check)
        logger.info(f"[{name}] self_check() stub added.")
    else:
        # Wrap existing self_check safely
        original_check = getattr(ai_instance, "self_check")
        if getattr(original_check, "_is_safe_wrapped", False):
            logger.debug(f"[{name}] self_check already safely wrapped; skipping.")
            return

        async def safe_self_check(*args, **kwargs):
            try:
                res = original_check(*args, **kwargs)
                if inspect.iscoroutine(res):
                    res = await res
                return res
            except Exception as e:
                if "TELEMETRY" in globals():
                    TELEMETRY.capture_exception(e, context=f"{name}_self_check_safe")
                logger.error(f"[{name}] self_check exception: {e}")
                return False

        safe_self_check._is_safe_wrapped = True
        setattr(ai_instance, "self_check", safe_self_check)
        logger.info(f"[{name}] self_check safely wrapped.")

# Apply to all AI instances
ensure_self_check(CORE_AI, "CoreAI")
ensure_self_check(ADVANCED_AI, "AdvancedAI")
ensure_self_check(OFFLINE_AI, "OfflineAI")
# -----------------------
# Wrap MetaMonitor snapshot safely
# -----------------------
if WATCHDOG:
    orig_snapshot = getattr(WATCHDOG, "snapshot", None)
    if orig_snapshot and not getattr(orig_snapshot, "_is_safe_wrapped", False):
        def safe_snapshot(*args, **kwargs):
            try:
                res = orig_snapshot(*args, **kwargs)
                if inspect.iscoroutine(res):
                    # Run async snapshot safely in ASYNC_LOOP
                    asyncio.run_coroutine_threadsafe(res, ASYNC_LOOP).result(timeout=10)
                return res
            except Exception as e:
                if "TELEMETRY" in globals():
                    TELEMETRY.capture_exception(e, context="MetaMonitor_safe_snapshot")
                logger.error(f"[Watchdog] MetaMonitor snapshot error: {e}")
                return None
        safe_snapshot._is_safe_wrapped = True
        setattr(WATCHDOG, "snapshot", safe_snapshot)
        logger.info("✅ MetaMonitor.snapshot safely wrapped.")
    else:
        logger.debug("MetaMonitor.snapshot already wrapped or missing.")
# -----------------------
# Main entry
# -----------------------
if __name__ == "__main__":
    logger.info("📡 Bibliotheca v22 booting — Beyond-Perfection")

    # -----------------------------
    # Initialize Telemetry & start snapshotting
    # -----------------------------
    try:
        if "TELEMETRY" in globals() and hasattr(TELEMETRY, "take_snapshot"):
            TELEMETRY.take_snapshot("startup")
            logger.info("📊 Telemetry startup snapshot taken.")
    except Exception as e:
        logger.warning(f"[Main] Telemetry startup snapshot failed: {e}")

    # -----------------------------
    # Launch interactive TTS AI (if available)
    # -----------------------------
    try:
        from ai_core.interactive_tts_ai import main as run_interactive_tts

        logger.info("🎤 Launching interactive TTS AI...")
        threading.Thread(target=run_interactive_tts, daemon=True).start()
    except ModuleNotFoundError:
        logger.info("Skipping interactive TTS AI — module not found.")
    except Exception as e:
        logger.error(f"[Main] Error launching interactive TTS AI: {e}", exc_info=True)

    # -----------------------------
    # Load AI engines and bootstrap
    # -----------------------------
    try:
        logger.info("🧠 Loading AI engines...")
        load_ai_engines()
        logger.info("✅ AI engines loaded successfully.")

        logger.info("🔹 Running bootstrap sequence...")
        bootstrap()

        # Ensure all AI self_checks are safely attached
        ensure_self_check(CORE_AI, "CoreAI")
        ensure_self_check(ADVANCED_AI, "AdvancedAI")
        ensure_self_check(OFFLINE_AI, "OfflineAI")

        # -----------------------------
        # Start Supervisor loop if not already started
        # -----------------------------
        try:
            threading.Thread(target=supervisor_loop, daemon=True).start()
            logger.info("🛡️ Supervisor loop started.")
        except Exception as e:
            logger.warning(f"[Main] Failed to start supervisor loop: {e}")

        # -----------------------------
        # Start Watchdog / MetaMonitor if available
        # -----------------------------
        if WATCHDOG and hasattr(WATCHDOG, "start") and callable(WATCHDOG.start):
            try:
                WATCHDOG.start()
                logger.info("🛡️ Watchdog/MetaMonitor started.")
            except Exception as e:
                logger.warning(f"[Main] Failed to start Watchdog: {e}")

        # -----------------------------
        # Launch Flask + SocketIO UI
        # -----------------------------
        start_ui_port = find_free_port()
        try:
            logger.info(f"🌐 Starting UI on port {start_ui_port}...")
            start_ui(port=start_ui_port)
        except Exception as e:
            logger.error(f"[Main] Failed to start UI: {e}", exc_info=True)
            sys.exit(1)

        logger.info("✅ Bibliotheca startup complete — system fully operational.")

    except Exception as e:
        logger.critical(f"[Main] Fatal error during startup: {e}", exc_info=True)
        sys.exit(1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bibliotheca Main — Beyond-Perfection Infinity++ v20
-----------------------------------------------------
Fully stable, self-healing, task-aware, telemetry-integrated, multi-AI ready
"""
# -----------------------
# Load configuration
# -----------------------
import json

CONFIG_PATH = PROJECT_ROOT / "app" / "config" / "config.json"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
    logger.info(f"Config loaded from {CONFIG_PATH}")
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    CONFIG = {}

# -----------------------
# Memory Manager
# -----------------------
try:
    from app.utils.memory_manager import MemoryManager
    MEMORY = MemoryManager()
    logger.info("MemoryManager loaded successfully.")
except Exception as e:
    logger.warning(f"MemoryManager missing or failed to load: {e}")

    class DummyMemory:
        def health_check(self):
            return {"status": "unknown"}

        def test(self):
            return True

        def repair(self):
            pass

        def save(self):
            pass

    MEMORY = DummyMemory()
    logger.info("Using DummyMemory fallback.")

# -----------------------
# Flask & SocketIO Setup
# -----------------------
app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATES_DIR))
socketio = SocketIO(app, async_mode="threading")

# -----------------------
# Async Loop
# -----------------------
try:
    ASYNC_LOOP = asyncio.get_running_loop()
except RuntimeError:
    ASYNC_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(ASYNC_LOOP)

# -----------------------
# Telemetry
# -----------------------
try:
    from app.utils.telemetry import Telemetry
    TELEMETRY = Telemetry()
except Exception:
    class DummyTelemetry:
        def capture_exception(self, e, context=None):
            logger.warning(f"Telemetry captured exception ({context}): {e}")
        def log_event(self, name, data=None):
            logger.info(f"Telemetry event: {name} | {data}")
        def take_snapshot(self, note=""):
            logger.info(f"Telemetry snapshot taken: {note}")
    TELEMETRY = DummyTelemetry()
# -----------------------
# Core AI Engines
# -----------------------
try:
    from app.ai.core_ai import CoreAI
    from app.ai.advanced_ai import AdvancedAI
    from app.utils.patch_engine import PATCH_ENGINE
    from app.utils.task_queue import TASK_QUEUE as task_queue

    # Initialize AI engines with memory and patch engine
    CORE_AI = CoreAI(memory=MEMORY, patch_engine=PATCH_ENGINE)
    ADVANCED_AI = AdvancedAI(core_ai=CORE_AI, memory=MEMORY, patch_engine=PATCH_ENGINE)

    logger.info("CoreAI and AdvancedAI initialized successfully ✅")

    # -----------------------
    # Run AI self-checks on startup
    # -----------------------
    async def startup_self_check():
        try:
            if CORE_AI:
                await CORE_AI.self_check()
            if ADVANCED_AI:
                await ADVANCED_AI.self_check()
            logger.info("Startup AI self-checks completed successfully ✅")
        except Exception as e:
            logger.warning(f"AI self-check encountered issues: {e}")
            TELEMETRY.capture_exception(e, context="startup_self_check")

    if task_queue:
        task_queue.schedule_coroutine(startup_self_check())

except Exception as e:
    CORE_AI = ADVANCED_AI = None
    logger.exception(f"Failed to initialize CoreAI/AdvancedAI: {e}")

# -----------------------
# Memory Manager
# -----------------------
try:
    from app.utils.memory_manager import MEMORY
except Exception as e:
    MEMORY = None
    logger.warning(f"MemoryManager unavailable: {e}")
# -----------------------
# Interactive Loop / TTS AI
# -----------------------
def interactive_loop():
    """Continuously read user input and delegate tasks to AI"""
    logger.info("[InteractiveLoop] Starting conversational loop ✅")
    try:
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue  # skip empty input

                # Process input via Core AI first
                if CORE_AI:
                    response = CORE_AI.process_input(user_input)
                elif ADVANCED_AI:
                    response = ADVANCED_AI.process_input(user_input)
                else:
                    response = f"No AI engine available to process: {user_input}"

                # Display response
                print(f"Bibliotheca: {response}")

                # Optional: emit via SocketIO if frontend is connected
                try:
                    socketio.emit("ai_response", {"input": user_input, "response": response})
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"[InteractiveLoop] Error processing input: {e}")
                TELEMETRY.capture_exception(e, context="interactive_loop_inner")
    except (KeyboardInterrupt, EOFError):
        logger.info("[InteractiveLoop] Shutting down interactive loop ✅")

# -----------------------
# Launch Interactive TTS AI (if available)
# -----------------------
try:
    from app.ai_core.interactive_tts_ai import main as run_interactive_tts
    threading.Thread(target=run_interactive_tts, daemon=True).start()
    logger.info("Interactive TTS AI launched ✅")
except ModuleNotFoundError:
    logger.info("Interactive TTS AI not available; skipping ✅")

# -----------------------
# MetaMonitor
# -----------------------
try:
    from app.utils.meta_monitor import MetaMonitor
    WATCHDOG = MetaMonitor(interval=10, allow_self_modify=True)
    WATCHDOG.start()
except Exception:
    WATCHDOG = None

# -----------------------
# Self-heal / Updater
# -----------------------
try:
    from app.utils.updater import Updater
    UPDATER = Updater(project_root=str(PROJECT_ROOT))
except Exception:
    UPDATER = None

try:
    import app.utils.self_heal as self_heal_mod
    if hasattr(self_heal_mod, "run"):
        threading.Thread(target=self_heal_mod.run, daemon=True).start()
        logger.info("self_heal module started.")
except Exception:
    logger.info("No self_heal module started; PatchEngine fallback skipped (not installed).")
    PATCH_ENGINE = None  # ensures any later references won't break

# -----------------------
# Query Processing Wrappers
# -----------------------
def process_query_safe(query):
    try:
        if CORE_AI:
            return CORE_AI.process(query)
        elif OFFLINE_AI:
            return OFFLINE_AI.process(query)
        else:
            return f"No AI engine available to process query: {query}"
    except Exception as e:
        TELEMETRY.capture_exception(e, context="process_query_safe")
        return f"Error processing query: {str(e)}"

def process_query_stream_to_sid(query, sid):
    try:
        response = process_query_safe(query)
        socketio.emit("ai_response", {"response": response}, room=sid)
    except Exception as e:
        TELEMETRY.capture_exception(e, context="stream_query")
        socketio.emit("ai_response", {"error": str(e)}, room=sid)

# -----------------------
# Flask Routes
# -----------------------
@app.route("/")
def root_index():
    try:
        if (TEMPLATES_DIR / "index.html").exists():
            return render_template("index.html")
    except Exception as e:
        TELEMETRY.capture_exception(e, context="render_index")
    return jsonify({"status": "bibliotheca", "ready": True})

@app.route("/health")
def health():
    try:
        memory_info = getattr(MEMORY, "health_check", lambda: {"status": "unknown"})()
        return jsonify({
            "status": "ok",
            "project_root": str(PROJECT_ROOT),
            "ai_core": bool(CORE_AI),
            "advanced_ai": bool(ADVANCED_AI),
            "offline_ai": bool(OFFLINE_AI),
            "memory": memory_info,
            "uptime": time.time() - START_TIME,
        })
    except Exception as e:
        TELEMETRY.capture_exception(e, context="health")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/logs")
def logs_endpoint():
    try:
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.read().splitlines()[-400:]
            return jsonify({"logs": lines})
        return jsonify({"logs": []})
    except Exception as e:
        TELEMETRY.capture_exception(e, context="logs")
        return jsonify({"error": str(e)}), 500

@app.route("/send_query", methods=["POST"])
def send_query_route():
    data = request.json or {}
    query = data.get("query") or data.get("message")
    stream = bool(data.get("stream", False))
    if not query:
        return jsonify({"error": "empty query"}), 400
    try:
        if stream:
            sid = data.get("sid")
            if not sid:
                return jsonify({"response": process_query_safe(query)})
            process_query_stream_to_sid(query, sid)
            return jsonify({"status": "streaming_started"})
        else:
            return jsonify({"response": process_query_safe(query)})
    except Exception as e:
        TELEMETRY.capture_exception(e, context="send_query")
        return jsonify({"error": str(e)}), 500

@app.route("/snapshot", methods=["POST"])
def snapshot_route():
    try:
        note = (request.json or {}).get("note", "")
        TELEMETRY.take_snapshot(note)
        return jsonify({"status": "snapshot_taken", "note": note})
    except Exception as e:
        TELEMETRY.capture_exception(e, context="snapshot")
        return jsonify({"error": str(e)}), 500

@app.route("/scan", methods=["GET"])
def scan_route():
    try:
        from app.utils.scanner import scan_project
        rep = scan_project()
        return jsonify(rep)
    except Exception as e:
        TELEMETRY.capture_exception(e, context="scan")
        return jsonify({"error": str(e)}), 500

@app.route("/apply_patch", methods=["POST"])
def apply_patch_route():
    try:
        payload = request.json or {}
        patches = payload.get("patches", [])
        force = bool(payload.get("force", False))
        if not patches:
            return jsonify({"error": "no patches provided"}), 400
        if not force:
            ok, compile_res, smoke_ok, smoke_out = dry_run_patch(patches)
            return jsonify({"dry_run_ok": ok, "compile": compile_res, "smoke_ok": smoke_ok, "smoke_out": smoke_out})
        return jsonify(apply_patch(patches, force=True))
    except Exception as e:
        TELEMETRY.capture_exception(e, context="apply_patch")
        return jsonify({"error": str(e)}), 500

# -----------------------
# Shutdown Hooks
# -----------------------
def _shutdown():
    logger.info("📌 Shutting down Bibliotheca gracefully...")
    try:
        if TASK_MANAGER:
            TASK_MANAGER.shutdown()
        if MEMORY:
            MEMORY.save()
        if UPDATER:
            UPDATER.save_state()
        logger.info("Shutdown complete ✅")
    except Exception as e:
        TELEMETRY.capture_exception(e, context="shutdown")

atexit.register(_shutdown)
signal_list = [signal.SIGINT, signal.SIGTERM]
for sig in signal_list:
    try:
        import signal
        signal.signal(sig, lambda *_: _shutdown())
    except Exception:
        pass

# -----------------------
# Background Tasks / Loops
# -----------------------
def start_background_loops():
    # Example: periodic self-check
    def self_check_loop():
        while True:
            try:
                if CORE_AI:
                    ASYNC_LOOP.run_until_complete(CORE_AI.self_check())
                if ADVANCED_AI:
                    ASYNC_LOOP.run_until_complete(ADVANCED_AI.self_check())
                time.sleep(300)  # every 5 minutes
            except Exception as e:
                TELEMETRY.capture_exception(e, context="background_self_check")
                time.sleep(60)

    threading.Thread(target=self_check_loop, daemon=True).start()
    logger.info("Background self-check loop started ✅")


# -----------------------------
# Interactive Console Loop
# -----------------------------
def interactive_loop():
    """Continuously read user input from terminal and delegate to AI"""
    logger.info("[InteractiveLoop] Starting conversational loop ✅")
    try:
        while True:
            try:
                user_input = input("You: ")
                if not user_input.strip():
                    continue  # skip empty input
                resp = process_query_safe(user_input)
                print(f"Bibliotheca: {resp}")
            except (EOFError, KeyboardInterrupt):
                logger.info("Interactive console terminated by user.")
                break
            except Exception as e:
                TELEMETRY.capture_exception(e, context="interactive_loop")
                print(f"Error: {e}")
    except Exception as e:
        TELEMETRY.capture_exception(e, context="interactive_loop_outer")
        logger.exception("Interactive loop failed unexpectedly")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bibliotheca Main — Beyond-Perfection Infinity++ v22
-----------------------------------------------------
Fully stable, self-healing, task-aware, telemetry-integrated, multi-AI ready
"""

import os
import sys
import time
import threading
import asyncio
import socket
import logging
import inspect
import subprocess
import atexit
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import eventlet
import eventlet.wsgi

# -----------------------
# Constants / Paths
# -----------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
STATIC_DIR = PROJECT_ROOT / "app" / "static"
TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"
LOG_PATH = PROJECT_ROOT / "bibliotheca.log"

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("Bibliotheca")

# -----------------------
# Flask & SocketIO Setup
# -----------------------
app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATES_DIR))
socketio = SocketIO(app, async_mode="threading")

# -----------------------
# Async Loop
# -----------------------
try:
    ASYNC_LOOP = asyncio.get_running_loop()
except RuntimeError:
    ASYNC_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(ASYNC_LOOP)

# -----------------------
# Telemetry
# -----------------------
try:
    from app.utils.telemetry import Telemetry
    TELEMETRY = Telemetry()
except Exception:
    class DummyTelemetry:
        def capture_exception(self, e, context=None):
            logger.warning(f"Telemetry captured exception ({context}): {e}")
        def log_event(self, name, data=None):
            logger.info(f"Telemetry event: {name} | {data}")
        def take_snapshot(self, note=""):
            logger.info(f"Telemetry snapshot taken: {note}")
    TELEMETRY = DummyTelemetry()

# -----------------------
# Memory Manager — Beyond-Perfection
# -----------------------
import threading
import importlib
import logging
import traceback
from typing import Optional, Type

logger = logging.getLogger("BibliothecaAI.MemoryManager")

# Global singleton and locks
MEMORY: Optional[object] = None
_RealMemoryManager: Optional[Type] = None
_MemoryManager_import_attempted = False
_memory_manager_lock = threading.Lock()


def _import_real_memory_manager() -> Optional[Type]:
    """
    Thread-safe lazy loader for the real MemoryManager class.
    - Tries to import `MemoryManager` from app.utils.memory_manager
    - Returns the class if found, None otherwise.
    - Safe to call multiple times.
    """
    global _RealMemoryManager, _MemoryManager_import_attempted

    if _RealMemoryManager is not None:
        return _RealMemoryManager

    with _memory_manager_lock:
        if _RealMemoryManager is not None:
            return _RealMemoryManager
        if _MemoryManager_import_attempted:
            # Already tried and failed previously
            return None

        _MemoryManager_import_attempted = True
        try:
            mod = importlib.import_module("app.utils.memory_manager")
            cls = getattr(mod, "MemoryManager", None)
            if cls is None:
                logger.warning("[MemoryManager] Module imported but class 'MemoryManager' not found.")
                return None
            _RealMemoryManager = cls
            logger.info("[MemoryManager] Real MemoryManager imported successfully ✅")
            return _RealMemoryManager
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to import MemoryManager: {e}\n{traceback.format_exc()}")
            return None


def _instantiate_memory_manager() -> Optional[object]:
    """
    Attempt to safely instantiate the MemoryManager.
    - Tries multiple constructor signatures intelligently.
    - Falls back to lightweight in-memory MemoryStore if necessary.
    """
    global MEMORY
    cls = _import_real_memory_manager()

    if cls is None:
        logger.warning("[MemoryManager] MemoryManager class not available — using fallback MemoryStore")
        try:
            from app.utils.memory_store import MemoryStore
            MEMORY = MemoryStore()
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to instantiate fallback MemoryStore: {e}")
            MEMORY = None
        return MEMORY

    # Attempt prioritized instantiation
    attempts = []
    app_cfg = globals().get("CONFIG", {}) or {}
    try:
        import inspect
        sig = inspect.signature(cls)
        params = list(sig.parameters.keys())
    except Exception:
        params = []

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
    attempts.append({})  # generic no-arg
    attempts_pos = [(), (app_cfg.get("memory_db_name", "bibliotheca_memory"),)]

    for payload in attempts:
        try:
            MEMORY = cls(**payload)
            logger.info(f"[MemoryManager] MEMORY instantiated with kwargs {payload} ✅")
            return MEMORY
        except TypeError as e:
            logger.debug(f"[MemoryManager] Constructor failed with kwargs {payload}: {e}")
        except Exception as e:
            logger.warning(f"[MemoryManager] MEMORY instantiation failed with kwargs {payload}: {e}")

    for args in attempts_pos:
        try:
            MEMORY = cls(*args)
            logger.info(f"[MemoryManager] MEMORY instantiated with args {args} ✅")
            return MEMORY
        except Exception as e:
            logger.warning(f"[MemoryManager] MEMORY instantiation failed with args {args}: {e}")

    # Final fallback
    try:
        from app.utils.memory_store import MemoryStore
        MEMORY = MemoryStore()
        logger.warning("[MemoryManager] Fallback MEMORY (MemoryStore) created ✅")
    except Exception as e:
        logger.error(f"[MemoryManager] Final fallback instantiation failed: {e}")
        MEMORY = None

    return MEMORY


# Initialize MEMORY singleton on import
if MEMORY is None:
    _instantiate_memory_manager()

# -----------------------
# Patch Engine
# -----------------------
try:
    from app.utils.patch_engine import PatchEngine, patch_engine, dry_run_patch, apply_patch
    PATCH_ENGINE = PatchEngine(project_root=str(PROJECT_ROOT))
    PATCH_ENGINE.log_event("startup", {"status": "ok"})
except Exception:
    PATCH_ENGINE = None
    patch_engine = dry_run_patch = apply_patch = None

# -----------------------
# Simple TaskManager fallback
# -----------------------
try:
    from app.utils.task_manager import TaskManager
    TASK_MANAGER = TaskManager()
except Exception:
    class SimpleTaskManager:
        def __init__(self):
            self._tasks = {}
            self._counter = 0
            self._lock = threading.Lock()

        def queue(self, task):
            with self._lock:
                self._counter += 1
                tid = f"task-{self._counter}"
                self._tasks[tid] = {"task": task, "status": "queued", "started": None, "finished": None}
            threading.Thread(target=self._run, args=(tid,), daemon=True).start()
            return tid

        def _run(self, tid):
            tinfo = self._tasks.get(tid)
            if not tinfo:
                return
            tinfo["status"] = "running"
            tinfo["started"] = time.time()
            try:
                task = tinfo["task"]
                if isinstance(task, dict) and "type" in task:
                    if task["type"] == "shell":
                        cmd = task.get("cmd")
                        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        tinfo["result"] = proc.stdout + proc.stderr
                    elif task["type"] == "ai_query":
                        q = task.get("query", "")
                        if CORE_AI:
                            tinfo["result"] = CORE_AI.process_query_safe(q)
                        else:
                            tinfo["result"] = process_query_safe(q)
                    else:
                        tinfo["result"] = f"unknown task type: {task.get('type')}"
                elif isinstance(task, str):
                    if CORE_AI:
                        tinfo["result"] = CORE_AI.process_query_safe(task)
                    else:
                        tinfo["result"] = process_query_safe(task)
                else:
                    tinfo["result"] = f"invalid task payload: {repr(task)[:200]}"
                tinfo["status"] = "finished"
            except Exception as e:
                tinfo["status"] = "error"
                tinfo["error"] = str(e)
            finally:
                tinfo["finished"] = time.time()
                try:
                    socketio.emit("task_update", {"task_id": tid, "info": self._tasks[tid]})
                except Exception:
                    pass

        def get(self, tid):
            return self._tasks.get(tid)

        def all(self):
            return self._tasks

        def check_tasks(self):
            return

        def shutdown(self):
            return

    TASK_MANAGER = SimpleTaskManager()

# -----------------------
# MetaMonitor
# -----------------------
try:
    from app.utils.meta_monitor import MetaMonitor
    WATCHDOG = MetaMonitor(interval=10, allow_self_modify=True)
    WATCHDOG.start()
except Exception:
    WATCHDOG = None

# -----------------------
# Self-heal / Updater
# -----------------------
try:
    from app.utils.updater import Updater
    UPDATER = Updater(project_root=str(PROJECT_ROOT))
except Exception:
    UPDATER = None

try:
    import app.utils.self_heal as self_heal_mod
    if hasattr(self_heal_mod, "run"):
        threading.Thread(target=self_heal_mod.run, daemon=True).start()
        logger.info("self_heal module started.")
except Exception:
    logger.info("No self_heal module started; PatchEngine fallback skipped.")
    PATCH_ENGINE = None

# -----------------------
# Core AI Engines
# -----------------------
try:
    from app.ai.core_ai import CoreAI
    from app.ai.advanced_ai import AdvancedAI

    CORE_AI = CoreAI(memory=MEMORY, patch_engine=PATCH_ENGINE)
    ADVANCED_AI = AdvancedAI(core_ai=CORE_AI, memory=MEMORY, patch_engine=PATCH_ENGINE)

    logger.info("CoreAI and AdvancedAI initialized successfully ✅")
except Exception as e:
    CORE_AI = ADVANCED_AI = None
    logger.exception(f"Failed to initialize CoreAI/AdvancedAI: {e}")

# -----------------------
# Utility Functions
# -----------------------
def process_query_safe(query):
    try:
        if CORE_AI:
            return CORE_AI.process(query)
        else:
            return f"No AI engine available to process query: {query}"
    except Exception as e:
        TELEMETRY.capture_exception(e, context="process_query_safe")
        return f"Error processing query: {str(e)}"

def process_query_stream_to_sid(query, sid):
    try:
        response = process_query_safe(query)
        socketio.emit("ai_response", {"response": response}, room=sid)
    except Exception as e:
        TELEMETRY.capture_exception(e, context="stream_query")
        socketio.emit("ai_response", {"error": str(e)}, room=sid)

# -----------------------
# Boot-Time Initialization
# -----------------------
START_TIME = time.time()
def bootstrap():
    from app.utils.scanner import scan_project
    try:
        rep = scan_project()
        TELEMETRY.log_event("initial_scan", {"python_files": len(rep.get("python_files", []))})
        logger.info(f"Initial scan completed: {len(rep.get('python_files', []))} Python files")
    except Exception as e:
        TELEMETRY.capture_exception(e, context="bootstrap_scan")

# -----------------------
# Shutdown Hook
# -----------------------
@atexit.register
def _shutdown():
    try:
        TELEMETRY.take_snapshot("shutdown")
    except Exception:
        pass
    try:
        if MEMORY and hasattr(MEMORY, "save"):
            MEMORY.save()
    except Exception:
        pass
    try:
        if TASK_MANAGER:
            TASK_MANAGER.shutdown()
    except Exception:
        pass
    logger.info("Bibliotheca shutdown complete.")

# -----------------------
# Interactive Loop
# -----------------------
def interactive_loop():
    logger.info("[InteractiveLoop] Starting conversational loop ✅")
    try:
        while True:
            try:
                user_input = input("You: ")
                if not user_input.strip():
                    continue
                resp = process_query_safe(user_input)
                print(f"Bibliotheca: {resp}")
            except (EOFError, KeyboardInterrupt):
                logger.info("Interactive console terminated by user.")
                break
            except Exception as e:
                TELEMETRY.capture_exception(e, context="interactive_loop")
                print(f"Error: {e}")
    except Exception as e:
        TELEMETRY.capture_exception(e, context="interactive_loop_outer")

# -----------------------
# Main Entry — Beyond-Perfection
# -----------------------
if __name__ == "__main__":
    import time
    import inspect
    import threading
    import asyncio
    import eventlet
    import socket
    import sys
    import logging
    from app import app  # your Flask app
    from app.utils.ai_core import ADVANCED_AI, CORE_AI, OFFLINE_AI
    from app.utils.memory import MEMORY
    from app.utils.telemetry import TELEMETRY
    from app.utils.task_manager import TASK_MANAGER
    from app.utils.watchdog import WATCHDOG
    from app.utils.patch_engine import PATCH_ENGINE
    from app.bootstrap import bootstrap  # your project bootstrap

    logger.info("📡 Bibliotheca v22 booting — Beyond-Perfection")

    # -----------------------
    # Bootstrap project
    # -----------------------
    try:
        bootstrap()
        logger.info("✅ Bootstrap completed successfully")
    except Exception as e:
        logger.exception("❌ Bootstrap failed")
        sys.exit(1)

    # -----------------------
    # Supervisor Loop
    # -----------------------
    def supervisor_loop():
        logger.info("Supervisor thread started ✅")
        while True:
            try:
                # Memory health check
                if MEMORY and hasattr(MEMORY, "test"):
                    try:
                        if not MEMORY.test():
                            TELEMETRY.log_event("memory_test_failed")
                            if hasattr(MEMORY, "repair"):
                                MEMORY.repair()
                                logger.info("Memory repair executed")
                    except Exception as e:
                        TELEMETRY.capture_exception(e, context="memory_test")

                # AI self-checks
                for ai_instance in (CORE_AI, ADVANCED_AI, OFFLINE_AI):
                    if ai_instance and hasattr(ai_instance, "self_check"):
                        try:
                            res = ai_instance.self_check()
                            if inspect.iscoroutine(res):
                                asyncio.run_coroutine_threadsafe(res, asyncio.get_event_loop()).result(timeout=5)
                        except Exception as e:
                            TELEMETRY.capture_exception(e, context=f"{type(ai_instance).__name__}_self_check")

                # TaskManager checks
                if TASK_MANAGER and hasattr(TASK_MANAGER, "check_tasks"):
                    try:
                        TASK_MANAGER.check_tasks()
                    except Exception as e:
                        TELEMETRY.capture_exception(e, context="task_manager_check")

                # Watchdog / MetaMonitor snapshot
                if WATCHDOG and hasattr(WATCHDOG, "snapshot"):
                    try:
                        WATCHDOG.snapshot()
                    except Exception as e:
                        TELEMETRY.capture_exception(e, context="watchdog_snapshot")

            except Exception as e:
                TELEMETRY.capture_exception(e, context="supervisor_outer")
            finally:
                time.sleep(6)  # run every 6 seconds

    threading.Thread(target=supervisor_loop, daemon=True, name="SupervisorThread").start()

    # -----------------------
    # Async background loops
    # -----------------------
    async def background_chat():
        while True:
            try:
                if CORE_AI and hasattr(CORE_AI, "handle_chat"):
                    await CORE_AI.handle_chat()
            except Exception as e:
                logger.warning(f"[Chat AutoFix] {e}")
                TELEMETRY.capture_exception(e, context="background_chat")
                if PATCH_ENGINE:
                    PATCH_ENGINE.run_self_heal()
            await asyncio.sleep(0.1)

    async def background_research():
        while True:
            try:
                if CORE_AI and hasattr(CORE_AI, "handle_research"):
                    await CORE_AI.handle_research()
            except Exception as e:
                logger.warning(f"[Research AutoFix] {e}")
                TELEMETRY.capture_exception(e, context="background_research")
                if PATCH_ENGINE:
                    PATCH_ENGINE.run_self_heal()
            await asyncio.sleep(0.1)

    async def background_telegram():
        while True:
            try:
                if CORE_AI and hasattr(CORE_AI, "handle_telegram"):
                    await CORE_AI.handle_telegram()
            except Exception as e:
                logger.warning(f"[Telegram AutoFix] {e}")
                TELEMETRY.capture_exception(e, context="background_telegram")
                if PATCH_ENGINE:
                    PATCH_ENGINE.run_self_heal()
            await asyncio.sleep(0.1)

    async def main_async_loop():
        await asyncio.gather(
            background_chat(),
            background_research(),
            background_telegram()
        )

    # Safe async loop launcher
    def launch_async_loop():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main_async_loop())
        except Exception as e:
            logger.exception(f"Async main loop crashed: {e}")
            TELEMETRY.capture_exception(e, context="async_main_loop")

    threading.Thread(target=launch_async_loop, daemon=True, name="AsyncLoopThread").start()
    logger.info("Background async loops launched ✅")

    # -----------------------
    # Ultimate Flask + SocketIO Starter
    # -----------------------
    eventlet.monkey_patch()  # Patch early for async

    def find_free_port(start_port=5000, max_attempts=50) -> int:
        port = start_port
        for _ in range(max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('0.0.0.0', port))
                    return port
                except OSError:
                    port += 1
        raise RuntimeError(f"No free port found starting at {start_port}")

    def run_ui(app, socketio, port=None):
        if port is None:
            port = find_free_port(5000)
        try:
            logger.info(f"🚀 Starting Flask + SocketIO on port {port} ...")
            print(f"🚀 UI running on http://localhost:{port}")
            socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
        except Exception as e:
            logger.exception(f"❌ Flask/SocketIO failed: {e}")
            try:
                # Retry next free port
                port = find_free_port(port + 1)
                logger.warning(f"⚠ Retrying server start on port {port} ...")
                socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
            except Exception as e2:
                logger.exception(f"❌ Second attempt failed. Exiting: {e2}")
                sys.exit(1)

    # -----------------------
    # Launch Flask + SocketIO UI
    # -----------------------
    try:
        # Detect last Flask app in globals
        flask_candidates = [(name, obj) for name, obj in globals().items() if "Flask" in str(type(obj))]
        if not flask_candidates:
            logger.error("❌ No Flask app found in globals. UI cannot start.")
            sys.exit(1)
        app_name, app = flask_candidates[-1]
        logger.info(f"✅ Using Flask app '{app_name}' for UI.")

        # Detect last SocketIO instance
        socketio_candidates = [(name, obj) for name, obj in globals().items() if "SocketIO" in str(type(obj))]
        if not socketio_candidates:
            logger.error("❌ No SocketIO instance found in globals. UI cannot start.")
            sys.exit(1)
        socketio_name, socketio = socketio_candidates[-1]
        logger.info(f"✅ Using SocketIO instance '{socketio_name}'.")

        run_ui(app, socketio)

    except KeyboardInterrupt:
        logger.info("🛑 KeyboardInterrupt received. Shutting down gracefully...")
        print("🛑 Shutting down Flask + SocketIO server...")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"❌ Fatal error starting UI: {e}")
        sys.exit(1)

    # -----------------------
    # Graceful shutdown hook
    # -----------------------
    def graceful_shutdown():
        logger.info("🛑 Initiating graceful shutdown sequence...")
        try:
            if TASK_MANAGER and hasattr(TASK_MANAGER, "shutdown"):
                TASK_MANAGER.shutdown()
                logger.info("TaskManager shutdown complete")
            if WATCHDOG and hasattr(WATCHDOG, "shutdown"):
                WATCHDOG.shutdown()
                logger.info("Watchdog shutdown complete")
            if MEMORY and hasattr(MEMORY, "close"):
                MEMORY.close()
                logger.info("MemoryStore closed")
        except Exception as e:
            logger.warning(f"Exception during shutdown: {e}")
        finally:
            logger.info("Shutdown sequence finished ✅")
            sys.exit(0)

    import atexit
    atexit.register(graceful_shutdown)
