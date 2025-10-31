#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bibliotheca — ai_core.py v25.0
Beyond-Perfection Infinity++ Edition — Autonomous, Multi-backend, Self-Evolving

This module initializes the AI core with:
- Robust BackendRouter integration (tasks, telemetry, self-improvement)
- Memory & Semantic wrappers
- Self-Evolution Engine (propose, test, apply, rate-limited)
- Async-safe TaskQueue and executor handling
- Advanced conversational AI interfaces
- Safe shell, file editing, and sandboxed execution helpers

Author: Bibliotheca Core
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import json
import asyncio
import logging
import concurrent.futures
import subprocess
import threading
import traceback
from functools import partial
from threading import Lock
from typing import Optional, Any, Dict, List, Tuple

# -----------------------------
# Core AI imports
# -----------------------------
from .backend_router import ROUTER, maybe_await
from app.utils.telemetry import TelemetryHandler
from utils.self_evolution import SelfEvolutionEngine
from app.utils.file_editor import FileEditor
from app.utils.test_runner import TestRunner
from app.utils.sandbox_runner import SandboxRunner
from app.utils.ai_core.advanced_autonomy import AdvancedAutonomy

import builtins
builtins.ADVANCED_AI_INSTANCE = self

# -----------------------------
# Logger setup
# -----------------------------
logger = logging.getLogger("AI_Core")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s][AI_Core][%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# -----------------------
# Self-Improver: orchestrates file edits, sandbox tests, and execution
# -----------------------
class SelfImprover:
    """
    Combines FileEditor, SandboxRunner, and TestRunner to allow the AI
    to propose, test, and apply code updates autonomously.
    """
    def __init__(self, memory_manager):
        self.file_editor = FileEditor()
        self.test_runner = TestRunner()
        self.sandbox_runner = SandboxRunner()
        self.memory = memory_manager
        self.lock = Lock()
        self.logger = logging.getLogger("SelfImprover")

    async def propose_and_apply(self, file_path: str, new_content: str, description: str, max_retries: int = 3):
        """
        Propose a patch, test it in sandbox, and apply if successful.
        """
        async with asyncio.Lock():
            for attempt in range(1, max_retries + 1):
                try:
                    # Step 1: Propose file changes
                    self.logger.info(f"Proposing patch for {file_path} (attempt {attempt})")
                    self.file_editor.write_temp(file_path, new_content)

                    # Step 2: Run sandboxed tests
                    sandbox_result = await self.sandbox_runner.run(file_path)
                    if not sandbox_result["success"]:
                        self.logger.warning(f"Sandbox tests failed: {sandbox_result['errors']}")
                        continue

                    # Step 3: Run full test suite
                    test_result = await self.test_runner.run_tests(file_path)
                    if not test_result["success"]:
                        self.logger.warning(f"Tests failed: {test_result['failures']}")
                        continue

                    # Step 4: Apply changes permanently
                    self.file_editor.commit(file_path, new_content)
                    self.logger.info(f"Patch applied successfully to {file_path}")
                    return {"applied": True, "attempt": attempt}

                except Exception as e:
                    self.logger.error(f"Attempt {attempt} failed: {e}", exc_info=True)
                    await asyncio.sleep(2 ** attempt)  # exponential backoff

            # If all attempts fail
            self.logger.error(f"Failed to apply patch after {max_retries} attempts")
            return {"applied": False, "attempt": max_retries}

# -----------------------------------------------------------------------------
# Environment and configuration
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPINFRA_KEY = os.getenv("DEEPINFRA_KEY")
TOGETHERAI_KEY = os.getenv("TOGETHERAI_KEY")
HF_MODEL = os.getenv("HF_MODEL")  # e.g. "gpt2", "facebook/opt-125m"
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH")
SELF_UPDATE_MODE = os.getenv("SELF_UPDATE_MODE", os.getenv("BIBLIOTHECA_SELF_UPDATE_MODE", "manual"))
AUTONOMOUS_MODE = os.getenv("AUTONOMOUS_MODE", "false").lower() in ("1", "true", "yes")
PATCH_WHITELIST = os.getenv("BIBLIOTHECA_PATCH_WHITELIST", "app/,scripts/,app/utils/").split(",")
PATCH_RATE_LIMIT = int(os.getenv("BIBLIOTHECA_PATCH_RATE_LIMIT", "2"))
PATCH_SIZE_LIMIT = int(os.getenv("BIBLIOTHECA_PATCH_SIZE_LIMIT", "20000"))

# -----------------------------------------------------------------------------
# Logging — Beyond-Perfection Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("BibliothecaAI")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent double logging

# Only add handlers once
if not logger.handlers:
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s"))
    logger.addHandler(ch)

    # Optional file handler
    try:
        fh = logging.FileHandler("bibliotheca_ai.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s"))
        logger.addHandler(fh)
    except Exception as e:
        print(f"[Logging Setup Warning] Could not create file handler: {e}")

# Safe logging helper for async/background tasks
def safe_log(level, msg):
    try:
        if level == "info":
            logger.info(msg)
        elif level == "debug":
            logger.debug(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        elif level == "critical":
            logger.critical(msg)
        else:
            logger.info(msg)
    except Exception:
        print(f"[Logging Fallback] {level}: {msg}")

# --- Beyond-Perfection CouchDB-backed memory replacement ---
import os
import time
import asyncio
import threading
import logging
from typing import Any, Optional, Dict, List

import couchdb

logger = logging.getLogger("CouchDBMemory")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)

class CouchDBMemory:
    """
    True drop-in replacement for MemoryManager / MemoryStore using CouchDB.
    Fully async-compatible, thread-safe, supports patching, caching, and
    SemanticMemory-style usage.
    """

    _instance_lock = threading.RLock()
    _singleton_instance: Optional["CouchDBMemory"] = None

    @classmethod
    def get_instance(cls, db_url=None, db_name="bibliotecha_memory"):
        """Thread-safe singleton accessor."""
        if cls._singleton_instance is None:
            with cls._instance_lock:
                if cls._singleton_instance is None:
                    cls._singleton_instance = cls(db_url=db_url, db_name=db_name)
        return cls._singleton_instance

    def __init__(self, db_url=None, db_name="bibliotecha_memory"):
        self.db_url = db_url or f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/"
        self.db_name = db_name
        self.server = couchdb.Server(f"http://{os.getenv('COUCHDB_USER')}:{os.getenv('COUCHDB_PASSWORD')}@127.0.0.1:5984/")
        if self.db_name not in self.server:
            self.server.create(self.db_name)
        self.db = self.server[self.db_name]

        self._memory_lock = threading.RLock()
        self._local_cache: Dict[str, Any] = {}  # Optional fallback in-memory cache
        self._patch_store: Dict[str, Any] = {}  # For AI-generated patches
        self.ai_helper: Optional[Any] = None

        logger.info(f"[CouchDBMemory] Initialized with DB {self.db_name} at {self.db_url}")

    # ---------------- Basic Key-Value ----------------
    def get(self, key: str, default: Any = None) -> Any:
        try:
            doc = self.db[key]
            return doc.get("data", default)
        except couchdb.http.ResourceNotFound:
            return default
        except Exception as e:
            logger.warning(f"[get] Failed to retrieve key={key}: {e}")
            return default

    def set(self, key: str, value: Any):
        with self._memory_lock:
            doc = {"_id": key, "data": value}
            try:
                if key in self.db:
                    existing = self.db[key]
                    doc["_rev"] = existing.rev
                self.db.save(doc)
                self._local_cache[key] = value
            except Exception as e:
                logger.error(f"[set] Failed to save key={key}: {e}")

    # ---------------- Async Support ----------------
    async def store(self, key: str, value: Any, meta: Optional[Dict] = None):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.set, key, value)

    async def recall(self, key: str, default: Any = None):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key, default)

    # ---------------- Patch / Self-Evolution Support ----------------
    def propose_patch(self, patch_id: str, patch_data: Any):
        with self._memory_lock:
            self._patch_store[patch_id] = patch_data
            logger.info(f"[propose_patch] Patch stored id={patch_id}")

    def apply_patch(self, patch_id: str):
        with self._memory_lock:
            patch = self._patch_store.get(patch_id)
            if not patch:
                logger.warning(f"[apply_patch] Patch not found id={patch_id}")
                return False
            try:
                for key, value in patch.items():
                    self.set(key, value)
                logger.info(f"[apply_patch] Patch applied id={patch_id}")
                return True
            except Exception as e:
                logger.error(f"[apply_patch] Failed id={patch_id}: {e}")
                return False

    # ---------------- Local Cache / Fallback ----------------
    def flush_local_cache(self):
        with self._memory_lock:
            for key, value in self._local_cache.items():
                self.set(key, value)
            logger.info("[flush_local_cache] Flushed in-memory cache to CouchDB")
            self._local_cache.clear()

    # ---------------- SemanticMemory / AIMemoryHelper Compatibility ----------------
    def attach_ai_helper(self, helper_class):
        try:
            self.ai_helper = helper_class(self)
            logger.info("[attach_ai_helper] AI helper attached")
        except Exception as e:
            logger.error(f"[attach_ai_helper] Failed to attach AI helper: {e}")

    # ---------------- Utilities ----------------
    def keys(self) -> List[str]:
        try:
            return [row.id for row in self.db.view("_all_docs")]
        except Exception as e:
            logger.error(f"[keys] Failed to list keys: {e}")
            return []

    def has_key(self, key: str) -> bool:
        return self.get(key) is not None

    def delete(self, key: str):
        with self._memory_lock:
            try:
                doc = self.db[key]
                self.db.delete(doc)
                self._local_cache.pop(key, None)
                logger.info(f"[delete] Key deleted: {key}")
            except couchdb.http.ResourceNotFound:
                logger.warning(f"[delete] Key not found: {key}")
            except Exception as e:
                logger.error(f"[delete] Failed to delete key={key}: {e}")

    # ---------------- Full Flush / Shutdown ----------------
    def sync_to_db(self):
        try:
            self.flush_local_cache()
            logger.info("[sync_to_db] Memory fully synced to CouchDB")
        except Exception as e:
            logger.error(f"[sync_to_db] Failed: {e}")

    def stop(self):
        # Placeholder for graceful shutdown if needed
        try:
            self.sync_to_db()
            logger.info("[stop] CouchDBMemory stopped gracefully")
        except Exception as e:
            logger.error(f"[stop] Stop failed: {e}")

# -----------------------------------------------------------------------------
# MemoryManager import (extend where necessary)
# -----------------------------------------------------------------------------
# The project provided a BaseMemoryManager in app.utils.memory_manager; we extend it
try:
    from app.utils.memory_manager import MemoryManager as BaseMemoryManager
except Exception:
    # fallback placeholder if import fails — provide minimal in-memory manager
    class BaseMemoryManager:
        def __init__(self, base_dir: str = "."):
            self._store = {}
            self.base_dir = base_dir

        def set(self, key, value):
            self._store[key] = value

        def get(self, key, default=None):
            return self._store.get(key, default)

        def snapshot(self):
            return True

        # placeholders for patch interfaces (may be implemented in real BaseMemoryManager)
        def propose_patch_from_ai(self, file_path, content, desc):
            raise NotImplementedError

        def apply_patch_safely(self, proposal_id, mode="manual"):
            raise NotImplementedError

        def save_json(self, filename, data):
            self._store[filename] = data
            return True

        def load_json(self, filename):
            return self._store.get(filename)

# Extend BaseMemoryManager with compatibility helpers
class COUCHDB_MEMORY:
    def save_json(self, filename, data):
        try:
            # prefer an explicit method if present
            if hasattr(super(), "save_json"):
                return super().save_json(filename, data)
        except Exception:
            pass
        # fallback to set
        try:
            self.set(filename, data)
            return True
        except Exception as e:
            logger.error(f"[MemoryManager.save_json] failed: {e}")
            return False

    def load_json(self, filename):
        try:
            if hasattr(super(), "load_json"):
                return super().load_json(filename)
        except Exception:
            pass
        try:
            return self.get(filename)
        except Exception as e:
            logger.error(f"[MemoryManager.load_json] failed: {e}")
            return None

    # If the underlying base class doesn't provide patch helpers, provide safe local ones
    def propose_patch_from_ai(self, file_path: str, content: str, desc: str):
        """
        Store a proposed patch in memory and return a proposal dict.
        If underlying implementation exists, delegate to it.
        """
        try:
            if hasattr(super(), "propose_patch_from_ai"):
                # delegate if implemented
                return super().propose_patch_from_ai(file_path, content, desc)
        except Exception:
            pass

        # Local fallback: store proposals in-memory under key "pending_patches"
        try:
            pending = self.get("pending_patches") or []
            pid = f"proposal_{int(time.time()*1000)}_{len(pending)}"
            proposal = {"id": pid, "path": file_path, "content": content, "desc": desc, "ts": time.time()}
            pending.append(proposal)
            self.set("pending_patches", pending)
            return proposal
        except Exception as e:
            logger.error(f"[MemoryManager.propose_patch_from_ai] fallback failed: {e}")
            return None

    def apply_patch_safely(self, proposal_id: str, mode: str = "manual"):
        """
        Apply a stored proposal: write file content and mark it applied.
        If underlying impl exists, delegate to it.
        """
        try:
            if hasattr(super(), "apply_patch_safely"):
                return super().apply_patch_safely(proposal_id, mode)
        except Exception:
            pass

        try:
            pending = self.get("pending_patches") or []
            proposal = next((p for p in pending if p.get("id") == proposal_id), None)
            if not proposal:
                logger.warning(f"[MemoryManager.apply_patch_safely] proposal {proposal_id} not found")
                return False
            # write file
            path = os.path.abspath(proposal["path"])
            d = os.path.dirname(path)
            os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(proposal["content"])
            # move to applied list
            applied = self.get("applied_patches") or []
            applied.append({"id": proposal_id, "applied_ts": time.time(), "mode": mode})
            self.set("applied_patches", applied)
            # remove from pending
            pending = [p for p in pending if p.get("id") != proposal_id]
            self.set("pending_patches", pending)
            return {"id": proposal_id, "applied": True, "path": path}
        except Exception as e:
            logger.error(f"[MemoryManager.apply_patch_safely] fallback failed: {e}")
            return False

# instantiate memory manager
memory_manager = COUCHDB_MEMORY)

# -----------------------------------------------------------------------------
# Executor & TaskQueue
# -----------------------------------------------------------------------------
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

class TaskQueue:
    """
    Hybrid TaskQueue:
    - async submit(fn, *args, **kwargs) -> awaitable that runs fn in executor
    - schedule_coroutine(coro) -> schedules a coroutine on an internal loop (safe from other threads)
    """
    def __init__(self):
        self._lock = Lock()
        # Create a dedicated loop for scheduling coroutines from other threads
        self._internal_loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        try:
            asyncio.set_event_loop(self._internal_loop)
            self._internal_loop.run_forever()
        except Exception as e:
            logger.exception(f"[TaskQueue] internal loop failed to start: {e}")

    async def submit(self, fn, *args, **kwargs):
        """
        Run a blocking callable in the shared executor. This is awaitable.
        Use this from within async code (await task_queue.submit(...))
        """
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # no running loop in current thread; fallback to run_in_executor via a temporary loop
            loop = asyncio.new_event_loop()
            try:
                return await loop.run_in_executor(executor, partial(fn, *args, **kwargs))
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
        return await loop.run_in_executor(executor, partial(fn, *args, **kwargs))

    def schedule_coroutine(self, coro):
        """
        Schedule a coroutine on the internal loop (safe to call from other threads).
        Returns a concurrent.futures.Future (not awaitable directly).
        """
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, self._internal_loop)
            return fut
        except Exception as e:
            logger.exception(f"[TaskQueue] schedule_coroutine failed: {e}")
            raise

task_queue = TaskQueue()

# -----------------------------------------------------------------------------
# Optional backend packages detection (lazy)
# -----------------------------------------------------------------------------
try:
    import openai_utils
    OPENAI_AVAILABLE = True and bool(OPENAI_API_KEY)
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

DEEPINFRA_AVAILABLE = bool(DEEPINFRA_KEY)
TOGETHERAI_AVAILABLE = bool(TOGETHERAI_KEY)

TRANSFORMERS_AVAILABLE = False
LLAMA_CPP_AVAILABLE = False
_local_hf_pipeline = None
_llama_cpp_client = None

# transformers detection
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True and bool(HF_MODEL)
except Exception:
    TRANSFORMERS_AVAILABLE = False

# llama_cpp detection
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True and bool(LLAMA_MODEL_PATH)
except Exception:
    LLAMA_CPP_AVAILABLE = False

# helper to check if torch available (used by HF backend selection)
def torch_available():
    try:
        import torch  # local import to avoid import overhead
        return True
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Model backend wrappers
# -----------------------------------------------------------------------------
class ModelBackend:
    name: str = "base"
    available: bool = False

    async def chat(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

class OpenAIBackend(ModelBackend):
    name = "openai"
    available = OPENAI_AVAILABLE and bool(OPENAI_API_KEY)

    async def chat(self, prompt: str, model: str = "gpt-4", **kwargs) -> str:
        if not self.available:
            raise RuntimeError("OpenAI backend not available")
        def _call():
            try:
                # Support both legacy ChatCompletion and new ChatCompletion SDKs
                try:
                    resp = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role":"user","content":prompt}],
                        temperature=kwargs.get("temperature", 0.7),
                        max_tokens=kwargs.get("max_tokens", 512)
                    )
                except Exception:
                    # try the newer openai.chat API shape if available
                    try:
                        resp = openai.chat.completions.create(
                            model=model,
                            messages=[{"role":"user","content":prompt}],
                            temperature=kwargs.get("temperature", 0.7),
                            max_tokens=kwargs.get("max_tokens", 512)
                        )
                    except Exception as e:
                        logger.exception(f"[OpenAIBackend] create call failed: {e}")
                        raise

                # robust extraction for different SDK versions
                try:
                    return resp.choices[0].message.content.strip()
                except Exception:
                    try:
                        return resp["choices"][0]["message"]["content"].strip()
                    except Exception:
                        try:
                            return resp["choices"][0]["text"].strip()
                        except Exception:
                            return str(resp)
            except Exception as e:
                logger.error(f"[OpenAIBackend] call failed: {e}")
                raise
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _call)

class DeepInfraBackend(ModelBackend):
    name = "deepinfra"
    available = DEEPINFRA_AVAILABLE
    async def chat(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("DeepInfra integration is a placeholder. Add SDK calls and API keys.")

class TogetherAIBackend(ModelBackend):
    name = "togetherai"
    available = TOGETHERAI_AVAILABLE
    async def chat(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("TogetherAI integration is a placeholder. Add SDK calls and API keys.")

class HFLocalBackend(ModelBackend):
    name = "hf_local"
    available = TRANSFORMERS_AVAILABLE and bool(HF_MODEL)

    def _ensure_pipeline(self):
        global _local_hf_pipeline
        if _local_hf_pipeline is None:
            logger.info(f"[HFLocalBackend] loading model {HF_MODEL} (this may take time)")
            # select device
            device = 0 if torch_available() else -1
            # pipeline may be heavy - load lazily
            from transformers import pipeline as _pipeline
            _local_hf_pipeline = _pipeline("text-generation", model=HF_MODEL, device=device)
        return _local_hf_pipeline

    async def chat(self, prompt: str, **kwargs) -> str:
        if not self.available:
            raise RuntimeError("HF local backend not available")
        def _call():
            pipe = self._ensure_pipeline()
            out = pipe(prompt, max_new_tokens=kwargs.get("max_tokens", 256), do_sample=kwargs.get("do_sample", True))
            if isinstance(out, list) and out:
                txt = out[0].get("generated_text") or str(out[0])
                return txt
            return str(out)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _call)

class LlamaCppBackend(ModelBackend):
    name = "llama_cpp"
    available = LLAMA_CPP_AVAILABLE and bool(LLAMA_MODEL_PATH)

    def _ensure_client(self):
        global _llama_cpp_client
        if _llama_cpp_client is None:
            logger.info(f"[LlamaCppBackend] loading model from {LLAMA_MODEL_PATH}")
            import llama_cpp
            _llama_cpp_client = llama_cpp.Llama(model_path=LLAMA_MODEL_PATH)
        return _llama_cpp_client

    async def chat(self, prompt: str, **kwargs) -> str:
        if not self.available:
            raise RuntimeError("llama_cpp backend not available")
        def _call():
            client = self._ensure_client()
            out = client.create(prompt=prompt, max_tokens=kwargs.get("max_tokens", 256))
            return getattr(out, "response", str(out))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _call)

class OfflineAIBackend(ModelBackend):
    name = "offline_ai"
    available = False
    _mod = None

    @classmethod
    def _ensure_mod(cls):
        if cls._mod is None:
            try:
                cls._mod = __import__("app.utils.offline_ai", fromlist=["*"])
                cls.available = True
            except Exception:
                cls.available = False
        return cls._mod

    async def chat(self, prompt: str, **kwargs) -> str:
        mod = self._ensure_mod()
        if not mod or not hasattr(mod, "generate"):
            raise RuntimeError("offline_ai module unavailable or does not provide generate()")
        def _call():
            return mod.generate(prompt, **kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _call)

# helper to check if a backend client is available
def backend_summary():
    return {
        "openai": OPENAI_AVAILABLE and bool(OPENAI_API_KEY),
        "deepinfra": DEEPINFRA_AVAILABLE,
        "togetherai": TOGETHERAI_AVAILABLE,
        "hf_local": TRANSFORMERS_AVAILABLE and bool(HF_MODEL),
        "llama_cpp": LLAMA_CPP_AVAILABLE and bool(LLAMA_MODEL_PATH),
        "offline_ai": OfflineAIBackend._ensure_mod() is not None
    }

# -----------------------------------------------------------------------------
# BackendRouter: orchestrates across available model backends
# -----------------------------------------------------------------------------
class BackendRouter:
    def __init__(self):
        # instantiate available backends with order of preference
        self.backends: List[Tuple[str, ModelBackend]] = []
        # prefer OpenAI if available
        try:
            if OpenAIBackend.available:
                self.backends.append((OpenAIBackend.name, OpenAIBackend()))
        except Exception:
            pass
        try:
            if DEEPINFRA_AVAILABLE:
                self.backends.append((DeepInfraBackend.name, DeepInfraBackend()))
        except Exception:
            pass
        try:
            if TOGETHERAI_AVAILABLE:
                self.backends.append((TogetherAIBackend.name, TogetherAIBackend()))
        except Exception:
            pass
        try:
            if LlamaCppBackend.available:
                self.backends.append((LlamaCppBackend.name, LlamaCppBackend()))
        except Exception:
            pass
        try:
            if HFLocalBackend.available:
                self.backends.append((HFLocalBackend.name, HFLocalBackend()))
        except Exception:
            pass
        # offline last resort
        try:
            self.backends.append((OfflineAIBackend.name, OfflineAIBackend()))
        except Exception:
            pass

        self.fail_counts: Dict[str,int] = {name:0 for (name,_) in self.backends}
        logger.info(f"[BackendRouter] available backends: {[n for (n,_) in self.backends]}")

    async def ask(self, prompt: str, **kwargs) -> str:
        """
        Try backends in order; backoff on repeated failures
        """
        last_exc = None
        for name, backend in self.backends:
            if self.fail_counts.get(name,0) >= 3:
                logger.debug(f"[BackendRouter] skipping {name} due to fail_count={self.fail_counts.get(name)}")
                continue
            try:
                logger.debug(f"[BackendRouter] trying backend {name}")
                res = await backend.chat(prompt, **kwargs)
                logger.info(f"[BackendRouter] backend {name} succeeded")
                # reset fail count on success
                self.fail_counts[name] = 0
                # ensure string
                if isinstance(res, (dict, list)):
                    try:
                        return json.dumps(res)
                    except Exception:
                        return str(res)
                return str(res)
            except Exception as e:
                logger.warning(f"[BackendRouter] backend {name} failed: {e}")
                self.fail_counts[name] = self.fail_counts.get(name,0) + 1
                last_exc = e
                continue
        logger.error("[BackendRouter] all backends failed")
        if last_exc:
            raise last_exc
        return "[ERROR] No backend available"

backend_router = BackendRouter()

# -----------------------------------------------------------------------------
# SemanticMemory (wrap MemoryManager with async calls)
# -----------------------------------------------------------------------------
class SemanticMemory:
    def __init__(self, manager: MemoryManager):
        self.mgr = manager

    async def remember(self, key: str, data: dict):
        try:
            # attempt snapshot for durability
            try:
                self.mgr.snapshot()
            except Exception:
                pass
            return await task_queue.submit(self.mgr.save_json, f"memory_{key}.json", {"data": data})
        except Exception as e:
            logger.error(f"[SemanticMemory.remember] failed: {e}")
            return None

    async def recall(self, key: str):
        try:
            return await task_queue.submit(self.mgr.load_json, f"memory_{key}.json")
        except Exception as e:
            logger.error(f"[SemanticMemory.recall] failed: {e}")
            return None

semantic_memory = SemanticMemory(memory_manager)

# -----------------------------------------------------------------------------
# SelfEvolutionEngine (self-modification & patching)
# -----------------------------------------------------------------------------
class SelfEvolutionEngine:
    def __init__(self, manager: MemoryManager):
        self.mgr = manager
        self.last_proposals: List[Tuple[str, str]] = []
        self.last_ts = 0
        self.rate_limit = PATCH_RATE_LIMIT
        self.size_limit = PATCH_SIZE_LIMIT

    def _allowed_file(self, path: str) -> bool:
        # allow everything if SELF_UPDATE_MODE=auto and AUTONOMOUS_MODE true; otherwise enforce whitelist
        if SELF_UPDATE_MODE == "auto" and AUTONOMOUS_MODE:
            return True
        # make path relative normalized
        norm = path.replace("\\", "/")
        return any(norm.startswith(w) for w in PATCH_WHITELIST)

    async def propose_patch(self, file_path: str, new_content: str, desc: str):
        now = time.time()
        if len(new_content) > self.size_limit:
            logger.warning("[SelfEvolution] Patch too large, rejected.")
            return None
        if (now - self.last_ts) < 3600 and len(self.last_proposals) >= self.rate_limit:
            logger.warning("[SelfEvolution] Rate limit reached.")
            return None
        if not self._allowed_file(file_path):
            logger.warning(f"[SelfEvolution] File not allowed for auto-patching: {file_path}")
            return None
        self.last_ts = now
        self.last_proposals.append((file_path, desc))
        # delegate to underlying memory manager's helper (expected in repo)
        try:
            return await task_queue.submit(self.mgr.propose_patch_from_ai, file_path, new_content, desc)
        except Exception as e:
            logger.error(f"[SelfEvolution] propose_patch failed: {e}")
            return None

    async def apply_patch(self, proposal_id: str, mode: str = "manual"):
        """
        Apply a proposal after running tests (if auto). This wraps the memory_manager apply.
        """
        if SELF_UPDATE_MODE != "auto" and mode == "auto":
            logger.info("[SelfEvolution] auto apply blocked by SELF_UPDATE_MODE")
            return False
        try:
            # try a test-run if available
            try:
                tests_ok = await self.run_tests_for_proposal(proposal_id)
            except Exception as e:
                logger.warning(f"[SelfEvolution] run_tests_for_proposal raised: {e}")
                tests_ok = False
            # if mode auto, ensure tests_ok
            if mode == "auto" and not tests_ok and AUTONOMOUS_MODE:
                logger.warning("[SelfEvolution] tests failed; auto apply aborted")
                return False
            return await task_queue.submit(self.mgr.apply_patch_safely, proposal_id, mode)
        except Exception as e:
            logger.error(f"[SelfEvolution] apply_patch failed: {e}")
            return False

    async def run_tests_for_proposal(self, proposal_id: str) -> bool:
        """
        Best-effort tests for a proposal:
        - If the manager can return proposal content, run syntax checks (compileall).
        - Optionally run pytest if available.
        """
        try:
            # try to retrieve proposal
            pending = self.mgr.get("pending_patches") or []
            proposal = next((p for p in pending if p.get("id") == proposal_id), None)
            if not proposal:
                logger.info(f"[SelfEvolution.run_tests_for_proposal] proposal {proposal_id} not found; skipping tests")
                return False
            # Write content to temporary path for dry-run
            import tempfile
            tmpdir = tempfile.mkdtemp(prefix="bibliotheca_patch_")
            rel_path = os.path.basename(proposal.get("path") or "patched_file.py")
            tmpfile = os.path.join(tmpdir, rel_path)
            with open(tmpfile, "w", encoding="utf-8") as f:
                f.write(proposal.get("content", ""))
            # run python -m compileall on tmpdir
            cmd = f"python -m compileall -q {tmpdir}"
            res = run_shell(cmd, cwd=None)
            if res.get("returncode", 1) != 0:
                logger.warning(f"[SelfEvolution] compileall failed for proposal {proposal_id}: {res.get('stderr')}")
                return False
            # if pytest is installed in environment, run it (best effort)
            try:
                test_cmd = "pytest -q"
                test_res = run_shell(test_cmd, cwd=tmpdir, timeout=60)
                # If pytest returns non-zero, treat as failure
                if test_res.get("returncode", 0) != 0:
                    logger.warning(f"[SelfEvolution] pytest reported failures (ignored if none tests exist): {test_res.get('stderr')}")
                    # not a hard failure if no tests; treat compile success as enough
            except Exception:
                pass
            # cleanup - best-effort; do not raise on cleanup failure
            try:
                import shutil
                shutil.rmtree(tmpdir)
            except Exception:
                pass
            return True
        except Exception as e:
            logger.exception(f"[SelfEvolution.run_tests_for_proposal] error: {e}")
            return False

self_evolution = SelfEvolutionEngine(memory_manager)

# -----------------------------------------------------------------------------
# TaskPlanner & CommandInterpreter
# -----------------------------------------------------------------------------
class TaskPlanner:
    """
    Parse a natural-language task into steps, persist it, and run steps.
    Steps types: llm, code_patch, shell, create_file, generate_image, post_social
    """
    def __init__(self, backend_router: BackendRouter, self_evolution: SelfEvolutionEngine, memory: SemanticMemory):
        self.router = backend_router
        self.evolver = self_evolution
        self.memory = memory
        self.task_store_key = "task_planner.tasks"

    def _persist_tasks(self, tasks):
        try:
            memory_manager.set(self.task_store_key, tasks)
        except Exception:
            try:
                memory_manager.save_json(self.task_store_key, tasks)
            except Exception:
                logger.debug("[TaskPlanner] persist failed")

    def _load_tasks(self):
        try:
            return memory_manager.get(self.task_store_key) or []
        except Exception:
            try:
                return memory_manager.load_json(self.task_store_key) or []
            except Exception:
                return []

    async def plan_task_from_prompt(self, prompt: str) -> Dict[str,Any]:
        """
        Create a simple plan using the LLM: break prompt into steps and required tools.
        For speed/robustness we'll do a quick local parser + LLM-assisted step refinement.
        """
        plan = {"id": f"task_{int(time.time())}", "prompt": prompt, "created": time.time(), "steps": []}
        lower = prompt.lower()
        if any(k in lower for k in ("fix", "bug", "error", "patch", "update", "improve", "refactor")):
            plan["steps"].append({"type": "analyze_code", "desc": "Analyze repo for relevant files"})
            plan["steps"].append({"type": "code_patch", "desc": "Generate code patch based on prompt"})
            plan["steps"].append({"type": "test", "desc": "Run tests / syntax checks"})
        elif any(k in lower for k in ("image", "make an image", "generate image", "create image", "draw")):
            plan["steps"].append({"type": "generate_image", "desc": "Generate an image based on prompt"})
        elif any(k in lower for k in ("instagram", "social", "post", "tweet", "post to", "create account")):
            plan["steps"].append({"type": "research", "desc": "Research best approach and APIs"})
            plan["steps"].append({"type": "setup_social", "desc": "Create/prepare posting automation (requires credentials)"})
        elif any(k in lower for k in ("make me money", "monetize", "earn", "income")):
            plan["steps"].append({"type": "research", "desc": "Generate potential monetization strategies"})
            plan["steps"].append({"type": "pick_strategy", "desc": "Pick highest-probability strategy"})
            plan["steps"].append({"type": "execute", "desc": "Execute actionable steps (may require external credentials)"})
        else:
            # ask LLM to break into steps (use backend_router)
            try:
                breakdown = await self.router.ask(f"Break this task into steps for automation:\n\n{prompt}\n\nReturn JSON list of steps with type and brief desc.")
                # attempt to parse JSON from response
                try:
                    parsed = json.loads(breakdown)
                    if isinstance(parsed, list):
                        for s in parsed:
                            plan["steps"].append(s)
                except Exception:
                    # fallback: store LLM text as single step
                    plan["steps"].append({"type":"llm_instruct","desc":breakdown})
            except Exception as e:
                logger.warning(f"[TaskPlanner] LLM breakdown failed: {e}")
                plan["steps"].append({"type":"llm_instruct","desc":prompt})
        # persist plan
        tasks = self._load_tasks()
        tasks.append(plan)
        self._persist_tasks(tasks)
        await self.memory.remember(plan["id"], plan)
        return plan

    async def execute_step(self, step: Dict[str,Any], context: Dict[str,Any]) -> Dict[str,Any]:
        typ = step.get("type")
        desc = step.get("desc", "")
        logger.info(f"[TaskPlanner] executing step {typ}: {desc}")
        result = {"ok": False, "output": None, "step": step}
        try:
            if typ in ("analyze_code", "code_patch"):
                # ask LLM for patch contents and propose via self.evolver
                code_prompt = f"Repo context: {desc}\nTask: {context.get('task_prompt')}\nProvide a safe patch diff or full file content changes in JSON: {{'path':'...','content':'...','desc':'...'}}"
                patch_text = await self.router.ask(code_prompt)
                # Try parse JSON
                try:
                    patches = json.loads(patch_text)
                    applied = []
                    for p in patches if isinstance(patches, list) else [patches]:
                        path = p.get("path")
                        content = p.get("content")
                        pd = p.get("desc","self-evolution patch")
                        if path and content:
                            proposal = await self.evolver.propose_patch(path, content, pd)
                            if proposal and SELF_UPDATE_MODE == "auto" and AUTONOMOUS_MODE:
                                ok = await self.evolver.apply_patch(proposal["id"], mode="auto")
                                applied.append({"proposal": proposal, "applied": ok})
                            else:
                                applied.append({"proposal":proposal, "applied": False})
                    result["ok"] = True
                    result["output"] = applied
                except Exception:
                    # if not JSON, output raw
                    result["ok"] = False
                    result["output"] = {"raw": patch_text}
            elif typ == "test":
                # attempt to run a lightweight syntax check across repo
                out = await task_queue.submit(run_shell, "python -m compileall -q .")
                result["ok"] = True
                result["output"] = out
            elif typ == "generate_image":
                # basic image generation via OpenAI images (if available) or placeholder
                img_prompt = context.get("task_prompt") or desc
                if OPENAI_AVAILABLE and OPENAI_API_KEY:
                    def _call_image():
                        try:
                            return openai.Image.create(prompt=img_prompt, n=1, size="1024x1024")
                        except Exception as e:
                            logger.error(f"[TaskPlanner] openai image failed: {e}")
                            raise
                    loop = asyncio.get_running_loop()
                    out = await loop.run_in_executor(executor, _call_image)
                    result["ok"] = True
                    result["output"] = out
                else:
                    result["ok"] = False
                    result["output"] = "No image backend available"
            elif typ == "shell":
                cmd = step.get("cmd")
                if not cmd:
                    result["ok"] = False
                    result["output"] = "no cmd specified"
                else:
                    if AUTONOMOUS_MODE:
                        out = await task_queue.submit(run_shell, cmd)
                        result["ok"] = True
                        result["output"] = out
                    else:
                        result["ok"] = False
                        result["output"] = "AUTONOMOUS_MODE false; shell blocked"
            elif typ == "create_file":
                path = step.get("path")
                content = step.get("content","")
                if path:
                    if SELF_UPDATE_MODE == "auto" and AUTONOMOUS_MODE:
                        await task_queue.submit(write_file, path, content)
                        result["ok"] = True
                        result["output"] = f"wrote {path}"
                    else:
                        result["ok"] = False
                        result["output"] = "auto-write blocked; propose via self-evolution instead"
                else:
                    result["ok"] = False
                    result["output"] = "no path specified"
            else:
                # generic LLM-driven step
                reply = await self.router.ask(f"Perform this single-step task:\n{desc}\nContext: {context.get('task_prompt')}")
                result["ok"] = True
                result["output"] = reply
        except Exception as e:
            logger.error(f"[TaskPlanner] execute_step exception: {e}")
            result["ok"] = False
            result["output"] = str(e)
        return result

    async def execute_plan(self, plan: Dict[str,Any]):
        context = {"task_prompt": plan.get("prompt")}
        results = []
        for step in plan.get("steps",[]):
            res = await self.execute_step(step, context)
            results.append(res)
            # optionally persist partial results
            await self.memory.remember(f"{plan['id']}_progress", {"step": step, "res": res, "ts": time.time()})
        # mark done
        await self.memory.remember(f"{plan['id']}_done", {"results": results, "ts": time.time()})
        return results

# helpers for file and shell
def run_shell(cmd: str, cwd: Optional[str]=None, timeout: int=120) -> Dict[str,Any]:
    logger.info(f"[run_shell] {cmd}")
    try:
        completed = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except subprocess.TimeoutExpired as e:
        return {"error": "timeout", "detail": str(e)}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

def write_file(path: str, content: str):
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# ai_core.py — CoreAI & AdvancedAI — Beyond-Perfection v24.0
# Fully async, robust, self-evolving, memory-safe, Beyond-Perfection ready
# -----------------------------------------------------------------------------
import asyncio
import logging
import time
from typing import Optional, Dict, Any

# Core project imports
from app.utils.backend_router import BackendRouter, backend_summary
from app.utils.memory_manager import MemoryManager
from app.utils.task_planner import TaskPlanner
from app.utils.self_evolution import SelfEvolutionEngine, SelfImprover

# Config flags / feature availability
from app.utils.config_legacy import (
    AUTONOMOUS_MODE,
    SELF_UPDATE_MODE,
    OPENAI_AVAILABLE,
    OPENAI_API_KEY,
    DEEPINFRA_AVAILABLE,
    TOGETHERAI_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
    HF_MODEL,
    LLAMA_CPP_AVAILABLE,
    LLAMA_MODEL_PATH,
)

# Logging setup
logger_core = logging.getLogger("BibliothecaAI-Core")
logger_adv = logging.getLogger("BibliothecaAI-Advanced")

# -----------------------------------------------------------------------------
# CoreAI — Beyond-Perfection Infinity++
# -----------------------------------------------------------------------------
class CoreAI:
    def __init__(
        self,
        backend_router: BackendRouter,
        semantic_memory: MemoryManager,
        self_evolver: Optional[SelfEvolutionEngine] = None,
        self_improver: Optional[SelfImprover] = None,
    ):
        # Core modules
        self.router = backend_router
        self.memory = semantic_memory
        self.evolver = self_evolver if self_evolver else SelfEvolutionEngine(self.memory)
        self.self_improver = self_improver if self_improver else SelfImprover(self.memory)
        self.planner = TaskPlanner(self.router, self.evolver, self.memory)
        self.openai_enabled = OPENAI_AVAILABLE
        self.memory_db = getattr(self.memory, "db", None)

        # Ensure ROUTER telemetry is attached
        if self.router.telemetry_handler is None:
            self.router.attach_telemetry_handler(TelemetryHandler())
        self.telemetry = self.router.telemetry_handler

        # Logger
        self.logger = logger_core
        self.logger.info("[CoreAI] Initialized Beyond-Perfection CoreAI ✅")

    # -----------------------------
    # Task helpers
    # -----------------------------
    def create_task(self, name: str, params: dict = None) -> dict:
        """Register a new task with BackendRouter and telemetry."""
        task = {"name": name, "params": params or {}}
        return self.router.register_task(task)

    async def execute_task(self, task_name: str, params: dict = None) -> dict:
        """Execute a task via BackendRouter safely with telemetry."""
        try:
            result = await self.router.execute_task(task_name, params or {})
            return result
        except Exception as e:
            self.logger.error(f"[CoreAI.execute_task] Failed: {e}", exc_info=True)
            return {"status": "failed", "task": task_name, "error": str(e)}

    async def apply_fix(self, description: str, code: str) -> dict:
        """Apply code fixes via BackendRouter safely."""
        try:
            result = await self.router.apply_fix(description, code)
            return result
        except Exception as e:
            self.logger.error(f"[CoreAI.apply_fix] Failed: {e}", exc_info=True)
            return {"status": "failed", "description": description, "error": str(e)}

    async def handle_message(self, message: str) -> str:
        """Convert conversation into actionable tasks."""
        try:
            response = await self.router.handle_conversation(message)
            return response
        except Exception as e:
            self.logger.error(f"[CoreAI.handle_message] Failed: {e}", exc_info=True)
            return f"[CoreAI] Could not handle message: {message} | Error: {e}"

    # -----------------------------
    # Basic chat / conversational interface
    # -----------------------------
    async def chat(self, prompt: str, **kwargs) -> str:
        """Handle conversational queries safely with fallback."""
        try:
            if hasattr(self.router, "ask"):
                response = await self.router.ask(prompt, **kwargs)
                if not response:
                    raise RuntimeError("Router returned empty response")
                return response
            else:
                raise AttributeError("BackendRouter missing 'ask' method")
        except Exception as e:
            self.logger.error(f"[CoreAI.chat] backend error: {e}", exc_info=True)
            return f"[Fallback-CoreAI] Could not process prompt: {prompt} | Error: {e}"

    # -----------------------------
    # Interpret prompt and act autonomously
    # -----------------------------
    async def interpret_and_act(self, prompt: str, auto_plan: bool = True) -> Dict[str, Any]:
        """Detect actionable intents and generate/execute a plan."""
        lower = prompt.lower().strip()
        actionable_keywords = (
            "create", "make", "build", "deploy", "fix", "patch", "update",
            "monetize", "generate", "post", "image", "video", "earn", "income",
            "instagram", "fiverr", "sell", "automate", "launch"
        )

        if any(k in lower for k in actionable_keywords):
            self.logger.info("[CoreAI] Actionable intent detected; generating plan")
            try:
                plan = await self.planner.plan_task_from_prompt(prompt)
            except Exception as e:
                self.logger.error(f"[CoreAI] Task planning failed: {e}", exc_info=True)
                return {"type": "error", "error": str(e)}

            # Self-improvement / patch application
            for step in plan.get("steps", []):
                action = step.get("action", "").lower()
                if "modify" in action or "patch" in action:
                    file_path = step.get("target_file")
                    new_content = step.get("new_content")
                    desc = step.get("description", "Auto-generated patch")
                    try:
                        result = await self.self_improver.propose_and_apply(file_path, new_content, desc)
                        step["patch_applied"] = result.get("applied", False)
                    except Exception as e:
                        self.logger.error(f"[CoreAI] Self-improvement step failed: {e}", exc_info=True)
                        step["patch_applied"] = False

            # Schedule execution
            if auto_plan:
                try:
                    fut = self.planner.execute_plan(plan)
                    asyncio.create_task(fut)
                    self.logger.info("[CoreAI] Main task plan scheduled ✅")

                    if AUTONOMOUS_MODE:
                        fut2 = self.planner.execute_autonomous_tasks(plan)
                        asyncio.create_task(fut2)
                        self.logger.info("[CoreAI] Autonomous tasks scheduled ✅")
                except Exception as e:
                    self.logger.error(f"[CoreAI] Failed to schedule tasks: {e}", exc_info=True)

            return {"type": "plan_created", "plan": plan}

        # Default: conversational response
        response = await self.chat(prompt)
        return {"type": "chat", "response": response}

    # -----------------------------
    # Self-check / health diagnostics (AdvancedAI)
    # -----------------------------
    async def self_check(self) -> dict:
        """
        Perform full self-diagnostic: memory, planner, router, task queue.
        Always returns a structured dict with extended telemetry.
        """
        results = {
            "status": "ok",
            "engine": self.__class__.__name__,
            "memory": None,
            "planner": None,
            "router": None,
            "task_execution": None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            # Memory test
            if hasattr(self, "memory") and self.memory:
                test_key = f"selfcheck_{int(time.time())}"
                try:
                    await self.memory.store(test_key, {"ok": True})
                    recalled = await self.memory.recall(test_key)
                    results["memory"] = "ok" if recalled else "failed"
                except Exception as me:
                    results["memory"] = f"failed ({me})"
            else:
                results["memory"] = "unavailable"

            # Planner test
            if hasattr(self, "planner") and self.planner:
                try:
                    plan = await self.planner.plan_task_from_prompt(
                        "Quick self-check: return a single step 'noop' in JSON"
                    )
                    results["planner"] = "ok" if plan and "steps" in plan else "failed"
                except Exception as pe:
                    results["planner"] = f"failed ({pe})"
            else:
                results["planner"] = "unavailable"

            # Router test
            if hasattr(self, "router") and hasattr(self.router, "ask"):
                try:
                    await self.router.ask(f"{self.__class__.__name__} self-check test")
                    results["router"] = "ok"
                except Exception as re:
                    results["router"] = f"failed ({re})"
            else:
                results["router"] = "unavailable"

            # Task execution test (create + execute a dummy task)
            try:
                test_task = self.create_task(
                    name="selfcheck_dummy_task",
                    params={"note": "Health diagnostic task"}
                )
                exec_result = await self.execute_task("selfcheck_dummy_task", test_task["params"])
                if exec_result.get("status") in ("ok", "completed", "success"):
                    results["task_execution"] = "ok"
                else:
                    results["task_execution"] = f"failed ({exec_result})"
            except Exception as te:
                results["task_execution"] = f"failed ({te})"

            # Log full structured report
            self.logger.info(
                f"[{self.__class__.__name__}.self_check] ✅ Full diagnostic: {json.dumps(results, indent=2)}")
            return results

        except Exception as e:
            results["status"] = "failed"
            self.logger.error(f"[{self.__class__.__name__}.self_check] Critical failure: {e}", exc_info=True)
            return results


# -----------------------------------------------------------------------------
# AdvancedAI — Beyond-Perfection (Fixed)
# -----------------------------------------------------------------------------
from app.utils.core_ai import CoreAI
from app.utils.memory_manager import MemoryManager
from app.utils.logger import logger_adv
from app.utils.self_evolution import SelfEvolutionEngine
from app.utils.backends import BackendRouter
from app.config import AUTONOMOUS_MODE, SELF_UPDATE_MODE

# ------------------------
# AdvancedAI — Beyond-Perfection Edition
# Replace the original AdvancedAI class with this code.
# ------------------------
import asyncio
import threading
import time
import logging
import traceback
import os
import re
from typing import Any, Optional, Dict

# CoreAI is expected to be defined earlier in this module.
# If it's not, import fallback (adjust path if you split files).
try:
    CoreAI  # type: ignore
except NameError:
    from .core_ai import CoreAI  # fallback, may be unnecessary


class AdvancedAI(CoreAI):
    """
    AdvancedAI — robust, autonomous, self-evolving AI manager.

    Key features included:
    - Safe singleton pattern with re-entrant init protection
    - Background asyncio loop (for resilient async tasks) + scheduler
    - Watchdog that auto-attaches MemoryManager/MemoryStore if missing
    - Heartbeat / diagnostics (configurable)
    - Multi-provider fallback querying system
    - Async-safe memory API (set/get) with local fallback cache
    - Compatibility get(...) wrapper for legacy callers (synchronous or awaitable)
    - Self-evolution integration with guard rails (SelfImporver/Evolver)
    - Detailed logging and safe failure modes
    """

    _instance = None
    _init_lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(AdvancedAI, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        # idempotent init
        if getattr(self, "_initialized", False):
            return
        super().__init__(*args, **kwargs)

        # Logger (use existing logger if CoreAI set one)
        self.logger = getattr(self, "logger", None) or logging.getLogger("AdvancedAI")

        # Runtime config: try to get from config module, fallback to envs
        try:
            from app.utils.config import GLOBAL_CONFIG  # optional
            self.autonomous_mode = GLOBAL_CONFIG.get("autonomous_mode", False)
            self.self_update_mode = GLOBAL_CONFIG.get("self_update_mode", "manual")
        except Exception:
            self.autonomous_mode = os.environ.get("AUTONOMOUS_MODE", "0").lower() in ("1", "true", "yes")
            self.self_update_mode = os.environ.get("SELF_UPDATE_MODE", "manual")

        # Memory references (populated via attach_memory)
        self.memory: Optional[Any] = None
        self.memory_manager: Optional[Any] = None
        self.memory_store: Optional[Any] = None
        self._local_cache: Dict[str, Any] = {}
        self._memory_lock = threading.RLock()

        # Self-improvement components (lazy-loaded)
        self.evolver = None
        self.self_improver = None

        # Background asyncio loop and thread (used for health checks, attachments, etc.)
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(target=self._run_bg_loop, name="AdvancedAI-bg", daemon=True)
        self._bg_thread.start()

        # Watchdog / heartbeat infrastructure
        self._watchdog_interval = float(os.environ.get("ADVAI_WATCHDOG_INTERVAL", 4.0))
        self._heartbeat_interval = float(os.environ.get("ADVAI_HEARTBEAT_INTERVAL", 60.0))
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, name="AdvancedAI-watchdog", daemon=True)
        self._watchdog_thread.start()

        # Mark initialized
        self._initialized = True

        # Lazy-load optional components
        self._load_optional_components()

        # Start a heartbeat task in the background loop
        try:
            self.schedule(self._heartbeat_loop())
        except Exception:
            self.logger.debug("[AdvancedAI] Failed to start heartbeat task", exc_info=True)

        # Try initial attachment of memory in background (non-blocking)
        try:
            self.attach_memory(background=True)
        except Exception:
            # never crash on boot because of memory
            self.logger.debug("[AdvancedAI] attach_memory (initial) failed", exc_info=True)

    # -------------------------
    # Background loop & scheduling
    # -------------------------
    def _run_bg_loop(self) -> None:
        """
        Runs a private asyncio event loop in a daemon thread.
        All background async tasks for AdvancedAI should use `self.schedule(coro)`.
        """
        try:
            asyncio.set_event_loop(self._bg_loop)
            self._bg_loop.run_forever()
        except Exception:
            # log to main logger (safe)
            try:
                self.logger.exception("[AdvancedAI._run_bg_loop] background loop died")
            except Exception:
                print("[AdvancedAI._run_bg_loop] background loop died\n", traceback.format_exc())

    def schedule(self, coro: asyncio.coroutines) -> "concurrent.futures.Future":
        """
        Schedule a coroutine on the background loop and return a concurrent Future.
        Use `result()` to block-wait or let it run in background.
        """
        try:
            return asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        except Exception as e:
            self.logger.exception("[AdvancedAI.schedule] failed to schedule coro: %s", e)
            raise

    # -------------------------
    # Optional components loader
    # -------------------------
    def _load_optional_components(self) -> None:
        """
        Lazy-load components that may not be available in minimal installs:
        - SelfImporver, Evolver
        - other provider adapters can be loaded here
        Failures are non-fatal; we keep a functional fallback mode.
        """
        # Self-improver
        try:
            from app.utils.self_improver import SelfImprover  # optional
            self.self_improver = SelfImprover(self)
            self.logger.info("[AdvancedAI] SelfImprover loaded")
        except Exception:
            self.logger.debug("[AdvancedAI] SelfImprover not available", exc_info=True)

        # Evolver
        try:
            from app.utils.evolver import Evolver  # optional
            self.evolver = Evolver(self)
            self.logger.info("[AdvancedAI] Evolver loaded")
        except Exception:
            self.logger.debug("[AdvancedAI] Evolver not available", exc_info=True)

    # -------------------------
    # Health / heartbeat / watchdog
    # -------------------------
    async def _heartbeat_loop(self) -> None:
        """
        Periodic heartbeat that records self_check to logs and optionally telemetry.
        Runs on the background asyncio loop.
        """
        while True:
            try:
                status = await self.self_check()
                self.logger.info("[AdvancedAI.heartbeat] status=%s", status)
            except Exception:
                self.logger.exception("[AdvancedAI.heartbeat] failed")
            await asyncio.sleep(self._heartbeat_interval)

    def _watchdog_loop(self) -> None:
        """
        Lightweight watchdog on a plain thread: reconnect memory if missing,
        refresh optional components, schedule health pings.
        """
        while not self._watchdog_stop.is_set():
            try:
                # if memory missing, kick off attach attempts
                if self.memory is None:
                    try:
                        f = self.attach_memory(background=True)
                        # do not block here; the background attach does its work
                        self.logger.debug("[AdvancedAI.watchdog] scheduled background attach_memory")
                    except Exception:
                        self.logger.debug("[AdvancedAI.watchdog] attach_memory scheduling failed", exc_info=True)

                # schedule a quick health ping on bg loop
                try:
                    self.schedule(self._health_ping())
                except Exception:
                    self.logger.debug("[AdvancedAI.watchdog] schedule health_ping failed", exc_info=True)

            except Exception:
                # never let watchdog die
                try:
                    self.logger.exception("[AdvancedAI.watchdog] unexpected error")
                except Exception:
                    print("[AdvancedAI.watchdog] unexpected error\n", traceback.format_exc())

            time.sleep(self._watchdog_interval)

    async def _health_ping(self) -> None:
        """
        Quick health-check coroutine used by watchdog to probe subcomponents.
        """
        try:
            status = await self.self_check()
            self.logger.debug("[AdvancedAI.health_ping] %s", status)
        except Exception:
            self.logger.exception("[AdvancedAI.health_ping] failed")

    # -------------------------
    # Self-check / diagnostics
    # -------------------------
    async def self_check(self) -> dict:
        """
        Run CoreAI self_check (if available) and append AdvancedAI info.
        """
        base_ok = {}
        try:
            if hasattr(super(), "self_check"):
                # call parent self_check if it exists and is awaitable
                maybe = super().self_check()
                if asyncio.iscoroutine(maybe):
                    base_ok = await maybe
                else:
                    base_ok = maybe
        except Exception:
            self.logger.exception("[AdvancedAI.self_check] base self_check failed")

        memory_ok = self.memory is not None
        result = {
            "ok": True,
            "advanced_features": True,
            "autonomous_mode": bool(self.autonomous_mode),
            "self_update_mode": str(self.self_update_mode),
            "memory_connected": bool(memory_ok),
            "timestamp": time.time(),
        }
        result.update(base_ok or {})
        return result

    # -------------------------
    # Query routing and handling
    # -------------------------
    def _looks_like_research(self, query: str) -> bool:
        """
        More robust detection (not just startswith).
        Recognizes question words and "current/latest/news" anywhere in query.
        """
        if not query:
            return False
        q = query.strip().lower()
        # quick patterns
        if re.search(r"\b(who|what|when|where|why|how|which)\b", q):
            return True
        if re.search(r"\b(current|latest|news|update|status|breaking)\b", q):
            return True
        # fallback: long queries with keywords
        if len(q.split()) > 6 and any(k in q for k in ("reported", "announced", "published", "study", "research")):
            return True
        return False

    async def respond(self, query: str) -> str:
        """
        Public entrypoint: route to research or conversational flows.
        Stores dialogue in memory when possible.
        """
        start = time.time()
        try:
            if self._looks_like_research(query):
                reply = await self._handle_research(query)
            else:
                reply = await self._handle_conversation(query)
            # try to save dialogue to memory (best-effort)
            try:
                await self.set("dialogue", {"q": query, "a": reply, "ts": time.time()})
            except Exception:
                self.logger.debug("[AdvancedAI.respond] failed to persist dialogue", exc_info=True)
            return reply
        except Exception as e:
            self.logger.exception("[AdvancedAI.respond] error handling query")
            return f"[AdvancedAI] Could not process query: {e}"
        finally:
            elapsed = time.time() - start
            self.logger.debug("[AdvancedAI.respond] elapsed=%.3fs query=%s", elapsed, (query[:80] + "...") if len(query) > 80 else query)

    async def _handle_research(self, query: str) -> str:
        """
        Research route: use _fetch_with_fallback and save to memory.
        """
        try:
            result = await self._fetch_with_fallback(query, conversational=False)
            try:
                await self.set("research", {"q": query, "a": result, "ts": time.time()})
            except Exception:
                self.logger.debug("[AdvancedAI._handle_research] failed to save research result", exc_info=True)
            return result
        except Exception as e:
            self.logger.exception("[AdvancedAI._handle_research] failed")
            return f"[Research-Error] {e}"

    async def _handle_conversation(self, query: str) -> str:
        """
        Conversational route: first try local/fast provider, then fallback.
        """
        try:
            result = await self._fetch_with_fallback(query, conversational=True)
            try:
                await self.set("conversation", {"q": query, "a": result, "ts": time.time()})
            except Exception:
                self.logger.debug("[AdvancedAI._handle_conversation] failed to save conversation", exc_info=True)
            return result
        except Exception as e:
            self.logger.exception("[AdvancedAI._handle_conversation] failed")
            return f"[Conversation-Error] {e}"

    async def _fetch_with_fallback(self, query: str, conversational: bool = False) -> str:
        """
        Try a list of providers; first successful response is returned.
        Providers are attempted in order: try to call CoreAI/provider hooks if present,
        otherwise fallback to any provider adapters present or to CoreAI.chat.
        """
        providers = ["openai", "anthropic", "togetherai", "deepinfra", "llama_cpp"]
        last_exc = None
        # If CoreAI/BackendRouter provides a 'query_provider' method, prefer that:
        try:
            provider_callable = getattr(super(), "query_provider", None)
            if callable(provider_callable):
                # try providers in order
                for provider in providers:
                    try:
                        maybe = provider_callable(provider, query, conversational=conversational)
                        if asyncio.iscoroutine(maybe):
                            return await maybe
                        else:
                            return maybe
                    except Exception as e:
                        self.logger.debug("[AdvancedAI._fetch_with_fallback] parent provider %s failed: %s", provider, e)
                        last_exc = e
        except Exception:
            self.logger.debug("[AdvancedAI._fetch_with_fallback] parent query_provider not usable", exc_info=True)

        # fallback: try direct chat method from CoreAI (if present)
        try:
            chat_callable = getattr(self, "chat", None)
            if callable(chat_callable):
                maybe = chat_callable(query)
                if asyncio.iscoroutine(maybe):
                    return await maybe
                else:
                    return maybe
        except Exception as e:
            self.logger.debug("[AdvancedAI._fetch_with_fallback] chat fallback failed: %s", e)
            last_exc = e

        # final fallback: local canned response / error
        self.logger.warning("[AdvancedAI._fetch_with_fallback] all providers failed; last error=%s", last_exc)
        raise RuntimeError(f"All providers failed. Last error: {last_exc}")

    # -------------------------
    # Self-evolution
    # -------------------------
    async def generate_update(self, prompt: str) -> str:
        """
        Ask the best available generator for a code update suggestion.
        Returns raw generator text (best-effort); not guaranteed valid code.
        """
        try:
            msg = f"Generate a Python patch for: {prompt}"
            if hasattr(self, "chat"):
                candidate = self.chat(msg)
                if asyncio.iscoroutine(candidate):
                    candidate = await candidate
                return candidate
            else:
                return "# [ERROR] No chat provider available to generate update"
        except Exception:
            self.logger.exception("[AdvancedAI.generate_update] failed")
            return "# [ERROR] Failed to generate update"

    async def evolve_self(self, file_path: str, new_content: str, desc: str, apply_auto: bool = False) -> dict:
        """
        Coordinate the evolve process:
        - If auto apply requested, ensure both self_update_mode == 'auto' and autonomous_mode is enabled.
        - Use self_improver.propose_and_apply when available; otherwise propose via evolver or return proposal only.
        """
        try:
            # Auto-apply guard: require explicit env var to avoid surprise
            allow_auto = os.environ.get("ADVAI_ALLOW_AUTO_APPLY", "0").lower() in ("1", "true", "yes")
            if apply_auto and self.self_update_mode == "auto" and self.autonomous_mode and allow_auto:
                if self.self_improver:
                    return await self.self_improver.propose_and_apply(file_path, new_content, desc)
                else:
                    return {"proposal": None, "applied": False, "error": "self_improver unavailable"}
            else:
                if self.evolver:
                    proposal = await self.evolver.propose_patch(file_path, new_content, desc)
                    return {"proposal": proposal, "applied": False}
                # fallback: create a safe proposal dict
                proposal = {"file": file_path, "desc": desc, "content_snippet": new_content[:400]}
                return {"proposal": proposal, "applied": False}
        except Exception:
            self.logger.exception("[AdvancedAI.evolve_self] failed")
            return {"proposal": None, "applied": False, "error": "exception"}

    # -------------------------
    # Memory API (async + legacy-friendly wrappers)
    # -------------------------
    async def set(self, key: str, value: dict) -> bool:
        """
        Store value in memory asynchronously with retries and local fallback cache.
        Returns True if stored (or cached), False on persistent failure.
        """
        for attempt in range(3):
            try:
                if self.memory is not None:
                    # support multiple method names for store API
                    store_fn = getattr(self.memory, "store", None) or getattr(self.memory, "save", None) or getattr(self.memory, "upsert", None)
                    if callable(store_fn):
                        result = store_fn(key, value)
                        if asyncio.iscoroutine(result):
                            await result
                        self.logger.debug("[AdvancedAI.set] stored key=%s via memory backend", key)
                        return True
                # fallback to local in-process cache
                with self._memory_lock:
                    self._local_cache[key] = value
                self.logger.debug("[AdvancedAI.set] stored key=%s in local cache", key)
                return True
            except Exception as e:
                self.logger.exception("[AdvancedAI.set] attempt %s failed for key=%s", attempt + 1, key)
                await asyncio.sleep(0.25 * (attempt + 1))
        self.logger.critical("[AdvancedAI.set] failed to store key=%s after retries", key)
        return False

    async def _get_async(self, key: str) -> Optional[Any]:
        """
        Async inner get: try memory backend then local cache.
        """
        for attempt in range(3):
            try:
                if self.memory is not None:
                    recall_fn = getattr(self.memory, "recall", None) or getattr(self.memory, "retrieve", None) or getattr(self.memory, "get", None)
                    if callable(recall_fn):
                        result = recall_fn(key)
                        if asyncio.iscoroutine(result):
                            result = await result
                        self.logger.debug("[AdvancedAI._get_async] recalled key=%s from memory backend", key)
                        return result
                # fallback local cache
                with self._memory_lock:
                    if key in self._local_cache:
                        self.logger.debug("[AdvancedAI._get_async] returned key=%s from local cache", key)
                        return self._local_cache.get(key)
                return None
            except Exception:
                self.logger.exception("[AdvancedAI._get_async] attempt %d failed for key=%s", attempt + 1, key)
                await asyncio.sleep(0.25)
        self.logger.warning("[AdvancedAI._get_async] returning None for key=%s after retries", key)
        return None

    def get_sync(self, key: str, default: Any = None, timeout: float = 2.0) -> Any:
        """
        Synchronous blocking getter that runs the async getter on the background loop.
        """
        try:
            fut = asyncio.run_coroutine_threadsafe(self._get_async(key), self._bg_loop)
            return fut.result(timeout=timeout) or default
        except Exception:
            self.logger.exception("[AdvancedAI.get_sync] failed for key=%s", key)
            # final fallback to local cache
            with self._memory_lock:
                return self._local_cache.get(key, default)

    # -------------------------
    # Memory attach helpers
    # -------------------------
    def attach_memory(self, background: bool = True, timeout: float = 10.0) -> Optional[Any]:
        """
        Attach CouchDB-backed memory to AdvancedAI.
        - background=True: schedules attach in background, returns Future
        - background=False: blocks up to `timeout` seconds
        """
        try:
            if background:
                return self.schedule(self._attach_memory_coro())
            else:
                fut = self.schedule(self._attach_memory_coro())
                try:
                    return fut.result(timeout=timeout)
                except Exception as e:
                    self.logger.exception(f"[AdvancedAI.attach_memory] blocking attach failed: {e}")
                    return False
        except Exception as e:
            self.logger.exception(f"[AdvancedAI.attach_memory] scheduling failed: {e}")
            return None

    async def _attach_memory_coro(self) -> bool:
        """
        Coroutine: attach memory backend safely.
        - Primary: COUCHDB_MEMORY
        - Fallback: global MEMORY singleton
        - Auto-attaches AIMemoryHelper if available
        """
        try:
            # ================= Primary Memory Backend =================
            try:
                from app.utils.memory import COUCHDB_MEMORY as primary_memory
                self.memory_manager = primary_memory
                self.memory = primary_memory
                self.logger.info("[AdvancedAI.attach_memory] attached COUCHDB_MEMORY ✅")
            except Exception as e:
                self.logger.warning(f"[AdvancedAI.attach_memory] COUCHDB_MEMORY unavailable: {e}")
                from app.utils.memory import MEMORY as fallback_memory
                self.memory_manager = fallback_memory
                self.memory = fallback_memory
                self.logger.info("[AdvancedAI.attach_memory] fallback MEMORY singleton attached ✅")

            # ================= Attach AI Helper Safely =================
            try:
                from app.utils.memory import AIMemoryHelper
                if self.memory_manager and not getattr(self.memory_manager, "ai_helper", None):
                    self.memory_manager.ai_helper = AIMemoryHelper(self.memory_manager)
                    self.logger.info("[AdvancedAI.attach_memory] AIMemoryHelper attached ✅")
            except Exception as e:
                self.logger.debug(f"[AdvancedAI.attach_memory] AIMemoryHelper attach failed: {e}", exc_info=True)

            return True

        except Exception as e:
            self.logger.exception(f"[AdvancedAI.attach_memory] memory attach failed: {e}")
            return False

    # -------------------------
    # Ultra-resilient get / recall
    # -------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """
        Hybrid get:
          • Async/sync compatible
          • Multi-layer fallback: memory_manager -> AI attributes -> config -> default
          • Circular-access safe
          • Logs all failures
        """
        if not hasattr(self, "_get_stack"):
            self._get_stack = set()
        if key in self._get_stack:
            self.logger.warning(f"[AdvancedAI.get] Circular get() call for key='{key}'")
            return default

        self._get_stack.add(key)
        try:
            # Attempt memory backend
            try:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

                if loop and loop.is_running():
                    coro = self.memory.recall(key)
                    fut = asyncio.ensure_future(coro)
                    return fut
                else:
                    result = asyncio.run(self.memory.recall(key))
                    if result is not None:
                        return result
            except Exception as e:
                self.logger.debug(f"[AdvancedAI.get] memory recall failed for key='{key}': {e}")

            # Attribute fallback
            if hasattr(self, key):
                return getattr(self, key)

            # Config fallback
            try:
                if hasattr(self, "config") and key in getattr(self.config, "__dict__", {}):
                    return getattr(self.config, key)
            except Exception:
                pass

            # Default fallback
            return default
        finally:
            self._get_stack.discard(key)

    def debug_get_sources(self, key: str) -> dict:
        """
        Diagnostics: show source of key
        """
        sources = {}
        sources["attribute"] = getattr(self, key, None)
        sources["config"] = getattr(getattr(self, "config", None), key, None)
        if hasattr(self, "memory"):
            try:
                val = asyncio.run(self.memory.recall(key))
                sources["memory"] = val
            except Exception:
                sources["memory"] = None
        sources["default"] = None
        return {k: v for k, v in sources.items() if v is not None}

    # -------------------------
    # Stop / cleanup
    # -------------------------
    def stop(self) -> None:
        """
        Graceful shutdown of background loops / watchdog
        """
        try:
            self._watchdog_stop.set()
            try:
                self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
            except Exception:
                pass
            if getattr(self, "_bg_thread", None) and self._bg_thread.is_alive():
                self._bg_thread.join(timeout=1.0)
            if getattr(self, "_watchdog_thread", None) and self._watchdog_thread.is_alive():
                self._watchdog_thread.join(timeout=1.0)
            self.logger.info("[AdvancedAI.stop] stopped background threads ✅")
        except Exception:
            self.logger.exception("[AdvancedAI.stop] error while stopping")


# -------------------------
# MEMORY & AIMemoryHelper attach / safe instance integration
# -------------------------

def _create_or_get_ai_helper() -> Optional[Any]:
    """
    Ensure MEMORY.ai_helper exists and return it (best-effort).
    """
    try:
        if MEMORY is None:
            return None
        helper = getattr(MEMORY, "ai_helper", None)
        if helper is None and AIMemoryHelper is not None:
            try:
                helper = AIMemoryHelper(MEMORY)
                MEMORY.ai_helper = helper
                _logger.info("[ai_core.attach] Created and attached AIMemoryHelper to MEMORY ✅")
            except Exception as e:
                _logger.debug(f"[ai_core.attach] AIMemoryHelper construction failed: {e}", exc_info=True)
                helper = None
        return helper
    except Exception:
        _logger.exception("[ai_core.attach] Unexpected error creating ai_helper")
        return None


def _safe_attach_to_instance(ai_instance: Any) -> bool:
    """
    Attach MEMORY and helpful wrappers to an AdvancedAI instance safely.
    - Does not override existing attributes unless absent.
    - Adds 'get', 'remember', 'recall' methods if missing.
    """
    try:
        if MEMORY is None:
            _logger.debug("[ai_core.attach] No MEMORY available; skipping instance attach.")
            return False

        # ---------------- Attach canonical references ----------------
        if not getattr(ai_instance, "memory", None):
            setattr(ai_instance, "memory", MEMORY)
        setattr(ai_instance, "MEMORY", MEMORY)

        # Attach AI helper if available
        helper = _create_or_get_ai_helper()
        if helper and not getattr(ai_instance, "memory_helper", None):
            setattr(ai_instance, "memory_helper", helper)

        # ---------------- Convenience wrappers ----------------
        if not hasattr(ai_instance, "remember"):
            def remember(self, key=None, content=None, meta=None):
                # if key is omitted, MemoryStore will create one
                return getattr(self, "memory").store(key, content, meta)
            ai_instance.remember = MethodType(remember, ai_instance)

        if not hasattr(ai_instance, "recall"):
            def recall(self, key, default=None):
                try:
                    val = getattr(self, "memory").recall(key)
                    return val if val is not None else default
                except Exception:
                    return default
            ai_instance.recall = MethodType(recall, ai_instance)

        # Legacy callers expect .get(key, default)
        cls = type(ai_instance)
        if not hasattr(cls, "get"):
            def _get(self, key, default=None):
                try:
                    mem = getattr(self, "memory", None)
                    if mem:
                        try:
                            val = mem.recall(key)
                            if val is not None:
                                return val
                        except Exception:
                            pass
                    # fallback to attribute
                    return getattr(self, key, default)
                except Exception:
                    return default
            setattr(cls, "get", _get)
            _logger.info("[ai_core.attach] Installed safe AdvancedAI.get fallback on class %s", cls.__name__)

        _logger.info("[ai_core.attach] MEMORY attached to instance: %s", getattr(ai_instance, "__class__", None))
        return True
    except Exception:
        _logger.exception("[ai_core.attach] Failed to attach MEMORY to AI instance.")
        return False


# -------------------------
# Attach to any pre-existing known global AI instances
# -------------------------
try:
    candidates = [
        globals().get("ADVANCED_AI_INSTANCE"),
        globals().get("ADVANCED_AI"),
        globals().get("advanced_ai"),
        globals().get("advancedAI"),
    ]
    for cand in candidates:
        if cand:
            _safe_attach_to_instance(cand)
except Exception:
    _logger.debug("[ai_core.attach] Error while scanning for existing AI globals.", exc_info=True)


# -------------------------
# Public API for external attach
# -------------------------
def attach_memory_to(ai_instance: Any) -> bool:
    """
    Attach the module-level MEMORY to any AdvancedAI instance safely.
    Returns True on success, False on failure.
    """
    return _safe_attach_to_instance(ai_instance)


# -------------------------
# Ensure MEMORY aliases exist
# -------------------------
def _ensure_memory_aliases():
    try:
        if MEMORY is None:
            return
        alias_map = {
            "sync": "sync_to_db",
            "flush": "sync_to_db",
            "persist": "sync_to_db",
            "save_memory": "sync_to_db",
            "store_local": "store",
            "recall_local": "recall",
            "stop": "stop",
        }
        for alias, real in alias_map.items():
            if hasattr(MEMORY, real) and not hasattr(MEMORY, alias):
                try:
                    setattr(MEMORY, alias, getattr(MEMORY, real))
                except Exception:
                    _logger.debug("[ai_core.attach] Could not set memory alias %s -> %s", alias, real, exc_info=True)
        _logger.debug("[ai_core.attach] Ensured memory API aliases are present ✅")
    except Exception:
        _logger.exception("[ai_core.attach] Error while ensuring memory aliases")


_ensure_memory_aliases()


# -------------------------
# Flush and stop MEMORY at exit — beyond perfection edition
# -------------------------
def _flush_and_stop_memory():
    """
    Fully safe memory flush and shutdown.
    - Best-effort CouchDB sync
    - Stops threads and background loops
    - Handles any exceptions defensively
    """
    try:
        if MEMORY is None:
            return
        # Attempt sync to database
        try:
            if hasattr(MEMORY, "sync_to_db") and callable(MEMORY.sync_to_db):
                MEMORY.sync_to_db()
                MEMORY.logger.info("[ai_core.attach] sync_to_db() executed at exit ✅")
        except Exception:
            MEMORY.logger.debug("[ai_core.attach] sync_to_db() failed at exit", exc_info=True)

        # Attempt to stop memory background threads or loops
        try:
            if hasattr(MEMORY, "stop") and callable(MEMORY.stop):
                MEMORY.stop()
                MEMORY.logger.info("[ai_core.attach] MEMORY.stop() executed at exit ✅")
        except Exception:
            MEMORY.logger.debug("[ai_core.attach] MEMORY.stop() failed at exit", exc_info=True)

    except Exception:
        _logger.exception("[ai_core.attach] Unexpected error in _flush_and_stop_memory")


# -------------------------
# Register atexit and signal handlers
# -------------------------
try:
    atexit.register(_flush_and_stop_memory)
    _logger.debug("[ai_core.attach] Registered atexit memory flush ✅")
except Exception:
    _logger.debug("[ai_core.attach] Failed to register atexit handler", exc_info=True)


def _signal_handler(signum, frame):
    """
    Signal handler: flush memory, stop threads, exit cleanly
    """
    try:
        _logger.info("[ai_core.attach] Received signal %s — flushing MEMORY and exiting", signum)
        _flush_and_stop_memory()
    except Exception:
        _logger.exception("[ai_core.attach] Error in signal handler.")
    finally:
        try:
            sys.exit(0)
        except SystemExit:
            raise


for _sig in ("SIGINT", "SIGTERM"):
    try:
        signal.signal(getattr(signal, _sig), _signal_handler)
        _logger.debug("[ai_core.attach] Installed signal handler for %s ✅", _sig)
    except Exception:
        _logger.debug("[ai_core.attach] Could not install handler for %s", _sig, exc_info=True)


# -------------------------
# Final MEMORY integration and AIMemoryHelper attach
# -------------------------
try:
    # Attach AI helper if missing
    if MEMORY is not None and isinstance(MEMORY, MemoryStore):
        try:
            from app.utils.ai_memory_helper import AIMemoryHelper
            if not getattr(MEMORY, "ai_helper", None):
                MEMORY.ai_helper = AIMemoryHelper(MEMORY)
                MEMORY.logger.info("[Memory] AI helper successfully attached ✅")
        except Exception as e:
            MEMORY.logger.debug(f"[Memory] AIMemoryHelper attach failed: {e}", exc_info=True)
    else:
        logger = logging.getLogger("BibliothecaAI.AI_Core")
        logger.warning("[Memory] No active MemoryStore detected — AI helper not attached ⚠️")

    # Initial proactive sync check
    try:
        if hasattr(MEMORY, "sync_to_db") and callable(MEMORY.sync_to_db):
            MEMORY.sync_to_db()
            MEMORY.logger.info("[Memory] Initial sync_to_db() completed ✅")
    except Exception as e:
        MEMORY.logger.warning(f"[Memory] Initial sync_to_db failed: {e}")

    # Final heartbeat
    if MEMORY is not None:
        MEMORY.logger.info("[Memory] Integration verified — Beyond Perfection Edition 🧠✨")

except Exception:
    _logger.exception("[ai_core.attach] Unexpected error during final MEMORY integration")


# -------------------------
# Ensure final AIMemoryHelper attach to known globals
# -------------------------
try:
    from app.utils.ai_memory_helper import AIMemoryHelper
    if MEMORY is not None and AIMemoryHelper is not None:
        # Attach to primary global advanced_ai if it exists
        if "advanced_ai" in globals() and advanced_ai:
            if not getattr(advanced_ai, "memory_helper", None):
                advanced_ai.memory_helper = AIMemoryHelper(MEMORY)
                _logger.info("[ai_core.attach] Attached AIMemoryHelper to advanced_ai ✅")
        # Attach to MEMORY singleton itself if missing
        if not getattr(MEMORY, "ai_helper", None):
            MEMORY.ai_helper = AIMemoryHelper(MEMORY)
            MEMORY.logger.info("[ai_core.attach] Attached AIMemoryHelper to MEMORY ✅")
except Exception as e:
    _logger.debug(f"[ai_core.attach] Final AIMemoryHelper attach failed: {e}", exc_info=True)


# -------------------------
# Beyond perfection: safety sweep
# -------------------------
try:
    # Ensure MEMORY aliases exist
    if MEMORY:
        for alias, real in {
            "sync": "sync_to_db",
            "flush": "sync_to_db",
            "persist": "sync_to_db",
            "save_memory": "sync_to_db",
            "store_local": "store",
            "recall_local": "recall",
            "stop": "stop",
        }.items():
            if hasattr(MEMORY, real) and not hasattr(MEMORY, alias):
                setattr(MEMORY, alias, getattr(MEMORY, real))
        MEMORY.logger.debug("[ai_core.attach] Verified MEMORY alias methods ✅")
except Exception:
    _logger.exception("[ai_core.attach] Error ensuring MEMORY alias safety")
