"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SANA — Smart Autonomous Natural Agent  (OpenRouter Edition)                 ║
║  Edge-based Eutrophication Detection & Response System                       ║
║  AI backend: OpenRouter free Gemma models (no local GPU required)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEPENDENCIES (all pip-installable, no Ollama needed):
    pip install customtkinter requests

SETUP:
    1. Get a free API key from https://openrouter.ai/keys
    2. Either:
       a) Set environment variable:   export OPENROUTER_API_KEY=sk-or-v1-...
       b) Or paste it into the key dialog that appears on first launch.

FREE MODELS AVAILABLE (all $0/token):
    google/gemma-3-27b-it:free    — best quality, 128K ctx
    google/gemma-4-31b-it:free    — latest Gemma 4, vision+reasoning
    google/gemma-4-26b-a4b-it:free — MoE, near-31B quality at 4B cost
    google/gemma-3-4b-it:free     — fastest, lightest

ARCHITECTURE CHANGES vs the Ollama edition:
    • `call_ollama_agent()` → `call_openrouter_agent()`
      Uses `requests.post()` to https://openrouter.ai/api/v1/chat/completions
      with Bearer token auth. Fully OpenAI-compatible payload format.
    • API key stored in `SimulationEngine.api_key`, set via GUI dialog or env var.
    • Model selector updated to OpenRouter free Gemma IDs.
    • `OPENROUTER_AVAILABLE` flag replaces `OLLAMA_AVAILABLE`.
    • All simulation physics, GUI layout, and queue architecture unchanged.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import math
import os
import queue
import random
import threading
import time
from datetime import datetime, timedelta

try:
    import spidev
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 2000000
    spi.mode = 0
    SPI_AVAILABLE = True
except Exception as e:
    print(f"[WARN] SPI/LoRa not available: {e}")
    SPI_AVAILABLE = False

def w(r,v):
    if SPI_AVAILABLE: spi.xfer2([r|0x80,v])
def r(r):
    if SPI_AVAILABLE: return spi.xfer2([r&0x7F,0])[1]
    return 0
def b(r,n):
    if SPI_AVAILABLE: return spi.xfer2([r&0x7F]+[0]*n)[1:]
    return [0]*n

def init_lora():
    if not SPI_AVAILABLE: return
    w(0x01,0x00); time.sleep(0.01)
    w(0x01,0x80); time.sleep(0.01)
    w(0x06,0x6C); w(0x07,0x80); w(0x08,0x00)
    w(0x09,0x8F); w(0x1D,0x72); w(0x1E,0x74)
    w(0x33,0x27); w(0x3B,0x1D)
    w(0x0E,0x00); w(0x0F,0x00)

# ── Third-party ───────────────────────────────────────────────────────────────

import customtkinter as ctk

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARN] `python-dotenv` not found.  pip install python-dotenv for .env support")

# requests is almost certainly already installed; we fail gracefully if not
try:
    import requests as req_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] `requests` library not found.  pip install requests")

# ═════════════════════════════════════════════════════════════════════════════
#  OPENROUTER CONFIG
# ═════════════════════════════════════════════════════════════════════════════

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERRER = "https://github.com/sana-hydro-monitor"   # optional leaderboard header
OPENROUTER_APP_NAME = "SANA-Dashboard"

# Free Gemma models available on OpenRouter (as of April 2025)
FREE_MODELS = [
    "google/gemma-3-27b-it:free",    # recommended — best quality free
    "google/gemma-4-31b-it:free",    # Gemma 4, reasoning-capable
    "google/gemma-4-26b-a4b-it:free",# MoE variant
    "google/gemma-3-4b-it:free",     # fastest / lowest latency
]

# ═════════════════════════════════════════════════════════════════════════════
#  THEME & COLOUR CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_dark":      "#0A0E1A",
    "bg_panel":     "#0D1220",
    "bg_card":      "#111827",
    "bg_terminal":  "#080C18",
    "border":       "#1E2D45",
    "border_bright":"#2A4070",

    "green":        "#00FF94",
    "green_dim":    "#00C070",
    "yellow":       "#FFD700",
    "orange":       "#FF8C00",
    "red":          "#FF3355",
    "red_dim":      "#8B0000",

    "cyan":         "#00D4FF",
    "cyan_dim":     "#0088AA",
    "blue":         "#4488FF",
    "purple":       "#AA66FF",

    "text_bright":  "#E8F4FF",
    "text_normal":  "#8BA0C0",
    "text_dim":     "#3D5070",
}

SEVERITY_COLORS = {
    "LOW":      COLORS["green"],
    "MODERATE": COLORS["yellow"],
    "HIGH":     COLORS["orange"],
    "CRITICAL": COLORS["red"],
    "IDLE":     COLORS["text_dim"],
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTOR PHYSICS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class SectorState:
    """
    Living environmental state for one water sector.

    bloom_level (0.0–1.0) is the master state variable.
    All spectral/chemical indices are derived as calibrated noisy functions
    of bloom_level, mimicking real multispectral sensor physics.
    """

    def __init__(self, node_id: int, initial_bloom: float, trend: float):
        self.node_id        = node_id
        self.bloom_level    = initial_bloom
        self.trend          = trend
        self.aeration_ticks = 0
        self.severity       = "LOW"
        self.last_action    = "IDLE"

    def _noise(self, scale: float = 0.01) -> float:
        return random.gauss(0, scale)

    def tick(self):
        """Advance one 15-minute simulated interval."""
        if self.aeration_ticks > 0:
            # Aerators fight growth AND actively reduce bloom
            effective_trend  = self.trend - 0.04
            self.bloom_level = max(0.0, self.bloom_level + effective_trend)
            self.aeration_ticks -= 1
        else:
            self.bloom_level = max(0.0, min(1.0, self.bloom_level + self.trend))

        # Logistic ceiling brake
        if self.bloom_level > 0.85:
            self.trend = max(self.trend - 0.002, -0.005)

        # Climatic noise
        self.trend += random.gauss(0, 0.001)
        self.trend  = max(-0.02, min(0.03, self.trend))

    def activate_aeration(self, duration_ticks: int = 6):
        self.aeration_ticks = duration_ticks
        self.last_action    = "ACTIVATE_AERATOR"

    def get_indices(self) -> dict:
        """Derive all spectral and chemical indices from bloom_level."""
        b = self.bloom_level

        ndvi      = 0.05 + b * 0.65 + self._noise(0.015)
        sabi      = -0.1  + b * 1.1  + self._noise(0.02)
        ndwi      = 0.3   - b * 0.7  + self._noise(0.015)
        ndci      = -0.05 + b * 0.85 + self._noise(0.02)
        bci       = 0.0   + b * 0.9  + self._noise(0.025)
        fai       = -0.02 + b * 0.55 + self._noise(0.01)

        chla      = max(1.0,  2.0 + math.exp(b * 4.6) * 1.5 + random.gauss(0, 2.0))
        turbidity = max(0.5,  1.5 + b * 48.5 + random.gauss(0, 1.5))
        secchi    = max(0.2,  4.5 - b * 4.0  + self._noise(0.1))
        coverage  = max(0.0,  min(100.0, b * 100.0 + random.gauss(0, 2.0)))
        cyano     = max(0,    min(100, b * 110 + random.gauss(0, 3)))
        abi       = max(0,    b * 0.95 + self._noise(0.02))
        nutrient  = 0.1 + b * 0.9 + self._noise(0.02)

        return {
            "NDVI":     round(ndvi,      3),
            "SABI":     round(sabi,      3),
            "NDWI":     round(ndwi,      3),
            "NDCI":     round(ndci,      3),
            "BCI":      round(bci,       3),
            "FAI":      round(fai,       3),
            "CHL-a":    round(chla,      2),
            "Turbidity":round(turbidity, 2),
            "Secchi":   round(secchi,    2),
            "Coverage": round(coverage,  1),
            "CYANO":    round(cyano,     1),
            "ABI":      round(abi,       3),
            "Nutrient": round(nutrient,  3),
        }


def classify_severity(indices: dict) -> str:
    chla  = indices["CHL-a"]
    cyano = indices["CYANO"]
    sabi  = indices["SABI"]
    if chla >= 50 or cyano >= 80 or sabi >= 0.7:  return "CRITICAL"
    elif chla >= 25 or cyano >= 50 or sabi >= 0.45: return "HIGH"
    elif chla >= 10 or cyano >= 20 or sabi >= 0.2:  return "MODERATE"
    else:                                            return "LOW"


def format_index_status(key: str, value: float) -> str:
    thresholds = {
        "NDVI":      [(0.1,"CLEAR"),(0.3,"SPARSE VEG"),(0.5,"ACTIVE VEG"),(1.0,"DENSE MAT")],
        "SABI":      [(-0.05,"CLEAN"),(0.2,"EARLY BLOOM"),(0.5,"SURFACE BLOOM"),(1.0,"SEVERE BLOOM")],
        "NDWI":      [(-0.5,"COVERED"),(0.0,"TURBID"),(0.2,"MIXED"),(1.0,"CLEAR WATER")],
        "CYANO":     [(20,"SAFE"),(50,"CAUTION"),(80,"TOXIC"),(100,"EXTREME TOXIC")],
        "CHL-a":     [(10,"NORMAL"),(25,"ELEVATED"),(50,"HIGH"),(200,"CRITICAL")],
        "Turbidity": [(5,"CLEAR"),(15,"MODERATE"),(30,"TURBID"),(100,"OPAQUE")],
    }
    if key not in thresholds:
        return "OK" if value < 0.5 else "ELEVATED"
    for threshold, label in thresholds[key]:
        if value <= threshold:
            return label
    return thresholds[key][-1][1]


# ═════════════════════════════════════════════════════════════════════════════
#  LORA TELEMETRY ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def build_lora_payload(sector: SectorState, sim_time: datetime) -> dict:
    indices  = sector.get_indices()
    severity = classify_severity(indices)
    sector.severity = severity

    return {
        "node_id":        f"NODE-{sector.node_id:02d}",
        "timestamp":      sim_time.strftime("%H:%M:%S"),
        "date":           sim_time.strftime("%Y-%m-%d"),
        "severity":       severity,
        "bloom_level":    round(sector.bloom_level, 3),
        "aeration_active":sector.aeration_ticks > 0,
        "indices": {
            "NDVI":       {"value": indices["NDVI"],      "status": format_index_status("NDVI",      indices["NDVI"])},
            "SABI":       {"value": indices["SABI"],      "status": format_index_status("SABI",      indices["SABI"])},
            "NDWI":       {"value": indices["NDWI"],      "status": format_index_status("NDWI",      indices["NDWI"])},
            "NDCI":       {"value": indices["NDCI"],      "status": "OK"},
            "BCI":        {"value": indices["BCI"],       "status": "OK"},
            "FAI":        {"value": indices["FAI"],       "status": "OK"},
            "CHL-a":      {"value": indices["CHL-a"],     "unit": "μg/L",  "status": format_index_status("CHL-a",     indices["CHL-a"])},
            "Turbidity":  {"value": indices["Turbidity"], "unit": "NTU",   "status": format_index_status("Turbidity", indices["Turbidity"])},
            "Secchi":     {"value": indices["Secchi"],    "unit": "m",     "status": "OK"},
            "Coverage":   {"value": indices["Coverage"],  "unit": "%",     "status": "OK"},
            "CYANO-PROXY":{"value": indices["CYANO"],     "unit": "risk%", "status": format_index_status("CYANO",     indices["CYANO"])},
            "ABI":        {"value": indices["ABI"],       "status": "OK"},
            "Nutrient":   {"value": indices["Nutrient"],  "status": "OK"},
        }
    }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENTIC AI CONTROLLER  —  OpenRouter Edition
# ═════════════════════════════════════════════════════════════════════════════

SANA_SYSTEM_PROMPT = """You are SANA-BRAIN, the agentic AI controller of the SANA (Smart Autonomous Natural Agent) environmental monitoring network deployed on a freshwater lake.

You receive real-time telemetry from autonomous surface nodes measuring eutrophication and algal bloom conditions.

Your job is to:
1. Analyze the incoming telemetry data
2. Determine the severity and nature of the bloom threat
3. Issue an automated response command
4. Generate a brief public safety bulletin

You MUST respond with ONLY a single valid JSON object — no markdown, no explanations outside the JSON. The JSON must have exactly these four fields:

{
  "reasoning": "<2-3 sentences of chain-of-thought analysis of the key indices>",
  "action": "<one of: IDLE | ACTIVATE_AERATOR | DEPLOY_CHEMICALS | CRITICAL_HUMAN_INTERVENTION>",
  "severity": "<one of: LOW | MODERATE | HIGH | CRITICAL>",
  "bulletin": "<1-2 sentences for the public status board, clear plain English>"
}

Action selection guidelines:
- IDLE: LOW severity, no intervention needed
- ACTIVATE_AERATOR: MODERATE or HIGH — deploy dissolved-oxygen aerators to disrupt stratification
- DEPLOY_CHEMICALS: HIGH with cyanotoxin risk — apply algaecide/flocculant
- CRITICAL_HUMAN_INTERVENTION: CRITICAL — bloom is toxic, immediate human response required

Respond ONLY with the JSON object. No preamble, no markdown fences."""


def call_openrouter_agent(payload: dict, api_key: str,
                          model: str = "google/gemma-3-27b-it:free") -> dict:
    """
    Send telemetry payload to OpenRouter's free Gemma endpoint and parse response.

    OpenRouter is OpenAI-compatible:
        POST https://openrouter.ai/api/v1/chat/completions
        Authorization: Bearer <api_key>
        Content-Type: application/json

    Threading contract: always called from a worker thread, never the GUI thread.

    Returns a dict: {reasoning, action, severity, bulletin}
    Falls back to rule-based response on any error (network, auth, parse).
    """

    # ── Rule-based fallback (used when API is unavailable or fails) ───────────
    def rule_based_fallback(payload: dict, note: str = "") -> dict:
        severity = payload.get("severity", "LOW")
        node     = payload["node_id"]
        chla     = payload["indices"]["CHL-a"]["value"]
        cyano    = payload["indices"]["CYANO-PROXY"]["value"]
        sabi     = payload["indices"]["SABI"]["value"]
        tag      = f" [FALLBACK{': ' + note if note else ''}]"

        if severity == "CRITICAL":
            return {
                "action":    "CRITICAL_HUMAN_INTERVENTION",
                "severity":  severity,
                "reasoning": f"CHL-a={chla}μg/L and CYANO={cyano}% exceed WHO toxic thresholds. SABI={sabi} confirms active surface bloom. Emergency protocols triggered.{tag}",
                "bulletin":  f"⚠ CRITICAL: {node} reports toxic bloom (CHL-a={chla}μg/L). Avoid all water contact. Emergency services notified.{tag}",
            }
        elif severity == "HIGH":
            return {
                "action":    "DEPLOY_CHEMICALS",
                "severity":  severity,
                "reasoning": f"SABI={sabi} and CHL-a={chla}μg/L indicate active mid-stage bloom. Cyanobacterial risk elevated (CYANO={cyano}%). Chemical intervention warranted.{tag}",
                "bulletin":  f"HIGH ALERT: {node} — elevated algal activity. Algaecide deployment authorised. Recreational use suspended.{tag}",
            }
        elif severity == "MODERATE":
            return {
                "action":    "ACTIVATE_AERATOR",
                "severity":  severity,
                "reasoning": f"CHL-a={chla}μg/L and SABI={sabi} show early bloom development. Aeration initiated to disrupt thermal stratification.{tag}",
                "bulletin":  f"MODERATE: {node} — algal levels rising. Aerators activated. Monitor closely.{tag}",
            }
        else:
            return {
                "action":    "IDLE",
                "severity":  severity,
                "reasoning": f"All indices within safe ranges. CHL-a={chla}μg/L, SABI={sabi}, CYANO={cyano}%. Monitoring continues.{tag}",
                "bulletin":  f"NOMINAL: {node} — water quality within safe parameters. No intervention required.{tag}",
            }

    # ── Guard: requests library must be available ─────────────────────────────
    if not REQUESTS_AVAILABLE:
        return rule_based_fallback(payload, "requests library missing")

    # ── Guard: API key must be provided ───────────────────────────────────────
    if not api_key or not api_key.strip().startswith("sk-"):
        return rule_based_fallback(payload, "no valid API key")

    # ── Build compact telemetry summary for the prompt ────────────────────────
    idx = payload["indices"]
    key_data = {
        "node":          payload["node_id"],
        "time":          payload["timestamp"],
        "severity":      payload["severity"],
          parsed = json.loads(raw_text)

        # Validate required keys
        for key in ("reasoning", "action", "severity", "bulletin"):
            if key not in parsed:
                raise ValueError(f"Missing field: '{key}'")

        return parsed

    except req_lib.exceptions.Timeout:
        return rule_based_fallback(payload, "request timeout — try faster model")
    except req_lib.exceptions.ConnectionError:
        return rule_based_fallback(payload, "network unreachable")
    except json.JSONDecodeError as e:
        fb = rule_based_fallback(payload, f"JSON parse error: {e}")
        return fb
    except Exception as e:
        return rule_based_fallback(payload, f"{type(e).__name__}: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class SimulationEngine:
    """
    Orchestrates the four-sector environment, LoRa telemetry generation,
    and AI inference scheduling.

    Thread model:
        • _run_loop  : background daemon thread — advances state, produces payloads
        • _run_ai_analysis : per-tick daemon thread — calls OpenRouter, applies actions
        • GUI polls three queues via after() callbacks:
            telemetry_queue, ai_queue, event_queue
    """

    def __init__(self):
        self.sectors = [
            SectorState(1, initial_bloom=0.05, trend=+0.008),
            SectorState(2, initial_bloom=0.15, trend=+0.020),
            SectorState(3, initial_bloom=0.02, trend=-0.002),
            SectorState(4, initial_bloom=0.40, trend=+0.012),
        ]

        self.sim_time    = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
        self.tick_count  = 0

        self.telemetry_queue = queue.Queue()
        self.ai_queue        = queue.Queue()
        self.event_queue     = queue.Queue()

        self._running    = False
        self._paused     = False
        self._ai_pending = False
        self.tick_interval = 6.0    # a bit slower default — free tier has latency
        self.model_name  = FREE_MODELS[0]
        self.api_key     = os.environ.get("OPENROUTER_API_KEY", "")

    def start(self):
        self._running = True
        self._paused  = False
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._running = False

    def set_speed(self, interval_seconds: float):
        self.tick_interval = max(1.0, float(interval_seconds))

    def _run_loop(self):
        self.event_queue.put("[RX] LORA RADIO LISTENING...")
        if SPI_AVAILABLE:
            init_lora()
            
        while self._running:
            if self._paused:
                time.sleep(0.2)
                continue

            if SPI_AVAILABLE:
                w(0x01,0x85)
                irq = r(0x12)

                if irq & 0x40:
                    length = r(0x13)
                    fifo = r(0x10)

                    w(0x0D,fifo)
                    data = b(0x00,length)

                    w(0x12,0xFF)

                    text = bytes(data).decode(errors="ignore")

                    try:
                        obj = json.loads(text)
                        bloom = float(obj.get("indices",{}).get("BLOOM",0))
                        node = obj.get("node","unknown")
                    except:
                        bloom = 0
                        node = "unknown"
                        obj = {}

                    node_num = 1
                    try:
                        if "NODE-" in node:
                            node_num = int(node.split("-")[1])
                    except:
                        pass
                    if not (1 <= node_num <= 4):
                        node_num = 1

                    sector = self.sectors[node_num - 1]
                    
                    # Align simulation physics with new telemetry data
                    sector.bloom_level = bloom
                    
                    self.sim_time += timedelta(minutes=15)
                    self.tick_count += 1

                    self.event_queue.put(
                        f"[{self.sim_time.strftime('%H:%M')}] ── LORA PACKET ──"
                        f" {'🔴' if bloom>0.7 else '🟢'}"
                    )
                    
                    payload = build_lora_payload(sector, self.sim_time)
                    payload["node_id"] = node
                    if obj.get("raw"):
                        payload["raw"] = obj.get("raw")

                    self.event_queue.put(
                        f"  ↗ LoRa RX  {node} → QUEEN  "
                        f"[bloom={sector.bloom_level:.2f}  sev={sector.severity}]"
                    )
                    self.telemetry_queue.put(payload)

                    if not self._ai_pending:
                        self._ai_pending = True
                        threading.Thread(
                            target=self._run_ai_analysis,
                            args=(payload,),
                            daemon=True
                        ).start()

                    init_lora()
            else:
                pass

            time.sleep(0.02)

    def _run_ai_analysis(self, payload: dict):
        node_num = int(payload["node_id"].split("-")[1])
        sector   = self.sectors[node_num - 1]

        self.event_queue.put(
            f"  🤖 SANA-BRAIN → OpenRouter [{self.model_name.split('/')[-1]}] …"
        )

        ai_result = call_openrouter_agent(
            payload,
            api_key=self.api_key,
            model=self.model_name,
        )

        action = ai_result.get("action", "IDLE")
        if action == "ACTIVATE_AERATOR":
            sector.activate_aeration(duration_ticks=6)
            self.event_queue.put(
                f"  ⚡ AERATOR ACTIVATED  {payload['node_id']}  (6-tick)"
            )
        elif action in ("DEPLOY_CHEMICALS", "CRITICAL_HUMAN_INTERVENTION"):
            sector.activate_aeration(duration_ticks=10)
            sector.trend = min(sector.trend, -0.01)
            self.event_queue.put(
                f"  ☣  {action}  {payload['node_id']}  [ALERT DISPATCHED]"
            )

        self.ai_queue.put({
            "node_id":   payload["node_id"],
            "timestamp": payload["timestamp"],
            "result":    ai_result,
            "payload":   payload,
        })
        self._ai_pending = False


# ═════════════════════════════════════════════════════════════════════════════
#  API KEY DIALOG  (shown on startup if no key found in environment)
# ═════════════════════════════════════════════════════════════════════════════

class APIKeyDialog(ctk.CTkToplevel):
    """
    Modal dialog asking for the OpenRouter API key.
    Dismissed by clicking Save, pressing Enter, or closing (uses fallback mode).
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("SANA — OpenRouter API Key")
        self.geometry("520x280")
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg_panel"])
        self.grab_set()   # modal
        self.api_key = ""

        ctk.CTkLabel(
            self,
            text="◈ SANA  OpenRouter Configuration",
            font=ctk.CTkFont(family="Courier New", size=14, weight="bold"),
            text_color=COLORS["cyan"],
        ).pack(pady=(22, 4))

        ctk.CTkLabel(
            self,
            text=(
                "Paste your OpenRouter API key below.\n"
                "Get a free key at  openrouter.ai/keys\n"
                "Free Gemma models are available at $0/token."
            ),
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_normal"],
            justify="center",
        ).pack(pady=(0, 12))

        self.key_entry = ctk.CTkEntry(
            self,
            placeholder_text="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            width=440, height=36,
            font=ctk.CTkFont(family="Courier New", size=11),
            fg_color=COLORS["bg_terminal"],
            text_color=COLORS["text_bright"],
            border_color=COLORS["border_bright"],
            show="•",
        )
        self.key_entry.pack(pady=4)
        self.key_entry.bind("<Return>", lambda e: self._save())

        # Reveal toggle
        self._reveal = False
        ctk.CTkButton(
            self,
            text="👁 Show / Hide Key",
            width=160, height=26,
            font=ctk.CTkFont(family="Courier New", size=9),
            fg_color="transparent",
            text_color=COLORS["text_dim"],
            hover_color=COLORS["bg_card"],
            command=self._toggle_reveal,
        ).pack(pady=2)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=14)

        ctk.CTkButton(
            btn_row,
            text="✓  SAVE & CONNECT",
            width=180, height=34,
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            fg_color=COLORS["green_dim"],
            hover_color=COLORS["green"],
            text_color=COLORS["bg_dark"],
            command=self._save,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_row,
            text="Skip (fallback mode)",
            width=160, height=34,
            font=ctk.CTkFont(family="Courier New", size=10),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["border"],
            text_color=COLORS["text_dim"],
            command=self._skip,
        ).pack(side="left", padx=8)

    def _toggle_reveal(self):
        self._reveal = not self._reveal
        self.key_entry.configure(show="" if self._reveal else "•")

    def _save(self):
        self.api_key = self.key_entry.get().strip()
        self.grab_release()
        self.destroy()

    def _skip(self):
        self.api_key = ""
        self.grab_release()
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
#  GUI DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

class SANADashboard(ctk.CTk):
    """
    Main application window — SANA Hydro-Informatics Command Centre.

    Layout:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  HEADER  [logo · subtitle · status · clock]                         │
    ├──────────────┬──────────────────────┬──────────────────────────────┤
    │  SECTOR MAP  │  LIVE TELEMETRY      │  AI AGENT TERMINAL           │
    │              │  STREAM              │  [reasoning · commands]      │
    ├──────────────┤                      │                              │
    │  PUBLIC      │                      │                              │
    │  BULLETIN    │                      │                              │
    ├──────────────┴──────────────────────┴──────────────────────────────┤
    │  CONTROL BAR  [Play | Pause | Speed | Model | API key btn | Tick]  │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, engine: SimulationEngine):
        super().__init__()
        self.engine = engine

        self.title("SANA · Smart Autonomous Natural Agent  |  OpenRouter Edition  v1.1")
        self.geometry("1440x860")
        self.minsize(1200, 720)
        self.configure(fg_color=COLORS["bg_dark"])

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self._running    = True
        self._sim_active = False

        self._build_header()
        self._build_main_grid()
        self._build_control_bar()
        self._poll_queues()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── If no API key in environment, show dialog after window appears ────
        if not self.engine.api_key:
            self.after(400, self._prompt_api_key)

    # ─────────────────────────────────────────────────────────────────────────
    #  LAYOUT
    # ─────────────────────────────────────────────────────────────────────────

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                              corner_radius=0, height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="◈ SANA",
            font=ctk.CTkFont(family="Courier New", size=22, weight="bold"),
            text_color=COLORS["cyan"],
        ).pack(side="left", padx=18, pady=8)

        ctk.CTkLabel(
            header,
            text="SMART AUTONOMOUS NATURAL AGENT  ·  FOG-COMPUTING SWARM  ·  EUTROPHICATION MONITOR",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_dim"],
        ).pack(side="left", padx=4)

        # OpenRouter badge
        ctk.CTkLabel(
            header,
            text="[ OpenRouter ]",
            font=ctk.CTkFont(family="Courier New", size=9),
            text_color=COLORS["purple"],
        ).pack(side="left", padx=8)

        self.clock_label = ctk.CTkLabel(
            header, text="",
            font=ctk.CTkFont(family="Courier New", size=12),
            text_color=COLORS["text_normal"],
        )
        self.clock_label.pack(side="right", padx=18)
        self._update_clock()

        self.status_dot = ctk.CTkLabel(
            header, text="● OFFLINE",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            text_color=COLORS["text_dim"],
        )
        self.status_dot.pack(side="right", padx=12)

        # API key status indicator (right side)
        self.key_status_label = ctk.CTkLabel(
            header, text="🔑 NO KEY",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["yellow"],
        )
        self.key_status_label.pack(side="right", padx=8)
        self._refresh_key_status()

    def _build_main_grid(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=3)
        main.columnconfigure(2, weight=3)
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)

        left_col = ctk.CTkFrame(main, fg_color="transparent")
        left_col.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0,4))
        left_col.rowconfigure(0, weight=3)
        left_col.rowconfigure(1, weight=2)

        self._build_sector_map(left_col)
        self._build_bulletin_board(left_col)
        self._build_telemetry_panel(main)
        self._build_ai_terminal(main)

    def _build_sector_map(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=0, sticky="nsew", pady=(0,4))

        tb = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=0, height=26)
        tb.pack(fill="x", side="top")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="  ⬡ SECTOR MAP  ·  LAKE OVERVIEW",
                     font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
                     text_color=COLORS["cyan_dim"], anchor="w").pack(side="left", fill="y")

        self.map_canvas = ctk.CTkCanvas(content, bg=COLORS["bg_dark"], highlightthickness=0)
        self.map_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.map_canvas.bind("<Configure>", self._redraw_map)

        self.node_positions = [(0.25,0.28),(0.72,0.28),(0.25,0.72),(0.72,0.72)]

    def _redraw_map(self, event=None):
        c = self.map_canvas
        c.delete("all")
        w = c.winfo_width();  h = c.winfo_height()
        if w < 10 or h < 10: return

        cx, cy = w*0.5, h*0.5
        rx, ry = w*0.36, h*0.42
        c.create_oval(cx-rx, cy-ry, cx+rx, cy+ry,
                      fill="#0A1828", outline=COLORS["border_bright"], width=1)
        c.create_text(cx, cy, text="◈ LAKE", fill=COLORS["text_dim"],
                      font=("Courier New", 10))
        c.create_oval(cx-10, cy-10, cx+10, cy+10,
                      fill=COLORS["cyan_dim"], outline=COLORS["cyan"], width=2)
        c.create_text(cx, cy+20, text="QUEEN", fill=COLORS["cyan"],
                      font=("Courier New", 8, "bold"))

        for i, (fx, fy) in enumerate(self.node_positions):
            sector  = self.engine.sectors[i]
            x, y    = w*fx, h*fy
            color   = SEVERITY_COLORS.get(sector.severity, COLORS["green"])
            pulse_r = 14 + int(sector.bloom_level * 10)

            c.create_oval(x-pulse_r, y-pulse_r, x+pulse_r, y+pulse_r,
                          fill="", outline=color, width=1, dash=(3,3))
            c.create_oval(x-10, y-10, x+10, y+10,
                          fill=COLORS["bg_card"], outline=color, width=2)
            c.create_text(x, y, text=f"{i+1}", fill=color,
                          font=("Courier New", 9, "bold"))
            c.create_text(x, y+18, text=f"N-{i+1:02d}", fill=COLORS["text_normal"],
                          font=("Courier New", 7))
            c.create_text(x, y+28, text=sector.severity, fill=color,
                          font=("Courier New", 7, "bold"))

            bw = 36
            c.create_rectangle(x-bw//2, y+34, x+bw//2, y+38,
                                fill=COLORS["bg_terminal"], outline="")
            filled = int(bw * sector.bloom_level)
            if filled > 0:
                c.create_rectangle(x-bw//2, y+34, x-bw//2+filled, y+38,
                                   fill=color, outline="")

            c.create_line(x, y, cx, cy, fill=COLORS["border"], width=1, dash=(2,4))

            if sector.aeration_ticks > 0:
                c.create_text(x, y-20, text="⚡AER", fill=COLORS["blue"],
                              font=("Courier New", 7, "bold"))

    def _build_bulletin_board(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=1, column=0, sticky="nsew")

        tb = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=0, height=26)
        tb.pack(fill="x", side="top")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="  📢 PUBLIC BULLETIN BOARD",
                     font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
                     text_color=COLORS["cyan_dim"], anchor="w").pack(side="left", fill="y")

        self.bulletin_text = ctk.CTkTextbox(
            content, fg_color=COLORS["bg_terminal"],
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_bright"],
            corner_radius=0, wrap="word", state="disabled",
        )
        self.bulletin_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._configure_text_tags(self.bulletin_text)

    def _build_telemetry_panel(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(0,4))

        tb = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=0, height=26)
        tb.pack(fill="x", side="top")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="  📡 LIVE TELEMETRY STREAM  ·  LoRa RX → QUEEN NODE",
                     font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
                     text_color=COLORS["cyan_dim"], anchor="w").pack(side="left", fill="y")

        self.telem_text = ctk.CTkTextbox(
            content, fg_color=COLORS["bg_terminal"],
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_bright"],
            corner_radius=0, wrap="none", state="disabled",
        )
        self.telem_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._configure_text_tags(self.telem_text)
        self._append(self.telem_text,
            "  SANA QUEEN NODE  —  WAITING FOR TELEMETRY\n"
            "  ─────────────────────────────────────────\n"
            "  Press START to begin receiving LoRa packets.\n\n", "dim")

    def _build_ai_terminal(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=2, rowspan=2, sticky="nsew")

        tb = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=0, height=26)
        tb.pack(fill="x", side="top")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="  🤖 SANA-BRAIN  ·  AGENTIC AI  (OpenRouter)",
                     font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
                     text_color=COLORS["purple"], anchor="w").pack(side="left", fill="y")

        self.ai_text = ctk.CTkTextbox(
            content, fg_color=COLORS["bg_terminal"],
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_bright"],
            corner_radius=0, wrap="word", state="disabled",
        )
        self.ai_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._configure_text_tags(self.ai_text)

        key_state = "KEY SET ✓" if self.engine.api_key else "⚠ NO KEY — fallback mode"
        self._append(self.ai_text,
            "  SANA-BRAIN OFFLINE\n"
            "  ──────────────────────────────────────\n"
            f"  API key : {key_state}\n"
            f"  Model   : {self.engine.model_name}\n"
            "  Backend : OpenRouter (free Gemma)\n\n", "dim")

    def _build_control_bar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=0, height=50)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        # Play / pause
        self.play_btn = ctk.CTkButton(
            bar, text="▶  START", width=120, height=34,
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            fg_color=COLORS["green_dim"], hover_color=COLORS["green"],
            text_color=COLORS["bg_dark"], command=self._toggle_simulation,
        )
        self.play_btn.pack(side="left", padx=12, pady=8)

        # Speed
        ctk.CTkLabel(bar, text="SPEED:",
                     font=ctk.CTkFont(family="Courier New", size=10),
                     text_color=COLORS["text_normal"]).pack(side="left", padx=(12,2))

        self.speed_label = ctk.CTkLabel(bar, text="6s/tick",
                                        font=ctk.CTkFont(family="Courier New", size=10),
                                        text_color=COLORS["cyan"], width=60)
        self.speed_label.pack(side="left", padx=(0,4))

        self.speed_slider = ctk.CTkSlider(
            bar, from_=2, to=30, width=160, height=18,
            command=self._on_speed_change,
            button_color=COLORS["cyan"], button_hover_color=COLORS["cyan_dim"],
            progress_color=COLORS["border_bright"], fg_color=COLORS["border"],
        )
        self.speed_slider.set(6)
        self.speed_slider.pack(side="left", padx=4)

        ctk.CTkLabel(bar, text="│",
                     text_color=COLORS["border_bright"]).pack(side="left", padx=10)

        # Model selector — only free OpenRouter Gemma models
        ctk.CTkLabel(bar, text="MODEL:",
                     font=ctk.CTkFont(family="Courier New", size=10),
                     text_color=COLORS["text_normal"]).pack(side="left", padx=(0,4))

        self.model_var = ctk.StringVar(value=FREE_MODELS[0])
        ctk.CTkOptionMenu(
            bar, values=FREE_MODELS,
            variable=self.model_var,
            width=230, height=30,
            font=ctk.CTkFont(family="Courier New", size=9),
            fg_color=COLORS["bg_card"],
            button_color=COLORS["border_bright"],
            button_hover_color=COLORS["border"],
            text_color=COLORS["text_bright"],
            command=self._on_model_change,
        ).pack(side="left", padx=4)

        ctk.CTkLabel(bar, text="│",
                     text_color=COLORS["border_bright"]).pack(side="left", padx=10)

        # API key button
        ctk.CTkButton(
            bar, text="🔑 API KEY", width=100, height=30,
            font=ctk.CTkFont(family="Courier New", size=9),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["border_bright"],
            text_color=COLORS["text_normal"],
            command=self._prompt_api_key,
        ).pack(side="left", padx=4)

        # Tick counter
        self.tick_label = ctk.CTkLabel(bar,
            text="TICK: 0000  SIM: --:--",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_dim"])
        self.tick_label.pack(side="right", padx=16)

    # ─────────────────────────────────────────────────────────────────────────
    #  TAG CONFIG  (shared across all textboxes)
    # ─────────────────────────────────────────────────────────────────────────

    def _configure_text_tags(self, widget: ctk.CTkTextbox):
        tb = widget._textbox
        tb.tag_configure("header",    foreground=COLORS["cyan"])
        tb.tag_configure("critical",  foreground=COLORS["red"])
        tb.tag_configure("high",      foreground=COLORS["orange"])
        tb.tag_configure("moderate",  foreground=COLORS["yellow"])
        tb.tag_configure("low",       foreground=COLORS["green"])
        tb.tag_configure("dim",       foreground=COLORS["text_dim"])
        tb.tag_configure("label",     foreground=COLORS["text_normal"])
        tb.tag_configure("value",     foreground=COLORS["text_bright"])
        tb.tag_configure("action",    foreground=COLORS["cyan"])
        tb.tag_configure("reasoning", foreground=COLORS["text_normal"])
        tb.tag_configure("event",     foreground=COLORS["blue"])
        tb.tag_configure("timestamp", foreground=COLORS["text_dim"])
        tb.tag_configure("node",      foreground=COLORS["cyan"])
        tb.tag_configure("purple",    foreground=COLORS["purple"])

    # ─────────────────────────────────────────────────────────────────────────
    #  CONTROL HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_simulation(self):
        if not self._sim_active:
            self._sim_active = True
            self.engine.start()
            self.play_btn.configure(text="⏸  PAUSE",
                                    fg_color=COLORS["orange"],
                                    hover_color=COLORS["yellow"],
                                    text_color=COLORS["bg_dark"])
            self.status_dot.configure(text="● ONLINE", text_color=COLORS["green"])
        else:
            if self.engine._paused:
                self.engine.resume()
                self.play_btn.configure(text="⏸  PAUSE",
                                        fg_color=COLORS["orange"],
                                        hover_color=COLORS["yellow"],
                                        text_color=COLORS["bg_dark"])
            else:
                self.engine.pause()
                self.play_btn.configure(text="▶  RESUME",
                                        fg_color=COLORS["green_dim"],
                                        hover_color=COLORS["green"],
                                        text_color=COLORS["bg_dark"])

    def _on_speed_change(self, value):
        val = int(value)
        self.engine.set_speed(val)
        self.speed_label.configure(text=f"{val}s/tick")

    def _on_model_change(self, value):
        self.engine.model_name = value
        self._append(self.ai_text,
                     f"\n  ↻ Model switched to: {value}\n", "event")

    def _prompt_api_key(self):
        """Open the API key dialog, then save the key to the engine."""
        dlg = APIKeyDialog(self)
        self.wait_window(dlg)
        if dlg.api_key:
            self.engine.api_key = dlg.api_key
            self._append(self.ai_text,
                         "  ✓ API key saved — OpenRouter connection active.\n", "low")
        self._refresh_key_status()

    def _refresh_key_status(self):
        if self.engine.api_key:
            masked = self.engine.api_key[:8] + "…" + self.engine.api_key[-4:]
            self.key_status_label.configure(
                text=f"🔑 {masked}", text_color=COLORS["green"])
        else:
            self.key_status_label.configure(
                text="🔑 NO KEY", text_color=COLORS["yellow"])

    # ─────────────────────────────────────────────────────────────────────────
    #  QUEUE POLLING
    # ─────────────────────────────────────────────────────────────────────────

    def _poll_queues(self):
        processed = 0
        while not self.engine.telemetry_queue.empty() and processed < 8:
            payload = self.engine.telemetry_queue.get_nowait()
            self._render_telemetry(payload)
            processed += 1

        processed = 0
        while not self.engine.ai_queue.empty() and processed < 2:
            item = self.engine.ai_queue.get_nowait()
            self._render_ai_result(item)
            self._refresh_key_status()
            processed += 1

        processed = 0
        while not self.engine.event_queue.empty() and processed < 20:
            msg = self.engine.event_queue.get_nowait()
            self._append(self.ai_text, msg + "\n", "event")
            processed += 1

        if self._sim_active:
            self._redraw_map()
            self.tick_label.configure(
                text=f"TICK: {self.engine.tick_count:04d}  "
                     f"SIM: {self.engine.sim_time.strftime('%H:%M')}"
            )

        if self._running:
            self.after(100, self._poll_queues)

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDER HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _append(self, widget: ctk.CTkTextbox, text: str, tag: str = ""):
        widget.configure(state="normal")
        if tag:
            widget._textbox.insert("end", text, tag)
        else:
            widget._textbox.insert("end", text)
        widget._textbox.see("end")
        lc = int(widget._textbox.index("end-1c").split(".")[0])
        if lc > 600:
            widget._textbox.delete("1.0", f"{lc-500}.0")
        widget.configure(state="disabled")

    def _render_telemetry(self, payload: dict):
        node    = payload["node_id"]
        ts      = payload["timestamp"]
        sev     = payload["severity"]
        bloom   = payload["bloom_level"]
        aer     = "⚡AER " if payload.get("aeration_active") else ""
        sev_tag = sev.lower()

        self._append(self.telem_text, f"\n{'─'*52}\n", "dim")
        self._append(self.telem_text, f"  ↗ {node}  @{ts}  {aer}", "header")
        self._append(self.telem_text, f"[{sev}]", sev_tag)
        self._append(self.telem_text, f"  bloom={bloom:.2f}\n", "dim")

        idx  = payload["indices"]
        rows = [
            ("NDVI",     idx["NDVI"]["value"],       "",       idx["NDVI"]["status"]),
            ("SABI",     idx["SABI"]["value"],       "",       idx["SABI"]["status"]),
            ("NDWI",     idx["NDWI"]["value"],       "",       idx["NDWI"]["status"]),
            ("CHL-a",    idx["CHL-a"]["value"],      "μg/L",   idx["CHL-a"]["status"]),
            ("TURBIDITY",idx["Turbidity"]["value"],  "NTU",    idx["Turbidity"]["status"]),
            ("SECCHI",   idx["Secchi"]["value"],     "m",      ""),
            ("COVERAGE", idx["Coverage"]["value"],   "%",      ""),
            ("CYANO",    idx["CYANO-PROXY"]["value"],"risk%",  idx["CYANO-PROXY"]["status"]),
        ]

        for label, value, unit, status in rows:
            v_str = f"{value:>7.2f} {unit:<5}"
            if label == "CHL-a":
                v_tag = "critical" if value>=50 else "high" if value>=25 else "moderate" if value>=10 else "low"
            elif label == "CYANO":
                v_tag = "critical" if value>=80 else "high" if value>=50 else "moderate" if value>=20 else "low"
            elif label == "SABI":
                v_tag = "critical" if value>=0.7 else "high" if value>=0.45 else "moderate" if value>=0.2 else "low"
            else:
                v_tag = "value"

            self._append(self.telem_text, f"    {label:<12}", "label")
            self._append(self.telem_text, v_str, v_tag)
            if status:
                self._append(self.telem_text, f"  {status}", "dim")
            self._append(self.telem_text, "\n")

    def _render_ai_result(self, item: dict):
        node    = item["node_id"]
        ts      = item["timestamp"]
        result  = item["result"]
        action  = result.get("action", "IDLE")
        sev     = result.get("severity", "LOW")
        reason  = result.get("reasoning", "—")
        bulletin= result.get("bulletin", "—")
        sev_tag = sev.lower() if sev != "IDLE" else "dim"

        self._append(self.ai_text, f"\n{'═'*50}\n", "dim")
        self._append(self.ai_text, f"  SANA-BRAIN  {node}  @{ts}\n", "purple")
        self._append(self.ai_text, "  SEVERITY : ", "dim")
        self._append(self.ai_text, f"{sev}\n", sev_tag)
        self._append(self.ai_text, "  ACTION   : ", "dim")
        self._append(self.ai_text, f"{action}\n", "action")
        self._append(self.ai_text, "\n  REASONING:\n", "dim")
        self._append(self.ai_text, f"  {reason}\n", "reasoning")

        self._append(self.bulletin_text, f"\n[{ts}] ", "timestamp")
        self._append(self.bulletin_text, f"{node} ", "node")
        self._append(self.bulletin_text, f"[{sev}]", sev_tag)
        self._append(self.bulletin_text,
                     f"\n{bulletin}\n",
                     sev_tag if sev in ("CRITICAL","HIGH") else "")

    def _update_clock(self):
        self.clock_label.configure(
            text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._update_clock)

    def _on_close(self):
        self._running = False
        self.engine.stop()
        self.after(200, self.destroy)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SANA — Smart Autonomous Natural Agent  v1.1             ║")
    print("║  Backend : OpenRouter (free Gemma models)                ║")
    print("╠══════════════════════════════════════════════════════════╣")

    # Check environment for pre-set key
    env_key = os.environ.get("OPENROUTER_API_KEY", "")
    if env_key:
        print(f"║  API key  : found in environment ({env_key[:8]}…)              ║")
    else:
        print("║  API key  : not set — will prompt in GUI dialog          ║")

    if not REQUESTS_AVAILABLE:
        print("║  ⚠ WARNING: `requests` not installed — fallback mode     ║")
        print("║    Run:  pip install requests                             ║")

    print("╚══════════════════════════════════════════════════════════╝\n")

    engine = SimulationEngine()
    if env_key:
        engine.api_key = env_key

    app = SANADashboard(engine)
    app.mainloop()


if __name__ == "__main__":
    main()