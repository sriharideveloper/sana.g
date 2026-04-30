"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SANA — Smart Autonomous Natural Agent                                       ║
║  Edge-based Eutrophication Detection & Response System                       ║
║  Fog-Computing Swarm Simulation with Agentic AI Controller                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEPENDENCIES:
    pip install customtkinter ollama

USAGE:
    python sana_dashboard.py

    Make sure `ollama` is running locally with gemma4 or llama3:
        ollama serve
        ollama pull gemma4
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import math
import queue
import random
import threading
import time
from datetime import datetime, timedelta

# ── Third-party ───────────────────────────────────────────────────────────────
import customtkinter as ctk

# Try importing ollama; fall back gracefully if not installed
try:
    import ollama as ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARN] `ollama` library not found. Running in fallback mode.")

# ── Model Configuration ──────────────────────────────────────────────────────
DEFAULT_MODEL = "gemma3:1b"   # Must match an installed model from `ollama list`

def validate_ollama_model(model_name: str) -> bool:
    """
    Verify that the requested model is actually installed in Ollama.
    Prevents silent fallback or 404 errors from wrong model names.
    Returns True if the model is found, False otherwise.
    """
    if not OLLAMA_AVAILABLE:
        return False
    try:
        models = ollama_lib.list()
        installed = [m.model for m in models.models]
        # Check exact match and tag-less match (e.g. 'gemma4' matches 'gemma4:latest')
        for installed_name in installed:
            if model_name == installed_name or model_name == installed_name.split(':')[0]:
                print(f"[OK] Model '{model_name}' found as '{installed_name}'")
                return True
        print(f"[ERROR] Model '{model_name}' NOT FOUND. Installed models: {installed}")
        return False
    except Exception as e:
        print(f"[WARN] Could not validate model: {e}")
        return False

# ═════════════════════════════════════════════════════════════════════════════
#  THEME & COLOUR CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_dark":      "#0A0E1A",   # Deep navy — main window background
    "bg_panel":     "#0D1220",   # Slightly lighter panel background
    "bg_card":      "#111827",   # Card / widget backgrounds
    "bg_terminal":  "#080C18",   # Ultra-dark terminal areas
    "border":       "#1E2D45",   # Subtle border lines
    "border_bright":"#2A4070",   # Highlighted borders

    "green":        "#00FF94",   # OK / Low severity
    "green_dim":    "#00C070",
    "yellow":       "#FFD700",   # Moderate severity
    "orange":       "#FF8C00",   # High severity
    "red":          "#FF3355",   # Critical severity
    "red_dim":      "#8B0000",

    "cyan":         "#00D4FF",   # Primary accent / headers
    "cyan_dim":     "#0088AA",
    "blue":         "#4488FF",   # Secondary accent
    "purple":       "#AA66FF",   # AI / LLM output accent

    "text_bright":  "#E8F4FF",   # Primary text
    "text_normal":  "#8BA0C0",   # Secondary text
    "text_dim":     "#3D5070",   # Disabled / background text

    "node_colors":  ["#00FF94", "#FFD700", "#FF8C00", "#FF3355"],
}

SEVERITY_COLORS = {
    "LOW":      COLORS["green"],
    "MODERATE": COLORS["yellow"],
    "HIGH":     COLORS["orange"],
    "CRITICAL": COLORS["red"],
    "IDLE":     COLORS["text_dim"],
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTOR PHYSICS ENGINE  (Simulated Environmental State)
# ═════════════════════════════════════════════════════════════════════════════

class SectorState:
    """
    Holds the living environmental state for one water sector.

    Physics model:
    ─────────────
    Each sector has a `bloom_level` (0.0–1.0) that acts as the master driver.
    All spectral indices are derived as noisy functions of bloom_level, mimicking
    how real multispectral satellite/drone sensors respond to eutrophication:

      • NDVI  ↑ with bloom (floating vegetation reflects NIR)
      • SABI  ↑ with bloom (surface algae index)
      • NDWI  ↓ with bloom (water reflectance drops under mats)
      • NDCI  ↑ with bloom (chlorophyll difference)
      • BCI   ↑ with bloom (blue-green cyanobacteria)
      • FAI   ↑ with bloom (floating algae)
      • CHL-a ↑ with bloom (μg/L, exponential at high bloom)
      • Turbidity ↑ with bloom (NTU)
      • Secchi ↓ with bloom (visibility depth drops)
      • Coverage ↑ with bloom (% surface covered)
      • Cyano-Proxy ↑ with bloom (cyanotoxin risk proxy)
      • ABI   ↑ with bloom (Algal Bloom Index)
      • Nutrient ↑ with bloom (nitrogen / phosphorus proxy)
    """

    def __init__(self, node_id: int, initial_bloom: float, trend: float):
        self.node_id      = node_id
        self.bloom_level  = initial_bloom   # 0.0 → pristine, 1.0 → toxic bloom
        self.trend        = trend           # rate of change per tick (can be negative)
        self.aeration_ticks = 0             # how many more ticks of aeration remain
        self.severity     = "LOW"
        self.last_action  = "IDLE"

    def _noise(self, scale: float = 0.01) -> float:
        """Gaussian noise injection for sensor realism."""
        return random.gauss(0, scale)

    def tick(self):
        """
        Advance the sector by one simulated 15-minute interval.

        If an aerator is running, it fights the trend and reduces bloom
        faster than natural decay, simulating dissolved-oxygen injection
        disrupting stratification and cyanobacterial buoyancy.
        """
        if self.aeration_ticks > 0:
            # Aeration effect: suppresses bloom growth, actively reduces bloom
            effective_trend = self.trend - 0.04   # aerators counter growth
            self.bloom_level = max(0.0, self.bloom_level + effective_trend)
            self.aeration_ticks -= 1
        else:
            # Natural drift: bloom grows or decays following its trend
            self.bloom_level = max(0.0, min(1.0, self.bloom_level + self.trend))

        # Slow natural recovery after bloom peaks — logistic-like brake
        if self.bloom_level > 0.85:
            self.trend = max(self.trend - 0.002, -0.005)  # slow at ceiling

        # Small random climatic variation each tick
        self.trend += random.gauss(0, 0.001)
        self.trend  = max(-0.02, min(0.03, self.trend))   # clamp trend

    def activate_aeration(self, duration_ticks: int = 6):
        """Trigger aerator; effect lasts `duration_ticks` simulation steps."""
        self.aeration_ticks = duration_ticks
        self.last_action    = "ACTIVATE_AERATOR"

    def get_indices(self) -> dict:
        """
        Derive all spectral and chemical indices from the current bloom_level.
        Each formula is a calibrated transfer function with sensor noise.
        """
        b = self.bloom_level  # shorthand

        # ── Spectral Indices ─────────────────────────────────────────────────

        # NDVI: Normalized Difference Vegetation Index
        # Water → ~0.0, dense algae → ~0.7
        ndvi = 0.05 + b * 0.65 + self._noise(0.015)

        # SABI: Surface Algal Bloom Index
        # Pristine → slightly negative, bloom → positive/high
        sabi = -0.1 + b * 1.1 + self._noise(0.02)

        # NDWI: Normalized Difference Water Index
        # Clean water → positive, covered → negative
        ndwi = 0.3 - b * 0.7 + self._noise(0.015)

        # NDCI: Normalized Difference Chlorophyll Index
        ndci = -0.05 + b * 0.85 + self._noise(0.02)

        # BCI: Blue-Cyanobacterial Index
        bci = 0.0 + b * 0.9 + self._noise(0.025)

        # FAI: Floating Algae Index
        fai = -0.02 + b * 0.55 + self._noise(0.01)

        # ── Chemical / Biological Parameters ─────────────────────────────────

        # CHL-a: Chlorophyll-a concentration (μg/L)
        # Exponential relationship to bloom intensity
        chla = 2.0 + math.exp(b * 4.6) * 1.5 + random.gauss(0, 2.0)
        chla = max(1.0, chla)

        # Turbidity (NTU): increases with bloom density
        turbidity = 1.5 + b * 48.5 + random.gauss(0, 1.5)
        turbidity = max(0.5, turbidity)

        # Secchi Depth (m): transparency decreases with bloom
        secchi = 4.5 - b * 4.0 + self._noise(0.1)
        secchi = max(0.2, secchi)

        # Coverage (%): surface covered by algal mats
        coverage = b * 100.0 + random.gauss(0, 2.0)
        coverage = max(0.0, min(100.0, coverage))

        # Cyano-Proxy: cyanobacterial bloom risk (0–100)
        cyano = max(0, min(100, b * 110 + random.gauss(0, 3)))

        # ABI: Algal Bloom Index composite
        abi = max(0, b * 0.95 + self._noise(0.02))

        # Nutrient Index: proxy for nitrogen + phosphorus load
        nutrient = 0.1 + b * 0.9 + self._noise(0.02)

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
    """
    Determine sector severity level from spectral/chemical indices.

    Thresholds based on WHO / EPA eutrophication guidelines:
      CHL-a: LOW<10, MODERATE<25, HIGH<50, CRITICAL≥50 μg/L
      CYANO: risk proxy amplifies severity
    """
    chla  = indices["CHL-a"]
    cyano = indices["CYANO"]
    sabi  = indices["SABI"]

    if chla >= 50 or cyano >= 80 or sabi >= 0.7:
        return "CRITICAL"
    elif chla >= 25 or cyano >= 50 or sabi >= 0.45:
        return "HIGH"
    elif chla >= 10 or cyano >= 20 or sabi >= 0.2:
        return "MODERATE"
    else:
        return "LOW"


def format_index_status(key: str, value: float) -> str:
    """Return a concise human-readable status label for a given index value."""
    thresholds = {
        "NDVI":      [(0.1, "CLEAR"), (0.3, "SPARSE VEG"), (0.5, "ACTIVE VEG"), (1.0, "DENSE MAT")],
        "SABI":      [(-0.05,"CLEAN"), (0.2, "EARLY BLOOM"), (0.5, "SURFACE BLOOM"), (1.0, "SEVERE BLOOM")],
        "NDWI":      [(-0.5,"COVERED"), (0.0, "TURBID"), (0.2, "MIXED"), (1.0, "CLEAR WATER")],
        "CYANO":     [(20, "SAFE"), (50, "CAUTION"), (80, "TOXIC"), (100, "EXTREME TOXIC")],
        "CHL-a":     [(10, "NORMAL"), (25, "ELEVATED"), (50, "HIGH"), (200, "CRITICAL")],
        "Turbidity": [(5, "CLEAR"), (15, "MODERATE"), (30, "TURBID"), (100, "OPAQUE")],
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
    """
    Build the JSON payload a Worker Node would transmit via LoRa radio.

    In a real deployment each node compresses this to ~256 bytes
    at 125kHz bandwidth, SF=10, for ~1.5km range over water.
    """
    indices = sector.get_indices()
    severity = classify_severity(indices)
    sector.severity = severity

    payload = {
        "node_id":   f"NODE-{sector.node_id:02d}",
        "timestamp": sim_time.strftime("%H:%M:%S"),
        "date":      sim_time.strftime("%Y-%m-%d"),
        "severity":  severity,
        "bloom_level": round(sector.bloom_level, 3),
        "aeration_active": sector.aeration_ticks > 0,
        "indices": {
            "NDVI":    {"value": indices["NDVI"],     "status": format_index_status("NDVI", indices["NDVI"])},
            "SABI":    {"value": indices["SABI"],     "status": format_index_status("SABI", indices["SABI"])},
            "NDWI":    {"value": indices["NDWI"],     "status": format_index_status("NDWI", indices["NDWI"])},
            "NDCI":    {"value": indices["NDCI"],     "status": "OK"},
            "BCI":     {"value": indices["BCI"],      "status": "OK"},
            "FAI":     {"value": indices["FAI"],      "status": "OK"},
            "CHL-a":   {"value": indices["CHL-a"],    "unit": "μg/L",  "status": format_index_status("CHL-a", indices["CHL-a"])},
            "Turbidity":{"value": indices["Turbidity"],"unit":"NTU",   "status": format_index_status("Turbidity", indices["Turbidity"])},
            "Secchi":  {"value": indices["Secchi"],   "unit": "m",     "status": "OK"},
            "Coverage":{"value": indices["Coverage"], "unit": "%",     "status": "OK"},
            "CYANO-PROXY":{"value": indices["CYANO"],"unit":"risk%", "status": format_index_status("CYANO", indices["CYANO"])},
            "ABI":     {"value": indices["ABI"],      "status": "OK"},
            "Nutrient":{"value": indices["Nutrient"], "status": "OK"},
        }
    }
    return payload


# ═════════════════════════════════════════════════════════════════════════════
#  AGENTIC AI CONTROLLER  (Ollama Integration)
# ═════════════════════════════════════════════════════════════════════════════

SANA_SYSTEM_PROMPT = """You are SANA-BRAIN, the agentic AI controller of the SANA (Smart Autonomous Natural Agent) environmental monitoring network deployed on a freshwater lake.

You receive real-time telemetry from autonomous surface nodes measuring eutrophication and algal bloom conditions.

Your job is to:
1. Analyze the incoming telemetry data
2. Determine the severity and nature of the bloom threat
3. Issue an automated response command
4. Generate a brief public safety bulletin

You MUST respond with ONLY a single valid JSON object — no markdown, no explanations outside the JSON. The JSON must have exactly these fields:

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

Respond ONLY with the JSON object."""


def call_ollama_agent(payload: dict, model: str = DEFAULT_MODEL) -> dict:
    """
    Send telemetry payload to the local Ollama LLM and parse the response.

    Threading contract: this function is ALWAYS called from a worker thread,
    never from the GUI thread, to avoid blocking the event loop.

    Returns a dict with keys: reasoning, action, severity, bulletin
    Falls back to a rule-based response if Ollama is unavailable.
    """
    # ── Fallback: rule-based response if ollama not available ─────────────────
    def rule_based_fallback(payload: dict) -> dict:
        """
        Deterministic fallback controller when the LLM is unreachable.
        Mirrors the logic the LLM would ideally apply.
        """
        severity = payload.get("severity", "LOW")
        node     = payload["node_id"]
        chla     = payload["indices"]["CHL-a"]["value"]
        cyano    = payload["indices"]["CYANO-PROXY"]["value"]
        sabi     = payload["indices"]["SABI"]["value"]

        if severity == "CRITICAL":
            action  = "CRITICAL_HUMAN_INTERVENTION"
            bulletin = (f"⚠ CRITICAL ALERT: {node} reports toxic bloom. "
                        f"CHL-a={chla}μg/L. Avoid all water contact. "
                        f"Emergency services notified. [FALLBACK MODE]")
            reason = (f"CHL-a at {chla}μg/L and CYANO-PROXY at {cyano}% exceed WHO toxic thresholds. "
                      f"SABI={sabi} confirms surface bloom. Immediate human intervention required.")
        elif severity == "HIGH":
            action  = "DEPLOY_CHEMICALS"
            bulletin = (f"HIGH ALERT: {node} — elevated algal activity detected. "
                        f"Algaecide deployment authorized. Recreational use suspended. [FALLBACK MODE]")
            reason = (f"SABI={sabi} and CHL-a={chla}μg/L indicate active bloom. "
                      f"Cyanobacterial risk elevated. Chemical intervention warranted.")
        elif severity == "MODERATE":
            action  = "ACTIVATE_AERATOR"
            bulletin = (f"MODERATE: {node} — algal levels rising. "
                        f"Aerators activated as precaution. Monitor closely. [FALLBACK MODE]")
            reason = (f"CHL-a={chla}μg/L and SABI={sabi} show early-stage bloom development. "
                      f"Aeration initiated to disrupt thermal stratification.")
        else:
            action  = "IDLE"
            bulletin = (f"NOMINAL: {node} — water quality within safe parameters. "
                        f"CHL-a={chla}μg/L. No intervention required. [FALLBACK MODE]")
            reason = (f"All indices within safe ranges. "
                      f"CHL-a={chla}μg/L, SABI={sabi}, CYANO={cyano}%. Monitoring continues.")

        return {"reasoning": reason, "action": action,
                "severity": severity, "bulletin": bulletin}

    # ── Attempt real Ollama call ───────────────────────────────────────────────
    if not OLLAMA_AVAILABLE:
        return rule_based_fallback(payload)

    try:
        # Build a compact but information-rich prompt
        key_indices = {
            "node":     payload["node_id"],
            "time":     payload["timestamp"],
            "severity": payload["severity"],
            "bloom%":   payload["bloom_level"],
            "NDVI":     payload["indices"]["NDVI"]["value"],
            "SABI":     payload["indices"]["SABI"]["value"],
            "NDWI":     payload["indices"]["NDWI"]["value"],
            "CHL-a_ugL":payload["indices"]["CHL-a"]["value"],
            "Turbidity_NTU": payload["indices"]["Turbidity"]["value"],
            "Secchi_m": payload["indices"]["Secchi"]["value"],
            "Coverage%":payload["indices"]["Coverage"]["value"],
            "CYANO%":   payload["indices"]["CYANO-PROXY"]["value"],
            "ABI":      payload["indices"]["ABI"]["value"],
        }

        user_message = (
            f"Telemetry received from {key_indices['node']} at {key_indices['time']}:\n"
            f"{json.dumps(key_indices, indent=2)}\n\n"
            f"Analyze this data and provide your JSON response."
        )

        response = ollama_lib.chat(
            model   = model,
            messages=[
                {"role": "system",  "content": SANA_SYSTEM_PROMPT},
                {"role": "user",    "content": user_message},
            ],
            options={"temperature": 0.2, "num_predict": 512},
        )

        # ── Verify correct model was used ─────────────────────────────────────
        served_model = response["model"] if isinstance(response, dict) else getattr(response, "model", None)
        if served_model and model not in served_model:
            print(f"[WARN] Model mismatch! Requested '{model}' but got '{served_model}'")

        raw_text = response["message"]["content"].strip()

        # ── Log raw LLM response to terminal ──────────────────────────────────
        print(raw_text)

        # ── JSON extraction: strip markdown code fences if present ─────────────
        if "```" in raw_text:
            # Extract content between first ``` and last ```
            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            raw_text = raw_text[start:end] if start != -1 else raw_text

        parsed = json.loads(raw_text)

        # Validate required fields; fall back on missing keys
        for key in ("reasoning", "action", "severity", "bulletin"):
            if key not in parsed:
                raise ValueError(f"Missing key '{key}' in LLM response")

        # Tag response with model metadata for traceability
        parsed["_model_used"] = served_model or model

        return parsed

    except json.JSONDecodeError as e:
        # LLM returned malformed JSON → use fallback but note the error
        fallback = rule_based_fallback(payload)
        fallback["reasoning"] = f"[JSON PARSE ERROR: {e}] " + fallback["reasoning"]
        return fallback

    except Exception as e:
        err_str = str(e)
        # ── Detect model-not-found specifically ───────────────────────────────
        if "404" in err_str or "not found" in err_str.lower():
            print(f"[CRITICAL] Model '{model}' not found on Ollama server! "
                  f"Run: ollama pull {model}")
            fallback = rule_based_fallback(payload)
            fallback["reasoning"] = (f"[MODEL NOT FOUND: '{model}'] "
                                     f"Run `ollama pull {model}` to fix. "
                                     + fallback["reasoning"])
            return fallback

        # Ollama server unreachable, timeout, etc.
        fallback = rule_based_fallback(payload)
        fallback["reasoning"] = f"[OLLAMA ERROR: {type(e).__name__}: {e}] " + fallback["reasoning"]
        return fallback


# ═════════════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE  (ties all subsystems together)
# ═════════════════════════════════════════════════════════════════════════════

class SimulationEngine:
    """
    Orchestrates the four-sector environment, LoRa telemetry generation,
    and AI inference scheduling.

    Design:
    ───────
    A background thread runs the simulation loop. Results are placed into
    thread-safe queues that the GUI polls via `after()` callbacks — this
    keeps the Tkinter main thread exclusively for rendering.

    Queues:
        telemetry_queue  → raw LoRa payloads (one per sector per tick)
        ai_queue         → AI analysis results (one per tick, best sector)
        event_queue      → log/event strings for the terminal panel
    """

    def __init__(self):
        # Initialise the four water sectors with varied bloom states and trends
        self.sectors = [
            SectorState(1, initial_bloom=0.05, trend=+0.008),  # Sector 1: slowly worsening
            SectorState(2, initial_bloom=0.15, trend=+0.020),  # Sector 2: developing bloom ⚠
            SectorState(3, initial_bloom=0.02, trend=-0.002),  # Sector 3: recovering
            SectorState(4, initial_bloom=0.40, trend=+0.012),  # Sector 4: already elevated
        ]

        # Simulated clock — starts at 06:00 today, advances 15 min per tick
        self.sim_time  = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
        self.tick_count = 0

        # Queues for GUI communication
        self.telemetry_queue = queue.Queue()
        self.ai_queue        = queue.Queue()
        self.event_queue     = queue.Queue()

        # Simulation state
        self._running    = False
        self._paused     = False
        self._thread     = None
        self._ai_thread  = None
        self.tick_interval = 4.0   # seconds between ticks (adjustable via slider)
        self.model_name  = DEFAULT_MODEL

        # Track which sector the AI is currently analysing
        self._ai_pending = False

    def start(self):
        """Start the background simulation thread."""
        self._running = True
        self._paused  = False
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False

    def set_speed(self, interval_seconds: float):
        """Adjust how long each simulated 15-minute interval takes in real time."""
        self.tick_interval = max(1.0, float(interval_seconds))

    def _run_loop(self):
        """
        Main simulation loop — runs in a daemon thread.

        Each iteration:
          1. Advance all sector states by one tick
          2. Generate LoRa payloads for every sector
          3. Push payloads to telemetry_queue
          4. Pick the highest-severity sector for AI analysis
          5. Spawn AI inference in a separate thread
          6. Sleep for tick_interval seconds
        """
        while self._running:
            if self._paused:
                time.sleep(0.2)
                continue

            # ── Advance time ─────────────────────────────────────────────────
            self.sim_time  += timedelta(minutes=15)
            self.tick_count += 1

            self.event_queue.put(
                f"[{self.sim_time.strftime('%H:%M')}] ── TICK {self.tick_count:04d} ──"
                f" {'🔴' if any(s.bloom_level>0.7 for s in self.sectors) else '🟢'}"
            )

            # ── Advance all sectors ───────────────────────────────────────────
            payloads = []
            for sector in self.sectors:
                sector.tick()
                payload = build_lora_payload(sector, self.sim_time)
                payloads.append(payload)

                # LoRa transmission event
                self.event_queue.put(
                    f"  ↗ LoRa RX  NODE-{sector.node_id:02d} → QUEEN  "
                    f"[bloom={sector.bloom_level:.2f}  sev={sector.severity}]"
                )
                self.telemetry_queue.put(payload)

                # Short stagger between node transmissions (simulates LoRa timing)
                time.sleep(0.12)

            # ── Select most critical sector for AI deep-analysis ──────────────
            worst = max(payloads, key=lambda p: (
                {"LOW":0,"MODERATE":1,"HIGH":2,"CRITICAL":3}[p["severity"]]
            ))

            # Run AI inference in a separate thread to avoid blocking sim loop
            if not self._ai_pending:
                self._ai_pending = True
                ai_t = threading.Thread(
                    target=self._run_ai_analysis,
                    args=(worst,),
                    daemon=True
                )
                ai_t.start()

            # ── Sleep until next tick ─────────────────────────────────────────
            elapsed = 0
            while elapsed < self.tick_interval and self._running and not self._paused:
                time.sleep(0.1)
                elapsed += 0.1

    def _run_ai_analysis(self, payload: dict):
        """
        Call the Ollama AI controller, parse the response, apply any
        interventions back to the sector state, and push results to ai_queue.
        """
        node_num = int(payload["node_id"].split("-")[1])
        sector   = self.sectors[node_num - 1]

        self.event_queue.put(
            f"  🤖 SANA-BRAIN analysing {payload['node_id']}..."
        )

        ai_result = call_ollama_agent(payload, model=self.model_name)

        # ── Apply AI action back to the physical sector state ─────────────────
        action = ai_result.get("action", "IDLE")
        if action == "ACTIVATE_AERATOR":
            sector.activate_aeration(duration_ticks=6)
            self.event_queue.put(
                f"  ⚡ AERATOR ACTIVATED  {payload['node_id']}  "
                f"(6-tick intervention)"
            )
        elif action in ("DEPLOY_CHEMICALS", "CRITICAL_HUMAN_INTERVENTION"):
            # Strong chemical intervention: larger bloom reduction
            sector.activate_aeration(duration_ticks=10)
            sector.trend = min(sector.trend, -0.01)
            self.event_queue.put(
                f"  ☣  {action}  {payload['node_id']}  "
                f"[ALERT DISPATCHED]"
            )

        # Push result to GUI queue
        self.ai_queue.put({
            "node_id":   payload["node_id"],
            "timestamp": payload["timestamp"],
            "result":    ai_result,
            "payload":   payload,
        })

        self._ai_pending = False


# ═════════════════════════════════════════════════════════════════════════════
#  GUI DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

class SANADashboard(ctk.CTk):
    """
    Main application window — SANA Hydro-Informatics Command Centre.

    Layout (grid-based, dark-mode):
    ┌─────────────────────────────────────────────────────────────┐
    │  HEADER BAR                                                 │
    ├──────────────┬──────────────────────┬───────────────────────┤
    │  SECTOR MAP  │  LIVE TELEMETRY      │  AI AGENT TERMINAL    │
    │  (4 nodes)   │  STREAM              │                       │
    ├──────────────┴──────────────────────┤  (reasoning + cmds)   │
    │  PUBLIC BULLETIN BOARD              │                       │
    ├─────────────────────────────────────┴───────────────────────┤
    │  CONTROL BAR  [Play | Pause | Speed slider | Model select]  │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, engine: SimulationEngine):
        super().__init__()
        self.engine = engine

        # ── Window config ─────────────────────────────────────────────────────
        self.title("SANA · Smart Autonomous Natural Agent  |  Eutrophication Monitor v1.0")
        self.geometry("1440x860")
        self.minsize(1200, 720)
        self.configure(fg_color=COLORS["bg_dark"])

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # ── Internal state ────────────────────────────────────────────────────
        self._running         = True
        self._sim_active      = False
        self.bulletin_entries = []   # list of bulletin dicts for the board

        # ── Build all panels ──────────────────────────────────────────────────
        self._build_header()
        self._build_main_grid()
        self._build_control_bar()

        # ── Start GUI polling loop ────────────────────────────────────────────
        self._poll_queues()

        # ── Window close handler ─────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────────────────
    #  LAYOUT CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def _build_header(self):
        """Top banner with logo, status indicator, and clock."""
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                              corner_radius=0, height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Logo text
        logo = ctk.CTkLabel(
            header,
            text="◈ SANA",
            font=ctk.CTkFont(family="Courier New", size=22, weight="bold"),
            text_color=COLORS["cyan"],
        )
        logo.pack(side="left", padx=18, pady=8)

        sub = ctk.CTkLabel(
            header,
            text="SMART AUTONOMOUS NATURAL AGENT  ·  FOG-COMPUTING SWARM  ·  EUTROPHICATION MONITOR",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_dim"],
        )
        sub.pack(side="left", padx=4)

        # Live clock (updates every second)
        self.clock_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(family="Courier New", size=12),
            text_color=COLORS["text_normal"],
        )
        self.clock_label.pack(side="right", padx=18)
        self._update_clock()

        # System status
        self.status_dot = ctk.CTkLabel(
            header,
            text="● OFFLINE",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            text_color=COLORS["text_dim"],
        )
        self.status_dot.pack(side="right", padx=12)

    def _build_main_grid(self):
        """Create the three-column main content area."""
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        main.columnconfigure(0, weight=2)   # Sector map + bulletins
        main.columnconfigure(1, weight=3)   # Telemetry stream
        main.columnconfigure(2, weight=3)   # AI terminal
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)

        # ── LEFT column ───────────────────────────────────────────────────────
        left_col = ctk.CTkFrame(main, fg_color="transparent")
        left_col.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0,4))
        left_col.rowconfigure(0, weight=3)
        left_col.rowconfigure(1, weight=2)

        self._build_sector_map(left_col)
        self._build_bulletin_board(left_col)

        # ── CENTRE column ─────────────────────────────────────────────────────
        self._build_telemetry_panel(main)

        # ── RIGHT column ──────────────────────────────────────────────────────
        self._build_ai_terminal(main)

    def _make_panel(self, parent, title: str, row: int, col: int,
                    rowspan=1, sticky="nsew", padx=(0,4), pady=(0,4)) -> ctk.CTkFrame:
        """Helper: create a labelled panel card."""
        outer = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                             corner_radius=6, border_width=1,
                             border_color=COLORS["border"])
        outer.grid(row=row, column=col, rowspan=rowspan,
                   sticky=sticky, padx=padx, pady=pady)

        title_bar = ctk.CTkFrame(outer, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        ctk.CTkLabel(
            title_bar, text=f"  {title}",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["cyan_dim"], anchor="w",
        ).pack(side="left", fill="y")

        content = ctk.CTkFrame(outer, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=6, pady=4)

        return content

    # ── SECTOR MAP ────────────────────────────────────────────────────────────

    def _build_sector_map(self, parent):
        """
        Visual lake map: a canvas with 4 node indicators positioned around
        a stylised water body.  Colors update in real-time based on severity.
        """
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=0, sticky="nsew", padx=(0,0), pady=(0,4))

        title_bar = ctk.CTkFrame(content, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar,
            text="  ⬡ SECTOR MAP  ·  LAKE OVERVIEW",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["cyan_dim"], anchor="w",
        ).pack(side="left", fill="y")

        self.map_canvas = ctk.CTkCanvas(
            content, bg=COLORS["bg_dark"],
            highlightthickness=0,
        )
        self.map_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.map_canvas.bind("<Configure>", self._redraw_map)

        # Node positions as fractions of canvas size (NW, NE, SW, SE)
        self.node_positions = [
            (0.25, 0.28),  # Node 1 — NW
            (0.72, 0.28),  # Node 2 — NE
            (0.25, 0.72),  # Node 3 — SW
            (0.72, 0.72),  # Node 4 — SE
        ]

        # Per-node label refs for dynamic updates
        self.map_node_items = {}

    def _redraw_map(self, event=None):
        """Redraw the entire map canvas (called on resize or state change)."""
        c = self.map_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            return

        # ── Water body ────────────────────────────────────────────────────────
        cx, cy = w * 0.5, h * 0.5
        rx, ry = w * 0.36, h * 0.42
        c.create_oval(cx-rx, cy-ry, cx+rx, cy+ry,
                      fill="#0A1828", outline=COLORS["border_bright"], width=1)
        c.create_text(cx, cy, text="◈ LAKE", fill=COLORS["text_dim"],
                      font=("Courier New", 10))

        # ── Queen node (centre) ───────────────────────────────────────────────
        c.create_oval(cx-10, cy-10, cx+10, cy+10,
                      fill=COLORS["cyan_dim"], outline=COLORS["cyan"], width=2)
        c.create_text(cx, cy+20, text="QUEEN", fill=COLORS["cyan"],
                      font=("Courier New", 8, "bold"))

        # ── Worker nodes ──────────────────────────────────────────────────────
        self.map_node_items = {}
        for i, (fx, fy) in enumerate(self.node_positions):
            sector   = self.engine.sectors[i]
            x, y     = w * fx, h * fy
            severity = sector.severity
            color    = SEVERITY_COLORS.get(severity, COLORS["green"])
            pulse_r  = 14 + int(sector.bloom_level * 10)

            # Pulsing ring (bloom radius visual)
            c.create_oval(x-pulse_r, y-pulse_r, x+pulse_r, y+pulse_r,
                          fill="", outline=color, width=1, dash=(3,3))

            # Node body
            c.create_oval(x-10, y-10, x+10, y+10,
                          fill=COLORS["bg_card"], outline=color, width=2)
            c.create_text(x, y, text=f"{i+1}", fill=color,
                          font=("Courier New", 9, "bold"))
            c.create_text(x, y+18, text=f"N-{i+1:02d}", fill=COLORS["text_normal"],
                          font=("Courier New", 7))
            c.create_text(x, y+28, text=severity, fill=color,
                          font=("Courier New", 7, "bold"))

            # Bloom % bar
            bar_w = 36
            bar_h = 4
            c.create_rectangle(x-bar_w//2, y+34, x+bar_w//2, y+34+bar_h,
                                fill=COLORS["bg_terminal"], outline="")
            filled = int((bar_w) * sector.bloom_level)
            if filled > 0:
                c.create_rectangle(x-bar_w//2, y+34,
                                   x-bar_w//2+filled, y+34+bar_h,
                                   fill=color, outline="")

            # LoRa spokes to queen
            c.create_line(x, y, cx, cy,
                          fill=COLORS["border"], width=1, dash=(2,4))

            # Aerator indicator
            if sector.aeration_ticks > 0:
                c.create_text(x, y-20, text="⚡AER", fill=COLORS["blue"],
                              font=("Courier New", 7, "bold"))

    # ── BULLETIN BOARD ───────────────────────────────────────────────────────

    def _build_bulletin_board(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=1, column=0, sticky="nsew", padx=(0,0), pady=(0,0))

        title_bar = ctk.CTkFrame(content, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar,
            text="  📢 PUBLIC BULLETIN BOARD",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["cyan_dim"], anchor="w",
        ).pack(side="left", fill="y")

        self.bulletin_text = ctk.CTkTextbox(
            content, fg_color=COLORS["bg_terminal"],
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_bright"],
            corner_radius=0, wrap="word", state="disabled",
        )
        self.bulletin_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._configure_bulletin_tags()

    def _configure_bulletin_tags(self):
        """Set up colour tags on the bulletin textbox."""
        tb = self.bulletin_text._textbox
        for sev, color in SEVERITY_COLORS.items():
            tb.tag_configure(sev, foreground=color)
        tb.tag_configure("timestamp", foreground=COLORS["text_dim"])
        tb.tag_configure("node",      foreground=COLORS["cyan"])

    # ── TELEMETRY PANEL ──────────────────────────────────────────────────────

    def _build_telemetry_panel(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=1, rowspan=2, sticky="nsew",
                     padx=(0,4), pady=(0,0))

        title_bar = ctk.CTkFrame(content, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar,
            text="  📡 LIVE TELEMETRY STREAM  ·  LoRa RX → QUEEN NODE",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["cyan_dim"], anchor="w",
        ).pack(side="left", fill="y")

        self.telem_text = ctk.CTkTextbox(
            content, fg_color=COLORS["bg_terminal"],
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_bright"],
            corner_radius=0, wrap="none", state="disabled",
        )
        self.telem_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._configure_telem_tags()

        # Initial placeholder
        self._telem_append(
            "  SANA QUEEN NODE  —  WAITING FOR TELEMETRY\n"
            "  ─────────────────────────────────────────\n"
            "  Start simulation to begin receiving LoRa packets.\n\n",
            "dim"
        )

    def _configure_telem_tags(self):
        tb = self.telem_text._textbox
        tb.tag_configure("header",   foreground=COLORS["cyan"])
        tb.tag_configure("critical", foreground=COLORS["red"])
        tb.tag_configure("high",     foreground=COLORS["orange"])
        tb.tag_configure("moderate", foreground=COLORS["yellow"])
        tb.tag_configure("low",      foreground=COLORS["green"])
        tb.tag_configure("dim",      foreground=COLORS["text_dim"])
        tb.tag_configure("label",    foreground=COLORS["text_normal"])
        tb.tag_configure("value",    foreground=COLORS["text_bright"])

    # ── AI AGENT TERMINAL ────────────────────────────────────────────────────

    def _build_ai_terminal(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=2, rowspan=2, sticky="nsew",
                     padx=(0,0), pady=(0,0))

        title_bar = ctk.CTkFrame(content, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar,
            text="  🤖 SANA-BRAIN  ·  AGENTIC AI CONTROLLER",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["purple"], anchor="w",
        ).pack(side="left", fill="y")

        self.ai_text = ctk.CTkTextbox(
            content, fg_color=COLORS["bg_terminal"],
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_bright"],
            corner_radius=0, wrap="word", state="disabled",
        )
        self.ai_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._configure_ai_tags()

        self._ai_append(
            "  SANA-BRAIN OFFLINE\n"
            "  ──────────────────────────────────────────────\n"
            "  Awaiting first telemetry packet.\n"
            f"  Ollama status: {'AVAILABLE' if OLLAMA_AVAILABLE else '⚠ NOT FOUND (fallback mode)'}\n\n",
            "dim"
        )

    def _configure_ai_tags(self):
        tb = self.ai_text._textbox
        tb.tag_configure("header",   foreground=COLORS["purple"])
        tb.tag_configure("reasoning",foreground=COLORS["text_normal"])
        tb.tag_configure("action",   foreground=COLORS["cyan"])
        tb.tag_configure("critical", foreground=COLORS["red"])
        tb.tag_configure("high",     foreground=COLORS["orange"])
        tb.tag_configure("moderate", foreground=COLORS["yellow"])
        tb.tag_configure("low",      foreground=COLORS["green"])
        tb.tag_configure("dim",      foreground=COLORS["text_dim"])
        tb.tag_configure("event",    foreground=COLORS["blue"])

    # ── CONTROL BAR ──────────────────────────────────────────────────────────

    def _build_control_bar(self):
        """Bottom bar: Play/Pause, speed slider, model selector."""
        bar = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                           corner_radius=0, height=50)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        # Left: play/pause
        self.play_btn = ctk.CTkButton(
            bar, text="▶  START",
            width=120, height=34,
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            fg_color=COLORS["green_dim"],
            hover_color=COLORS["green"],
            text_color=COLORS["bg_dark"],
            command=self._toggle_simulation,
        )
        self.play_btn.pack(side="left", padx=12, pady=8)

        # Speed label + slider
        ctk.CTkLabel(
            bar, text="SPEED:",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_normal"],
        ).pack(side="left", padx=(12,2))

        self.speed_label = ctk.CTkLabel(
            bar, text="4s/tick",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["cyan"],
            width=60,
        )
        self.speed_label.pack(side="left", padx=(0,4))

        self.speed_slider = ctk.CTkSlider(
            bar, from_=1, to=20,
            width=180, height=18,
            command=self._on_speed_change,
            button_color=COLORS["cyan"],
            button_hover_color=COLORS["cyan_dim"],
            progress_color=COLORS["border_bright"],
            fg_color=COLORS["border"],
        )
        self.speed_slider.set(4)
        self.speed_slider.pack(side="left", padx=4)

        # Separator
        ctk.CTkLabel(
            bar, text="│",
            text_color=COLORS["border_bright"],
        ).pack(side="left", padx=12)

        # Model selector
        ctk.CTkLabel(
            bar, text="MODEL:",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_normal"],
        ).pack(side="left", padx=(0,4))

        self.model_var = ctk.StringVar(value="gemma3:1b")
        self.model_menu = ctk.CTkOptionMenu(
            bar,
            values=["gemma3:1b", "gemma4:1b", "gemma:4b", "llama3", "mistral", "phi3", "llama3.2"],
            variable=self.model_var,
            width=130, height=30,
            font=ctk.CTkFont(family="Courier New", size=10),
            fg_color=COLORS["bg_card"],
            button_color=COLORS["border_bright"],
            button_hover_color=COLORS["border"],
            text_color=COLORS["text_bright"],
            command=self._on_model_change,
        )
        self.model_menu.pack(side="left", padx=4)

        # Right: tick counter
        self.tick_label = ctk.CTkLabel(
            bar,
            text="TICK: 0000  SIM TIME: --:--",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_dim"],
        )
        self.tick_label.pack(side="right", padx=16)

        # Ollama status
        ollama_status = "● OLLAMA OK" if OLLAMA_AVAILABLE else "⚠ FALLBACK"
        ollama_color  = COLORS["green"] if OLLAMA_AVAILABLE else COLORS["yellow"]
        ctk.CTkLabel(
            bar,
            text=ollama_status,
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=ollama_color,
        ).pack(side="right", padx=12)

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

    # ─────────────────────────────────────────────────────────────────────────
    #  QUEUE POLLING  (GUI thread update loop)
    # ─────────────────────────────────────────────────────────────────────────

    def _poll_queues(self):
        """
        Poll the three engine queues every 100ms from the GUI thread.
        This is the safe way to update Tkinter widgets from data produced
        by background threads (direct cross-thread widget calls crash Tkinter).
        """
        # ── Process telemetry queue ───────────────────────────────────────────
        processed = 0
        while not self.engine.telemetry_queue.empty() and processed < 8:
            payload = self.engine.telemetry_queue.get_nowait()
            self._render_telemetry(payload)
            processed += 1

        # ── Process AI result queue ───────────────────────────────────────────
        processed = 0
        while not self.engine.ai_queue.empty() and processed < 2:
            item = self.engine.ai_queue.get_nowait()
            self._render_ai_result(item)
            processed += 1

        # ── Process event log queue ───────────────────────────────────────────
        processed = 0
        while not self.engine.event_queue.empty() and processed < 20:
            msg = self.engine.event_queue.get_nowait()
            self._ai_append(msg + "\n", "event")
            processed += 1

        # ── Refresh map and tick counter ──────────────────────────────────────
        if self._sim_active:
            self._redraw_map()
            self.tick_label.configure(
                text=f"TICK: {self.engine.tick_count:04d}  "
                     f"SIM TIME: {self.engine.sim_time.strftime('%H:%M')}"
            )

        # Schedule next poll
        if self._running:
            self.after(100, self._poll_queues)

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDER HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _telem_append(self, text: str, tag: str = ""):
        """Append styled text to the telemetry textbox."""
        tb = self.telem_text
        tb.configure(state="normal")
        if tag:
            tb._textbox.insert("end", text, tag)
        else:
            tb._textbox.insert("end", text)
        tb._textbox.see("end")
        # Trim to ~600 lines for performance
        line_count = int(tb._textbox.index("end-1c").split(".")[0])
        if line_count > 600:
            tb._textbox.delete("1.0", f"{line_count-500}.0")
        tb.configure(state="disabled")

    def _ai_append(self, text: str, tag: str = ""):
        """Append styled text to the AI terminal textbox."""
        tb = self.ai_text
        tb.configure(state="normal")
        if tag:
            tb._textbox.insert("end", text, tag)
        else:
            tb._textbox.insert("end", text)
        tb._textbox.see("end")
        line_count = int(tb._textbox.index("end-1c").split(".")[0])
        if line_count > 600:
            tb._textbox.delete("1.0", f"{line_count-500}.0")
        tb.configure(state="disabled")

    def _bulletin_append(self, text: str, tag: str = ""):
        """Append styled text to the bulletin board."""
        tb = self.bulletin_text
        tb.configure(state="normal")
        if tag:
            tb._textbox.insert("end", text, tag)
        else:
            tb._textbox.insert("end", text)
        tb._textbox.see("end")
        tb.configure(state="disabled")

    def _render_telemetry(self, payload: dict):
        """Format and display a LoRa payload in the telemetry stream panel."""
        node     = payload["node_id"]
        ts       = payload["timestamp"]
        sev      = payload["severity"]
        bloom    = payload["bloom_level"]
        aer      = "⚡AER " if payload.get("aeration_active") else ""
        sev_tag  = sev.lower()

        # Divider + header
        self._telem_append(f"\n{'─'*52}\n", "dim")
        self._telem_append(f"  ↗ {node}  @{ts}  {aer}", "header")
        self._telem_append(f"[{sev}]", sev_tag)
        self._telem_append(f"  bloom={bloom:.2f}\n", "dim")

        # Key indices table
        idx = payload["indices"]
        rows = [
            ("NDVI",     idx["NDVI"]["value"],      "",      idx["NDVI"]["status"]),
            ("SABI",     idx["SABI"]["value"],      "",      idx["SABI"]["status"]),
            ("NDWI",     idx["NDWI"]["value"],      "",      idx["NDWI"]["status"]),
            ("CHL-a",    idx["CHL-a"]["value"],     "μg/L",  idx["CHL-a"]["status"]),
            ("TURBIDITY",idx["Turbidity"]["value"], "NTU",   idx["Turbidity"]["status"]),
            ("SECCHI",   idx["Secchi"]["value"],    "m",     ""),
            ("COVERAGE", idx["Coverage"]["value"],  "%",     ""),
            ("CYANO",    idx["CYANO-PROXY"]["value"],"risk%",idx["CYANO-PROXY"]["status"]),
        ]

        for label, value, unit, status in rows:
            # Colour value by severity contribution
            v_str  = f"{value:>7.2f} {unit:<5}"
            s_str  = status

            # Determine value colour
            if label == "CHL-a":
                v_tag = "critical" if value>=50 else "high" if value>=25 else "moderate" if value>=10 else "low"
            elif label == "CYANO":
                v_tag = "critical" if value>=80 else "high" if value>=50 else "moderate" if value>=20 else "low"
            elif label == "SABI":
                v_tag = "critical" if value>=0.7 else "high" if value>=0.45 else "moderate" if value>=0.2 else "low"
            else:
                v_tag = "value"

            self._telem_append(f"    {label:<12}", "label")
            self._telem_append(f"{v_str}", v_tag)
            if s_str:
                self._telem_append(f"  {s_str}", "dim")
            self._telem_append("\n")

    def _render_ai_result(self, item: dict):
        """Format and display an AI analysis result in the AI terminal panel."""
        node    = item["node_id"]
        ts      = item["timestamp"]
        result  = item["result"]
        action  = result.get("action", "IDLE")
        sev     = result.get("severity", "LOW")
        reason  = result.get("reasoning", "—")
        bulletin= result.get("bulletin", "—")

        sev_tag = sev.lower() if sev != "IDLE" else "dim"

        # ── AI terminal output ────────────────────────────────────────────────
        self._ai_append(f"\n{'═'*50}\n", "dim")
        self._ai_append(f"  SANA-BRAIN  {node}  @{ts}\n", "header")
        self._ai_append(f"  SEVERITY : ", "dim")
        self._ai_append(f"{sev}\n", sev_tag)
        self._ai_append(f"  ACTION   : ", "dim")
        self._ai_append(f"{action}\n", "action")
        self._ai_append(f"\n  REASONING:\n", "dim")
        self._ai_append(f"  {reason}\n", "reasoning")

        # ── Bulletin board update ─────────────────────────────────────────────
        self._bulletin_append(f"\n[{ts}] ", "timestamp")
        self._bulletin_append(f"{node} ", "node")
        self._bulletin_append(f"[{sev}]", sev_tag)
        self._bulletin_append(f"\n{bulletin}\n", sev_tag if sev in ("CRITICAL","HIGH") else "")

    # ─────────────────────────────────────────────────────────────────────────
    #  UTILITY
    # ─────────────────────────────────────────────────────────────────────────

    def _update_clock(self):
        """Update the real-world clock in the header every second."""
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self.clock_label.configure(text=now)
        self.after(1000, self._update_clock)

    def _on_close(self):
        """Clean shutdown."""
        self._running = False
        self.engine.stop()
        self.after(200, self.destroy)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SANA — Smart Autonomous Natural Agent  v1.0             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Ollama available : {'YES — LLM inference active' if OLLAMA_AVAILABLE else 'NO  — using rule-based fallback':<34}║")
    print(f"║  Target model     : {DEFAULT_MODEL:<34}║")

    # ── Startup model validation ─────────────────────────────────────────────
    if OLLAMA_AVAILABLE:
        model_ok = validate_ollama_model(DEFAULT_MODEL)
        status = "VERIFIED ✓" if model_ok else "NOT FOUND ✗ (will use fallback)"
        print(f"║  Model status     : {status:<34}║")
    else:
        print(f"║  Model status     : {'SKIPPED (no ollama)':<34}║")

    print("╚══════════════════════════════════════════════════════════╝\n")

    engine = SimulationEngine()
    app    = SANADashboard(engine)
    app.mainloop()


if __name__ == "__main__":
    main()
