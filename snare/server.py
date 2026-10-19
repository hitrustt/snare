"""HTTP and WebSocket API for the live demo, and the built web app.

uvicorn snare.server:app            # or: python -m snare.server

GET  /api/stream        WebSocket: scored payments, four times a second
POST /api/attacks       launch an attack {"kind": "card_testing"}
GET  /api/cases/{id}    why a payment was scored the way it was
GET  /api/model         how the model was trained and how well it does
GET  /api/health        liveness check for the host
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import scenarios, snapshot
from .engine import SPEED, LiveEngine, compact
from .geo import ALL_CITIES
from .model import FraudModel
from .simulator import DAY
from .world import CATEGORIES

WEB = Path(__file__).resolve().parent.parent / "web" / "dist"
TICK_SECONDS = 0.25
DEFAULT_CUSTOMERS = 12_000
CUSTOMERS = int(os.getenv("SNARE_CUSTOMERS", str(DEFAULT_CUSTOMERS)))
# The network is rebuilt after this many simulated days of viewing, so card
# histories and account groups stay in the range the model was trained on
# and memory stays flat on a long-running server.
RESET_AFTER_DAYS = float(os.getenv("SNARE_RESET_DAYS", "4"))
# Limits on launching attacks, to keep a shared demo usable.
ATTACK_COOLDOWN_SECONDS = 4.0
GLOBAL_ATTACK_SECONDS = 1.0
MAX_ACTIVE_VISITOR_ATTACKS = 6
# A viewer this many ticks behind is on a connection too slow to keep up.
CLIENT_QUEUE = 16


class Client:
    """One viewer, with its own send queue so a slow connection never holds up the others."""

    def __init__(self, socket: WebSocket) -> None:
        self.socket = socket
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=CLIENT_QUEUE)
        self.closed = asyncio.Event()

    def offer(self, payload: str) -> None:
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            self.closed.set()  # it will reconnect and start fresh

    async def pump(self) -> None:
        try:
            while not self.closed.is_set():
                await self.socket.send_text(await self.queue.get())
        except Exception:
            pass
        finally:
            self.closed.set()


class Hub:
    """Runs the engine while anyone is watching and fans results out to them."""

    def __init__(self) -> None:
        self.model: FraudModel | None = None
        self.engine: LiveEngine | None = None
        self.ready = asyncio.Event()
        self.clients: set[Client] = set()
        self.tasks: list[asyncio.Task] = []
        self.generation = 0
        self.last_attack: dict[str, float] = {}
        self.last_any_attack = 0.0

    async def start(self) -> None:
        # Start serving (and answering health checks) while the network loads.
        self.tasks.append(asyncio.create_task(self._boot()))

    async def _boot(self) -> None:
        self.model = FraudModel()
        self.engine = await asyncio.to_thread(self._build)
        self.ready.set()
        self.tasks.append(asyncio.create_task(self._run()))

    def _build(self) -> LiveEngine:
        assert self.model is not None
        self.generation += 1
        seed = 2026 + self.generation
        # A saved warm network (built during the Docker build) loads in about a
        # second; building one from scratch takes about 30 CPU-seconds.
        if snapshot.PATH.exists() and CUSTOMERS == DEFAULT_CUSTOMERS:
            return snapshot.load(self.model, seed)
        return LiveEngine(self.model, seed=seed, customers=CUSTOMERS)

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _reset(self, totals: dict) -> LiveEngine:
        """Swap in a fresh network. The caller has already let go of the old
        one, so a small instance never holds two in memory at once."""
        self.engine = None
        gc.collect()
        engine = await asyncio.to_thread(self._build)
        for key, value in totals.items():  # keep the scoreboard running
            engine.totals[key] += value
        self.engine = engine
        hello = self.hello(engine)
        for client in list(self.clients):
            client.offer(hello)
        return engine

    async def _run(self) -> None:
        last = time.perf_counter()
        last_seq = 0
        engine = self.engine
        assert engine is not None
        batch = None
        while True:
            await asyncio.sleep(TICK_SECONDS)
            now = time.perf_counter()
            elapsed, last = min(now - last, 1.0), now
            if not self.clients:
                last_seq = engine.event_seq
                continue  # nobody watching: pause the world
            if engine.sim.now - engine.started_at > RESET_AFTER_DAYS * DAY:
                totals, engine, batch = dict(engine.totals), None, None
                engine = await self._reset(totals)
                last_seq, last = engine.event_seq, time.perf_counter()
                continue
            batch = engine.advance(elapsed)
            new_events = [event for event in engine.events if event["seq"] > last_seq]
            last_seq = engine.event_seq
            payload = json.dumps(
                {
                    "type": "tick",
                    "now": engine.sim.now,
                    "txns": [compact(item, engine.threshold, engine.world) for item in batch],
                    "events": new_events,
                    "attacks": list(engine.attacks.values())[-12:],
                    "stats": engine.stats(),
                },
                separators=(",", ":"),
            )
            for client in list(self.clients):
                client.offer(payload)

    @staticmethod
    def hello(engine: LiveEngine) -> str:
        # New viewers start with the recent past so the screen isn't empty.
        recent = list(engine.recent.values())[-4000:]
        return json.dumps(
            {
                "type": "hello",
                "now": engine.sim.now,
                "threshold": engine.threshold,
                "txns": [compact(item, engine.threshold, engine.world) for item in recent],
                "attacks": list(engine.attacks.values())[-12:],
                "stats": engine.stats(),
            },
            separators=(",", ":"),
        )


hub = Hub()


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    await hub.start()
    yield
    await hub.stop()


app = FastAPI(title="Snare", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")


def _engine() -> LiveEngine:
    if hub.engine is None:
        raise HTTPException(503, "The payment network is still starting up.")
    return hub.engine


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "ready": hub.engine is not None}


@app.get("/api/model")
def model_info() -> dict:
    engine = _engine()
    return {
        "metrics": engine.model.metrics,
        "threshold": engine.threshold,
        "speed": SPEED,
        "customers": CUSTOMERS,
        "categories": [{"name": c.name, "label": c.label} for c in CATEGORIES],
        "attack_types": [{"kind": kind, "label": scenarios.ATTACK_LABELS[kind]} for kind in scenarios.ATTACK_TYPES],
        "cities": [{"name": c.name, "country": c.country, "lat": c.lat, "lng": c.lng} for c in ALL_CITIES],
    }


class AttackRequest(BaseModel):
    kind: str


# These handlers are async on purpose: they run on the event loop, the same
# thread that advances the simulation, so they never interleave with it.
@app.post("/api/attacks")
async def launch_attack(body: AttackRequest, request: Request) -> JSONResponse:
    engine = _engine()
    if body.kind not in scenarios.ATTACK_TYPES:
        raise HTTPException(400, f"Unknown attack. Choose one of: {', '.join(scenarios.ATTACK_TYPES)}.")
    now = time.time()
    # Behind the host's proxy the visitor is the first forwarded address. That
    # header can be spoofed, so there is also a global limit.
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    visitor = forwarded or (request.client.host if request.client else "unknown")
    wait = max(hub.last_attack.get(visitor, 0.0) + ATTACK_COOLDOWN_SECONDS, hub.last_any_attack + GLOBAL_ATTACK_SECONDS) - now
    if wait > 0:
        return JSONResponse({"error": f"One attack every {ATTACK_COOLDOWN_SECONDS:.0f} seconds. Try again in {max(wait, 1):.0f}s."}, status_code=429)
    active = [a for a in engine.attacks.values() if a["source"] == "visitor" and a["end"] > engine.sim.now]
    if len(active) >= MAX_ACTIVE_VISITOR_ATTACKS:
        return JSONResponse({"error": "Lots of attacks are running right now. Try again in a few seconds."}, status_code=429)
    hub.last_attack = {ip: t for ip, t in hub.last_attack.items() if now - t < ATTACK_COOLDOWN_SECONDS}
    hub.last_attack[visitor] = hub.last_any_attack = now
    return JSONResponse(engine.launch(body.kind))


@app.get("/api/cases/{txn_id}")
async def case(txn_id: int) -> dict:
    engine = _engine()
    scored = engine.recent.get(txn_id)
    found = engine.case(txn_id)
    if scored is None or found is None:
        raise HTTPException(404, "That payment has scrolled out of the recent history.")
    # SHAP is the slow part; it uses its own copy of the model, so it runs off the event loop.
    found["explanation"] = await asyncio.to_thread(engine.model.explain, scored.row, scored.txn.amount, scored.txn.category)
    return found


@app.websocket("/api/stream")
async def stream(socket: WebSocket) -> None:
    await socket.accept()
    client = Client(socket)
    pump = asyncio.create_task(client.pump())
    receive = closer = None
    try:
        await hub.ready.wait()
        while hub.engine is None:  # mid-reset
            await asyncio.sleep(0.2)
        client.offer(hub.hello(hub.engine))
        hub.clients.add(client)
        # Keep-alive pings from the client; a disconnect ends the loop.
        receive = asyncio.create_task(_drain(socket))
        closer = asyncio.create_task(client.closed.wait())
        await asyncio.wait({receive, closer}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        hub.clients.discard(client)
        client.closed.set()
        for task in (pump, receive, closer):
            if task:
                task.cancel()
        with contextlib.suppress(Exception):
            await socket.close()


async def _drain(socket: WebSocket) -> None:
    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
        while True:
            await socket.receive_text()


# The built web app, with client-side routes falling back to index.html.
if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = WEB / path
        if path and candidate.is_file() and WEB in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
