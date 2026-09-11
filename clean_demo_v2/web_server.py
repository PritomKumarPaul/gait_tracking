import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT = ROOT / "clean_demo_v2"
sys.path.insert(0, str(CLEAN_ROOT))

from realtime_gait_engine import RealtimeGaitEngine

app = FastAPI(title="OpenGait Live Tracking")
engine: Optional[RealtimeGaitEngine] = None


class EnrollRequest(BaseModel):
    track_id: int
    name: str


class ThresholdRequest(BaseModel):
    threshold: float


class DeleteRequest(BaseModel):
    name: str


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenGait Live Camera Tracking</title>
    <style>
        :root {
            --bg: #0d1117;
            --card-bg: rgba(22, 27, 34, 0.9);
            --border: #30363d;
            --accent: #58a6ff;
            --green: #2ea043;
            --green-glow: #3fb950;
            --yellow: #d29922;
            --red: #f85149;
            --text: #c9d1d9;
            --text-dim: #8b949e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 16px; min-height: 100vh; }
        
        .top-nav {
            display: flex; justify-content: space-between; align-items: center;
            background: var(--card-bg); border: 1px solid var(--border);
            padding: 12px 20px; border-radius: 10px; margin-bottom: 16px;
        }
        .brand { display: flex; align-items: center; gap: 10px; font-size: 1.15rem; font-weight: 700; color: #fff; }
        .live-dot {
            width: 10px; height: 10px; background: #3fb950; border-radius: 50%;
            box-shadow: 0 0 8px #3fb950; animation: pulse 1.8s infinite;
        }
        @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; } 100% { opacity: 1; transform: scale(1); } }
        
        .stats-strip { display: flex; gap: 16px; font-size: 0.88rem; }
        .stat-pill { background: #161b22; border: 1px solid var(--border); padding: 4px 10px; border-radius: 16px; }
        .stat-val { font-weight: bold; color: var(--accent); }

        .main-grid {
            display: grid; grid-template-columns: 1fr 360px; gap: 16px;
        }
        @media (max-width: 1080px) { .main-grid { grid-template-columns: 1fr; } }

        .panel {
            background: var(--card-bg); border: 1px solid var(--border);
            border-radius: 10px; padding: 16px;
        }
        .panel-title {
            font-size: 1rem; font-weight: 600; color: #f0f6fc; margin-bottom: 12px;
            display: flex; justify-content: space-between; align-items: center;
        }

        .stream-container {
            position: relative; width: 100%; border-radius: 8px; overflow: hidden;
            background: #000; border: 1px solid var(--border);
        }
        .stream-container img {
            width: 100%; height: auto; display: block; object-fit: contain; aspect-ratio: 4/3;
        }
        .live-badge {
            position: absolute; top: 10px; left: 10px;
            background: rgba(0,0,0,0.7); border: 1px solid var(--green-glow);
            color: var(--green-glow); padding: 3px 8px; border-radius: 4px;
            font-size: 0.75rem; font-weight: bold;
        }

        .controls-card {
            margin-top: 14px; background: #161b22; border: 1px solid var(--border);
            border-radius: 8px; padding: 12px;
        }
        .form-row { display: flex; gap: 10px; margin-bottom: 10px; align-items: center; }
        .form-row label { font-size: 0.85rem; color: var(--text-dim); min-width: 85px; }
        input[type="text"], select {
            flex: 1; background: #0d1117; border: 1px solid var(--border); color: #fff;
            padding: 7px 10px; border-radius: 6px; font-size: 0.88rem; outline: none;
        }
        input[type="text"]:focus, select:focus { border-color: var(--accent); }
        
        .slider-row { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
        .slider-row input[type="range"] { flex: 1; accent-color: var(--accent); cursor: pointer; }

        .btn {
            background: #21262d; border: 1px solid var(--border); color: #fff;
            padding: 8px 14px; border-radius: 6px; font-weight: 600; cursor: pointer;
            transition: background 0.2s; font-size: 0.88rem;
        }
        .btn:hover { background: #30363d; }
        .btn-green { background: #238636; }
        .btn-green:hover { background: #2ea043; }
        .btn-red { background: #da3633; }
        .btn-red:hover { background: #b62324; }

        .gallery-table {
            width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 0.84rem;
        }
        .gallery-table th, .gallery-table td {
            padding: 7px 8px; text-align: left; border-bottom: 1px solid var(--border);
        }
        .gallery-table th { color: var(--text-dim); font-weight: 600; }
        .del-btn {
            background: transparent; border: none; color: var(--red); cursor: pointer; font-size: 0.9rem;
        }

        .log-box {
            background: #0d1117; border: 1px solid var(--border); border-radius: 6px;
            padding: 8px; height: 210px; overflow-y: auto; font-family: monospace; font-size: 0.78rem;
        }
        .log-entry { margin-bottom: 5px; line-height: 1.35; border-bottom: 1px solid #1c2128; padding-bottom: 3px; }
        .log-time { color: var(--text-dim); }
        .log-match { color: var(--green-glow); }
        .log-enroll { color: var(--accent); }

        .toast {
            position: fixed; bottom: 20px; right: 20px; background: #1f6feb; color: #fff;
            padding: 8px 16px; border-radius: 6px; font-size: 0.85rem; opacity: 0;
            transition: opacity 0.3s; z-index: 100;
        }
    </style>
</head>
<body>
    <div class="top-nav">
        <div class="brand">
            <div class="live-dot"></div>
            <span>OpenGait Live Camera Tracking</span>
        </div>
        <div class="stats-strip">
            <div class="stat-pill">FPS: <span id="stat-fps" class="stat-val">0.0</span></div>
            <div class="stat-pill">Active: <span id="stat-tracks" class="stat-val">0</span></div>
            <div class="stat-pill">Gallery: <span id="stat-gallery" class="stat-val">0</span></div>
        </div>
    </div>

    <div class="main-grid">
        <div>
            <div class="panel">
                <div class="panel-title">
                    <span>Camera Stream</span>
                    <span style="font-size: 0.8rem; color: var(--green-glow);">Active</span>
                </div>
                <div class="stream-container">
                    <img src="/video_feed" alt="Live Camera Stream">
                    <div class="live-badge">LIVE</div>
                </div>

                <div class="controls-card">
                    <div style="font-weight: 600; margin-bottom: 10px; font-size: 0.92rem; color: #f0f6fc;">
                        Enroll Subject
                    </div>
                    <div class="form-row">
                        <label for="track-select">Track ID:</label>
                        <select id="track-select">
                            <option value="">Detecting...</option>
                        </select>
                    </div>
                    <div class="form-row">
                        <label for="person-name">Name:</label>
                        <input type="text" id="person-name" placeholder="e.g. Subject 1">
                    </div>
                    <div class="slider-row">
                        <label style="font-size:0.85rem; color:var(--text-dim); min-width:85px;">Threshold:</label>
                        <input type="range" id="thresh-slider" min="0.50" max="0.99" step="0.01" value="0.85" oninput="updateThreshDisplay(this.value)">
                        <span id="thresh-val" style="font-weight:bold; width: 40px;">0.85</span>
                        <button class="btn" style="padding: 3px 8px; font-size: 0.78rem;" onclick="setThreshold()">Set</button>
                    </div>
                    <div style="display: flex; gap: 8px; margin-top: 8px;">
                        <button class="btn btn-green" style="flex: 2;" onclick="enrollCurrentPerson()">
                            Enroll Person
                        </button>
                        <button class="btn btn-red" style="flex: 1;" onclick="clearGallery()">
                            Clear Gallery
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 16px;">
            <div class="panel">
                <div class="panel-title">
                    <span>Gallery</span>
                    <span id="gallery-count-badge" class="stat-pill" style="font-size:0.75rem;">0 subjects</span>
                </div>
                <div style="max-height: 200px; overflow-y: auto;">
                    <table class="gallery-table">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Frames</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody id="gallery-body">
                            <tr><td colspan="3" style="text-align:center; color:var(--text-dim);">No gallery entries enrolled yet.</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="panel" style="flex: 1;">
                <div class="panel-title">
                    <span>Events</span>
                    <span style="font-size: 0.75rem; color: var(--accent);">Live</span>
                </div>
                <div class="log-box" id="event-log">
                    <div class="log-entry"><span class="log-time">[System]</span> Ready.</div>
                </div>
            </div>
        </div>
    </div>

    <div id="toast" class="toast"></div>

    <script>
        function showToast(msg) {
            const t = document.getElementById("toast");
            t.innerText = msg;
            t.style.opacity = "1";
            setTimeout(() => { t.style.opacity = "0"; }, 3000);
        }

        function updateThreshDisplay(v) {
            document.getElementById("thresh-val").innerText = Number(v).toFixed(2);
        }

        async function setThreshold() {
            const v = parseFloat(document.getElementById("thresh-slider").value);
            try {
                const res = await fetch("/api/set_threshold", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ threshold: v })
                });
                showToast("Threshold set to " + v.toFixed(2));
            } catch(e) { console.error(e); }
        }

        async function enrollCurrentPerson() {
            const select = document.getElementById("track-select");
            const trackId = parseInt(select.value);
            const nameInput = document.getElementById("person-name");
            const name = nameInput.value.trim();

            if (!trackId || isNaN(trackId)) {
                showToast("No active track selected.");
                return;
            }
            if (!name) {
                showToast("Please enter a subject name.");
                return;
            }

            try {
                const res = await fetch("/api/enroll", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ track_id: trackId, name: name })
                });
                const data = await res.json();
                showToast(data.message);
                pollStatus();
            } catch(e) {
                showToast("Enrollment failed.");
                console.error(e);
            }
        }

        async function clearGallery() {
            if (!confirm("Clear enrolled gallery?")) return;
            try {
                const res = await fetch("/api/clear_gallery", { method: "POST" });
                const data = await res.json();
                showToast(data.message);
                pollStatus();
            } catch(e) { console.error(e); }
        }

        async function deleteSubject(name) {
            try {
                const res = await fetch("/api/delete_identity", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name: name })
                });
                const data = await res.json();
                showToast(data.message);
                pollStatus();
            } catch(e) { console.error(e); }
        }

        async function pollStatus() {
            try {
                const res = await fetch("/api/status");
                const data = await res.json();

                document.getElementById("stat-fps").innerText = data.fps.toFixed(1);
                document.getElementById("stat-tracks").innerText = data.active_tracks.length;
                document.getElementById("stat-gallery").innerText = data.gallery.length;
                document.getElementById("gallery-count-badge").innerText = data.gallery.length + " subjects";

                const select = document.getElementById("track-select");
                const prevVal = select.value;
                select.innerHTML = "";
                if (data.active_tracks.length === 0) {
                    const opt = document.createElement("option");
                    opt.value = "";
                    opt.innerText = "No person detected";
                    select.appendChild(opt);
                } else {
                    data.active_tracks.forEach(t => {
                        const opt = document.createElement("option");
                        opt.value = t.track_id;
                        opt.innerText = `Track #${t.track_id} - ${t.identity} (${t.frames}/30 frames)`;
                        if (String(t.track_id) === prevVal) opt.selected = true;
                        select.appendChild(opt);
                    });
                }

                const tbody = document.getElementById("gallery-body");
                if (data.gallery.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; color:var(--text-dim);">No gallery entries enrolled yet.</td></tr>`;
                } else {
                    tbody.innerHTML = "";
                    data.gallery.forEach(item => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td style="font-weight:600; color:#58a6ff;">${item.name}</td>
                            <td>${item.meta.frames || 30}</td>
                            <td><button class="del-btn" onclick="deleteSubject('${item.name}')">×</button></td>
                        `;
                        tbody.appendChild(tr);
                    });
                }

                const logBox = document.getElementById("event-log");
                if (data.events && data.events.length > 0) {
                    logBox.innerHTML = "";
                    data.events.forEach(ev => {
                        const div = document.createElement("div");
                        div.className = "log-entry";
                        let cls = "log-match";
                        if (ev.event === "ENROLL") cls = "log-enroll";
                        div.innerHTML = `<span class="log-time">[${ev.time}]</span> <span class="${cls}">[${ev.event}]</span> ${ev.message}`;
                        logBox.appendChild(div);
                    });
                }

            } catch(e) { console.error("Poll error:", e); }
        }

        setInterval(pollStatus, 1000);
        pollStatus();
    </script>
</body>
</html>
"""


@app.on_event("startup")
def startup_event():
    global engine
    engine = RealtimeGaitEngine(camera_id=0, threshold=0.85)
    engine.start()


@app.on_event("shutdown")
def shutdown_event():
    global engine
    if engine:
        engine.stop()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_DASHBOARD


@app.get("/video_feed")
def video_feed():
    if not engine:
        return JSONResponse({"error": "Engine not initialized"}, status_code=500)
    return StreamingResponse(
        engine.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
def get_status():
    if not engine:
        return JSONResponse({"error": "Engine not running"}, status_code=500)

    with engine.lock:
        tracks_info = [
            {
                "track_id": t.track_id,
                "identity": t.identity,
                "confidence": float(t.confidence),
                "status": t.status,
                "frames": len(t.sil_buffer),
                "score": float(t.score)
            }
            for t in engine.tracks.values()
            if t.track_id in engine.active_track_ids
        ]
        gallery_info = [
            {"name": k, "meta": v}
            for k, v in engine.gallery_meta.items()
        ]
        events_info = list(engine.recent_events)
        fps_val = float(engine.fps)
        thresh_val = float(engine.threshold)

    return {
        "fps": fps_val,
        "active_tracks": tracks_info,
        "gallery": gallery_info,
        "events": events_info,
        "threshold": thresh_val
    }


@app.post("/api/enroll")
def enroll(req: EnrollRequest):
    if not engine:
        return JSONResponse({"error": "Engine not running"}, status_code=500)
    success, msg = engine.enroll_person(req.track_id, req.name)
    return {"success": success, "message": msg}


@app.post("/api/clear_gallery")
def clear_gallery():
    if not engine:
        return JSONResponse({"error": "Engine not running"}, status_code=500)
    msg = engine.clear_gallery()
    return {"success": True, "message": msg}


@app.post("/api/delete_identity")
def delete_identity(req: DeleteRequest):
    if not engine:
        return JSONResponse({"error": "Engine not running"}, status_code=500)
    msg = engine.delete_identity(req.name)
    return {"success": True, "message": msg}


@app.post("/api/set_threshold")
def set_threshold(req: ThresholdRequest):
    if not engine:
        return JSONResponse({"error": "Engine not running"}, status_code=500)
    engine.threshold = float(req.threshold)
    return {"success": True, "threshold": engine.threshold}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
