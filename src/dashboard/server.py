"""Web dashboard: live camera with helmet labels, detection results, alerts and recordings.

The dashboard itself is served on http://localhost:<web_port> and only this computer can open it.
The phone camera page is served separately over HTTPS on <phone_port> (phone browsers only allow
the camera on HTTPS pages) and only accepts pictures that carry the secret key in its address.
"""

from __future__ import annotations

import io
import secrets
import socket
import threading
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from src.camera import check_source, describe_source, is_phone, list_screens, list_webcams
from src.config import Config, save_config
from src.engine import MonitorEngine

STATIC = Path(__file__).resolve().parent / "static"
PROJECT = Path(__file__).resolve().parents[2]
DEMO_VIDEO = PROJECT / "samples" / "demo.mp4"
CERT_DIR = PROJECT / ".certs"


def lan_ip() -> str:
    """This computer's address on the local network (no data is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def ensure_certificate(ip: str) -> tuple[str, str]:
    """Self-signed HTTPS certificate for the phone page (created once, in .certs/)."""
    cert_path, key_path = CERT_DIR / "phone-cert.pem", CERT_DIR / "phone-key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Safety Helmet Detection")])
    alt_names = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    if ip != "127.0.0.1":
        alt_names.append(x509.IPAddress(ipaddress.ip_address(ip)))
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
            .sign(key, hashes.SHA256()))

    CERT_DIR.mkdir(exist_ok=True)
    for path, data in ((key_path, key.private_bytes(serialization.Encoding.PEM,
                                                    serialization.PrivateFormat.TraditionalOpenSSL,
                                                    serialization.NoEncryption())),
                       (cert_path, cert.public_bytes(serialization.Encoding.PEM))):
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return str(cert_path), str(key_path)


def qr_svg(text: str) -> str:
    import qrcode
    import qrcode.image.svg

    image = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
    buffer = io.BytesIO()
    image.save(buffer)
    return buffer.getvalue().decode("utf-8")


def parse_saved_camera(entry: str) -> tuple[str, str]:
    name, _, address = entry.partition("|")
    return (name.strip(), address.strip()) if address else (describe_source(name.strip()), name.strip())


class Dashboard:
    def __init__(self, config: Config, config_path: str, engine: MonitorEngine | None = None):
        self.config = config
        self.config_path = config_path
        self.engine = engine or MonitorEngine(config)
        self.phone_key = secrets.token_urlsafe(16)
        self.phone_server = None
        self._phone_lock = threading.Lock()   # The page and the camera switch may both start it at once
        self._webcams: list[int] = []
        self._webcam_lock = threading.Lock()
        self.app = self._make_app()

    # ---------- Cameras ----------

    def scan_webcams(self) -> list[int]:
        with self._webcam_lock:
            in_use = int(self.config.source) if str(self.config.source).isdigit() else None
            found = list_webcams()
            if in_use is not None and in_use not in found and self.engine.status["state"] == "running":
                found.append(in_use)   # Busy because we are using it
            self._webcams = sorted(found)
            return self._webcams

    def sources(self) -> list[dict]:
        items = []
        for index in self._webcams:
            label = "Laptop / USB camera" if index == 0 else f"Camera {index + 1} (USB or DroidCam)"
            items.append({"source": str(index), "label": label, "kind": "webcam"})
        items.append({"source": "phone", "label": "Phone camera (scan QR code)", "kind": "phone"})
        for entry in self.config.saved_cameras:
            name, address = parse_saved_camera(entry)
            items.append({"source": address, "label": name, "kind": "network",
                          "detail": describe_source(address)})
        for screen in list_screens():
            source = "screen" if screen["number"] == 1 else f"screen:{screen['number']}"
            items.append({"source": source, "label": f"This computer's screen {screen['number']}",
                          "kind": "screen"})
        if DEMO_VIDEO.exists():
            items.append({"source": str(DEMO_VIDEO), "label": "Demo video", "kind": "file"})

        current = str(self.config.source)
        if not any(item["source"] == current for item in items):
            items.insert(0, {"source": current, "label": describe_source(current), "kind": "other"})
        return items

    def switch(self, source: str, save: bool = True) -> str | None:
        """Start monitoring `source`. Returns an error message, or None."""
        if not is_phone(source):
            if source == str(self.config.source):
                self.engine.stop()   # A webcam can only be opened once; free it before testing it
            problem = check_source(source)
            if problem:
                return problem
        else:
            self.start_phone_server()
        # Saved network cameras log under their own name; other cameras use the default name.
        saved = dict(parse_saved_camera(entry)[::-1] for entry in self.config.saved_cameras)
        if source in saved:
            self.config.camera_id = saved[source]
        elif self.config.camera_id in saved.values():
            self.config.camera_id = Config.camera_id
        self.engine.start(source)
        if save:
            save_config(self.config, self.config_path)
        return None

    # ---------- Phone page (HTTPS) ----------

    def phone_url(self) -> str:
        return f"https://{lan_ip()}:{self.config.phone_port}/phone?k={self.phone_key}"

    def start_phone_server(self) -> None:
        with self._phone_lock:
            if self.phone_server is None:
                self._start_phone_server()

    def _start_phone_server(self) -> None:
        from werkzeug.serving import make_server

        cert = ensure_certificate(lan_ip())
        app = Flask("phone", static_folder=None)
        feed = self.engine.browser_feed
        key = self.phone_key

        @app.get("/phone")
        def phone_page():
            if request.args.get("k") != key:
                abort(403)
            return send_from_directory(STATIC, "phone.html")

        @app.post("/phone/frame")
        def phone_frame():
            if request.args.get("k") != key:
                abort(403)
            if not feed.push_jpeg(request.get_data(), request.remote_addr or ""):
                abort(400)
            return jsonify(ok=True, monitoring=is_phone(self.config.source))

        self.phone_server = make_server("0.0.0.0", self.config.phone_port, app, threaded=True, ssl_context=cert)
        threading.Thread(target=self.phone_server.serve_forever, daemon=True).start()
        print(f"Phone camera page: {self.phone_url()}")

    # ---------- Web app ----------

    def _make_app(self) -> Flask:
        app = Flask("dashboard", static_folder=str(STATIC), static_url_path="/static")
        engine = self.engine

        @app.get("/")
        def index():
            return send_from_directory(STATIC, "index.html")

        @app.get("/stream.mjpg")
        def stream():
            def frames():
                number = 0
                while True:
                    number, data = engine.wait_frame(number, 1.0)
                    if data:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                               + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                            headers={"Cache-Control": "no-store"})

        @app.get("/api/frame.jpg")
        def frame():
            _, data = engine.wait_frame(-1, 0)
            return Response(data, mimetype="image/jpeg") if data else ("", 204)

        @app.get("/api/state")
        def state():
            return jsonify(engine.state())

        @app.get("/api/sources")
        def sources():
            if request.args.get("scan") == "1":
                self.scan_webcams()
            return jsonify(sources=self.sources(), current=str(self.config.source))

        @app.post("/api/source")
        def set_source():
            source = str((request.get_json(silent=True) or {}).get("source", "")).strip()
            if not source:
                return jsonify(error="No camera chosen."), 400
            problem = self.switch(source)
            if problem:
                return jsonify(error=problem), 400
            return jsonify(ok=True)

        @app.post("/api/cameras")
        def add_camera():
            data = request.get_json(silent=True) or {}
            address = str(data.get("address", "")).strip()
            name = str(data.get("name", "")).strip().replace("|", "/") or describe_source(address)
            if "://" not in address:
                return jsonify(error="The address must start with rtsp://, http:// or https://."), 400
            problem = check_source(address)
            if problem:
                return jsonify(error=problem + " Check that the camera is on the same network, and the "
                                               "address, username and password."), 400
            self.config.saved_cameras = [entry for entry in self.config.saved_cameras
                                         if parse_saved_camera(entry)[1] != address]
            self.config.saved_cameras.append(f"{name}|{address}")
            self.config.camera_id = name
            problem = self.switch(address)
            return (jsonify(error=problem), 400) if problem else jsonify(ok=True)

        @app.post("/api/stop")
        def stop():
            engine.stop()
            return jsonify(ok=True)

        @app.post("/api/start")
        def start():
            problem = self.switch(str(self.config.source), save=False)
            return (jsonify(error=problem), 400) if problem else jsonify(ok=True)

        @app.post("/api/record")
        def record():
            engine.set_recording(bool((request.get_json(silent=True) or {}).get("on")))
            return jsonify(ok=True)

        @app.post("/api/alerts-enabled")
        def alerts_enabled():
            engine.set_alerts_enabled(bool((request.get_json(silent=True) or {}).get("on")))
            return jsonify(ok=True)

        @app.post("/api/snapshot")
        def snapshot():
            name = engine.snapshot()
            return jsonify(file=name, url=f"/media/snapshots/{name}") if name else (jsonify(error="No picture yet."), 400)

        @app.post("/api/alerts/<int:alert_id>/dismiss")
        def dismiss(alert_id: int):
            return jsonify(ok=engine.dismiss(alert_id))

        @app.get("/api/alerts/<int:alert_id>.jpg")
        def alert_image(alert_id: int):
            data = engine.alert_image(alert_id)
            return Response(data, mimetype="image/jpeg") if data else abort(404)

        @app.get("/api/people/<int:track_id>.jpg")
        def person_photo(track_id: int):
            data = engine.thumbnail(track_id)
            return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"}) if data else abort(404)

        @app.get("/api/recordings")
        def recordings():
            folder = Path(self.config.recording_dir)
            files = sorted(folder.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True) if folder.exists() else []
            return jsonify(recordings=[{
                "file": path.name,
                "url": f"/media/recordings/{path.name}",
                "alert": path.name.startswith("alert_"),
                "size_mb": round(path.stat().st_size / 1e6, 1),
                "time": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            } for path in files[:50]])

        @app.get("/media/recordings/<path:name>")
        def recording_file(name: str):
            return send_from_directory(Path(self.config.recording_dir).resolve(), name, conditional=True)

        @app.get("/media/snapshots/<path:name>")
        def snapshot_file(name: str):
            return send_from_directory(Path(self.config.snapshot_dir).resolve(), name)

        @app.get("/api/phone")
        def phone_info():
            self.start_phone_server()
            url = self.phone_url()
            return jsonify(url=url, qr=qr_svg(url), connected=engine.browser_feed.connected)

        return app


def run_dashboard(config: Config, config_path: str, open_browser: bool = True) -> None:
    from werkzeug.serving import make_server

    dashboard = Dashboard(config, config_path)
    server = make_server("127.0.0.1", config.web_port, dashboard.app, threaded=True)
    url = f"http://localhost:{config.web_port}"
    print(f"Dashboard: {url}   (press Ctrl+C here to stop)")

    threading.Thread(target=dashboard.scan_webcams, daemon=True).start()
    if is_phone(config.source):
        dashboard.start_phone_server()
        dashboard.engine.start()
    else:
        problem = check_source(config.source)
        if problem:
            print(f"[info] {problem} Choose another camera in the dashboard.")
            dashboard.engine.status.update(state="error", message=problem)
        else:
            dashboard.engine.start()

    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    finally:
        dashboard.engine.stop()
        if dashboard.phone_server is not None:
            dashboard.phone_server.shutdown()
