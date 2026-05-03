# openSMT

Modulares Python-Projekt fuer den Aufbau eines Kommunikationssystems fuer einen SMD-Bestueckungsautomaten.

## Merkmale

- Voll asynchroner Message-Broker fuer Text- und Binaernachrichten
- SCPI-kompatibles Textschema mit Query, Set, WORKING und finaler Antwort
- Callback-Registrierung getrennt nach Query, Set, Response und Working
- Terminal-Monitor mit Senden, Anzeigen und Playback aus Datei
- Einfacher Qt-Monitor fuer Mehrfachinstanzen
- Serial/G-Code Modul mit beliebig vielen individuell konfigurierbaren Schnittstellen
- JSON-Konfiguration mit verschachtelten Includes ueber $include
- Einheitliche `system.json`-Laufzeit fuer Boards, Achsen, Kameras, Nozzles und Feeder
- Native Qt-Control-GUI fuer Maschinenbedienung ueber HTTP ohne Browser
- Kamera-Thumbs, XY-Jogging, Nozzle-Jogging, Offset-Kalibrierung und Vakuumsteuerung in der Qt-Control-GUI
- Externe Feeder-Konfiguration mit gemeinsamer Basiskonfiguration und feeder-typspezifischen Tabs in der GUI

## SCPI Nachrichten

- Query: :XXX:YYY?
- Query-Antwort: :XXX:YYY? <wert>
- Set: :XXX:YYY <wert>
- Zwischenstatus bei langer Ausfuehrung: :XXX:YYY WORKING
- Abschlussbestaetigung fuer Set: :XXX:YYY? <wert>

Werte duerfen Integer, Fliesskomma oder Strings in Anfuehrungszeichen sein.

## Schnellstart

1. Debian ohne pip-Installation

   Dieses Projekt kann direkt aus dem Quellcode gestartet werden:

   export PYTHONPATH=src

2. Konfiguration pruefen und an Hardware anpassen

   - Datei: config/examples/system.json
   - Wichtig: boards.*.device auf die realen seriellen Geraete setzen
   - Optional: camera.web_host / camera.web_port, locations und feeder include anpassen
   - Feeder sind im Beispiel extern definiert ueber: config/examples/feeders.json

3. System starten

   python3 -m opensmt run --config config/examples/system.json

4. Qt-Control-GUI starten

   python3 -m opensmt control-gui --host 127.0.0.1 --port 8080

Hinweis:
Die neue Laufzeit startet HardwareDriver + PositionStore + LocationStore + CameraVision direkt aus system.json.
Ein separater Broker ist dafuer nicht noetig. Die fruehere Browser-Oberflaeche wird nicht mehr verwendet; `opensmt run`
stellt die HTTP-API und Kamera-Thumb-Endpunkte fuer die Qt-Control-GUI bereit.

### Legacy Tools (optional)

Die folgenden CLI-Tools sind weiterhin verfuegbar, aber nicht Teil des neuen Standard-Startablaufs:

1. Broker starten

   python3 -m opensmt broker --host 127.0.0.1 --port 8765

2. Terminal-Monitor starten

   python3 -m opensmt monitor --host 127.0.0.1 --port 8765 --name MON1

3. Qt-Monitor starten

   python3 -m opensmt monitor-gui --host 127.0.0.1 --port 8765 --name MON_GUI

4. Qt Control GUI starten (Fernbedienung ohne Browser)

   python3 -m opensmt control-gui --host MACHINE_IP --port 8080

## Aktueller Status

- Primare Bedienoberflaeche ist die native Qt-Control-GUI.
- `opensmt run` liefert dafuer die HTTP-API auf `camera.web_host:camera.web_port` sowie Kamera-Thumbnails ueber `/thumb/<NAME>`.
- Die GUI enthaelt aktuell Kamerapane, General-Purpose-Pane mit Tabs, XY-Jogging, Nozzle-Karten und Feeders-Verwaltung.
- Nozzle-Vakuum wird ueber Board `XY` geschaltet, Indizes `2..5`, Werte `0` oder `255`.
- Feeder werden extern in `config/examples/feeders.json` konfiguriert und ueber `$include` in `system.json` eingebunden.

## Qt Control GUI (Remote Operation)

`opensmt control-gui` is a native Qt application that controls the machine over the network
without a browser. It connects directly to the HTTP API of a running `opensmt run` instance.

Typical use case: run the core software on the machine-side Linux host, and operate it
from an office machine on the same network.

### Requirements on the office machine

- Python >= 3.11
- PySide6 (for Qt)
- No serial hardware required — only network access to the machine host

### One-time setup on the office machine

```bash
# Create a virtual environment
python3 -m venv ~/.venvs/opensmt-gui

# Activate it
source ~/.venvs/opensmt-gui/bin/activate

# Install directly from the git repository
pip install -U "git+ssh://YOUR_GIT_HOST/YOUR_REPO_PATH.git"
```

Replace `YOUR_GIT_HOST/YOUR_REPO_PATH` with the actual SSH path to this repository.

If you prefer a local wheel instead:

```bash
# On the development/machine host — build a wheel
python3 -m build --wheel

# Copy the .whl file to the office machine, then install it there
pip install -U dist/opensmt-0.1.0-py3-none-any.whl
```

### Launching the GUI

```bash
# Activate the virtual environment
source ~/.venvs/opensmt-gui/bin/activate

# Start the control GUI, pointing at the machine host
opensmt control-gui --host MACHINE_FLOOR_IP --port 8080
```

The GUI polls `/api/status` every 800 ms and sends all commands through the existing REST
API, so control latency on a wired LAN is negligible.

Current Qt-Control-GUI surfaces:

- Cameras pane with live thumbnail refresh from `/thumb/<camera>`
- XY jogging pane with step selection, homing and machine-position shortcuts
- Nozzle pane with per-nozzle home, Z up/down, rotation, park, standard-down, offset calibration and vacuum on/off
- General Purpose pane with tabs for Setup & Configuration, Production, Feeders and Diagnostics Log
- Feeders tab with overall feeder list plus per-type detail tabs for tray, auto, push/pull, vibration, label and tube feeders

### Updating after code changes

```bash
source ~/.venvs/opensmt-gui/bin/activate
pip install -U "git+ssh://YOUR_GIT_HOST/YOUR_REPO_PATH.git"
```

### Optional: desktop launcher (Linux)

Create `~/.local/share/applications/opensmt-control.desktop`:

```ini
[Desktop Entry]
Name=openSMT Control
Exec=/bin/bash -c "source ~/.venvs/opensmt-gui/bin/activate && opensmt control-gui --host MACHINE_FLOOR_IP --port 8080"
Icon=utilities-system-monitor
Terminal=false
Type=Application
Categories=Utility;
```

Then run `update-desktop-database ~/.local/share/applications/` to register it.

### Machine-side prerequisite

Ensure `camera.web_host` is set to `0.0.0.0` (not `127.0.0.1`) in your `system.json`
so the HTTP API is reachable from the network:

```json
"camera": {
  "web_host": "0.0.0.0",
  "web_port": 8080,
  ...
}
```

### Feeder configuration

Feeders are defined outside the main machine config and merged in via `$include`.

Example in `config/examples/system.json`:

```json
{
   "$include": "feeders.json",
   ...
}
```

Each feeder currently shares these common fields:

- `feeder_id`: unique hexadecimal identifier with at least 16 digits
- `pick_location`: object with `x` / `y`
- `pick_height`: Z value used for pickup
- `manufacturer_part_number`: currently stored as the loaded part reference
- `feeder_type`: one of `tray_feeder`, `auto_feeder`, `push_pull_feeder`, `vibration_feeder`, `label_feeder`, `tube_feeder`

---

## Debian Pakete

Wenn Debian die Umgebung als externally managed markiert, werden Abhaengigkeiten ueber apt installiert.

Beispiel:

sudo apt install python3-serial python3-serial-asyncio

Fuer Qt wird zusaetzlich ein passendes PySide6-Paket aus den Debian-Repositories benoetigt.

## Playback-Datei fuer Monitor

- Kommentar: Zeile beginnt mit #
- Wartezeit: SLEEP <sekunden>
- Binaerdaten: BIN <hex bytes>
- Sonst: SCPI-Textnachricht

Beispiel in config/examples/playback.txt.

## Aktuelle Architektur und API

Dieser Abschnitt beschreibt den aktuellen Stand der produktiven Laufzeit (Qt-Control-GUI + HTTP API).
Die fruehere browserbasierte Bedienoberflaeche ist nicht mehr Teil des aktiven Bedienpfads.

### Architekturueberblick

- Prozessstart ueber `python3 -m opensmt run --config config/examples/system.json`
- Eine Laufzeit mit:
   - Board-Verbindungen (`XY`, `AB`, `CD`)
   - `HardwareDriver` (Bewegung + IO)
   - `PositionStore` / `LocationStore`
   - `CameraVisionModule` als HTTP API Schicht
- Bedienung primar ueber `opensmt control-gui`

### Konfiguration (heute relevant)

- `boards`: serielle Board-Endpunkte
- `driver`: Achs-Mapping, Geschwindigkeiten, Homing-Gruppen
- `locations`: Park/Dispose/Fiducials/Nozzle-Change/Calibration-Spot
- `camera`:
   - Kameraquellen
   - Nozzle-Geometrie und Offsets
   - Lichtdefinitionen
   - Web-Host/Port fuer API
- `feeders` (extern eingebunden ueber `$include`, Beispiel: `config/examples/feeders.json`)

### HTTP API (operator path)

Wichtige Endpunkte fuer die Qt-Control-GUI:

- `GET /api/status`
   - Gesamtstatus fuer Positionen, Kameras, Nozzles, Feeder
   - Primarer Poll-Endpunkt der GUI
- `GET /thumb/{name}`
   - JPEG-Thumbnail pro Kamera
- `POST /api/coord/jog`
- `POST /api/coord/home`
- `POST /api/coord/home-xy`
- `POST /api/coord/park`
- `POST /api/coord/dispose`
- `POST /api/coord/homing-fiducial-main`
- `POST /api/coord/secondary-fiducial`
- `POST /api/coord/nozzle-change`
- `POST /api/coord/calibration-spot`
- `POST /api/coord/set-home-here`
- `POST /api/coord/set-calibration-spot-here`
- `POST /api/head/nozzle/{name}/home`
- `POST /api/head/nozzle/{name}/move`
- `POST /api/head/nozzle/{name}/move-absolute`
- `POST /api/head/nozzle/{name}/move-standard-down`
- `POST /api/head/nozzle/{name}/rotate`
- `POST /api/head/nozzle/{name}/park`
- `POST /api/head/nozzle/{name}/vacuum`
- `POST /api/nozzle/{name}/move-to-camera`
- `POST /api/nozzle/{name}/move-to-bottom-camera`
- `POST /api/nozzle/{name}/move-camera-here`
- `POST /api/nozzle/{name}/calculate-offset-top`

### Wichtige Verhaltensregeln

- Nozzle-Vakuum wird ueber Board `XY` gefahren:
   - `N1 -> index 2`
   - `N2 -> index 3`
   - `N3 -> index 4`
   - `N4 -> index 5`
   - Werte: `255` (on), `0` (off)
- IO-Setzung nutzt `M106 P<index> S<value>` auf den Boards.

### API Beispiel

Vakuum fuer `N1` einschalten:

```bash
curl -sS -X POST \
   -H "Content-Type: application/json" \
   -d '{"on": true}' \
   http://127.0.0.1:8080/api/head/nozzle/N1/vacuum
```

Status abrufen:

```bash
curl -sS http://127.0.0.1:8080/api/status
```

### Legacy-Hinweis

Das Projekt enthaelt weiterhin Legacy-Module und SCPI-/Broker-Werkzeuge fuer Entwicklungs- und
Migrationszwecke. Fuer den aktuellen Produktionsfluss gilt jedoch:

- Steuerung: Qt-Control-GUI
- Transport: HTTP API
- Konfigurationsquelle: `system.json` mit Includes

---

## Lizenz

Public Domain (Unlicense), siehe LICENSE.
