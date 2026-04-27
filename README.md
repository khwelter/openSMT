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

2. Broker starten

   python3 -m opensmt broker --host 127.0.0.1 --port 8765

3. Terminal-Monitor starten

   python3 -m opensmt monitor --host 127.0.0.1 --port 8765 --name MON1

4. Qt-Monitor starten

   python3 -m opensmt monitor-gui --host 127.0.0.1 --port 8765 --name MON_GUI

5. Module aus Konfiguration starten

   python3 -m opensmt run --config config/examples/system.json

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

## SCPI Command Reference

This section describes all internal SCPI commands that are exchanged between modules on the message bus.
Each command uses the format `:MODULE:VERB[:SUBVERB]` with an optional value.

Legend:
- **Sender** — module that initiates the command
- **Receiver** — module that handles the command
- `?` suffix — this is a query; the receiver replies with a response
- `WORKING` — intermediate status sent during long-running operations
- `DONE` — final confirmation sent when an operation completes

---

### COORD module (coordinate_system)

The COORD module manages the machine coordinate cache, translates relative moves to absolute,
and dispatches motion commands to GCODE.

#### Queries answered by COORD

| Command | Response | Description |
|---------|----------|-------------|
| `:COORD:ABS:<axis>?` | `<value>` | Absolute position of axis (X, Y, Z1–Z4, R1–R4) |
| `:COORD:POS:<axis>?` | `<value>` | Same as ABS |
| `:COORD:PARK?` | `<x> <y>` | Configured Park position |
| `:COORD:DISPOSE?` | `<x> <y>` | Configured Dispose position |
| `:COORD:HOMINGFIDUCIALMAIN?` | `<x> <y>` | Homing Fiducial Main position |
| `:COORD:SECONDARYFIDUCIAL?` | `<x> <y>` | Secondary Fiducial position |
| `:COORD:NOZZLECHANGE?` | `<x> <y>` | Nozzle Change position |
| `:COORD:CALIBRATIONSPOT?` | `<x> <y>` | Calibration Spot position |

#### SET commands to COORD

| Command | Value | Description |
|---------|-------|-------------|
| `:COORD:ABS:<axis>` | `<position>` | Move axis to absolute position |
| `:COORD:ABS:XY` | `<x> <y>` | Move X and Y simultaneously to absolute position |
| `:COORD:REL:<axis>` | `<delta>` | Move axis by relative delta (added to cached position) |

#### ACTION commands to COORD

| Command | Description |
|---------|-------------|
| `:COORD:HOME` | Home all configured axis groups (XY, Z1Z2, Z3Z4) sequentially |
| `:COORD:HOME:XY` | Home X and Y axes simultaneously |
| `:COORD:HOME:Z1Z2` | Home Z1 and Z2 axes simultaneously |
| `:COORD:HOME:Z3Z4` | Home Z3 and Z4 axes simultaneously |
| `:COORD:PARK` | Move to Park position |
| `:COORD:DISPOSE` | Move to Dispose position |
| `:COORD:HOMINGFIDUCIALMAIN` | Move to Homing Fiducial Main position |
| `:COORD:SECONDARYFIDUCIAL` | Move to Secondary Fiducial position |
| `:COORD:NOZZLECHANGE` | Move to Nozzle Change position |
| `:COORD:CALIBRATIONSPOT` | Move to Calibration Spot position |

All action commands send `:COORD:... WORKING` during execution and respond with `DONE` or `TIMEOUT:...` on completion.

---

### GCODE module (serial_gcode)

The GCODE module translates SCPI commands to GCode and sends them over serial ports.
Each axis is mapped to a GCode axis letter on a specific serial port.

Axis–port–GCode mapping (configurable, default):

| Logical Axis | Port | GCode Axis |
|---|---|---|
| X | XY | X |
| Y | XY | Y |
| Z1, Z2 | AB | X, Y |
| R1, R2 | AB | A, B |
| Z3, Z4 | CD | X, Y |
| R3, R4 | CD | A, B |

#### Queries answered by GCODE

| Command | Response | Description |
|---------|----------|-------------|
| `:GCODE:STATUS?` | `"online"` | Module status |
| `:GCODE:SPEEDFACTOR?` | `<percent>` | Current speed factor (0–100) |
| `:GCODE:LASTRX?` | `"PORT=<text>;..."` | Last received line from each serial port |
| `:GCODE:DIGOUT:<n>?` | `0` or `1` | Current digital output state (n = 0–47) |
| `:GCODE:ANOUT:<n>?` | `<0–65535>` | Current analog output value (n = 0–47) |

#### SET commands to GCODE

| Command | Value | GCode emitted | Description |
|---------|-------|--------------|-------------|
| `:GCODE:POS:<axis>` | `<position>` | `G0 <ax><pos> F<velo>` then `M400` | Move single axis |
| `:GCODE:POS:XY` | `<x> <y>` | `G0 X<x> Y<y> F<velo>` then `M400` | Move X and Y simultaneously |
| `:GCODE:POS:Z1R1` | `<z> <r>` | `G0 X<z> A<r> F<velo>` then `M400` | Move Z1 and R1 together (AB port) |
| `:GCODE:POS:Z2R2` | `<z> <r>` | `G0 Y<z> B<r> F<velo>` then `M400` | Move Z2 and R2 together (AB port) |
| `:GCODE:VELO:<axis>` | `<mm_per_min>` | — | Set runtime velocity for axis |
| `:GCODE:SPEEDFACTOR` | `<0–100>` | — | Set global speed factor as percentage |
| `:GCODE:DIGOUT:<n>` | `0` or `1` | `M106 D<local> S<val>` | Set digital output (n = 0–47) |
| `:GCODE:ANOUT:<n>` | `<0–65535>` | `M106 D<local> S<val>` | Set analog output / PWM (n = 0–47) |

#### ACTION commands to GCODE

| Command | GCode emitted | Description |
|---------|--------------|-------------|
| `:GCODE:HOME:XY` | `M210 X<vhx> Y<vhy>` then `G28 X Y` | Home X and Y simultaneously |
| `:GCODE:HOME:Z1Z2` | `M210 X<vhz> Y<vhz>` then `G28 X Y` | Home Z1 and Z2 (AB port) |
| `:GCODE:HOME:Z3Z4` | `M210 X<vhz> Y<vhz>` then `G28 X Y` | Home Z3 and Z4 (CD port) |

`M210` sets homing velocities from `homing_velocity` config. `M400` waits for motion to complete.

#### Responses emitted by GCODE (position updates)

After every completed move or home, the GCODE module sends position reports:

| Command | Value | Description |
|---------|-------|-------------|
| `:GCODE:POS:<axis>` | `<position>` | Updated position for each moved axis |

These responses are consumed by the COORD module to keep its position cache current.

#### Raw serial access

| Command | Value | Description |
|---------|-------|-------------|
| `:SERIAL:<port>:TX` | `<gcode string>` | Send raw GCode line directly to a port |
| `:SERIAL:<port>:STATUS?` | `0` or `1` | Port connected state |
| `:SERIAL:<port>:LASTRX?` | `<text>` | Last received line from port |
| `:SERIAL:<port>:RX` | `<text>` | Broadcast of every incoming line from port |

---

### CAMERA module (camera_vision)

The CAMERA module provides the web dashboard, MJPEG camera streams, and light/pipeline control.

#### Queries answered by CAMERA

| Command | Response | Description |
|---------|----------|-------------|
| `:CAMERA:STATUS?` | `{"CAM": "online"|"offline", ...}` | Status of all cameras |
| `:CAMERA:<cam>:STATUS?` | `"online"` or `"offline"` | Status of one camera |
| `:CAMERA:<cam>:LIGHT:<light>?` | `<0–65535>` | Current light brightness |
| `:CAMERA:<cam>:PIPELINE?` | `<name>` | Active vision pipeline name |

#### SET commands to CAMERA

| Command | Value | Description |
|---------|-------|-------------|
| `:CAMERA:<cam>:LIGHT:<light>` | `<0–65535>` \| `ON` \| `OFF` | Set light brightness (standard, spec1, spec2) |
| `:CAMERA:<cam>:PIPELINE:<name>` | `<json params>` | Activate vision pipeline with optional JSON parameters |

When a light is set, CAMERA forwards the value to the appropriate GCODE ANOUT channel.

#### Outbound commands from CAMERA (to GCODE / COORD)

| Command | Target | Description |
|---------|--------|-------------|
| `:GCODE:ANOUT:<n>` | GCODE | Light brightness control |
| `:COORD:REL:X` | COORD | XY jog from web dashboard |
| `:COORD:REL:Y` | COORD | XY jog from web dashboard |
| `:COORD:HOME` | COORD | Home all axes (web button) |
| `:COORD:HOME:XY` | COORD | Home X and Y only (web button) |
| `:COORD:PARK` | COORD | Move to Park (web button) |
| `:COORD:DISPOSE` | COORD | Move to Dispose (web button) |
| `:COORD:HOMINGFIDUCIALMAIN` | COORD | Move to Homing Fiducial Main (web button) |
| `:COORD:SECONDARYFIDUCIAL` | COORD | Move to Secondary Fiducial (web button) |
| `:COORD:NOZZLECHANGE` | COORD | Move to Nozzle Change (web button) |
| `:COORD:CALIBRATIONSPOT` | COORD | Move to Calibration Spot (web button) |
| `:COORD:ABS:<axis>?` | COORD | Periodic coordinate polling (WebSocket) |

---

### Command Flow Examples

#### Moving to Homing Fiducial Main (simultaneous XY)

```
CAMERA → :COORD:HOMINGFIDUCIALMAIN       (ACTION)
COORD  → :COORD:HOMINGFIDUCIALMAIN WORKING
COORD  → :GCODE:POS:XY 262.4 120.5      (SET, simultaneous G0)
GCODE  → G0 X262.4 Y120.5 F25000        (serial)
GCODE  → M400                            (serial, wait)
GCODE  → :GCODE:POS:X 262.4             (response → COORD cache)
GCODE  → :GCODE:POS:Y 120.5             (response → COORD cache)
COORD  → :COORD:HOMINGFIDUCIALMAIN? DONE 262.4 120.5
```

#### Homing X and Y (simultaneous, with velocity preset)

```
CAMERA → :COORD:HOME:XY                 (ACTION)
COORD  → :COORD:HOME:XY WORKING
COORD  → :GCODE:HOME:XY                 (ACTION)
GCODE  → M210 X5000 Y5000               (serial, set homing velocities)
GCODE  → G28 X Y                        (serial, simultaneous home)
GCODE  → :GCODE:POS:X 0                 (response → COORD cache)
GCODE  → :GCODE:POS:Y 0                 (response → COORD cache)
COORD  → :COORD:HOME:XY? DONE
```

#### Setting a light

```
WEB UI → POST /api/camera/TOP/light/standard  {value: 32768}
CAMERA → :GCODE:ANOUT:2 32768           (SET)
GCODE  → M106 D2 S32768                 (serial)
GCODE  → :GCODE:ANOUT:2? 32768          (response)
```

---

## Lizenz

Public Domain (Unlicense), siehe LICENSE.
