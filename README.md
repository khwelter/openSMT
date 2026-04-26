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

## Lizenz

Public Domain (Unlicense), siehe LICENSE.
