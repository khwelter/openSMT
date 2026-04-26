from __future__ import annotations

import argparse
import asyncio

from opensmt.messaging.broker import run_broker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opensmt", description="openSMT communication runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    p_broker = sub.add_parser("broker", help="Start message broker")
    p_broker.add_argument("--host", default="127.0.0.1")
    p_broker.add_argument("--port", default=8765, type=int)

    p_monitor = sub.add_parser("monitor", help="Start terminal monitor")
    p_monitor.add_argument("--host", default="127.0.0.1")
    p_monitor.add_argument("--port", default=8765, type=int)
    p_monitor.add_argument("--name", default="MONITOR")
    p_monitor.add_argument("--playback", default=None)

    p_monitor_gui = sub.add_parser("monitor-gui", help="Start Qt monitor")
    p_monitor_gui.add_argument("--host", default="127.0.0.1")
    p_monitor_gui.add_argument("--port", default=8765, type=int)
    p_monitor_gui.add_argument("--name", default="MONITOR_QT")

    p_run = sub.add_parser("run", help="Run modules from configuration")
    p_run.add_argument("--config", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "broker":
        asyncio.run(run_broker(args.host, args.port))
        return

    if args.command == "monitor":
        from opensmt.monitor import run_terminal_monitor

        asyncio.run(run_terminal_monitor(args.host, args.port, args.name, args.playback))
        return

    if args.command == "monitor-gui":
        from opensmt.monitor import run_qt_monitor

        run_qt_monitor(args.host, args.port, args.name)
        return

    if args.command == "run":
        from opensmt.app import run_from_config

        asyncio.run(run_from_config(args.config))
        return


if __name__ == "__main__":
    main()
