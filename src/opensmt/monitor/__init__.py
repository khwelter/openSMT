def run_terminal_monitor(*args, **kwargs):
	from .cli_monitor import run_terminal_monitor as _run_terminal_monitor

	return _run_terminal_monitor(*args, **kwargs)


def run_qt_monitor(*args, **kwargs):
	from .qt_monitor import run_qt_monitor as _run_qt_monitor

	return _run_qt_monitor(*args, **kwargs)


def run_qt_control(*args, **kwargs):
	from .qt_control import run_qt_control as _run_qt_control

	return _run_qt_control(*args, **kwargs)


__all__ = ["run_terminal_monitor", "run_qt_monitor", "run_qt_control"]
