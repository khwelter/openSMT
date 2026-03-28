# =============================================================================
# utils/file_watcher.py
#
# Cross-platform file change detection using the watchdog library.
# Emits Qt signals when monitored files are modified, enabling hot-reload
# functionality for configuration files.
# =============================================================================

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent


class _FileChangeHandler(FileSystemEventHandler):
    """
    Internal handler that bridges watchdog events to a callback function.
    Filters events to only react to modifications of the specific target file.
    """

    def __init__(self, target_path: Path, callback: callable) -> None:
        """
        Initialize the file change handler.

        Args:
            target_path: Absolute path to the file being monitored.
            callback: Function to call when the target file is modified.
        """
        super().__init__()
        self._target_path = target_path.resolve()
        self._callback = callback

    def on_modified(self, event: FileModifiedEvent) -> None:
        """
        Called when a file modification is detected in the watched directory.

        Args:
            event: The file system event containing modification details.
        """
        # Ignore directory events
        if event.is_directory:
            return

        # Check if the modified file matches our target
        modified_path = Path(event.src_path).resolve()
        if modified_path == self._target_path:
            self._callback(modified_path)


class FileWatcher(QObject):
    """
    Monitors a file for changes and emits a Qt signal when modifications occur.
    
    This class wraps the watchdog library to provide Qt-compatible file
    monitoring. It watches the parent directory of the target file and filters
    events to only respond to changes in the specific file.

    Signals:
        file_changed: Emitted when the monitored file is modified.
                      Carries the absolute path to the changed file as a string.

    Example:
        watcher = FileWatcher()
        watcher.file_changed.connect(my_reload_function)
        watcher.start("/path/to/config.json")
    """

    file_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """
        Initialize the FileWatcher.

        Args:
            parent: Optional Qt parent object for memory management.
        """
        super().__init__(parent)
        self._observer: Optional[Observer] = None
        self._watched_path: Optional[Path] = None

    def start(self, file_path: str | Path) -> bool:
        """
        Begin monitoring the specified file for changes.

        If already monitoring a file, the previous watch is stopped first.

        Args:
            file_path: Path to the file to monitor.

        Returns:
            True if monitoring started successfully, False otherwise.
        """
        # Stop any existing watch
        self.stop()

        path = Path(file_path).resolve()

        # Verify the file exists
        if not path.is_file():
            return False

        self._watched_path = path

        # Create the event handler with a callback that emits our signal
        handler = _FileChangeHandler(path, self._on_file_changed)

        # Watch the parent directory (watchdog requirement)
        self._observer = Observer()
        self._observer.schedule(handler, str(path.parent), recursive=False)
        self._observer.start()

        return True

    def stop(self) -> None:
        """
        Stop monitoring the current file.

        Safe to call even if no file is being monitored.
        """
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
        self._watched_path = None

    def _on_file_changed(self, path: Path) -> None:
        """
        Internal callback triggered by the watchdog handler.

        Args:
            path: Path to the modified file.
        """
        self.file_changed.emit(str(path))

    @property
    def is_watching(self) -> bool:
        """Check if the watcher is currently monitoring a file."""
        return self._observer is not None and self._observer.is_alive()

    @property
    def watched_path(self) -> Optional[Path]:
        """Get the path currently being monitored, or None if not watching."""
        return self._watched_path

    def __del__(self) -> None:
        """Ensure the observer is stopped when the object is destroyed."""
        self.stop()
