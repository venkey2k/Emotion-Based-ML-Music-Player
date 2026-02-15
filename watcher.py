"""
═══════════════════════════════════════════════════
  MOODIFY — Live Library Sync (File System Watcher)
  Uses OS-level events via watchdog — zero polling.
═══════════════════════════════════════════════════
"""

import os
import time
import threading
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2, TPE1


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
SONGS_DIR = 'songs'
IMAGES_DIR = 'images'
DEBOUNCE_SECONDS = 1.5   # wait for rapid changes to settle


# ──────────────────────────────────────────────
# SONG SCANNER — Thread-safe with dirty cache
# ──────────────────────────────────────────────
class SongScanner:
    """
    Scans the songs/ directory for MP3 files.
    Uses a dirty-flag cache — only rescans when
    the filesystem has actually changed.
    """

    def __init__(self, songs_dir=SONGS_DIR, images_dir=IMAGES_DIR):
        self.songs_dir = songs_dir
        self.images_dir = images_dir
        self._lock = threading.Lock()
        self._cache = []
        self._dirty = True    # start dirty → first call always scans

        # Ensure directories exist
        os.makedirs(self.songs_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    # ── Public API ──

    def mark_dirty(self):
        """Mark cache stale. Next get_songs() will rescan."""
        self._dirty = True

    def get_songs(self, force=False):
        """
        Return list of song dicts.
        Uses cache if clean; rescans if dirty or forced.
        """
        with self._lock:
            if not force and not self._dirty and self._cache is not None:
                return self._cache

            self._cache = self._scan()
            self._dirty = False
            return self._cache

    # ── Internal scanning ──

    def _scan(self):
        """Walk songs directory, extract metadata, return sorted list."""
        result = []

        for root, dirs, files in os.walk(self.songs_dir):
            dirs.sort()                   # consistent traversal order
            for filename in sorted(files):  # consistent file order
                if not filename.lower().endswith('.mp3'):
                    continue

                file_path = os.path.join(root, filename).replace('\\', '/')
                folder_name = os.path.basename(root).lower()

                # Mood from folder name
                if os.path.abspath(root) == os.path.abspath(self.songs_dir):
                    moods = ['neutral']   # root-level = neutral
                else:
                    moods = [folder_name]

                title = filename[:-4]
                artist = "Unknown Artist"
                cover_path = "images/default.jpg"

                # Extract ID3 metadata
                try:
                    audio = MP3(file_path, ID3=ID3)

                    if 'TIT2' in audio:
                        title = str(audio['TIT2'].text[0])
                    if 'TPE1' in audio:
                        artist = str(audio['TPE1'].text[0])

                    # Extract embedded cover art
                    for tag in audio.tags.values():
                        if isinstance(tag, APIC):
                            h = hashlib.md5(
                                file_path.encode('utf-8')
                            ).hexdigest()
                            ext = 'png' if 'png' in tag.mime else 'jpg'
                            cover_fname = f"cover_{h}.{ext}"
                            cover_fpath = os.path.join(
                                self.images_dir, cover_fname
                            )

                            if not os.path.exists(cover_fpath):
                                with open(cover_fpath, 'wb') as img_f:
                                    img_f.write(tag.data)

                            cover_path = f"{self.images_dir}/{cover_fname}"
                            break
                except Exception:
                    pass

                result.append({
                    "id": "",
                    "filename": file_path,
                    "title": title,
                    "description": artist,
                    "cover": cover_path,
                    "moods": moods,
                    "emoji_bg": "1.png"
                })

        # Sort for stable ordering across calls
        result.sort(key=lambda s: s['filename'].lower())

        # Assign stable IDs after sort
        for i, song in enumerate(result):
            song['id'] = f"song_{i}"

        print(f"📂 Scanned: {len(result)} songs found")
        return result


# ──────────────────────────────────────────────
# FILE EVENT HANDLER — Filters for MP3 only
# ──────────────────────────────────────────────
class _MP3EventHandler(FileSystemEventHandler):
    """Listens to OS file events, fires callback for MP3 changes."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def _is_mp3(self, path):
        _, ext = os.path.splitext(path)
        return ext.lower() == '.mp3'

    def on_created(self, event):
        if not event.is_directory and self._is_mp3(event.src_path):
            name = os.path.basename(event.src_path)
            print(f"  ➕ Added: {name}")
            self._callback('created', event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_mp3(event.src_path):
            name = os.path.basename(event.src_path)
            print(f"  🗑️  Removed: {name}")
            self._callback('deleted', event.src_path)
        elif event.is_directory:
            print(f"  📂 Folder removed: {event.src_path}")
            self._callback('deleted', event.src_path)

    def on_moved(self, event):
        src_mp3 = self._is_mp3(event.src_path)
        dst_mp3 = self._is_mp3(event.dest_path)
        if not event.is_directory and (src_mp3 or dst_mp3):
            old_name = os.path.basename(event.src_path)
            new_name = os.path.basename(event.dest_path)
            print(f"  📝 Moved: {old_name} → {new_name}")
            self._callback('moved', event.dest_path)

    def on_modified(self, event):
        if not event.is_directory and self._is_mp3(event.src_path):
            # Tags changed, cover updated, etc.
            self._callback('modified', event.src_path)


# ──────────────────────────────────────────────
# LIBRARY WATCHER — Debounced, push-based
# ──────────────────────────────────────────────
class LibraryWatcher:
    """
    Watches songs/ directory using OS-level file events.
    Debounces rapid changes (e.g., copying 10 files at once).
    Notifies frontend via callback when library changes.

    Usage:
        scanner = SongScanner()
        watcher = LibraryWatcher(scanner, notify_callback=my_func)
        watcher.start()
        ...
        watcher.stop()
    """

    def __init__(self, scanner, notify_callback=None,
                 debounce_seconds=DEBOUNCE_SECONDS):
        self.scanner = scanner
        self._notify = notify_callback
        self._debounce = debounce_seconds
        self._observer = None
        self._timer = None
        self._timer_lock = threading.Lock()
        self._running = False

    # ── Event handling ──

    def _on_file_event(self, event_type, file_path):
        """Called by _MP3EventHandler. Debounces then notifies."""
        self.scanner.mark_dirty()
        self._schedule_notify()

    def _schedule_notify(self):
        """
        Debounce: restart timer on every event.
        Only fires after no events for debounce_seconds.
        """
        with self._timer_lock:
            # Cancel previous pending notification
            if self._timer is not None:
                self._timer.cancel()

            # Schedule new one
            self._timer = threading.Timer(
                self._debounce, self._do_notify
            )
            self._timer.daemon = True
            self._timer.start()

    def _do_notify(self):
        """Actually rescan and notify frontend."""
        songs = self.scanner.get_songs(force=True)
        count = len(songs)
        print(f"🔄 Library updated → {count} songs (pushing to UI)")

        if self._notify:
            try:
                self._notify(songs)
            except Exception as e:
                print(f"⚠️  Notify error: {e}")

    # ── Start / Stop ──

    def start(self):
        """Start watching the songs directory."""
        if self._running:
            return

        watch_dir = self.scanner.songs_dir
        os.makedirs(watch_dir, exist_ok=True)

        handler = _MP3EventHandler(self._on_file_event)
        self._observer = Observer()
        self._observer.schedule(handler, watch_dir, recursive=True)
        self._observer.daemon = True
        self._observer.start()
        self._running = True

        abs_path = os.path.abspath(watch_dir)
        print(f"👁️  Watching: {abs_path}")
        print(f"   Debounce: {self._debounce}s")

    def stop(self):
        """Stop watching and clean up."""
        if not self._running:
            return

        # Cancel pending timer
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

        # Stop observer
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=3)
            self._observer = None

        self._running = False
        print("⏹️  Watcher stopped")

    @property
    def is_running(self):
        return self._running