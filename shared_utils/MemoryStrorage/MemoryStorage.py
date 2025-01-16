from threading import Lock

class MemoryStorage:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls, *args, **kwargs)
                    cls._instance._data = {}
        return cls._instance

    def get(self, key):
        """Retrieve a single entity by its key."""
        return self._data.get(key)

    def set(self, key, value):
        """Store a single entity."""
        self._data[key] = value

    def delete(self, key):
        """Remove a single entity by its key."""
        self._data.pop(key, None)

    def get_many(self, keys):
        """Retrieve multiple entities by their keys."""
        return {key: self._data.get(key) for key in keys}

    def set_many(self, items):
        """Store multiple entities."""
        self._data.update(items)

    def delete_many(self, keys):
        """Remove multiple entities by their keys."""
        for key in keys:
            self._data.pop(key, None)

    def clear(self):
        """Clear all entities from storage."""
        self._data.clear()