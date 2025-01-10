from django.core.cache import cache

class CacheService:
    def __init__(self):
        self.cache = cache

    def get(self, key):
        """Get a single value from cache"""
        return self.cache.get(key)

    def set(self, key, value, timeout=None):
        """Set a single value in cache"""
        self.cache.set(key, value, timeout)

    def delete(self, key):
        """Delete a single key from cache"""
        self.cache.delete(key)

    def clear_all(self):
        """Clear all keys from cache"""
        self.cache.clear()

    def get_many(self, keys):
        """Get multiple values from cache"""
        return self.cache.get_many(keys)

    def set_many(self, key_value_dict, timeout=None):
        """Set multiple values in cache"""
        self.cache.set_many(key_value_dict, timeout)

    def delete_many(self, keys):
        """Delete multiple keys from cache"""
        self.cache.delete_many(keys)

    def get_session_key(self, session_id):
        """Generate a consistent key for session-related data"""
        return f"session_{session_id}" 