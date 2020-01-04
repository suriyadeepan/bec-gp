"""Caching Mechanism"""


def in_cache(key):
  return False


def cache(key, content=None):
  if content is None:
    return None
