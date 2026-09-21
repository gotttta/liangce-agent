"""Memory vocabulary. Status and provenance are independent of memory kind."""
from enum import StrEnum


class MemoryKind(StrEnum):
    SEMANTIC = 'semantic'
    PROCEDURAL = 'procedural'
    EPISODIC = 'episodic'


STATUSES = {'active', 'hypothesis', 'superseded', 'revoked'}
