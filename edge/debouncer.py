import time
from typing import Dict, Tuple

class Debouncer:
    def __init__(self, cooldown_seconds: float = 30.0):
        self.cooldown = cooldown_seconds
        self.seen: Dict[Tuple[str, str], float] = {}

    def should_emit(self, camera_id: str, plate: str) -> bool:
        now = time.time()
        key = (camera_id, plate)
        last_seen = self.seen.get(key)
        
        # Agar cooldown window ke andar dobara aayi, to emit mat karo
        if last_seen is not None and (now - last_seen) < self.cooldown:
            return False
            
        # Nayi sighting ya cooldown khatam hone par timestamp update karo
        self.seen[key] = now
        self._prune(now)
        return True

    def _prune(self, now: float) -> None:
        # Memory leak se bachne ke liye 2x cooldown se purani entries delete karo
        expired = [k for k, t in self.seen.items() if (now - t) > (self.cooldown * 2)]
        for k in expired:
            del self.seen[k]