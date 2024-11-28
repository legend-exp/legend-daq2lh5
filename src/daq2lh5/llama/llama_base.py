"""
General utilities for llamaDAQ data decoding
"""
from __future__ import annotations

import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

#build a unique flat identifier for fadc and channel together
def join_fadcid_chid(fadcid: int, chid: int) -> int:
    return (fadcid << 4) + chid

