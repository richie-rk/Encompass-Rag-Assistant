"""Sidebar-driven crawler for ICE MT Developer Connect (ReadMe-hosted) docs.

Pipeline: enumerate frontier from ssr-props sidebar -> fetch each page once ->
extract normalized record from ssr-props -> emit JSONL.
"""

__version__ = "0.1.0"
