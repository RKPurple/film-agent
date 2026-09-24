"""
cinemagent.chroma_client -- the one place a Chroma client is created, with
product telemetry fully off.

anonymized_telemetry=False alone stops Chroma sending anything, but
chromadb 0.6.3 still calls posthog.capture() positionally, which the
installed posthog (7.x) rejects -- so every client start and query logged
"Failed to send telemetry event ...". NoTelemetry replaces Chroma's posthog
client entirely: capture() is a no-op, so there's no outbound call and no
error log. Shared by the online retrieval index and pipeline/load_chroma.py.
"""

import chromadb
from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient
from overrides import override  # chromadb dependency; its base classes enforce it


class NoTelemetry(ProductTelemetryClient):
    @override
    def capture(self, event):
        pass


_NO_TELEMETRY = f"{__name__}.NoTelemetry"
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    chroma_product_telemetry_impl=_NO_TELEMETRY,
    chroma_telemetry_impl=_NO_TELEMETRY,
)


def persistent_client(path):
    return chromadb.PersistentClient(path=str(path), settings=CHROMA_SETTINGS)
