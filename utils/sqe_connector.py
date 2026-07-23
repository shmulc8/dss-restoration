"""Scripta Qumranica Electronica (SQE), Leon Levy IAA, and Qimron QTD Integration Connector.

Provides structured data interfaces for:
1. SQE (Scripta Qumranica Electronica) TEI-XML/JSON editions (Prof. Eshbal Ratzon et al.).
2. Leon Levy Digital Library (IAA) multispectral fragment metadata.
3. Qimron QTD (Ben-Gurion University) alternative epigraphic reconstructions.
"""
import json
import urllib.request
from pathlib import Path

# SQE GitHub Repository Base
SQE_API_REPO = "https://raw.githubusercontent.com/Scripta-Qumranica-Electronica/SQE_Database/master"
IAA_BASE_URL = "https://www.deadseascrolls.org.il/explore-the-archive"

class SQEConnector:
    """Interface for Scripta Qumranica Electronica (SQE) digital editions."""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "sqe_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_sqe_schema(self):
        """Fetch SQE database schema definition."""
        cache_file = self.cache_dir / "sqe_schema.md"
        if cache_file.is_file():
            return cache_file.read_text(encoding="utf-8")
        
        url = f"{SQE_API_REPO}/README.md"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8")
            cache_file.write_text(content, encoding="utf-8")
            return content

    def get_editor_concordance_schema(self):
        """Returns standard schema for comparing DJD, Qimron QTD, and SQE editor readings."""
        return {
            "source_edition": ["DJD (Discoveries in Judaean Desert)", "Qimron QTD (BGU)", "SQE (Qumranica)"],
            "features": [
                "fragment_pixel_coordinates",
                "physical_line_width_mm",
                "multi_editor_reconstruction_consensus",
                "morphological_alignment"
            ]
        }

class LeonLevyConnector:
    """Interface for Leon Levy Digital Library (IAA) multispectral image metadata."""

    @staticmethod
    def get_fragment_image_url(scroll_name, plate_number):
        """Construct Leon Levy archive query URL."""
        return f"{IAA_BASE_URL}?search=scroll:{scroll_name}+plate:{plate_number}"

class QimronQTDConnector:
    """Interface for Elisha Qimron's Qumran Text Database (Ben-Gurion University)."""

    @staticmethod
    def normalize_qimron_reading(text):
        """Normalize Qimron orthographic transcription markers."""
        if not text:
            return ""
        # Remove editor brackets [ ], ( ), { }
        cleaned = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        return cleaned.strip()

if __name__ == "__main__":
    sqe = SQEConnector()
    schema = sqe.fetch_sqe_schema()
    print("SQE Connector Initialized!")
    print(f"SQE Schema Documentation Loaded ({len(schema)} bytes)")
    print("Editor Concordance Schema:", sqe.get_editor_concordance_schema())
