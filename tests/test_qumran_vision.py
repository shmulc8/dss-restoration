import io
import tarfile
from pathlib import Path

import torch
from PIL import Image

from experiments.qumran_vision.audit_sqe_registration import query, target_id_sha256
from experiments.qumran_vision.download_data import safe_extract_images
from experiments.qumran_vision.modern_pilot import (
    BinaryPixelCNN,
    best_orientation,
    fit_image,
    inverse_transform,
    transform,
)


def test_modern_qumran_preprocessing_and_model(tmp_path: Path) -> None:
    source = tmp_path / "letter.png"
    Image.new("L", (8, 16), 255).save(source)
    image = fit_image(source)
    assert image.shape == (1, 32, 32)
    assert image[:, :, :8].sum() == 0
    output = BinaryPixelCNN(channels=4, layers=1)(torch.zeros(2, 1, 32, 32))
    assert output.shape == (2, 1, 32, 32)
    mask = torch.zeros(1, 32, 32)
    mask[:, :16] = 1
    code = best_orientation(mask)
    assert torch.equal(inverse_transform(transform(mask, code), code), mask)


def test_qumran_archive_extraction_ignores_pickle_and_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "images.tar.gz"
    image_bytes = io.BytesIO()
    Image.new("L", (2, 2), 255).save(image_bytes, format="PNG")
    with tarfile.open(archive, "w:gz") as bundle:
        for name, payload in (
            ("release/train/letters/one.png", image_bytes.getvalue()),
            ("release/train/cache.pkl", b"untrusted"),
            ("release/../escape.png", image_bytes.getvalue()),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    destination = tmp_path / "out"
    assert safe_extract_images(archive, destination, "release") == 1
    assert (destination / "release/train/letters/one.png").is_file()
    assert not (destination / "escape.png").exists()


def test_registration_query_uses_numeric_ids_and_sign_rois() -> None:
    ids = [7, 42]
    sql = query(ids)
    assert "INSERT INTO target_word VALUES (7),(42)" in sql
    assert "sign_interpretation_roi" in sql
    assert len(target_id_sha256(ids)) == 64
