import hashlib
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent
manifest = json.loads((root / "weights_manifest.json").read_text())
for entry in manifest["files"]:
    path = root / entry["path"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {entry['path']}")
    print("OK", entry["path"], flush=True)
print(f"Verified {len(manifest['files'])} files")
