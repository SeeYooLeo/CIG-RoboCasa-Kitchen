import pathlib
import shutil
import transformers


def main():
    if transformers.__version__ != "4.53.2":
        raise RuntimeError(
            "Install transformers==4.53.2 in the dedicated pi05 environment first"
        )
    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src/openpi/models_pytorch/transformers_replace"
    )
    target = pathlib.Path(transformers.__file__).parent
    for path in source.rglob("*.py"):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print("Installed the openpi transformers patch into", target)


if __name__ == "__main__":
    main()
