import glob
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.rag_engine import build_vector_store

DATA_DIR = ROOT / "data" / "cs"
PERSIST_DIR = ROOT / "vector_db"
DOMAIN = "cs"
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {"question", "answer", "topic", "difficulty", "type"}


def _validate_record(obj: dict[str, Any], source: str, line_no: int) -> tuple[bool, str]:
    missing = [field for field in REQUIRED_FIELDS if field not in obj]
    if missing:
        return False, f"{source}:{line_no} missing fields: {', '.join(missing)}"

    if not str(obj.get("question", "")).strip():
        return False, f"{source}:{line_no} empty question"
    if not str(obj.get("answer", "")).strip():
        return False, f"{source}:{line_no} empty answer"

    difficulty = str(obj.get("difficulty", "")).strip().lower()
    if difficulty not in ALLOWED_DIFFICULTIES:
        return False, f"{source}:{line_no} invalid difficulty: {difficulty}"

    return True, ""


def load_docs() -> list[dict]:
    docs = []
    errors: list[str] = []
    for path in glob.glob(str(DATA_DIR / "qa_*.jsonl")):
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{Path(path).name}:{line_no} invalid json: {exc}")
                    continue

                ok, err = _validate_record(obj, Path(path).name, line_no)
                if not ok:
                    errors.append(err)
                    continue

                content = f"{obj.get('question','')}\n{obj.get('answer','')}"
                metadata = {
                    "topic": str(obj.get("topic", "")).strip(),
                    "difficulty": str(obj.get("difficulty", "")).strip().lower(),
                    "type": str(obj.get("type", "")).strip().lower(),
                    "source": Path(path).name,
                }
                docs.append({"content": content, "metadata": metadata})

    if errors:
        err_preview = "\n".join(errors[:20])
        more = "" if len(errors) <= 20 else f"\n... and {len(errors) - 20} more errors"
        raise SystemExit(f"Data validation failed with {len(errors)} issue(s):\n{err_preview}{more}")

    return docs


def _on_rm_error(func, path, exc_info):
    # Handle read-only files on Windows when removing existing vector store
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def main():
    docs = load_docs()
    if not docs:
        raise SystemExit("No docs found in data/cs")
    target_dir = PERSIST_DIR / DOMAIN
    if target_dir.exists():
        shutil.rmtree(target_dir, onerror=_on_rm_error)
    db_path = build_vector_store(
        docs=docs,
        domain=DOMAIN,
        persist_dir=str(PERSIST_DIR),
        chunk_size=800,
        chunk_overlap=50,
    )
    print(f"Vector store built at: {db_path}")


if __name__ == "__main__":
    main()
