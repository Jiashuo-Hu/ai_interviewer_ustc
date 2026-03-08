import json
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DASHSCOPE_API_KEY

VDB_DIR = ROOT / "vector_db" / "cs"
EVAL_FILE = ROOT / "data" / "cs" / "eval_queries.jsonl"


def load_eval_items(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("query"):
                raise ValueError(f"{path.name}:{line_no} missing query")
            items.append(obj)
    return items


def evaluate(top_k: int = 6) -> None:
    if not VDB_DIR.exists():
        raise SystemExit(f"Vector DB not found: {VDB_DIR}")
    if not EVAL_FILE.exists():
        raise SystemExit(f"Eval file not found: {EVAL_FILE}")
    if not DASHSCOPE_API_KEY:
        raise SystemExit("DASHSCOPE_API_KEY is empty")

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    db = Chroma(persist_directory=str(VDB_DIR), embedding_function=embeddings)
    eval_items = load_eval_items(EVAL_FILE)

    topic_hit = 0
    type_hit = 0
    both_hit = 0

    print(f"Loaded {len(eval_items)} eval queries, top_k={top_k}")
    for idx, item in enumerate(eval_items, start=1):
        query = item["query"]
        expected_topic = str(item.get("expected_topic", "")).strip().lower()
        expected_type = str(item.get("expected_type", "")).strip().lower()

        docs = db.similarity_search(query, k=top_k)
        meta_list = [doc.metadata or {} for doc in docs]

        hit_topic = any(str(m.get("topic", "")).lower() == expected_topic for m in meta_list) if expected_topic else True
        hit_type = any(str(m.get("type", "")).lower() == expected_type for m in meta_list) if expected_type else True

        topic_hit += int(hit_topic)
        type_hit += int(hit_type)
        both_hit += int(hit_topic and hit_type)

        status = "OK" if (hit_topic and hit_type) else "MISS"
        print(f"[{idx:02d}] {status} | query={query}")

    total = len(eval_items)
    print("\n=== Recall Summary ===")
    print(f"Topic Recall@{top_k}: {topic_hit}/{total} = {topic_hit/total:.2%}")
    print(f"Type  Recall@{top_k}: {type_hit}/{total} = {type_hit/total:.2%}")
    print(f"Joint Recall@{top_k}: {both_hit}/{total} = {both_hit/total:.2%}")


if __name__ == "__main__":
    evaluate(top_k=6)
