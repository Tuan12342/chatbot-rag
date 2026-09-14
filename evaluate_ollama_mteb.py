"""Đánh giá model embedding chạy qua Ollama bằng benchmark MTEB."""

import argparse
import json
import os
from pathlib import Path

# MTEB tạo cache ngay khi import. Giữ cache bên trong dự án để không cần quyền admin.
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MTEB_CACHE", str(BASE_DIR / "results" / "mteb-cache"))

import mteb  # noqa: E402
import numpy as np  # noqa: E402
from mteb.models.abs_encoder import AbsEncoder  # noqa: E402
from ollama import Client  # noqa: E402


DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_TASK = "Banking77Classification.v2"


class OllamaEncoder(AbsEncoder):
    """Adapter để MTEB gọi model embedding thông qua Ollama local API."""

    mteb_model_meta = None

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        revision: str | None = None,
        *,
        host: str = "http://127.0.0.1:11434",
        **kwargs,
    ) -> None:
        del revision, kwargs
        self.model_name = model_name
        self.client = Client(host=host)

    def check_model(self) -> None:
        """Báo lỗi sớm nếu Ollama chưa chạy hoặc model chưa được tải."""
        try:
            self.client.show(self.model_name)
        except Exception as error:
            raise RuntimeError(
                "Không kết nối được model Ollama. Hãy chạy `ollama serve` và "
                f"`ollama pull {self.model_name}` trước. Chi tiết: {error}"
            ) from error

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        del task_metadata, hf_split, hf_subset, prompt_type
        keep_alive = kwargs.pop("keep_alive", "10m")
        kwargs.pop("batch_size", None)
        kwargs.pop("show_progress_bar", None)

        encoded_batches = []
        for batch in inputs:
            texts = list(batch["text"])
            response = self.client.embed(
                model=self.model_name,
                input=texts,
                truncate=True,
                keep_alive=keep_alive,
            )
            vectors = np.asarray(response.embeddings, dtype=np.float32)

            # Chuẩn hóa để cosine similarity ổn định giữa các batch.
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
            encoded_batches.append(vectors)

        if not encoded_batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(encoded_batches)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá model embedding Ollama trên một MTEB task."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = BASE_DIR / "results" / args.model.replace(":", "-")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = OllamaEncoder(model_name=args.model, host=args.ollama_host)
    model.check_model()
    tasks = mteb.get_tasks(tasks=[args.task])

    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print("Benchmark sẽ chạy trên CPU qua Ollama; quá trình có thể mất khá lâu.\n")

    result = mteb.evaluate(
        model,
        tasks=tasks,
        encode_kwargs={"batch_size": args.batch_size, "keep_alive": "10m"},
        cache=None,
        co2_tracker=False,
        overwrite_strategy="always",
    )

    result_file = output_dir / f"{args.task}.json"
    result_file.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    task_result = result.task_results[0]
    test_scores = task_result.scores.get("test", [])
    if not test_scores:
        raise RuntimeError("MTEB không trả về điểm cho test split.")

    scores = test_scores[0]
    print("\n===== KẾT QUẢ =====")
    for metric in ("accuracy", "f1", "f1_weighted", "precision", "recall"):
        value = scores.get(metric)
        if isinstance(value, (int, float)):
            print(f"{metric:16}: {value:.4f} ({value * 100:.2f}%)")
    print(f"\nKết quả đầy đủ: {result_file}")


if __name__ == "__main__":
    main()
