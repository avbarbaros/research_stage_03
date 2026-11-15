"""ScienceQA loading and formatting (Lu et al., NeurIPS 2022).

The Hugging Face copy (derek-thomas/ScienceQA) has no question id column, so an
item's id is "<split>-<row index>". Row order in the HF parquet files is stable,
so the id is reproducible and later experiments (E3.5) can reload the image with
dataset[row_index].
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

LETTERS = "ABCDEFGHIJ"

SUBJECT_CODES = {
    "natural science": "NAT",
    "social science": "SOC",
    "language science": "LAN",
}


@dataclass
class SQAItem:
    id: str
    row: int
    question: str
    choices: list[str]
    answer_idx: int
    hint: str
    lecture: str
    solution: str
    subject: str
    grade: str
    topic: str
    category: str
    skill: str
    image: Any = field(default=None, repr=False)  # PIL.Image or None

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def answer_letter(self) -> str:
        return LETTERS[self.answer_idx]

    # The standard ScienceQA breakdown columns used in every MCoT paper.
    def groups(self) -> list[str]:
        g = [SUBJECT_CODES.get(self.subject, self.subject)]
        if self.has_image:
            g.append("IMG")
        if self.hint.strip():
            g.append("TXT")
        if not self.has_image and not self.hint.strip():
            g.append("NO")
        try:
            grade_n = int(self.grade.replace("grade", ""))
            g.append("G1-6" if grade_n <= 6 else "G7-12")
        except ValueError:
            pass
        return g

    def context(self, include_lecture: bool = False) -> str:
        parts = []
        if self.hint.strip():
            parts.append(self.hint.strip())
        if include_lecture and self.lecture.strip():
            parts.append(self.lecture.strip())
        return "\n".join(parts) if parts else "N/A"

    def options_str(self) -> str:
        return " ".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(self.choices))

    def meta(self) -> dict:
        return {
            "id": self.id,
            "row": self.row,
            "gold": self.answer_letter,
            "n_choices": len(self.choices),
            "has_image": self.has_image,
            "groups": self.groups(),
            "subject": self.subject,
            "grade": self.grade,
            "topic": self.topic,
            "skill": self.skill,
        }


GROUP_ORDER = ["NAT", "SOC", "LAN", "TXT", "IMG", "NO", "G1-6", "G7-12"]


def load_scienceqa(
    hf_id: str,
    split: str,
    image_only: bool = False,
    max_samples: int | None = None,
    sample_seed: int = 42,
) -> list[SQAItem]:
    from datasets import load_dataset

    ds = load_dataset(hf_id, split=split)
    rows = list(range(len(ds)))
    if image_only:
        # Reading only the image column metadata is slow on HF Image features;
        # decode=False avoids decoding every PNG just to test for None.
        from datasets import Image as HFImage

        img_col = ds.cast_column("image", HFImage(decode=False))["image"]
        rows = [i for i in rows if img_col[i] is not None]
    if max_samples is not None and max_samples < len(rows):
        rng = random.Random(sample_seed)
        rows = sorted(rng.sample(rows, max_samples))

    items = []
    for i in rows:
        ex = ds[i]
        img = ex["image"]
        if img is not None and img.mode != "RGB":
            img = img.convert("RGB")
        items.append(
            SQAItem(
                id=f"{split}-{i}",
                row=i,
                question=ex["question"],
                choices=list(ex["choices"]),
                answer_idx=int(ex["answer"]),
                hint=ex.get("hint") or "",
                lecture=ex.get("lecture") or "",
                solution=ex.get("solution") or "",
                subject=ex.get("subject") or "",
                grade=ex.get("grade") or "",
                topic=ex.get("topic") or "",
                category=ex.get("category") or "",
                skill=ex.get("skill") or "",
                image=img,
            )
        )
    return items
