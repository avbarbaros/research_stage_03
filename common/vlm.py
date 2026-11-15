"""A thin VLM wrapper with two interchangeable backends.

    vlm = VLM(cfg["common"]["model"])
    outs = vlm.generate([Request(text="...", image=pil_or_None)], GenParams(max_new_tokens=16))
    outs[i].text, outs[i].n_tokens

Prompts are built with the model's own chat template (AutoProcessor), so the
vLLM and transformers backends see token-identical inputs. With temperature 0
both decode greedily; small numeric differences between the two kernels can
still change a few outputs, so do not mix backends within one run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tqdm import tqdm


@dataclass
class Request:
    text: str
    image: Any = None  # PIL.Image or None


@dataclass
class GenParams:
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None

    @classmethod
    def from_cfg(cls, d: dict, seed: int | None = None) -> "GenParams":
        return cls(
            max_new_tokens=int(d.get("max_new_tokens", 16)),
            temperature=float(d.get("temperature", 0.0)),
            top_p=float(d.get("top_p", 1.0)),
            seed=seed,
        )


@dataclass
class Output:
    text: str
    n_tokens: int
    finish_reason: str | None = None  # "stop" | "length" | None


class VLM:
    def __init__(self, model_cfg: dict, system_prompt: str | None = None):
        from transformers import AutoProcessor

        self.cfg = model_cfg
        self.name = model_cfg["name"]
        self.backend = model_cfg.get("backend", "vllm")
        self.system_prompt = system_prompt
        self.processor = AutoProcessor.from_pretrained(
            self.name,
            min_pixels=model_cfg.get("min_pixels"),
            max_pixels=model_cfg.get("max_pixels"),
        )
        if self.backend == "vllm":
            self._init_vllm()
        elif self.backend == "hf":
            self._init_hf()
        else:
            raise ValueError(f"unknown backend {self.backend!r} (use vllm or hf)")

    # ------------------------------------------------------------------ prompts
    def build_prompt(self, req: Request) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        content = []
        if req.image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": req.text})
        messages.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------ vLLM
    def _init_vllm(self) -> None:
        from vllm import LLM

        c = self.cfg
        self.llm = LLM(
            model=self.name,
            dtype=c.get("dtype", "bfloat16"),
            max_model_len=int(c.get("max_model_len", 8192)),
            gpu_memory_utilization=float(c.get("gpu_memory_utilization", 0.9)),
            enforce_eager=bool(c.get("enforce_eager", False)),
            max_num_seqs=int(c.get("max_num_seqs", 64)),
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": c.get("min_pixels"),
                "max_pixels": c.get("max_pixels"),
            },
            seed=0,
        )

    def _generate_vllm(self, reqs: list[Request], gp: GenParams, desc: str) -> list[Output]:
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=gp.max_new_tokens,
            temperature=gp.temperature,
            top_p=gp.top_p if gp.temperature > 0 else 1.0,
            seed=gp.seed,
        )
        inputs = []
        for r in reqs:
            item: dict[str, Any] = {"prompt": self.build_prompt(r)}
            if r.image is not None:
                item["multi_modal_data"] = {"image": r.image}
            inputs.append(item)
        results = self.llm.generate(inputs, sp, use_tqdm=True)
        outs = []
        for res in results:
            o = res.outputs[0]
            outs.append(Output(text=o.text, n_tokens=len(o.token_ids), finish_reason=o.finish_reason))
        return outs

    # ------------------------------------------------------------------ transformers
    def _init_hf(self) -> None:
        import torch

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoVLM

        c = self.cfg
        self.model = AutoVLM.from_pretrained(
            self.name,
            torch_dtype=getattr(torch, c.get("dtype", "bfloat16")),
            attn_implementation=c.get("attn_implementation", "sdpa"),
            device_map="cuda",
        ).eval()
        self.processor.tokenizer.padding_side = "left"

    def _generate_hf(self, reqs: list[Request], gp: GenParams, desc: str) -> list[Output]:
        import torch

        bs = int(self.cfg.get("hf_batch_size", 8))
        tok = self.processor.tokenizer
        eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}
        if gp.seed is not None:
            torch.manual_seed(gp.seed)
        outs: list[Output] = []
        for start in tqdm(range(0, len(reqs), bs), desc=desc):
            batch = reqs[start : start + bs]
            texts = [self.build_prompt(r) for r in batch]
            images = [r.image for r in batch if r.image is not None]
            enc = self.processor(
                text=texts,
                images=images or None,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            kwargs = dict(max_new_tokens=gp.max_new_tokens)
            if gp.temperature > 0:
                kwargs.update(do_sample=True, temperature=gp.temperature, top_p=gp.top_p)
            else:
                kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.inference_mode():
                gen = self.model.generate(**enc, **kwargs)
            new = gen[:, enc["input_ids"].shape[1] :]
            for row in new:
                ids = row.tolist()
                # trim at first EOS / padding
                n = len(ids)
                for j, t in enumerate(ids):
                    if t in eos_ids or t == tok.pad_token_id:
                        n = j
                        break
                finish = "stop" if n < len(ids) else "length"
                outs.append(
                    Output(text=tok.decode(ids[:n], skip_special_tokens=True), n_tokens=n, finish_reason=finish)
                )
        return outs

    # ------------------------------------------------------------------ public
    def generate(self, reqs: list[Request], gp: GenParams, desc: str = "generate") -> list[Output]:
        if not reqs:
            return []
        if self.backend == "vllm":
            return self._generate_vllm(reqs, gp, desc)
        return self._generate_hf(reqs, gp, desc)
