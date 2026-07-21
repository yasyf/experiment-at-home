from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from athome.train.spec import BaseModelSpec

SIDECAR: Path = Path(__file__).resolve()
PEFT_KEY = re.compile(r"base_model\.model\.(model\.layers\.\d+\.(.+?))\.lora_([AB])\.weight")
PEFT_CONFIG = "adapter_config.json"
GENERATION_FENCE = "=" * 10


def uvx(*args: str) -> tuple[str, ...]:
    """The argv that runs ``args`` inside the pinned mlx-lm environment."""
    from athome.config import load
    from athome.train.spec import TrainSettings

    return ("uvx", "--from", f"mlx-lm=={load(TrainSettings).mlx_lm_version}", *args)


def mlx_lm_command(*args: str) -> tuple[str, ...]:
    """The argv that runs an mlx-lm subcommand (``lora``, ``fuse``, ...) in the sidecar environment."""
    return uvx("python", "-m", "mlx_lm", *args)


def sidecar_command(*args: str) -> tuple[str, ...]:
    """The argv that runs this module's own ``convert`` subcommand in the sidecar environment."""
    return uvx("python", str(SIDECAR), *args)


async def run_process(command: Sequence[str]) -> None:
    import anyio

    await anyio.run_process(command)


async def run_output(command: Sequence[str]) -> str:
    """Run ``command`` in the sidecar and return its captured stdout, decoded."""
    import anyio

    return (await anyio.run_process(command)).stdout.decode()


def strip_generation(stdout: str) -> str:
    """The generated completion carried between mlx-lm's two ``==========`` fences.

    ``mlx_lm generate`` brackets the streamed completion with a line of ten ``=`` before and
    after it, then prints token-rate and peak-memory stats below the second fence. The text
    between the fences is exactly the completion; everything else is furniture.
    """
    lines = stdout.splitlines()
    fences = [index for index, line in enumerate(lines) if line == GENERATION_FENCE]
    opening, closing = fences[0], fences[1]
    return "\n".join(lines[opening + 1 : closing])


async def convert_peft_to_mlx(peft_dir: Path, out_dir: Path, *, base: BaseModelSpec) -> Path:
    """Convert a PEFT LoRA adapter into an mlx-lm adapter directory and return it.

    Tinker and Modal both train PEFT adapters, whose ``lora_A``/``lora_B`` matrices are the
    transpose of mlx-lm's ``lora_a``/``lora_b``. The shape written out is the archive's own —
    read from its ``adapter_config.json`` and its weight names — never the shape that was
    requested: a backend is free to resolve a request differently (Tinker picks the alpha
    itself), and fusing an adapter under a scale or a key set it was not trained with
    reinterprets the weights.

    Args:
        peft_dir: A directory holding ``adapter_model.safetensors`` and ``adapter_config.json``.
        out_dir: Where ``adapters.safetensors`` and ``adapter_config.json`` land.
        base: The base model the adapter was trained over.

    Returns:
        ``out_dir``, an mlx-lm adapter directory.

    Raises:
        UnservableBase: The base has no mlx-lm LoRA counterpart, so the PEFT adapter has no mlx-lm
            shape to convert into; nothing is converted.
    """
    from athome.train.spec import require_servable

    require_servable(base, kind="Tinker")
    await run_process(
        sidecar_command("convert", "--peft", str(peft_dir), "--out", str(out_dir), "--num-layers", str(base.num_layers))
    )
    return out_dir


async def fuse(adapter_dir: Path, out_dir: Path, *, base: BaseModelSpec) -> Path:
    """Merge an mlx-lm adapter into its base and return the standalone 4-bit MLX model directory.

    This is the servable artifact: ``rapid-mlx serve`` takes one model and has no adapter
    flag. Fusing a 4-bit base re-quantizes every merged layer at the base's own bit width,
    so the saved model is already 4-bit.

    Args:
        adapter_dir: An mlx-lm adapter directory.
        out_dir: Where the fused model is saved.
        base: The base model the adapter was trained over.

    Returns:
        ``out_dir``, a standalone 4-bit MLX model directory.

    Raises:
        UnservableBase: The base has no mlx-lm LoRA counterpart to fuse the adapter into; nothing
            is fused.
    """
    from athome.train.spec import require_servable

    require_servable(base, kind="Tinker")
    await run_process(
        mlx_lm_command("fuse", "--model", base.mlx, "--adapter-path", str(adapter_dir), "--save-path", str(out_dir))
    )
    return out_dir


async def generate(model: str, prompt: str, *, max_tokens: int, temperature: float, seed: int | None = None) -> str:
    """Sample a completion from ``model`` for ``prompt`` in the pinned mlx-lm sidecar.

    Runs ``mlx_lm generate`` out of process in the pinned environment and returns only the
    generated text, stripped of the ``==========`` fences and the token-rate and peak-memory
    lines mlx-lm prints around it. The prompt is fed verbatim (``--ignore-chat-template``), so
    the completion depends on ``(model, prompt, max_tokens, temperature, seed)`` alone and never
    on a model's embedded chat template — the caller owns any templating.

    Args:
        model: The mlx-lm model directory or Hugging Face repo to sample from.
        prompt: The prompt text, tokenized as-is with no chat-template wrapping.
        max_tokens: The generation cap.
        temperature: The sampling temperature; 0.0 is greedy.
        seed: The PRNG seed, or None to leave mlx-lm's generator unseeded.

    Returns:
        The generated completion text, furniture stripped.
    """
    return strip_generation(
        await run_output(
            mlx_lm_command(
                "generate",
                "--model",
                model,
                "--prompt",
                prompt,
                "--max-tokens",
                str(max_tokens),
                "--temp",
                str(temperature),
                "--ignore-chat-template",
                *(("--seed", str(seed)) if seed is not None else ()),
            )
        )
    )


def run_convert(peft_dir: Path, out_dir: Path, *, num_layers: int) -> dict[str, object]:
    import mlx.core as mx

    config = json.loads((peft_dir / PEFT_CONFIG).read_text())
    weights = mx.load(str(peft_dir / "adapter_model.safetensors"))
    matched = {key: match for key in weights if (match := PEFT_KEY.match(key))}
    if unfusable := sorted(key for key in weights if key not in matched):
        raise ValueError(f"{peft_dir}: mlx-lm cannot fuse {len(unfusable)} trained tensor(s): {unfusable}")
    adapter = {
        f"{match.group(1)}.lora_{'a' if match.group(3) == 'A' else 'b'}": weights[key].T
        for key, match in matched.items()
    }
    keys = sorted({match.group(2) for match in matched.values()})
    rank, alpha = config["r"], config["lora_alpha"]
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), adapter)
    (out_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": num_layers,
                "lora_parameters": {
                    "rank": rank,
                    "scale": alpha / rank,
                    "dropout": config["lora_dropout"],
                    "keys": keys,
                },
            },
            indent=2,
        )
        + "\n"
    )
    return {"n_lora_weights": len(adapter), "rank": rank, "alpha": alpha, "keys": keys}


def main() -> None:
    parser = argparse.ArgumentParser(prog="athome-train-sidecar")
    convert = parser.add_subparsers(dest="command", required=True).add_parser("convert")
    convert.add_argument("--peft", type=Path, required=True)
    convert.add_argument("--out", type=Path, required=True)
    convert.add_argument("--num-layers", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run_convert(args.peft, args.out, num_layers=args.num_layers)))


if __name__ == "__main__":
    main()
