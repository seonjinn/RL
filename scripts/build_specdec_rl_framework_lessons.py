#!/usr/bin/env python3
"""Build the offline cross-framework SpecDec RL lessons report."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "specdec_rl_framework_lessons_and_nemorl_gaps_20260709.html"
PUBLIC_OUT = (
    ROOT
    / "public"
    / "reports"
    / "specdec_rl_framework_lessons_and_nemorl_gaps_20260709.html"
)

SNAPSHOT_LABEL = "Evidence reviewed: 2026-07-09"
CURRENT_WORKTREE = ".worktrees/nemorl-vllm024-upgrade"
CURRENT_WORKTREE_COMMIT = "3d748edfbf3e7fc5168f3c0cfc618cd22c6d9e9a"
CURRENT_UPSTREAM_COMMIT = "9e01af64b3891e5bcc01885e10e9ca185b3e3690"
VLLM_024_COMMIT = "ee0da84ab9e04ac7610e28580af62c365e898389"
CURRENT_WORKTREE_PROVENANCE = (
    "origin: https://github.com/NVIDIA/NeMo-RL.git",
    "fork: git@github-seonjinn:seonjinn/RL.git",
    "branch: sna/nemorl-vllm024-upgrade",
)

FRAMEWORK_ORDER = ("veRL", "slime", "Miles", "SGLang/vLLM", "NeMo-RL gap")
STATUS_ORDER = ("merged", "open", "unresolved")
STATUS_LABELS = {
    "merged": "MERGED",
    "open": "OPEN",
    "unresolved": "UNRESOLVED",
}


@dataclass(frozen=True)
class EvidenceRecord:
    framework: str
    repo: str
    number: int
    kind: str
    bucket: str
    title: str
    author: str
    lesson: str

    @property
    def key(self) -> str:
        return f"{self.framework}:{self.number}"

    @property
    def url(self) -> str:
        suffix = "pull" if self.kind == "pr" else "issues"
        return f"https://github.com/{self.repo}/{suffix}/{self.number}"

    @property
    def anchor(self) -> str:
        return (
            self.framework.lower().replace("/", "-").replace(" ", "-").replace(".", "-")
            + f"-{self.number}"
        )

    @property
    def short_id(self) -> str:
        return f"{self.kind.upper()} #{self.number}"


@dataclass(frozen=True)
class MatrixCell:
    summary: str
    refs: tuple[str, ...]


@dataclass(frozen=True)
class MatrixRow:
    topic: str
    cells: dict[str, MatrixCell]


@dataclass(frozen=True)
class GapRow:
    label: str
    severity: str
    current_code_evidence: tuple[str, ...]
    user_visible_impact: str
    upstream_lesson: str
    upstream_refs: tuple[str, ...]
    implementation_status: str
    status_class: str
    validation_gate: str


def ref(framework: str, number: int) -> str:
    return f"{framework}:{number}"


EVIDENCE: tuple[EvidenceRecord, ...] = (
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        4936,
        "pr",
        "merged",
        "[megatron] feat: Using MTP in RL Training and Inference",
        "ArronHZG",
        "MTP RL training and inference became first-class rollout work.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        5116,
        "pr",
        "merged",
        "[sglang] fix: update wiki to support speculative decode rollout",
        "ArronHZG",
        "Rollout docs were updated alongside the runtime path.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        5561,
        "pr",
        "merged",
        "[megatron] feat: model engine support mtp",
        "ArronHZG",
        "The model engine learned explicit MTP support instead of hiding it in patches.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        5801,
        "pr",
        "merged",
        "[vllm, fsdp] fix: apply FSDP buffer updates during rollout weight sync",
        "chenshui223",
        "Weight sync correctness includes buffer ownership, not only parameter tensors.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        6432,
        "pr",
        "merged",
        "[megatron,rollout] fix: align MTP loss and rollout metrics",
        "xhx1022",
        "Loss accounting and rollout metrics were treated as one contract.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        6648,
        "pr",
        "merged",
        "[megatron] fix: MTP compatible with latest mcore",
        "HollowMan6",
        "Bridge compatibility drift keeps resurfacing and needs targeted fixes.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        6661,
        "pr",
        "merged",
        "[rollout, vllm] fix: preserve MTP drafter weights during hybrid sleep",
        "sunnweiwei",
        "Draft weights must survive rollout lifecycle transitions.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        4535,
        "pr",
        "merged",
        "[recipe] feat: accelerate rollout via model-free speculative decoding",
        "He-Jingkai",
        "Model-free speculation was promoted to a rollout recipe rather than a side experiment.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        4947,
        "pr",
        "unresolved",
        "[rfc]:add speculator training scripts and checkpoint support",
        "meichangsu1",
        "Speculator training scripts and checkpoints still lacked a closed path in the reviewed snapshot.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        5509,
        "pr",
        "open",
        "[rollout] feat: support eagle3 speculative decode in rollout",
        "miracle0517",
        "EAGLE3 rollout support remained active work.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        5925,
        "pr",
        "open",
        "[vllm, rollout, cfg, doc] feat: Accelerate RL rollouts with EAGLE/EAGLE3 speculative decoding",
        "alekseymalakhov11",
        "Runtime config and rollout acceleration were still evolving together.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        5757,
        "pr",
        "open",
        "[rollout] feat: Support decoupled speculation with dynamic adjustment feature for rollout (WIP)",
        "sisyphus111",
        "Dynamic decoupled speculation was still unfinished.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        6342,
        "pr",
        "open",
        "[rollout] feat: support suffix speculative decoding in vLLM rollout",
        "walterchenchn",
        "Suffix decode support was still open in rollout integration.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        4791,
        "issue",
        "open",
        "[RFC] Suffix Decoding + LSTM Speculative Decoding for Faster Generation",
        "vx120",
        "Suffix decoding remained a live design thread rather than closed infrastructure.",
    ),
    EvidenceRecord(
        "veRL",
        "verl-project/verl",
        6985,
        "issue",
        "open",
        "[roadmap] verl 26Q3 roadmap",
        "wuxibin89",
        "SpecDec rollout work stayed visible at roadmap scope.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        128,
        "pr",
        "merged",
        "Add XiaomiMiMo/MiMo-7B-RL with MTP support",
        "guapisolo",
        "Model-family onboarding included explicit MTP wiring.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        204,
        "pr",
        "merged",
        "[feat] support training with sglang draft model without mtp",
        "zhuzilin",
        "External draft models were supported directly, not only embedded MTP.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        640,
        "pr",
        "merged",
        "Support MTP training",
        "guapisolo",
        "MTP training graduated from issue queue to merged baseline behavior.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        691,
        "pr",
        "merged",
        "Fix Mimo-7B-RL special mtp structure",
        "guapisolo",
        "Model-specific bridge structure mismatches were fixed in-tree.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        707,
        "pr",
        "merged",
        "[Revert revert] Fix Mimo-7B-RL special mtp structure",
        "guapisolo",
        "The same bridge path needed follow-on correction, showing how fragile this surface is.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        709,
        "pr",
        "merged",
        "detach lm_head when training mtp",
        "zhuzilin",
        "Training semantics around MTP heads were explicitly managed.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        746,
        "pr",
        "merged",
        "Fix mtp rl detach",
        "zhuzilin",
        "RL-side MTP detach behavior needed a dedicated fix.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        751,
        "pr",
        "merged",
        "Fix MTP loss mask intersection",
        "guapisolo",
        "Loss masking correctness was part of the integration, not cleanup.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        971,
        "pr",
        "merged",
        "Super tiny enable draft-weights-cpu-backup to avoid MTP acc len issue",
        "fzyzcjy",
        "Draft-weight CPU backup was exposed as a concrete runtime lever.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1024,
        "pr",
        "merged",
        "Fix mimo speculative decoding oom",
        "guapisolo",
        "SpecDec memory pressure was fixed as a first-order rollout issue.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1207,
        "pr",
        "merged",
        "Add ci for mtp",
        "zhuzilin",
        "CI was added to keep MTP from regressing silently.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1313,
        "pr",
        "merged",
        "[docker] Fix sglang ima on mtp + pd disaggregation",
        "zhuzilin",
        "Container/runtime packaging was part of the MTP compatibility story.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1503,
        "pr",
        "merged",
        "fix: support mtp for qwen3-next",
        "huang3eng",
        "Qwen3-next required explicit MTP bridge support.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1702,
        "pr",
        "merged",
        "Fix/qwen3 5 mtp bridge",
        "huang3eng",
        "Qwen3.5 bridge drift produced another targeted MTP fix.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1712,
        "pr",
        "merged",
        "Add GLM-4.7-Flash MTP training support",
        "zhuzilin",
        "New model families were added only with explicit training support.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1938,
        "pr",
        "merged",
        "fix: guard sglang_speculative_algorithm read in --debug-train-only mode",
        "leofan-lab",
        "Even debug-only code paths needed guards around speculative config access.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        83,
        "issue",
        "unresolved",
        "[feature] MTP support",
        "zhuzilin",
        "The oldest MTP request stayed relevant deep into later fixes.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        654,
        "issue",
        "unresolved",
        "MTP support Training",
        "Seadawn",
        "Training support gaps remained visible even after major MTP merges.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        693,
        "issue",
        "unresolved",
        "Mimo MTP Issue",
        "sam571128",
        "MiMo-specific MTP behavior remained a recurring breakage class.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1017,
        "issue",
        "open",
        "Question about default arg config on /scripts/run-mimo-7B-rl-eagle.sh",
        "bingyang-lei",
        "Launch-argument defaults still needed clarification for practical use.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1022,
        "issue",
        "open",
        "Qwen3-8B RL Training with Eagle3 Spec Decoding OOM error",
        "gxlvera",
        "OOM remained a visible integration failure mode for EAGLE3 RL.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1177,
        "issue",
        "open",
        "Bugs in using an external draft model and set --sglang-speculative-algorithm EAGLE3",
        "bingyang-lei",
        "External-drafter correctness still produced open bug reports.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1316,
        "issue",
        "unresolved",
        "Mitigate content from sglang.patch to sglang",
        "PrinsYin",
        "Patch-carried runtime changes still needed upstream landing.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1501,
        "issue",
        "unresolved",
        "Cannot reproduce advanced feature online SFT for 'Speculative Decoding'",
        "xurui-del",
        "Online speculative flows were still hard to reproduce from public instructions.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1534,
        "issue",
        "open",
        "Is glm-4.7-flash MTP training in RL stage supported for now?",
        "ifififa",
        "Users still asked whether merged support actually covered RL-stage training.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1548,
        "issue",
        "open",
        "Speculative Decoding Standalone mode load draft model",
        "ruiqiRichard",
        "Draft model loading remained a user-visible lifecycle surface.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1556,
        "issue",
        "unresolved",
        "[Bug] AssertionError in fill_routing_replay when enabling MTP training for GLM-4.7",
        "liujiahua123123",
        "Routing and replay assumptions still broke under new MTP model paths.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        1870,
        "issue",
        "open",
        "[Question] GLM5----MTP training support",
        "Hevans123",
        "Model-support expansion continued past earlier GLM coverage.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        2131,
        "issue",
        "open",
        "[Bug] Multi-head MTP (--mtp-num-layers > 1) crashes at training-step logging",
        "ZiyiTsang",
        "Training-step logging still failed on multi-head MTP shapes.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        2132,
        "pr",
        "open",
        "fix(mtp): support multi-head MTP loss logging (mtp-num-layers > 1)",
        "ZiyiTsang",
        "A specific logging fix was proposed rather than broad refactoring.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        2154,
        "pr",
        "open",
        "fix: support multi-head MTP weight mapping in MimoBridge (closes #2131)",
        "botbikamordehai2-sketch",
        "Bridge-side weight mapping was still being repaired for multi-head MTP.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        2182,
        "pr",
        "open",
        "fix: Fixed the .item() crash on multi-element MTP loss tensors via a",
        "singlaamitesh",
        "Loss-shape assumptions still caused runtime crashes in open work.",
    ),
    EvidenceRecord(
        "slime",
        "THUDM/slime",
        777,
        "issue",
        "open",
        "[ci] CI coverage tracking",
        "zhuzilin",
        "Coverage tracking stayed an active maintenance thread.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        274,
        "pr",
        "merged",
        "Super tiny enable draft-weights-cpu-backup to avoid MTP acc len issue",
        "fzyzcjy",
        "Miles carried the same draft-weight CPU-backup lesson as slime/SGLang.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        278,
        "pr",
        "merged",
        "Add GB200, MTP, benchmark, fp8 rollout mode to glm script",
        "fzyzcjy",
        "Launch scripts treated MTP, benchmarking, and rollout mode as a shared surface.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        449,
        "pr",
        "merged",
        "Support speculative information in mock sglang server",
        "fzyzcjy",
        "Speculative metadata plumbing was added even to mock-server paths.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        619,
        "pr",
        "merged",
        "[fix] bypass r3 for mtp layer.",
        "guapisolo",
        "Bridge quirks around MTP layers needed targeted routing exceptions.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        623,
        "pr",
        "merged",
        "[feat] glm-4.7-flash support with r3+mtp ci",
        "guapisolo",
        "New model support was paired with CI rather than left manual.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        626,
        "pr",
        "merged",
        "[model] support GLM-5",
        "yueming-yuan",
        "Model-family coverage continued to expand through explicit bridge work.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        655,
        "pr",
        "merged",
        "[model] Support qwen3 next mtp training",
        "guapisolo",
        "Qwen3-next training support required explicit MTP integration.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        740,
        "pr",
        "merged",
        "[model] Add Qwen3.5 model support (4B, 9B, 27B and 35B-A3B)",
        "Zhichenzzz",
        "Qwen3.5 support widened the bridge surface substantially.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        984,
        "pr",
        "merged",
        "fix: add b300 qwen35 script and upd spec v2",
        "guapisolo",
        "Specs and scripts kept moving alongside model support.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1022,
        "pr",
        "merged",
        "Feat: Qwen3.6 RL support",
        "Zhichenzzz",
        "RL support expansion continued beyond earlier Qwen families.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1103,
        "pr",
        "merged",
        "doc [7/n]: add docs/advanced and docs/platforms",
        "Zhichenzzz",
        "Advanced/platform docs were considered part of the feature rollout.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1235,
        "pr",
        "merged",
        "feat: disk-delta weight sync for non-colocated rollout engines",
        "nanjiangwill",
        "Non-colocated rollout engines got explicit weight-sync mechanics.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1342,
        "pr",
        "merged",
        "optimize deepseek v4 sglang config",
        "yueming-yuan",
        "Spec runtime config tuning stayed visible in merged work.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1369,
        "pr",
        "merged",
        "[tiny] tune qwen3.5-35B-A3B MTP cp2 ep8 sglang/perf args",
        "guapisolo",
        "Performance tuning still had to touch MTP geometry explicitly.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1376,
        "pr",
        "merged",
        "Support GLM-5.2 744B-A40B",
        "yueming-yuan",
        "Very large-model bridge coverage kept expanding.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1512,
        "pr",
        "merged",
        "Megatron e2e: weight-check skip-list, Qwen3.5 MTP cases",
        "guapisolo",
        "End-to-end checks explicitly tracked MTP model cases.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        876,
        "pr",
        "open",
        "Fix Qwen3 Next MTP bridge block-spec wiring and add regression coverage",
        "taivu1998",
        "Bridge block-spec wiring still needed repair plus regression tests.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1007,
        "pr",
        "open",
        "[feat]support_init_random_mtp",
        "maocheng23",
        "Init-random MTP support remained open work.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1284,
        "pr",
        "open",
        "Nemotron RL support",
        "Zhichenzzz",
        "Nemotron RL support was still open in the reviewed snapshot.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1307,
        "pr",
        "open",
        "fix(mtp): track megatron mtp_model_layer rename in raw converters",
        "Zhichenzzz",
        "Raw converter naming drift still blocked clean MTP support.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1592,
        "pr",
        "open",
        "ci: re-enable GLM-4.7-Flash ckpt save/load e2e test",
        "guapisolo",
        "Checkpoint save/load e2e coverage was still being repaired.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        366,
        "issue",
        "open",
        "MTP not working with Qwen3 Next",
        "nutriarch",
        "User-facing Qwen3-next MTP failures persisted after initial support work.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1289,
        "issue",
        "open",
        "MTP naming mismatch: Megatron-LM fork uses transformer_layer but megatron-bridge expects mtp_model_layer...",
        "WindowsXp-Beta",
        "Naming mismatch in bridge code remained an active issue.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1360,
        "issue",
        "open",
        "[refactor] weight update refactor",
        "yueming-yuan",
        "Weight-update mechanics were still open for broader cleanup.",
    ),
    EvidenceRecord(
        "Miles",
        "radixark/miles",
        1583,
        "issue",
        "open",
        "deepseek-v4 branch: backport #1505 ...",
        "hvgazula",
        "Branch backports and runtime tuning were still in motion.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "sgl-project/sglang",
        10892,
        "pr",
        "merged",
        "Fix CUDA illegal memory access issues in speculative decoding",
        "ur4t",
        "Spec decode graph/runtime bugs were fixed directly in the engine.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "sgl-project/sglang",
        13318,
        "pr",
        "merged",
        "[RL] support only do cpu backup on draft model",
        "zhuzilin",
        "Draft-only CPU backup became an explicit RL runtime control.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "sgl-project/sglang",
        15726,
        "pr",
        "merged",
        "[Fix] Add --disable-draft-model-update to control draft model updates(especially in RL)",
        "bingyang-lei",
        "Draft model update control was exposed directly for RL flows.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "sgl-project/sglang",
        23037,
        "pr",
        "merged",
        "[Bug Fix] Resolve EAGLE cuda graph IMA under PD + DP + MTP with GLM-5.1",
        "zRzRzRzRzRzRzR",
        "CUDA graph correctness under RL-style parallelism was fixed upstream.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "sgl-project/sglang",
        26980,
        "pr",
        "merged",
        "[fix] Skip routed expert capture for draft model under spec v2",
        "guapisolo",
        "Draft-model capture exclusions were treated as engine behavior, not app glue.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "sgl-project/sglang",
        27696,
        "pr",
        "merged",
        "[RL] Handle Mooncake buffers across memory release",
        "zhuzilin",
        "Buffer lifetime handling across memory release is part of RL stability.",
    ),
)


MATRIX_ROWS: tuple[MatrixRow, ...] = (
    MatrixRow(
        "Draft weight lifecycle",
        {
            "veRL": MatrixCell(
                "veRL treats draft ownership and rollout-side weight sync as explicit runtime contracts.",
                (ref("veRL", 4535), ref("veRL", 5801), ref("veRL", 6661)),
            ),
            "slime": MatrixCell(
                "slime keeps draft loading and CPU-backup behavior explicit across standalone and RL paths.",
                (ref("slime", 204), ref("slime", 971), ref("slime", 1548)),
            ),
            "Miles": MatrixCell(
                "Miles extends the same theme through mock speculative plumbing and non-colocated weight sync.",
                (ref("Miles", 449), ref("Miles", 1235), ref("Miles", 1360)),
            ),
            "SGLang/vLLM": MatrixCell(
                "SGLang/vLLM exposes draft-only CPU backup, draft-update control, and buffer-release handling as engine knobs.",
                (
                    ref("SGLang/vLLM", 13318),
                    ref("SGLang/vLLM", 15726),
                    ref("SGLang/vLLM", 27696),
                ),
            ),
            "NeMo-RL gap": MatrixCell(
                "The safety branch separates target and external-draft loading, preserves an independent online Eagle LM head before CUDA-graph capture, and routes both absolute-layer and Qwen mtp.* state to the drafter. MTP startup uses a PP-aware tri-state contract.",
                (),
            ),
        },
    ),
    MatrixRow(
        "Online drafter training",
        {
            "veRL": MatrixCell(
                "veRL landed MTP rollout training, then kept the speculator script/checkpoint path open as unfinished work.",
                (
                    ref("veRL", 4936),
                    ref("veRL", 5561),
                    ref("veRL", 6432),
                    ref("veRL", 4947),
                ),
            ),
            "slime": MatrixCell(
                "slime has the broadest merged MTP training surface, but the open queue still shows bridge and multi-head expansion work.",
                (
                    ref("slime", 640),
                    ref("slime", 709),
                    ref("slime", 1712),
                    ref("slime", 1534),
                    ref("slime", 2131),
                ),
            ),
            "Miles": MatrixCell(
                "Miles keeps adding MTP model support while open work tracks naming drift, random-init flows, and Nemotron support.",
                (
                    ref("Miles", 623),
                    ref("Miles", 655),
                    ref("Miles", 876),
                    ref("Miles", 1007),
                    ref("Miles", 1284),
                    ref("Miles", 1289),
                    ref("Miles", 1307),
                ),
            ),
            "SGLang/vLLM": MatrixCell(
                "SGLang/vLLM focuses on whether and when the draft model updates plus draft-side capture exclusions under spec v2.",
                (ref("SGLang/vLLM", 15726), ref("SGLang/vLLM", 26980)),
            ),
            "NeMo-RL gap": MatrixCell(
                "Online refit is now limited to supported Eagle methods; unsafe fused-logprob, sequence-packing, and non-detached MTP combinations fail before training. MTP/Eagle GPU gradient and checkpoint gates remain pending.",
                (),
            ),
        },
    ),
    MatrixRow(
        "Sampling correctness",
        {
            "veRL": MatrixCell(
                "veRL keeps correctness visible through rollout docs, metric alignment, and still-open suffix or dynamic speculation threads.",
                (
                    ref("veRL", 5116),
                    ref("veRL", 6432),
                    ref("veRL", 5757),
                    ref("veRL", 6342),
                    ref("veRL", 4791),
                ),
            ),
            "slime": MatrixCell(
                "slime correctness work centers on mask semantics, external-draft EAGLE3 behavior, and reproducibility of online spec flows.",
                (
                    ref("slime", 751),
                    ref("slime", 1177),
                    ref("slime", 1501),
                    ref("slime", 1556),
                ),
            ),
            "Miles": MatrixCell(
                "Miles pairs speculative-info plumbing with open bridge mismatch reports that can break trainer-sampler agreement.",
                (
                    ref("Miles", 449),
                    ref("Miles", 366),
                    ref("Miles", 1289),
                    ref("Miles", 1307),
                ),
            ),
            "SGLang/vLLM": MatrixCell(
                "SGLang/vLLM fixes include illegal-memory-access repair and explicit draft-update control for RL-sensitive paths.",
                (ref("SGLang/vLLM", 10892), ref("SGLang/vLLM", 15726)),
            ),
            "NeMo-RL gap": MatrixCell(
                "Sync and async paths now require one finite chosen-token logprob per generated token. RL target sampling requires standard rejection, while greedy or probabilistic draft sampling is allowed and logged.",
                (),
            ),
        },
    ),
    MatrixRow(
        "CUDA graph capture",
        {
            "veRL": MatrixCell(
                "The reviewed veRL set shows active rollout evolution, but no closed item matches SGLang's explicit graph-capture fixes.",
                (ref("veRL", 5925), ref("veRL", 5757)),
            ),
            "slime": MatrixCell(
                "slime exposes graph-adjacent breakage through rollout OOM and training failures more than a dedicated graph API.",
                (ref("slime", 1022), ref("slime", 1024)),
            ),
            "Miles": MatrixCell(
                "Miles still reads more bridge-centric than graph-centric in this snapshot, so failures are the useful signal.",
                (ref("Miles", 1583), ref("Miles", 1592)),
            ),
            "SGLang/vLLM": MatrixCell(
                "This is the strongest upstream signal: speculative CUDA-graph IMA and capture-path fixes were merged in the engine.",
                (ref("SGLang/vLLM", 10892), ref("SGLang/vLLM", 23037)),
            ),
            "NeMo-RL gap": MatrixCell(
                "The generic vLLM 0.24 PARD proposer is still omitted from one CUDA-graph key-initialization condition. That engine change stays isolated until token/logprob parity passes; it is not part of the correctness patch.",
                (),
            ),
        },
    ),
    MatrixRow(
        "Buffer sizing",
        {
            "veRL": MatrixCell(
                "veRL weight-sync work explicitly calls out buffer updates and drafter preservation through rollout lifecycle events.",
                (ref("veRL", 5801), ref("veRL", 6661)),
            ),
            "slime": MatrixCell(
                "slime's OOM threads show what happens when draft-side memory growth is left opaque during rollout.",
                (ref("slime", 1022), ref("slime", 1024)),
            ),
            "Miles": MatrixCell(
                "Miles pushes buffer handling into weight-sync mechanics and open refactors.",
                (ref("Miles", 1235), ref("Miles", 1360)),
            ),
            "SGLang/vLLM": MatrixCell(
                "SGLang/vLLM now treats buffer lifetime as a tracked runtime surface, especially under Mooncake and draft-only CPU backup.",
                (ref("SGLang/vLLM", 13318), ref("SGLang/vLLM", 27696)),
            ),
            "NeMo-RL gap": MatrixCell(
                "MTP persistent buffers such as e_score_correction_bias now follow drafter ownership, and sleep/wake uses level 1. Long-context capacity, complete tensor equality, and release/resume GPU gates are still required.",
                (),
            ),
        },
    ),
    MatrixRow(
        "Metrics",
        {
            "veRL": MatrixCell(
                "veRL aligned rollout metrics with MTP loss semantics and kept roadmap-level work visible.",
                (ref("veRL", 6432), ref("veRL", 6985)),
            ),
            "slime": MatrixCell(
                "slime keeps MTP CI and bug reports close to training changes, which reduces silent regressions.",
                (
                    ref("slime", 1207),
                    ref("slime", 777),
                    ref("slime", 2131),
                    ref("slime", 2132),
                ),
            ),
            "Miles": MatrixCell(
                "Miles keeps CI and weight-update maintenance in the same conversation as support expansion.",
                (ref("Miles", 1103), ref("Miles", 1360), ref("Miles", 1592)),
            ),
            "SGLang/vLLM": MatrixCell(
                "The engine fixes are runtime-oriented, so downstream RL code still has to protect metric integrity itself.",
                (ref("SGLang/vLLM", 15726), ref("SGLang/vLLM", 26980)),
            ),
            "NeMo-RL gap": MatrixCell(
                "Chosen-token logprobs now fail closed, and the resolved method, sampling policy, TP, load formats, and CUDA-graph state are printed as a deterministic startup contract. End-to-end W&B parity remains a GPU gate.",
                (),
            ),
        },
    ),
    MatrixRow(
        "Failure recovery",
        {
            "veRL": MatrixCell(
                "veRL's merged lessons are mostly about preserving drafter state across disruptive lifecycle events while other rollout ideas remain open.",
                (ref("veRL", 5801), ref("veRL", 6661), ref("veRL", 6342)),
            ),
            "slime": MatrixCell(
                "slime's issue queue acts like a recovery map: crashes, bridge mismatches, and logging failures quickly turn into follow-on fixes.",
                (
                    ref("slime", 1177),
                    ref("slime", 1501),
                    ref("slime", 1556),
                    ref("slime", 2131),
                    ref("slime", 2154),
                    ref("slime", 2182),
                ),
            ),
            "Miles": MatrixCell(
                "Miles is still closing the loop on open bridge and CI cases, especially around Qwen naming and Nemotron coverage.",
                (
                    ref("Miles", 876),
                    ref("Miles", 1284),
                    ref("Miles", 1289),
                    ref("Miles", 1592),
                ),
            ),
            "SGLang/vLLM": MatrixCell(
                "SGLang/vLLM exposes direct runtime recovery levers: disable draft updates, release buffers correctly, and avoid graph-replay IMA paths.",
                (
                    ref("SGLang/vLLM", 10892),
                    ref("SGLang/vLLM", 15726),
                    ref("SGLang/vLLM", 27696),
                ),
            ),
            "NeMo-RL gap": MatrixCell(
                "Empty or partial collective results and missing drafters now fail instead of reporting success. PP rank translation is corrected, but MCore/GB200 topology and resume gates remain pending.",
                (),
            ),
        },
    ),
    MatrixRow(
        "Distributed topology and replay",
        {
            "veRL": MatrixCell(
                "veRL treats parameter and buffer ownership across rollout workers as an explicit distributed contract.",
                (ref("veRL", 5801), ref("veRL", 6432)),
            ),
            "slime": MatrixCell(
                "slime's graph and route fixes show that draft-only state must not leak into target execution or stale replay buffers.",
                (ref("slime", 1313),),
            ),
            "Miles": MatrixCell(
                "Miles explicitly bypasses router replay for MTP blocks and validates target/draft weight ownership.",
                (ref("Miles", 619), ref("Miles", 1512)),
            ),
            "SGLang/vLLM": MatrixCell(
                "SGLang excludes draft routing from target replay and repairs aliased or stale graph buffers in engine code.",
                (ref("SGLang/vLLM", 23037), ref("SGLang/vLLM", 26980)),
            ),
            "NeMo-RL gap": MatrixCell(
                "The safety worktree translates PP-local ranks before point-to-point transfer and excludes MTP routers from replay. Multi-rank MCore/GB200 validation remains mandatory.",
                (),
            ),
        },
    ),
    MatrixRow(
        "Training loss semantics",
        {
            "veRL": MatrixCell(
                "veRL aligns MTP labels, masks, detached base heads, synced buffers, and rollout metrics as one correctness surface.",
                (ref("veRL", 5801), ref("veRL", 6432)),
            ),
            "slime": MatrixCell(
                "slime isolates MTP gradients and intersects shifted masks so prompt, padding, and packed boundaries remain excluded.",
                (ref("slime", 709), ref("slime", 751)),
            ),
            "Miles": MatrixCell(
                "Miles keeps MTP routing separate from base-policy replay and checks draft state during synchronization.",
                (ref("Miles", 619), ref("Miles", 1512)),
            ),
            "SGLang/vLLM": MatrixCell(
                "Engine-side controls can disable external draft updates, but the RL trainer still owns loss masks and gradient isolation.",
                (ref("SGLang/vLLM", 15726),),
            ),
            "NeMo-RL gap": MatrixCell(
                "Unsafe fused-logprob, sequence-packed Eagle, and non-detached MTP training now fail fast. The third-party multi-horizon MTP mask replacement remains unresolved.",
                (),
            ),
        },
    ),
)


NEMO_AUDIT_GAPS: tuple[GapRow, ...] = (
    GapRow(
        "1. Generated-token chosen logprobs were silently zero-filled",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/utils.py:28-63",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:726-737",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker_async.py:1162-1181",
        ),
        "At the reviewed base commit, missing, short, wrong-token, or non-finite vLLM logprob records could become zero behavior logprobs and corrupt GRPO importance ratios, KL, entropy, and diagnostics. The local patch now requires one finite chosen-token value for every generated token.",
        "Rollout correctness and metrics must share one strict contract; malformed speculative output cannot be accepted as valid behavior data.",
        (ref("veRL", 6432),),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "Focused extraction unit coverage handles exact values, absent values, short lists, missing chosen tokens, and non-finite values. Run scripts/check_nemorl_specdec_parity.py for the GPU gate: greedy exact tokens/logprobs and sampled first-token TV plus mean logprob/reward. Its unit file tests/test_nemorl_specdec_parity.py passes 3 tests.",
    ),
    GapRow(
        "2. Refit and MTP startup trusted worker zero or ignored worker results",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:53-63",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:608-618",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:917-959",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker_async.py:429-443",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker_async.py:1419-1469",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:312-339",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/worker/gpu_model_runner.py",
        ),
        "A later TP or PP worker could fail while refit still reported success, leaving replicas on different weights. General refit requires a non-empty all-True result sequence. MTP startup is PP-aware because vLLM 0.24 creates the drafter only on the last PP stage: non-owner ranks return None, at least one owner result must exist, every owner must return True, and False, all-None, or empty results fail.",
        "Parameters and persistent buffers must be synchronized and acknowledged by every rollout worker before generation resumes.",
        (ref("veRL", 5801), ref("Miles", 1512)),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "Sync/async worker-result coverage passes 40 tests, including general all-worker refit and the MTP tri-state cases [None, None, True, True], owner False, all-None, and empty.",
    ),
    GapRow(
        "3. Non-empty online draft updates silently skipped a missing drafter",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:51-72",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:282-297",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:315-337",
        ),
        "A run could claim an online draft update while the owner had no vLLM drafter or while the loader reported no compatible weights. The local loader fails closed for non-empty owner updates; MTP disk loading returns None only on expected non-owner PP ranks and raises on the last-stage owner if its drafter is absent.",
        "RL runtimes expose draft-update controls because a stale or absent drafter is a correctness and acceptance-rate failure, not a harmless optimization miss.",
        (ref("SGLang/vLLM", 15726), ref("Miles", 1512)),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "The isolated vLLM backend gate passes 28 tests, covering owner/non-owner PP behavior, no-drafter failure, transport completeness, required MTP-from-refit ownership, empty loaded-name sets, and missing or unexpected keys.",
    ),
    GapRow(
        "4. Online MTP layers and buffers were not routed to the vLLM drafter",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:205-244",
            f"{CURRENT_WORKTREE}/tests/unit/models/generation/test_vllm_backend.py:108-133",
        ),
        "Bridge-exported MTP tensors retained either absolute HF layer names or Qwen's mtp.* namespace, so the old Eagle-only split left the vLLM MTP predictor dummy or stale. The local patch assigns both layouts, including persistent buffers such as e_score_correction_bias, to the drafter only when an MTP speculative method is active; MTP training-only and Eagle runs keep those tensors with the policy.",
        "Weight lifecycle includes persistent buffers and explicit target/draft ownership; name routing must be verified after every refit.",
        (ref("veRL", 5801), ref("veRL", 6432), ref("Miles", 1512)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "Focused absolute-layer, Qwen mtp.*, and non-MTP retention coverage passes. A GB200 post-refit target/draft name-set and tensor-equality gate, including e_score_correction_bias, is still required.",
    ),
    GapRow(
        "5. Eagle PP hidden-state transfer used PP-local ranks as global ranks",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/megatron/draft/hidden_capture.py:194-233",
            f"{CURRENT_WORKTREE}/tests/unit/models/megatron/draft/test_hidden_capture.py:49-95",
        ),
        "Nontrivial PP subgroups could send to the wrong process, error, or deadlock. The local patch converts group-local source and destination ranks through torch.distributed.get_global_rank before metadata and payload transfers.",
        "Speculative graph and distributed-topology fixes belong at the exact communication boundary and need subgroup-specific tests.",
        (ref("SGLang/vLLM", 23037),),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "Focused subgroup mapping gate passed 2 tests for send and receive metadata plus payload rank translation.",
    ),
    GapRow(
        "6. Router replay included MTP routers",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/megatron/router_replay.py:116-148",
            f"{CURRENT_WORKTREE}/tests/unit/models/megatron/test_router_replay.py:234-274",
        ),
        "MTP bonus-token forwards have no matching rollout route records, so replaying base-policy routes into MTP routers can train or execute the wrong experts. The local patch excludes modules marked is_mtp_layer from replay discovery and local-layer accounting.",
        "Draft routing must not pollute target-policy replay, and MTP blocks need an explicit bypass.",
        (ref("Miles", 619), ref("SGLang/vLLM", 26980)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "Focused exclusion coverage exists; MCore forward/backward replay on a GB200 TP/EP/PP topology remains pending.",
    ),
    GapRow(
        "7. Fused-linear-logprob forward bypasses MTP postprocessing",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/megatron/setup.py:945-952",
            f"{CURRENT_WORKTREE}/nemo_rl/distributed/model_utils.py:2348-2395",
        ),
        "The fused path bypasses GPTModel MTP postprocessing, so MTP layers receive no loss, gradients, or metrics. The local safety patch rejects MTP plus use_fused_linear_logprobs instead of silently training only the base path.",
        "MTP loss, gradient isolation, and rollout metrics are one contract; unsupported fused paths must fail before training.",
        (ref("veRL", 6432), ref("slime", 709)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "MCore setup rejection and an unfused control run with nonzero MTP gradients and metrics remain required on GB200.",
    ),
    GapRow(
        "8. Online MTP base-head isolation was opt-in",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/megatron/setup.py:945-958",
            f"{CURRENT_WORKTREE}/tests/unit/models/megatron/test_megatron_setup.py:1288-1319",
        ),
        "Without mtp_detach_heads, draft loss can update the base decoder, embeddings, and output head and change policy accuracy. The local patch defaults the isolation on and rejects an explicit false value during MTP training.",
        "Online MTP training must isolate draft gradients from the target policy unless base-policy training is an explicit, separately validated mode.",
        (ref("veRL", 6432), ref("slime", 709)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "A zero-policy-loss backward gate must show zero gradients for all non-MTP parameters and nonzero gradients for at least one MTP parameter.",
    ),
    GapRow(
        "9. Eagle plus sequence packing omitted DraftLossWrapper",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/megatron/setup.py:935-943",
            f"{CURRENT_WORKTREE}/nemo_rl/models/megatron/train.py:448-488",
        ),
        "Hidden-state capture and draft forward could execute while student logits contributed no loss or gradient. The local patch rejects online draft training with sequence packing until a packed draft-loss path exists.",
        "Shifted draft labels and masks must preserve prompt, padding, and packed-sequence boundaries; unsupported combinations should not degrade silently.",
        (ref("slime", 751),),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "MCore config/startup rejection is pending; later support requires packed-boundary draft-loss and gradient tests.",
    ),
    GapRow(
        "10. Partial Eagle checkpoints continued after missing or unexpected keys",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/megatron/draft/utils.py:1324-1348",
            f"{CURRENT_WORKTREE}/tests/unit/models/megatron/test_megatron_setup.py:2355-2437",
        ),
        "A checkpoint containing only a subset of recognized tensors could leave draft transformer weights or d2t randomly initialized. The local patch permits only the documented LM-head fallback and rejects all other missing or unexpected mapped keys.",
        "A drafter is usable only when the complete owned parameter and buffer set is proven loaded, not when one recognized tensor happens to match.",
        (ref("Miles", 1512),),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "MCore checkpoint fixtures must prove one missing transformer or d2t key fails while LM-head-only omission takes the documented policy-head fallback.",
    ),
    GapRow(
        "11. Multi-horizon MTP masks replace rather than intersect prior masks",
        "high",
        (
            f"{CURRENT_WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/multi_token_prediction.py:840-857",
        ),
        "At deeper MTP horizons, rolling the previous mask without intersecting it can re-enable positions excluded by prompt, padding, or packed boundaries. This remains unresolved in the pinned third-party dependency.",
        "Each shifted horizon must intersect the new valid-token mask with every prior exclusion boundary.",
        (ref("slime", 751),),
        "UNRESOLVED / EXPERIMENT ISOLATED",
        "status-unresolved",
        "Carry an isolated dependency patch, then run two-plus-horizon prompt, padding, and packed-boundary parity before considering an upstream change.",
    ),
    GapRow(
        "12. Hidden-state capture rebuilds hooks and owner maps every microbatch",
        "medium",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/megatron/draft/hidden_capture.py:319-393",
        ),
        "Repeated layer scans, hook registration, PP all-gather, and CUDA item synchronizations add avoidable online Eagle critical-path overhead even when correctness is intact.",
        "Static topology metadata and graph buffers should be initialized once, reset safely, and measured rather than rebuilt per microbatch.",
        (ref("slime", 1313), ref("SGLang/vLLM", 23037)),
        "UNRESOLVED / EXPERIMENT ISOLATED",
        "status-unresolved",
        "Use nsys to quantify owner-map and hook overhead, then compare a cache-once prototype with unchanged hidden-state and gradient parity.",
    ),
    GapRow(
        "13. Generic vLLM 0.24 PARD misses one CUDA-graph key initialization path",
        "medium",
        (
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/worker/gpu_model_runner.py",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/spec_decode/llm_base_proposer.py",
        ),
        "DraftModelProposer can fall outside the graph-key initialization condition used by Eagle or hidden-state extractors, which is a plausible reason PARD loses throughput when CUDA graphs are enabled. This is an engine performance gap, not evidence of an RL correctness defect.",
        "Speculative CUDA-graph coverage requires engine-specific initialization and stale-buffer discipline; performance patches must follow token and logprob parity.",
        (ref("SGLang/vLLM", 10892), ref("SGLang/vLLM", 23037)),
        "UNRESOLVED / EXPERIMENT ISOLATED",
        "status-unresolved",
        "Keep the vLLM change in a separate experiment patch. First run scripts/check_nemorl_specdec_parity.py, then compare graph-mode coverage and generation throughput with matched PARD K and TP.",
    ),
    GapRow(
        "14. Static external draft loading previously forced the whole target to auto",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:45-51",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:165-195",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/config/speculative.py#L196-L198",
        ),
        "Loading an external drafter could unnecessarily load the full target checkpoint instead of letting refit populate a dummy target. Worse, vLLM 0.24's generic draft-model and Medusa proposer paths ignored draft_load_config and could retain dummy draft weights. The local patch keeps the target dummy and makes those proposer paths honor the independent draft loader.",
        "Draft-only load and CPU-backup behavior should be explicit so target and draft lifecycle costs do not become coupled.",
        (ref("SGLang/vLLM", 13318), ref("SGLang/vLLM", 15726)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "Three proposer-patch unit tests pass, and the patches apply and compile against the exact vLLM 0.24.0 source. A GB200 startup trace must still verify target dummy, non-dummy draft checksums, and expected memory use.",
    ),
    GapRow(
        "15. Sampling, TP, load format, and CUDA-graph behavior were implicit",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:54-143",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:214-220",
            f"{CURRENT_WORKTREE}/examples/run_grpo.py:64-82",
            f"{CURRENT_WORKTREE}/examples/run_grpo.py:123-138",
        ),
        "Invalid PARD-2 or online-refit combinations, synthetic rejection sampling, and unsupported generic-draft TP could reach runtime with no compact resolved contract. The patch rejects those combinations, allows probabilistic draft sampling, materializes generic PARD TP, and logs the resolved contract. External explicit-MTP model loading is rejected only on the dummy/refit training path and remains available for evaluation with normal checkpoint loading.",
        "The RL target distribution must retain standard rejection semantics; proposal sampling may vary only when the verifier remains exact and the effective runtime contract is observable.",
        (ref("veRL", 6432), ref("SGLang/vLLM", 15726)),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "The final config, worker, lifecycle, and launcher safety slice passed 70 tests. Use scripts/check_nemorl_specdec_parity.py for greedy exact tokens/logprobs and sampled first-token TV plus mean logprob/reward under matched sampling; tests/test_nemorl_specdec_parity.py passes 3 tests.",
    ),
    GapRow(
        "16. Online Eagle dummy startup shared the target LM head",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:208-217",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/patches.py:178-208",
            f"{CURRENT_WORKTREE}/nemo_rl/models/policy/workers/megatron_policy_worker.py:1805-1813",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/spec_decode/llm_base_proposer.py#L1394-L1443",
        ),
        "With a dummy draft loader, vLLM did not observe lm_head.weight and shared the target head into the Eagle drafter. NeMo then loaded the independently trained draft head into that shared object, overwriting target logits. The patch marks online-refit Eagle as owning its LM head before vLLM sharing and before CUDA-graph capture, and rejects non-dummy online-refit draft loading so this ownership rule cannot be bypassed by a checkpoint that omits its head. Because vLLM 0.24 does not share target embeddings into a draft across PP ranks and NeMo does not transport them, online Eagle refit also rejects PP greater than one.",
        "Draft and target parameter ownership must be established before graph capture; rejection sampling cannot repair a verifier whose target head was overwritten.",
        (ref("veRL", 5801), ref("Miles", 1512), ref("SGLang/vLLM", 15726)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "The vLLM 0.24 source patch applies and compiles, its unit test proves the ownership marker is inserted before sharing, and config coverage proves PP greater than one fails before startup. GB200 PP=1 must verify distinct target/draft LM-head storage, unchanged target checksum after refit, and token/logprob parity with CUDA graphs enabled.",
    ),
    GapRow(
        "17. Multi-chunk MTP refit validated and post-processed each chunk incorrectly",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:206-269",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:464-609",
            f"{CURRENT_WORKTREE}/tests/unit/models/generation/test_vllm_backend.py:289-369",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/models/deepseek_mtp.py#L469-L488",
        ),
        "The transport splits large refits, but vLLM's DeepSeek MTP loader requires every expected MTP layer in one invocation. Per-chunk loading could fail or leave raw MoE layouts, while only the target received final post-load processing. The patch stages only atomic MTP draft tensors on CPU, validates the complete transport, loads once, and post-processes target and drafter once. Eagle still loads each chunk immediately. Non-owner PP ranks validate receipt without loading draft tensors, collective transport synchronizes the consumer CUDA stream before finalization, and an MTP-from-refit owner now fails if no draft tensor was routed.",
        "Weight synchronization has a transaction boundary: completeness validation and kernel-layout finalization belong after all chunks arrive, not inside each chunk callback.",
        (ref("veRL", 5801), ref("Miles", 1512)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "Multi-chunk unit coverage verifies one MTP draft load, immediate Eagle chunk loading, one drafter post-process, required owner draft receipt, PP owner/non-owner handling, collective stream synchronization, and fail-closed behavior for a missing chunk. GB200 must cover IPC and collective transport with multi-layer MTP, MoE layouts, repeated refits, and CUDA-graph storage stability; it must also record per-rank host RSS and D2H time because atomic MTP staging keeps a temporary CPU copy, and compare target/draft tensor checksums because fused, aliased, and EP-skipped loader names cannot be proven complete by strict string-set equality.",
    ),
    GapRow(
        "18. Training entrypoints could bypass online-draft and MTP ownership derivation",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:149-177",
            f"{CURRENT_WORKTREE}/examples/run_grpo.py:136-151",
            f"{CURRENT_WORKTREE}/examples/nemo_gym/run_grpo_nemo_gym.py:155-170",
            f"{CURRENT_WORKTREE}/examples/run_ppo.py:84-97",
        ),
        "Only the main GRPO launcher derived online draft and MTP refit ownership; other production launchers accepted false defaults. They could skip method validation or load MTP draft state from disk even when the trainer owned it. A shared resolver now derives both flags for GRPO, NeMo-Gym, PPO, VLM GRPO, distillation, and sliding-puzzle launchers.",
        "Safety decisions must be derived from the materialized policy config at every launcher, not supplied as optional booleans that silently default to false.",
        (ref("veRL", 4936), ref("slime", 971)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "Resolver unit tests cover valid online Eagle plus MTP and missing speculative config. Launcher smoke tests must confirm identical resolved runtime contracts across standard and NeMo-Gym GRPO before the branch is promoted.",
    ),
    GapRow(
        "19. IPC refit failures could strand the ZeroMQ request/reply state",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:550-635",
            f"{CURRENT_WORKTREE}/nemo_rl/models/policy/utils.py:101-119",
            f"{CURRENT_WORKTREE}/nemo_rl/models/policy/utils.py:401-483",
            f"{CURRENT_WORKTREE}/tests/unit/models/generation/test_vllm_backend.py:580-610",
            f"{CURRENT_WORKTREE}/tests/unit/models/policy/test_utils.py:38-47",
        ),
        "A generation worker could fail after receiving a REP request but before replying, leaving the policy-side REQ blocked and both sockets unusable for later refits. The patch defines explicit ACK and ERROR replies, returns an error for a failed request, and makes every policy sender validate each chunk and completion reply immediately.",
        "Weight-update failure recovery is part of the transaction protocol: each received request must receive exactly one terminal acknowledgement or explicit failure.",
        (ref("veRL", 5801), ref("Miles", 1512)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "Unit coverage verifies ACK acceptance, ERROR propagation, and a generation-side validation failure reply. A GB200 CUDA IPC round-trip must inject a failed chunk, prove the policy fails without hanging, then prove a fresh refit session succeeds.",
    ),
)

PERFORMANCE_HYPOTHESES: tuple[GapRow, ...] = (
    GapRow(
        "Historical online PARD-2 runs looked like correctness gates, not throughput wins",
        "medium",
        (
            "docs/specdec_status_update_20260616_current.md:261-267",
            "docs/nemorl_integrated_specdec_results_clean_20260617.md:45-49",
        ),
        "The historical qwen32 online PARD-2 notes show low weighted acceptance and near-1 accepted length. That demonstrates functional training, not a proven RL speedup.",
        "slime and Miles both kept iterating after initial correctness. A passing online path is not enough if acceptance stays weak.",
        (ref("slime", 640), ref("Miles", 449)),
        "HISTORICAL / MEASUREMENT PENDING",
        "status-hypothesis",
        "Matched baseline versus online PARD-2 needs a repeatable positive E2E result before the report should claim throughput upside.",
    ),
    GapRow(
        "Historical qwen235B integrated losses still looked system-bound more than sampler-bound",
        "medium",
        (
            "docs/specdec_status_update_20260616_current.md:268-276",
            "docs/nemorl_integrated_specdec_results_clean_20260617.md:46-49",
        ),
        "The large-model historical notes emphasize watchdog, wake-up, and worker-death instability before any durable speculative RL gain is visible.",
        "veRL and SGLang repeatedly treat lifecycle, buffer handling, and wake-sleep behavior as part of speculative-decoding performance, not peripheral cleanup.",
        (ref("veRL", 6661), ref("SGLang/vLLM", 27696)),
        "HISTORICAL / MEASUREMENT PENDING",
        "status-hypothesis",
        "A clean qwen235B integrated run with stable metrics is needed before claiming sampler-side limits rather than system limits.",
    ),
)


RECORDS_BY_KEY = {record.key: record for record in EVIDENCE}


def esc(value: object) -> str:
    return html.escape(str(value))


def render_ref_links(refs: Iterable[str]) -> str:
    parts: list[str] = []
    for key in refs:
        record = RECORDS_BY_KEY[key]
        label = f"{record.kind.upper()} #{record.number}"
        parts.append(f'<a class="ref" href="{esc(record.url)}">{esc(label)}</a>')
    return "".join(parts)


def render_matrix() -> str:
    rows: list[str] = []
    for row in MATRIX_ROWS:
        cells = [f'<th class="sticky-col scope">{esc(row.topic)}</th>']
        for framework in FRAMEWORK_ORDER:
            cell = row.cells[framework]
            refs_html = render_ref_links(cell.refs)
            refs_block = f'<div class="refs">{refs_html}</div>' if refs_html else ""
            cells.append(f"<td><p>{esc(cell.summary)}</p>{refs_block}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    headers = "".join(f"<th>{esc(name)}</th>" for name in FRAMEWORK_ORDER)
    return (
        '<div class="table-wrap"><table class="matrix">'
        '<thead><tr><th class="sticky-col">Topic</th>'
        f"{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_evidence_list(items: Iterable[str]) -> str:
    rows: list[str] = []
    for item in items:
        if item.startswith("http://") or item.startswith("https://"):
            rows.append(f'<li><a href="{esc(item)}">{esc(item)}</a></li>')
        else:
            rows.append(f"<li><code>{esc(item)}</code></li>")
    return f"<ul>{''.join(rows)}</ul>"


def render_gap_table(title: str, rows: Iterable[GapRow], note: str) -> str:
    body: list[str] = []
    for row in rows:
        upstream_refs = render_ref_links(row.upstream_refs)
        upstream_refs_block = (
            f'<div class="refs">{upstream_refs}</div>' if upstream_refs else ""
        )
        body.append(
            "<tr>"
            f'<th class="sticky-col scope">{esc(row.label)}</th>'
            f'<td><span class="status {esc(row.severity)}">{esc(row.severity.upper())}</span></td>'
            f"<td>{render_evidence_list(row.current_code_evidence)}</td>"
            f"<td><p>{esc(row.user_visible_impact)}</p></td>"
            f"<td><p>{esc(row.upstream_lesson)}</p>{upstream_refs_block}</td>"
            f'<td><span class="status {esc(row.status_class)}">'
            f"{esc(row.implementation_status)}</span></td>"
            f"<td><p>{esc(row.validation_gate)}</p></td>"
            "</tr>"
        )
    return (
        f'<section><h2>{esc(title)}</h2><p class="section-note">{esc(note)}</p>'
        '<div class="table-wrap"><table class="gaps"><thead><tr>'
        '<th class="sticky-col">Gap</th><th>Severity</th><th>Current Code Evidence</th>'
        "<th>User-Visible Impact</th><th>Upstream Lesson</th><th>Local Status</th><th>Validation Gate</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table></div></section>"
    )


def render_framework_appendix(framework: str) -> str:
    framework_records = [record for record in EVIDENCE if record.framework == framework]
    counts = {
        bucket: sum(1 for record in framework_records if record.bucket == bucket)
        for bucket in STATUS_ORDER
    }
    count_parts = [
        f"{STATUS_LABELS[bucket]} {counts[bucket]}"
        for bucket in STATUS_ORDER
        if counts[bucket]
    ]
    sections: list[str] = [
        f'<section><h3>{esc(framework)}</h3><p class="section-note">'
        f"Snapshot buckets: {esc(', '.join(count_parts))}.</p>"
    ]
    for bucket in STATUS_ORDER:
        bucket_records = [
            record for record in framework_records if record.bucket == bucket
        ]
        if not bucket_records:
            continue
        rows: list[str] = []
        for record in bucket_records:
            rows.append(
                "<tr>"
                f'<th class="sticky-col scope" id="{esc(record.anchor)}">'
                f'<a href="{esc(record.url)}">{esc(record.short_id)}</a></th>'
                f'<td><span class="status {esc(record.bucket)}">{esc(STATUS_LABELS[record.bucket])}</span></td>'
                f"<td>{esc(record.title)}</td>"
                f"<td>{esc(record.author)}</td>"
                f"<td><p>{esc(record.lesson)}</p></td>"
                "</tr>"
            )
        sections.append(
            f"<h4>{esc(STATUS_LABELS[bucket])}</h4>"
            '<div class="table-wrap"><table class="appendix"><thead><tr>'
            '<th class="sticky-col">Record</th><th>Status</th><th>Title</th><th>Author</th><th>Short Lesson</th>'
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )
    sections.append("</section>")
    return "".join(sections)


def build_html() -> str:
    appendix_html = "".join(
        render_framework_appendix(name)
        for name in ("veRL", "slime", "Miles", "SGLang/vLLM")
    )
    provenance_list = "".join(
        f"<li><code>{esc(item)}</code></li>" for item in CURRENT_WORKTREE_PROVENANCE
    )
    status_counts = {
        status_class: sum(row.status_class == status_class for row in NEMO_AUDIT_GAPS)
        for status_class in ("status-unit", "status-gpu", "status-unresolved")
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpecDec RL framework lessons and NeMo-RL gaps</title>
  <style>
    :root {{
      --bg: #f4f6f9;
      --panel: #ffffff;
      --ink: #101828;
      --muted: #475467;
      --line: #d0d5dd;
      --accent: #175cd3;
      --ok: #027a48;
      --warn: #b54708;
      --bad: #b42318;
      --soft: #eaecf0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      overflow-x: hidden;
    }}
    main {{
      width: 100%;
      max-width: 1480px;
      min-width: 0;
      margin: 0 auto;
      padding: 24px 16px 48px;
    }}
    h1, h2, h3, h4 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 31px; }}
    h2 {{ font-size: 22px; margin-top: 28px; }}
    h3 {{ font-size: 18px; margin-top: 22px; }}
    h4 {{ font-size: 15px; margin: 16px 0 8px; }}
    p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    ul {{
      margin: 8px 0 0 18px;
      padding: 0;
      color: var(--muted);
    }}
    li {{ margin: 4px 0; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
      overflow-wrap: anywhere;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .lede {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.85fr);
      gap: 18px;
      align-items: start;
      margin-top: 14px;
    }}
    .lede-block {{
      min-width: 0;
      padding: 0;
    }}
    .snapshot {{
      display: inline-flex;
      min-height: 28px;
      align-items: center;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }}
    .meta-list {{
      list-style: none;
      margin: 0;
      padding: 0;
    }}
    .meta-list li + li {{ margin-top: 6px; }}
    .meta-list code {{
      display: block;
      max-width: 100%;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .table-wrap {{
      width: 100%;
      max-width: 100%;
      min-width: 0;
      margin-top: 10px;
      overflow-x: auto;
      border: 1px solid var(--line);
      background: var(--panel);
    }}
    table {{
      width: 100%;
      min-width: 1180px;
      border-collapse: separate;
      border-spacing: 0;
    }}
    .matrix {{ min-width: 1460px; }}
    .gaps {{ min-width: 1520px; }}
    .appendix {{ min-width: 1100px; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      background: var(--panel);
      font-size: 13px;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: #f8fafc;
    }}
    .sticky-col {{
      position: sticky;
      left: 0;
      z-index: 2;
      background: var(--panel);
    }}
    thead .sticky-col {{
      z-index: 4;
      background: #f8fafc;
    }}
    tr:last-child td, tr:last-child th {{ border-bottom: 0; }}
    tr td:last-child, tr th:last-child {{ border-right: 0; }}
    .scope {{
      min-width: 220px;
      max-width: 220px;
    }}
    .section-note {{
      margin-top: 6px;
      font-size: 13px;
    }}
    .refs {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }}
    .ref {{
      display: inline-flex;
      min-height: 24px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #f8fafc;
      font-size: 11px;
      font-weight: 700;
    }}
    .status {{
      display: inline-flex;
      min-height: 24px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 800;
      line-height: 1.25;
      text-align: center;
      background: #f8fafc;
    }}
    .status.merged {{ color: var(--ok); }}
    .status.open {{ color: var(--warn); }}
    .status.unresolved {{ color: var(--bad); }}
    .status.critical {{ color: var(--bad); }}
    .status.high {{ color: var(--bad); }}
    .status.medium {{ color: var(--warn); }}
    .status.resolved {{ color: var(--ok); }}
    .status.status-unit {{ color: var(--ok); border-color: #6ce9a6; background: #ecfdf3; }}
    .status.status-gpu {{ color: var(--warn); border-color: #fec84b; background: #fffaeb; }}
    .status.status-unresolved {{ color: var(--bad); border-color: #fda29b; background: #fef3f2; }}
    .status.status-hypothesis {{ color: #344054; border-color: #98a2b3; background: #f2f4f7; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .note {{
      margin-top: 16px;
      padding: 12px 0 0;
      border-top: 1px solid var(--line);
    }}
    @media (max-width: 1024px) {{
      html, body, main {{
        max-width: 100vw;
      }}
      .lede {{
        grid-template-columns: 1fr;
      }}
      main {{
        padding: 18px 10px 36px;
      }}
      h1, .lede, section, .table-wrap {{
        width: calc(100vw - 20px);
        max-width: calc(100vw - 20px);
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>SpecDec RL framework lessons and NeMo-RL gaps</h1>
  <div class="lede">
    <div class="lede-block">
      <span class="snapshot">{esc(SNAPSHOT_LABEL)}</span>
      <p>This page is a deterministic offline snapshot. The appendix records are the authoritative merged/open/unresolved buckets reviewed on 2026-07-09. Current NeMo code proof is the <strong>pushed safety branch</strong> at <code>{
        esc(CURRENT_WORKTREE_COMMIT)
    }</code>, based on upstream main <code>{
        esc(CURRENT_UPSTREAM_COMMIT)
    }</code>. It is not represented as merged or upstreamed.</p>
      <p>The main matrix compares veRL, slime, Miles, SGLang/vLLM, and the matching NeMo-RL action. The deep audit exposes every confirmed defect, local patch state, and remaining validation gate in table rows rather than hiding them in prose.</p>
      <div class="legend" aria-label="Audit status legend">
        <span class="status status-unit">{
        status_counts["status-unit"]
    } PATCHED LOCALLY / UNIT VERIFIED</span>
        <span class="status status-gpu">{
        status_counts["status-gpu"]
    } PATCHED LOCALLY / GPU GATE PENDING</span>
        <span class="status status-unresolved">{
        status_counts["status-unresolved"]
    } UNRESOLVED / EXPERIMENT ISOLATED</span>
      </div>
    </div>
    <div class="lede-block">
      <ul class="meta-list">
        <li><code>current proof root: {esc(CURRENT_WORKTREE)}</code></li>
        <li><code>commit: {esc(CURRENT_WORKTREE_COMMIT)}</code></li>
        <li><code>upstream main: {esc(CURRENT_UPSTREAM_COMMIT)}</code></li>
        <li><code>vLLM v0.24.0: {esc(VLLM_024_COMMIT)}</code></li>
        {provenance_list}
      </ul>
    </div>
  </div>

  <section>
    <h2>Main matrix</h2>
    <p class="section-note">Each framework cell summarizes the reviewed lesson and links directly to the underlying PR or issue.</p>
    {render_matrix()}
  </section>

  {
        render_gap_table(
            "Deep NeMo-RL audit and action gates",
            NEMO_AUDIT_GAPS,
            f"These {len(NEMO_AUDIT_GAPS)} rows trace confirmed defects, the pushed safety patch at {CURRENT_WORKTREE_COMMIT[:8]}, direct upstream lessons, and the gate still required before merge or performance claims.",
        )
    }

  {
        render_gap_table(
            "Performance hypotheses",
            PERFORMANCE_HYPOTHESES,
            "These rows are historical-context hypotheses. They explain why the report does not overclaim speedups from older notes.",
        )
    }

  <section>
    <h2>Research notes</h2>
    <p class="section-note">b8zhong had SGLang-side work in the broader research trail, but this reviewed snapshot found no veRL, slime, or Miles appendix record involving b8zhong. Cross-framework proof on this page therefore comes from the mandatory record sets above, not from b8zhong-adjacent SGLang-only context.</p>
    <p class="section-note">Historical SGLang-only context links: <a href="https://github.com/sgl-project/sglang/issues/17371">sglang issue #17371</a>, <a href="https://github.com/sgl-project/sglang/issues/27286">sglang issue #27286</a>.</p>
  </section>

  <section>
    <h2>Appendices</h2>
    <p class="section-note">Complete mandatory evidence snapshot with direct GitHub links and explicit merged/open/unresolved labels.</p>
    {appendix_html}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    report = build_html()
    OUT.write_text(report, encoding="utf-8")
    PUBLIC_OUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
