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
CURRENT_WORKTREE_COMMIT = "4a776aa7579ea85d8be0bc7ea433138d8aa84d38"
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
        "The patch removed draft CPU backup and forced FA3 to avoid a FlashInfer IMA; the cross-step OOM in issue #1022 remained open.",
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
    EvidenceRecord(
        "SGLang/vLLM",
        "sgl-project/sglang",
        27718,
        "issue",
        "open",
        "[Bug] EAGLE/NEXTN/MTP draft weights never updated by update_weights_from_distributed",
        "ElliotXinqiWang",
        "A target-only distributed update can silently leave the online drafter stale while weight versions continue to advance.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "sgl-project/sglang",
        28257,
        "pr",
        "open",
        "[Fix] Update speculative draft weights on update_weights_from_distributed",
        "waynehacking8",
        "The proposed repair broadcasts once, then applies the received tensors to both target and draft owners.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "sgl-project/sglang",
        29144,
        "issue",
        "open",
        "EAGLE/NextN plus DP attention draft-subcommunicator deadlock",
        "WangTuoxytt",
        "Idle DP ranks can violate draft collective ordering and deadlock the first subcommunicator use.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        46725,
        "pr",
        "open",
        "Runtime Draft Weight Update for Speculative Decoding",
        "vx120",
        "Runtime updates need an explicit target-versus-draft ownership contract and lifecycle-safe restoration.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        44697,
        "issue",
        "open",
        "[RFC] MTP speculative decoding under pipeline parallelism (PP>1)",
        "atassis",
        "MTP PP requires sampler-count and draft-token broadcast plus accepted-token rollback on every pipeline stage.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        44698,
        "pr",
        "open",
        "[Spec][PP] Support MTP speculative decoding under pipeline parallelism",
        "atassis",
        "The V1 proposal implements the missing PP accounting and broadcast contract but remains open.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        46994,
        "pr",
        "open",
        "[Spec][V2] Support MTP speculative decoding under pipeline parallelism",
        "eastwood-c",
        "Model Runner V2 needs its own PP handler for speculative token and rollback state.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        44869,
        "pr",
        "open",
        "Fail closed on missing speculative draft probabilities",
        "masterFoad",
        "A missing cached q(token) row must abort probabilistic draft verification rather than silently change rejection behavior.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        47616,
        "pr",
        "open",
        "Trim token IDs and logprobs left past a stop string under speculative decoding",
        "Sunt-ing",
        "Token-count parity alone does not detect semantically invalid output retained beyond a matched stop string.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        44845,
        "pr",
        "open",
        "Fix discarded speculative rows leaking logprobs",
        "masterFoad",
        "Discarded speculative rows must clear their logprob state before trajectory output is materialized.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        37521,
        "pr",
        "open",
        "Fix speculative sampler warmup OOM when using EAGLE",
        "wangyxbh",
        "Warmup must profile the K-token rejection-sampler shape instead of one draft token per request.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        38220,
        "pr",
        "open",
        "Fix sampler peak memory utilization when using speculative decoding",
        "Flechman",
        "Speculative rejection-sampler peak memory must be included in available KV-cache capacity.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        48055,
        "pr",
        "open",
        "Fix int32 offset overflow in rejection sampler kernels",
        "paulsbrookes",
        "Large max-sequence, K, and vocabulary products can overflow int32 rejection-sampler offsets.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        47460,
        "pr",
        "open",
        "Initialize draft CUDA-graph keys for the native draft_model proposer",
        "avalliappan-nvidia",
        "The native generic draft proposer needs explicit dispatcher-key initialization before CUDA-graph replay can occur.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        46324,
        "pr",
        "open",
        "Align speculative-decode capture sizes for PIECEWISE mode",
        "davidzha712",
        "PIECEWISE capture sizes must include speculative K+1 verification shapes and partial-acceptance transitions.",
    ),
    EvidenceRecord(
        "SGLang/vLLM",
        "vllm-project/vllm",
        40768,
        "pr",
        "open",
        "Fix stale async placeholder tokens in speculative decoding",
        "z1ying",
        "Preemption and retries must clear speculative placeholder state before the next embedding lookup.",
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
                "SGLang/vLLM exposes draft-only CPU backup and update controls, while open work shows that distributed target updates can still leave the drafter stale.",
                (
                    ref("SGLang/vLLM", 13318),
                    ref("SGLang/vLLM", 15726),
                    ref("SGLang/vLLM", 27696),
                    ref("SGLang/vLLM", 27718),
                    ref("SGLang/vLLM", 28257),
                    ref("SGLang/vLLM", 46725),
                ),
            ),
            "NeMo-RL gap": MatrixCell(
                "The safety branch separates target and draft loading, supports V1 drafter and V2 speculator owners, rejects empty refit manifests, requires a loader receipt, and keeps online Eagle ownership explicit. Per-version target/draft checksums and a target/draft/all update selector remain unresolved.",
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
                "SGLang/vLLM focuses on whether and when the draft model updates, including the still-open distributed online-RL update path.",
                (
                    ref("SGLang/vLLM", 15726),
                    ref("SGLang/vLLM", 26980),
                    ref("SGLang/vLLM", 27718),
                    ref("SGLang/vLLM", 28257),
                    ref("SGLang/vLLM", 46725),
                ),
            ),
            "NeMo-RL gap": MatrixCell(
                "Online refit is limited to Eagle methods and policy/vLLM PP=1; unsafe fused-logprob, sequence-packing, and non-detached MTP combinations fail before training. End-to-end draft checksum and optimizer-delta gates remain pending.",
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
                "Open vLLM fixes show that missing q(token), post-stop tokens, and discarded-row logprobs can each invalidate otherwise well-shaped speculative output.",
                (
                    ref("SGLang/vLLM", 44869),
                    ref("SGLang/vLLM", 47616),
                    ref("SGLang/vLLM", 44845),
                ),
            ),
            "NeMo-RL gap": MatrixCell(
                "Sync and async paths require one finite chosen-token logprob per generated token, and a partial probabilistic-draft cache now fails closed. Per-sample stop strings are still merged into one shared SamplingParams object, while stop-boundary, discarded-row, streaming, and preemption parity remain GPU gates.",
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
                "Merged IMA fixes and open native-draft/capture-alignment patches show that key initialization and K+1 shape coverage are separate requirements.",
                (
                    ref("SGLang/vLLM", 10892),
                    ref("SGLang/vLLM", 23037),
                    ref("SGLang/vLLM", 47460),
                    ref("SGLang/vLLM", 46324),
                ),
            ),
            "NeMo-RL gap": MatrixCell(
                "An exact-v0.24 generic-draft dispatcher patch is available only behind NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH. It stays disabled until token/logprob parity, partial-acceptance graph hits, and matched PARD throughput pass.",
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
                "SGLang/vLLM treats lifecycle buffers and rejection-sampler peak memory as explicit runtime surfaces rather than relying on target-only profiling.",
                (
                    ref("SGLang/vLLM", 13318),
                    ref("SGLang/vLLM", 27696),
                    ref("SGLang/vLLM", 37521),
                    ref("SGLang/vLLM", 38220),
                    ref("SGLang/vLLM", 48055),
                ),
            ),
            "NeMo-RL gap": MatrixCell(
                "MTP persistent buffers follow drafter ownership and sleep/wake uses level 1. Full-K warmup memory, int32 offset limits, long-context capacity, and release/resume GPU gates remain required.",
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
                "SGLang/vLLM exposes direct recovery failures: draft communicator deadlock on idle ranks and stale speculative placeholders after preemption.",
                (
                    ref("SGLang/vLLM", 10892),
                    ref("SGLang/vLLM", 15726),
                    ref("SGLang/vLLM", 27696),
                    ref("SGLang/vLLM", 29144),
                    ref("SGLang/vLLM", 40768),
                ),
            ),
            "NeMo-RL gap": MatrixCell(
                "Empty or partial collective results, missing drafters, failed wake-up calls, and generation exceptions now fail instead of reporting success. A worker that sees a failed refit cannot generate again without restart. Transaction rollback, PP rank translation validation, and MCore/GB200 resume gates remain pending.",
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
                "SGLang excludes draft routing from target replay, while vLLM's open V1/V2 work documents the token accounting required for MTP PP correctness.",
                (
                    ref("SGLang/vLLM", 23037),
                    ref("SGLang/vLLM", 26980),
                    ref("SGLang/vLLM", 44697),
                    ref("SGLang/vLLM", 44698),
                    ref("SGLang/vLLM", 46994),
                ),
            ),
            "NeMo-RL gap": MatrixCell(
                "The safety worktree translates PP-local ranks and excludes MTP routers from replay, but rejects vLLM MTP PP>1 until sampler-count, draft-token broadcast, and rollback support is complete.",
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
        "DraftModelProposer falls outside the graph-key initialization condition used by Eagle or hidden-state extractors, so its dispatcher remains eager even when the target is graphed. The local exact-v0.24 source mutation installs once but checks the opt-in environment flag inside the runtime branch, so enabled and disabled runs remain isolated in a shared venv.",
        "Speculative CUDA-graph coverage requires engine-specific initialization and stale-buffer discipline; performance patches must follow token and logprob parity.",
        (
            ref("SGLang/vLLM", 10892),
            ref("SGLang/vLLM", 23037),
            ref("SGLang/vLLM", 47460),
            ref("SGLang/vLLM", 46324),
        ),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "The runtime-guarded source patch applies twice idempotently and compiles against exact vLLM 0.24.0. Startup records that the patch was requested, not that a graph replay occurred. Before enabling it, run greedy token/logprob parity, sampled distribution parity, partial-acceptance capture-shape checks, and TP1/TP2 graph-hit plus matched-throughput comparisons.",
    ),
    GapRow(
        "14. Static external draft loading previously forced the whole target to auto",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:45-51",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:165-195",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/config/speculative.py#L196-L198",
        ),
        "Loading an external drafter could unnecessarily load the full target checkpoint instead of letting refit populate a dummy target. Worse, vLLM 0.24's generic draft-model, Medusa, Model Runner V2 Eagle, and Model Runner V2 DFlash paths could ignore draft_load_config and retain dummy draft weights. The local patch keeps the target dummy and makes those proposer paths honor the independent draft loader.",
        "Draft-only load and CPU-backup behavior should be explicit so target and draft lifecycle costs do not become coupled.",
        (ref("SGLang/vLLM", 13318), ref("SGLang/vLLM", 15726)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "Generic, Medusa, V2 Eagle, and V2 DFlash proposer-patch unit tests pass, and the patches apply and compile against exact vLLM 0.24.0. A GB200 startup trace must still verify target dummy, non-dummy draft checksums, and expected memory use.",
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
    GapRow(
        "20. vLLM Model Runner V2 used a different draft owner and ignored independent Eagle and DFlash loaders",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:269-297",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/patches.py:208-288",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/worker/gpu/spec_decode/eagle/utils.py",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/worker/gpu/spec_decode/dflash/utils.py",
        ),
        "Qwen3 and Llama can use Model Runner V2, where the draft owner is speculator rather than drafter. Its Eagle and DFlash loaders inherited the dummy target load config, so online refit could report no drafter and static external drafters could start from random weights. The patch resolves both owner names, passes draft_load_config to both V2 loaders, and establishes online Eagle head ownership before sharing and capture.",
        "Runtime-generation variants must share one explicit draft ownership and loading contract; the target's dummy loader must never leak into a static external drafter.",
        (ref("SGLang/vLLM", 27718), ref("SGLang/vLLM", 46725)),
        "PATCHED LOCALLY / GPU GATE PENDING",
        "status-gpu",
        "Unit tests cover V2 owner resolution plus Eagle and DFlash source transformations; all ten source patches apply idempotently and compile on exact vLLM 0.24.0. GB200 must verify known static and online draft checksums, distinct target/draft head storage, level-1 sleep/wake, CUDA-graph generation, and unchanged target logits after refit.",
    ),
    GapRow(
        "21. Missing probabilistic draft rows silently changed rejection behavior",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/patches.py:257-290",
            "https://github.com/vllm-project/vllm/pull/44869",
        ),
        "vLLM 0.24 returned None when one drafted request lacked its cached q(token) row, silently falling back to legacy rejection behavior. The local source patch raises before trajectory output while preserving the intentional no-probability path for methods that never provide q(token).",
        "Probabilistic proposal sampling is valid only when every drafted request reaches verification with the matching proposal distribution.",
        (ref("SGLang/vLLM", 44869),),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "The transformation test proves only the partial-row fallback is replaced, and exact-source apply plus py_compile passes. GPU parity must still cover request reorder, preemption, and a deliberately missing cached row.",
    ),
    GapRow(
        "22. Online Eagle and MTP PP configurations exceeded the proven ownership contract",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:111-132",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:201-237",
        ),
        "vLLM MTP PP greater than one can crash or silently diverge because only the sampler rank owns accepted-token counts and draft tokens. Online Eagle also assumed both vLLM and policy Megatron PP equal one. The safety patch rejects all three unproven PP surfaces, and requires an explicit method for model auto-detection at PP greater than one so an MTP model cannot bypass the gate.",
        "Loader ownership is not enough for PP correctness; every pipeline stage must receive the sampler's token/count state and apply identical rollback.",
        (
            ref("SGLang/vLLM", 44697),
            ref("SGLang/vLLM", 44698),
            ref("SGLang/vLLM", 46994),
        ),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "Config tests cover vLLM MTP PP2, vLLM online Eagle PP2, policy Megatron PP2, and auto-detected-model PP2 rejection. PP support remains blocked until the full V1 or V2 broadcast, reconstruction, correction, and rollback patch set is carried and tested on multiple ranks.",
    ),
    GapRow(
        "23. Static dummy drafters and empty refit manifests could look like successful initialization",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:88-98",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:228-257",
        ),
        "A static neural drafter with draft_load_format=dummy can preserve target-distribution correctness under standard rejection but destroys acceptance and performance. An empty target manifest could also ACK an update without proving any target coverage. Both now fail before loading or post-processing.",
        "A valid update must identify real owned weights before version advancement; random draft state and empty target updates are not successful runs.",
        (ref("SGLang/vLLM", 27718), ref("Miles", 1360)),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "Train/eval static-dummy tests and prepare/begin empty-manifest tests pass. Full model-aware target/draft checksum coverage remains a GPU validation requirement because packed, aliased, and EP-sharded parameter names cannot be proven by naive string-set equality.",
    ),
    GapRow(
        "24. MiMo MTP routing and selective checkpoint loading omitted model.mtp_layers namespaces",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:89-159",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/models/mimo_mtp.py",
        ),
        "Both refit routing and disk loading recognized mtp.* and absolute transformer layers but not model.mtp_layers.*. Valid MiMo online-refit tensors could reach the target loader and be skipped, while valid MiMo or single-file checkpoints could fail before generation. Both paths now route the explicit namespace to the drafter, and the reader supports local sharded or single-file safetensors without loading unrelated base weights.",
        "Model-family MTP ownership must be expressed by checkpoint namespaces rather than assuming every drafter is encoded as extra base-model layers.",
        (ref("slime", 693), ref("Miles", 1289)),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "Refit routing tests cover mtp.* and model.mtp_layers.*, and four selective-reader tests cover absolute layers, both explicit namespaces, and a single safetensors file. Hugging Face repository IDs remain unsupported by this selective local reader and must be resolved to a local snapshot before startup.",
    ),
    GapRow(
        "25. Stop boundaries and discarded speculative rows can still emit semantically invalid behavior data",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:498-534",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker_async.py:1041-1076",
        ),
        "One logprob per returned token does not detect tokens retained after a matched stop string or stale logprobs from rows discarded during speculative verification. The pinned engine still has open fixes for both cases.",
        "Behavior-data validity includes semantic sequence boundaries, not only matching tensor lengths and finite values.",
        (ref("SGLang/vLLM", 47616), ref("SGLang/vLLM", 44845)),
        "UNRESOLVED / EXPERIMENT ISOLATED",
        "status-unresolved",
        "Run autoregressive-versus-SpecDec parity with stop strings at every draft position, partial acceptance, discarded rows, streaming, preemption, and retries. Do not enable nonstandard rejection or claim accuracy parity until these cases pass.",
    ),
    GapRow(
        "26. Refit lacks a target/draft/all selector, committed version, and end-to-end checksums",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:299-337",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:557-624",
        ),
        "Transport completeness is checked against the sender manifest, but a self-consistent incomplete exporter can still update only a subset. NeMo-RL does not yet publish independent target and draft checksums or commit a shared weight version only after both owners acknowledge the same update.",
        "Online RL needs a two-owner transaction: explicit update selection, per-owner coverage proof, and version advancement only after target and draft both commit.",
        (
            ref("SGLang/vLLM", 27718),
            ref("SGLang/vLLM", 28257),
            ref("SGLang/vLLM", 46725),
            ref("Miles", 1360),
        ),
        "UNRESOLVED / EXPERIMENT ISOLATED",
        "status-unresolved",
        "Define target, draft, and all update modes; record pre/post checksums and loaded/skipped aliases per owner; inject one-owner failure; and prove the version remains uncommitted until every required owner returns success across TP, PP, and EP.",
    ),
    GapRow(
        "27. Qwen3 Eagle-3 and DFlash loaders did not return an auditable load receipt",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_backend.py:65-82",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/patches.py:290-320",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/models/qwen3_eagle3.py",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/models/qwen3_dflash.py",
        ),
        "A receipt-less loader made a complete load, a partial load, and a silent skip indistinguishable. The safety patch rejects None and patches the exact vLLM 0.24 Qwen3 Eagle-3 and DFlash loaders to return the loaded-name set after required post-load buffer construction.",
        "A drafter update is valid only when the receiver independently reports what it loaded; sender-name equality alone cannot prove packed, aliased, PP-skipped, or EP-sharded coverage.",
        (ref("veRL", 5801), ref("Miles", 1512), ref("SGLang/vLLM", 46725)),
        "PATCHED LOCALLY / LINUX GPU GATE RUNNING",
        "status-gpu",
        "Exact vLLM 0.24 source apply, compile, and idempotence pass locally. Lyris Qwen3-32B baseline job 2322955 completed one full step. The unsafe Eagle job 2322962 was cancelled after its log proved async scheduling was enabled; corrected Eagle-3 K5 job 2323011 uses commit 4a776aa7. Matched token/logprob parity and a known draft checksum are still required.",
    ),
    GapRow(
        "28. Failed refit workers could be reused after partial mutation",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:77-95",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:962-1016",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker_async.py:1428-1491",
            f"{CURRENT_WORKTREE}/nemo_rl/algorithms/grpo.py:2030-2034",
            f"{CURRENT_WORKTREE}/nemo_rl/algorithms/grpo.py:2120-2124",
        ),
        "A rank-local load or wake failure could leave target or draft storage partially changed and still allow later generation. Sync and async workers now enter a terminal invalid state after any failed refit result or exception, and explicit false wake results abort the refit.",
        "A failed distributed update cannot be repaired by retrying through the same process state unless rollback is proven; restart is the minimum fail-closed boundary.",
        (ref("veRL", 5801), ref("veRL", 6661), ref("Miles", 1512)),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "Focused sync and async IPC/collective tests prove generation is rejected after failure, and weights/kv-cache wake failures abort. Full rollback is intentionally not claimed; inject a late-rank GB200 failure and verify every actor exits rather than hanging or emitting a trajectory.",
    ),
    GapRow(
        "29. Async rollout generation exceptions could still produce a successful SLURM job",
        "critical",
        (f"{CURRENT_WORKTREE}/nemo_rl/experience/rollouts.py:1010-1013",),
        "The sample loop printed an EngineCore or RPC failure and broke out, which could convert an empty or truncated trajectory into a nominally completed batch and let SLURM report COMPLETED. The exception now carries sample and turn context and aborts the run.",
        "Failure recovery must preserve failure semantics through the RL orchestration layer; logging an exception is not equivalent to failing the transaction.",
        (ref("SGLang/vLLM", 40768),),
        "PATCHED LOCALLY / UNIT VERIFIED",
        "status-unit",
        "A TimeoutError regression test proves the rollout raises instead of returning a partial sample. The Lyris smoke must now report FAILED if EngineCore dies; success requires an actual completed optimization step, not only SLURM exit code zero.",
    ),
    GapRow(
        "30. Runtime stop strings are merged across samples and can be ignored without tokenizer initialization",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:498-534",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker_async.py:1041-1076",
        ),
        "NeMo-RL currently unions disjoint sample stop strings into one shared SamplingParams object, so one sample can terminate on another sample's marker. Runtime stop strings can also be accepted while tokenizer initialization remains skipped, leaving the engine unable to enforce them correctly.",
        "Speculative verification must preserve each request's semantic stopping boundary; vLLM supports per-prompt SamplingParams and the RL adapter must not collapse them into a batch-wide union.",
        (ref("SGLang/vLLM", 47616),),
        "UNRESOLVED / PARITY BLOCKER",
        "status-unresolved",
        "Add disjoint per-sample SamplingParams and require tokenizer support whenever string stops are present. Test accepted bursts with K at least 3 where a stop occurs at every draft position, then compare emitted token IDs and chosen-token logprobs against autoregressive decoding.",
    ),
    GapRow(
        "31. vLLM 0.24 async scheduling can retain stale speculative placeholder tokens",
        "critical",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:209-210",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/__init__.py:286-294",
            "https://github.com/vllm-project/vllm/pull/40768",
        ),
        "vLLM 0.24 enabled async scheduling in Lyris Eagle smoke 2322962 while the upstream fix for stale -1 placeholder tokens remains open. The safety patch now forces async_scheduling=false for every SpecDec config and records the resolved value in the startup contract; non-SpecDec configs retain their requested behavior.",
        "Preemption and retry state must clear every speculative placeholder before the request returns to scheduling; performance defaults cannot override an unresolved correctness invariant.",
        (ref("SGLang/vLLM", 40768),),
        "PATCHED LOCALLY / GPU GATE RUNNING",
        "status-gpu",
        "Two focused tests and the 77-test safety regression pass. Corrected Lyris Eagle job 2323011 must show async_scheduling=false in both the NeMo runtime contract and vLLM engine config, then complete a real optimization step. Preemption, discard, retry, and request-reordering parity remain required before any future re-enable.",
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
    GapRow(
        "PARD CUDA-graph losses combine eager drafting with uncaptured K+1 verification shapes",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/patches.py:355-408",
            f"{CURRENT_WORKTREE}/experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh:140-165",
            "https://github.com/vllm-project/vllm/pull/47460",
            "https://github.com/vllm-project/vllm/pull/46324",
        ),
        "The target may replay CUDA graphs while the generic PARD drafter remains eager, and verification at approximately B times K+1 can exceed every configured capture size. For 128 requests, K5, K7, and K9 require 768, 1024, and 1280 verification tokens; for 256 requests they require 1536, 2048, and 2560. Those shapes exceed a default capture ceiling near 512, so CUDA graph enabled is not proof that either speculative subpath hit a graph.",
        "Graph-key initialization and capture-shape coverage are independent. Dynamic K can look better partly because it moves execution back under a capture threshold.",
        (ref("SGLang/vLLM", 47460), ref("SGLang/vLLM", 46324)),
        "PATCHED LOCALLY / MEASUREMENT PENDING",
        "status-hypothesis",
        "Profile TP1/TP2, B={1,8,32,64}, eager versus stock PIECEWISE versus the opt-in dispatcher patch. Record target/draft cudaGraphLaunch counts, partial-acceptance shapes, GPU idle gaps, TPOT, and identical output parity.",
    ),
    GapRow(
        "PARD speculative-slot reservation can collapse the scheduler token budget",
        "high",
        (
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:291-316",
            f"{CURRENT_WORKTREE}/nemo_rl/models/generation/vllm/vllm_worker.py:455-477",
            "https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/config/vllm.py",
        ),
        "Parallel drafting reserves K new slots for every allowed sequence. With max_num_batched_tokens=16384, max_num_seqs=1024, and K=12, only 4096 scheduler tokens remain, causing extra prefill iterations and low occupancy before draft compute is considered.",
        "Scheduler capacity must be derived from actual per-engine RL concurrency, while asynchronous serving configurations must retain their own larger concurrency contract.",
        (ref("SGLang/vLLM", 37521), ref("SGLang/vLLM", 38220)),
        "CONFIG EXPERIMENT / MEASUREMENT PENDING",
        "status-hypothesis",
        "Hold max_num_batched_tokens fixed and compare inherited max_num_seqs with the real per-engine rollout concurrency. Record effective token budget, scheduler iterations, prefill forwards, peak memory, and generation throughput before changing shared defaults.",
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
    .security-note {{
      margin-top: 14px;
      padding: 10px 12px;
      border-left: 4px solid var(--bad);
      background: #fef3f2;
      color: #7a271a;
      font-size: 13px;
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
      <p class="security-note"><strong>Operational blocker:</strong> a W&amp;B credential was observed in persisted Ray logs during the cluster audit. Rotate the credential before long runs and stop persisting full environment dumps. The credential is intentionally not reproduced here.</p>
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
