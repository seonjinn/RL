"""Backport of vLLM #58762, #58763 and #57102 onto vLLM 0.29 as an import-time source patch.

vLLM 0.29's V2 model runner builds Gated DeltaNet metadata once per KV cache group on every
step, and the spec-decode build path costs about 0.45 ms of CPU per group. A hybrid target
with a drafter can have dozens of GDN groups (Qwen3.5-35B-A3B with DFlash: 30 of 46), so the
metadata build adds 10-14 ms to every step. GDN groups with the same spec and builder differ
only in their state indices. Upstream builds the metadata for the first such group and
re-gathers the state indices for the others (#58762, v0.31.0), slices instead of gathering
with the host mask on pure spec-decode steps (#58763, v0.32.0), and shares the
token-to-request mapping across groups (#57102, v0.30.0).

The patch rewrites the two module sources as they are imported, so the installed vLLM files
stay untouched. Every anchor must match exactly once, or the import fails. Other vLLM
versions are left alone, because 0.32+ already ships the change.
"""

import importlib.abc
import importlib.machinery
import importlib.metadata
import sys
from collections.abc import Sequence
from types import CodeType, ModuleType

SUPPORTED_VLLM = "0.29."
# Modules this process loaded with the backport, for logs and bench records
PATCHED: list[str] = []

GDN_INIT_REUSE = """\
        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        # update_block_table() keeps the source group's batch-level FULL graph
        # buffers, so only MRV2, which also reuses metadata at capture, may use
        # it.
        self.supports_update_block_table = (
            vllm_config.use_v2_model_runner
            and device.type == "cuda"
            # Not isinstance: KDA's RecoverSSM/checkpoint metadata is per group.
            and type(self) is GDNAttentionMetadataBuilder
            # Align mode needs MRV2's aligned-index precompute, which 0.29 lacks for GDN.
            and vllm_config.cache_config.mamba_cache_mode != "align"
        )
        # attn_utils reuses metadata only for builders with this marker. 0.29's
        # Mamba1/Mamba2 update_block_table was written for the V1 runner.
        self.reuses_group_metadata = self.supports_update_block_table
"""

GDN_UPDATE_BLOCK_TABLE = """\
    def _stage_spec_decode(
        self,
        num_prefills: int,
        num_decodes: int,
        num_spec_decodes: int,
        num_spec_decode_tokens: int,
    ) -> bool:
        \"\"\"Whether spec-decode metadata goes into the FULL cudagraph buffers.\"\"\"
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _stage_decode(
        self, num_prefills: int, num_decodes: int, num_spec_decodes: int
    ) -> bool:
        \"\"\"Whether decode metadata goes into the FULL cudagraph buffers.\"\"\"
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        )

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        \"\"\"Re-gather this group's state indices. The other fields are
        batch-level and stay shared with ``metadata``.\"\"\"
        m = metadata
        masks = m.spec_sequence_masks_cpu
        spec_indices = non_spec_indices = prefill_indices = None
        if masks is None:
            non_spec_indices = blk_table[:, 0]
            if m.num_prefills > 0:
                prefill_indices = non_spec_indices[m.num_decodes :]
        elif m.num_prefills == 0:
            # Same as build(): padded sequences trail the spec decodes.
            spec_indices = blk_table[: m.num_spec_decodes, : self.num_spec + 1]
        else:
            spec_indices = blk_table[masks, : self.num_spec + 1]
            non_spec_indices = prefill_indices = blk_table[~masks, 0]

        if self._stage_spec_decode(
            m.num_prefills, m.num_decodes, m.num_spec_decodes, m.num_spec_decode_tokens
        ):
            assert m.spec_state_indices_tensor is not None
            batch_size = m.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:batch_size]
            spec_indices[m.num_spec_decodes :].fill_(NULL_BLOCK_ID)

        if self._stage_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes):
            assert m.non_spec_state_indices_tensor is not None
            batch_size = m.non_spec_state_indices_tensor.shape[0]
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices, non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        return replace(
            m,
            spec_state_indices_tensor=spec_indices,
            non_spec_state_indices_tensor=non_spec_indices,
            prefill_state_indices=prefill_indices,
        )

    def build_for_cudagraph_capture(
"""

PATCHES: dict[str, Sequence[tuple[str, str]]] = {
    "vllm.v1.attention.backends.gdn_attn": (
        (
            "from dataclasses import dataclass\n",
            "from dataclasses import dataclass, replace\n",
        ),
        (
            "    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]\n",
            "    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]\n"
            "    spec_sequence_masks_cpu: torch.Tensor | None = None  # shape: [batch,]\n",
        ),
        (
            "        self.use_full_cuda_graph: bool = (\n"
            "            self.compilation_config.cudagraph_mode.has_full_cudagraphs()\n"
            "        )\n",
            GDN_INIT_REUSE,
        ),
        (
            "                # Filter by spec_sequence_masks to exclude padded sequences\n"
            "                spec_state_indices_tensor = block_table_tensor[\n"
            "                    spec_sequence_masks_cpu, : self.num_spec + 1\n"
            "                ]\n",
            "                # Padded sequences trail the spec decodes, so slice them off\n"
            "                # rather than gather with the host mask (an H2D copy + a kernel).\n"
            "                spec_state_indices_tensor = block_table_tensor[\n"
            "                    :num_spec_decodes, : self.num_spec + 1\n"
            "                ]\n",
        ),
        (
            "            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]\n",
            "            if num_prefills == 0 and num_decodes == 0:\n"
            "                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]\n"
            "            else:\n"
            "                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]\n",
        ),
        (
            "        if (\n"
            "            self.use_full_cuda_graph\n"
            "            and num_prefills == 0\n"
            "            and num_decodes == 0\n"
            "            and num_spec_decodes <= self.decode_cudagraph_max_bs\n"
            "            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs\n"
            "        ):\n",
            "        if self._stage_spec_decode(\n"
            "            num_prefills, num_decodes, num_spec_decodes, num_spec_decode_tokens\n"
            "        ):\n",
        ),
        (
            "        if (\n"
            "            self.use_full_cuda_graph\n"
            "            and num_prefills == 0\n"
            "            and num_spec_decodes == 0\n"
            "            and num_decodes <= self.decode_cudagraph_max_bs\n"
            "        ):\n",
            "        if self._stage_decode(num_prefills, num_decodes, num_spec_decodes):\n",
        ),
        (
            "            spec_sequence_masks=spec_sequence_masks,\n",
            "            spec_sequence_masks=spec_sequence_masks,\n"
            "            spec_sequence_masks_cpu=spec_sequence_masks_cpu,\n",
        ),
        ("    def build_for_cudagraph_capture(\n", GDN_UPDATE_BLOCK_TABLE),
    ),
    "vllm.v1.worker.gpu.attn_utils": (
        (
            "    KVCacheSpec,\n    UniformTypeKVCacheSpecs,\n)\n",
            "    KVCacheSpec,\n    MambaSpec,\n    UniformTypeKVCacheSpecs,\n)\n",
        ),
        (
            "    attn_metadata: dict[str, Any] = {}\n"
            "    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)\n",
            "    attn_metadata: dict[str, Any] = {}\n"
            "    token_to_req_indices: torch.Tensor | None = None\n"
            "    # Mamba groups with the same spec and builder differ only in their state\n"
            "    # indices, so later groups re-gather those from the first group's metadata.\n"
            "    # Also at capture, so FULL graphs share the batch-level buffers.\n"
            "    cached_metadata: dict[tuple[KVCacheSpec, type], Any] = {}\n"
            "    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)\n",
        ),
        (
            "            rswa_prefix_lens=rswa_prefix_lens,\n"
            "            **common_attn_metadata_extra_kwargs,\n",
            "            rswa_prefix_lens=rswa_prefix_lens,\n"
            "            _token_to_req_indices_cache=token_to_req_indices,\n"
            "            **common_attn_metadata_extra_kwargs,\n",
        ),
        (
            "            attn_metadata_builder = attn_group.get_metadata_builder(0)\n"
            "            if for_cudagraph_capture:\n",
            "            attn_metadata_builder = attn_group.get_metadata_builder(0)\n"
            "            reuse_key = None\n"
            '            if getattr(attn_metadata_builder, "reuses_group_metadata", False) and isinstance(\n'
            "                attn_group.kv_cache_spec, MambaSpec\n"
            "            ):\n"
            "                reuse_key = (attn_group.kv_cache_spec, type(attn_metadata_builder))\n"
            "            if reuse_key in cached_metadata:\n"
            "                metadata = attn_metadata_builder.update_block_table(\n"
            "                    cached_metadata[reuse_key], block_table, slot_mapping\n"
            "                )\n"
            "            elif for_cudagraph_capture:\n",
        ),
        (
            "            for layer_name in attn_group.layer_names:\n"
            "                attn_metadata[layer_name] = metadata\n"
            "    return attn_metadata\n",
            "            if reuse_key is not None and reuse_key not in cached_metadata:\n"
            "                cached_metadata[reuse_key] = metadata\n"
            "            for layer_name in attn_group.layer_names:\n"
            "                attn_metadata[layer_name] = metadata\n"
            "        token_to_req_indices = common_attn_metadata._token_to_req_indices_cache\n"
            "    return attn_metadata\n",
        ),
    ),
}


def patch_source(module_name: str, source: str) -> str:
    for old, new in PATCHES[module_name]:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"{module_name}: expected one match for the backport anchor, found {count}. "
                f"The vLLM source changed; re-check the backport.\nAnchor:\n{old}"
            )
        source = source.replace(old, new)
    return source


class _PatchedSourceLoader(importlib.machinery.SourceFileLoader):
    # Compile from the patched source every time; a cached .pyc holds the unpatched module
    def get_code(self, fullname: str) -> CodeType:
        source = self.get_data(self.path).decode("utf-8")
        code = compile(
            patch_source(fullname, source), self.path, "exec", dont_inherit=True
        )
        PATCHED.append(fullname)
        print(f"[gdn_metadata_reuse] patched {fullname}", file=sys.stderr, flush=True)
        return code


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None:
            return None
        spec.loader = _PatchedSourceLoader(fullname, spec.origin)
        return spec


def install() -> bool:
    """Install the import hook. Returns False when this vLLM needs no backport."""
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return False
    if not version.startswith(SUPPORTED_VLLM):
        # 0.30 and 0.31 still lack part of the change, so a skip must show up in A/B logs
        print(
            f"[gdn_metadata_reuse] skipped: vLLM {version} is not {SUPPORTED_VLLM}x",
            file=sys.stderr,
            flush=True,
        )
        return False
    if any(isinstance(f, _PatchFinder) for f in sys.meta_path):
        return True
    imported = [name for name in PATCHES if name in sys.modules]
    if imported:
        raise RuntimeError(f"install() must run before vLLM imports {imported}")
    sys.meta_path.insert(0, _PatchFinder())
    return True
