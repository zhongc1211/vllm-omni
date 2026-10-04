# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contracts for per-layer schedule candidate preparation and startup validation.

The first half pins the candidate-preparation contract: with no schedule no candidates
are prepared and only the baseline implementation is built; with a schedule each
layer pre-constructs one deduplicated _PreparedCandidate record per distinct effective
profile, without replacing the baseline, reproducing the baseline construction-time
guards (marked-paged promotion, AllGather+TRTLLM rejection). The second half covers the
post-load startup compatibility traversal, validate_attention_schedule_candidates.
Calibration stamping is tested in test_trtllm_calibration.py.

No backend touches a GPU and no model weights load; backends and impls are fakes
resolved from the supplied AttentionConfig, mirroring test_attention_config.py.
"""

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.attention.layer as layer_mod
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionScheduleConfig,
    AttentionSpec,
    SkipSoftmaxSpec,
)
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeImpl:
    """Records construction kwargs; optionally accepts layer calibration."""

    # Mirrors AttentionImpl.supports_kv_cache_dtype, but keyed only by dtype so a CPU
    # test does not depend on the host platform key.
    _kv_cache_dtypes_ok: tuple[str, ...] = ()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calibration = None

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype, platform_key) -> bool:
        del platform_key
        return kv_cache_dtype is None or kv_cache_dtype in cls._kv_cache_dtypes_ok

    def forward(self, query, key, value, attn_metadata=None):
        return query

    def set_layer_calibration(self, a, b):
        self.calibration = (a, b)


_BACKEND_CACHE: dict[str, type] = {}
_BACKEND_IMPLS: dict[str, type[_FakeImpl]] = {}


def _fake_backend(name: str) -> type:
    """A distinct fake backend (and impl subclass) per backend name."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    impl_cls = type(f"Fake{name}Impl", (_FakeImpl,), {})
    _BACKEND_IMPLS[name] = impl_cls

    class _Backend:
        supports_paged_kv = False
        supports_piecewise_spans = False

        @staticmethod
        def get_name() -> str:
            return name

        @staticmethod
        def get_impl_cls():
            return impl_cls

        @staticmethod
        def supports_attention_mask(spec) -> bool:
            return True

    _Backend.__name__ = f"Fake{name}Backend"
    _BACKEND_CACHE[name] = _Backend
    return _Backend


def _fake_impl(name: str) -> type[_FakeImpl]:
    """Typed accessor for the impl class a fake backend resolves to.

    ``_fake_backend`` is annotated as a bare ``type`` because the backend class is built
    per name, so assertions that need the impl class go through here rather than calling
    ``.get_impl_cls()`` on an untyped class object.
    """
    _fake_backend(name)
    return _BACKEND_IMPLS[name]


def _fake_resolve(*, role, head_size, attention_config=None, role_category=None, allow_trtllm_default=True):
    """Resolve a fake backend from the supplied AttentionConfig, like the real selector."""
    spec = None
    if attention_config is not None:
        spec, _source = attention_config.resolve_with_source(role=role, role_category=role_category)
    if spec is None:
        return _fake_backend("PLATFORM_DEFAULT"), None
    return _fake_backend(spec.backend.upper()), spec


def _make_config(
    *, baseline=None, schedule=None, kv_mode=None, allgather_degree=1, kv_dtype=None, sp_size=1, ring_degree=1
):
    return SimpleNamespace(
        diffusion_attention_config=baseline if baseline is not None else AttentionConfig(),
        diffusion_attention_schedule=schedule,
        diffusion_kv_mode=kv_mode if kv_mode is not None else layer_mod.DiffusionKVCacheMode.DENSE_LEGACY,
        parallel_config=SimpleNamespace(
            ring_degree=ring_degree,
            allgather_degree=allgather_degree,
            sequence_parallel_size=sp_size,
            ulysses_degree=sp_size,
        ),
        diffusion_kv_cache_dtype=kv_dtype,
        diffusion_kv_cache_skip_step_indices=None,
        diffusion_kv_cache_skip_layer_indices=None,
    )


@pytest.fixture
def attention_env(monkeypatch):
    """Patch the layer module so Attention constructs against fake backends."""
    monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: False)

    calls: list[dict] = []

    def _counting(*, role, head_size, attention_config=None, role_category=None, allow_trtllm_default=True):
        calls.append({"role": role, "attention_config": attention_config})
        return _fake_resolve(
            role=role,
            head_size=head_size,
            attention_config=attention_config,
            role_category=role_category,
            allow_trtllm_default=allow_trtllm_default,
        )

    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", _counting)

    def build(config, **kwargs):
        with set_current_diffusion_config(config):
            return Attention(num_heads=4, head_size=64, causal=False, softmax_scale=1.0, **kwargs)

    return SimpleNamespace(build=build, calls=calls, monkeypatch=monkeypatch)


def test_no_schedule_prepares_no_candidates_and_single_resolution(attention_env):
    attention = attention_env.build(_make_config(schedule=None))

    assert attention._schedule_profiles == ()
    assert attention._schedule_candidates == {}
    # Baseline only: exactly one backend resolution, no extra candidates.
    assert len(attention_env.calls) == 1
    assert attention.attn_backend is _fake_backend("PLATFORM_DEFAULT")
    assert attention.backend_explicit is False


def test_schedule_prepares_one_candidate_record_per_profile(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={
            "quant": AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=7)),
            "sparse": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE")),
        },
        default=[{"start": 2, "end": None, "profile": "quant"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    assert attention._schedule_profiles == ("quant", "sparse")
    assert set(attention._schedule_candidates) == {"quant", "sparse"}
    # Baseline + two distinct profiles -> three resolutions.
    assert len(attention_env.calls) == 3
    # The baseline impl is unchanged and is not any candidate's impl.
    assert type(attention.attention) is _fake_impl("PLATFORM_DEFAULT")
    assert all(rec.impl is not attention.attention for rec in attention._schedule_candidates.values())


def test_candidate_backend_identity_matches_profile(attention_env):
    # Assert each candidate carries the backend/spec its profile selects, so a
    # wrong-backend candidate would be caught.
    schedule = AttentionScheduleConfig(
        profiles={
            "quant": AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=7)),
            "sparse": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE")),
        },
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    quant = attention._schedule_candidates["quant"]
    sparse = attention._schedule_candidates["sparse"]
    assert quant.backend_cls is _fake_backend("FASTVIDEO_VSA")
    assert quant.backend_pref == "FASTVIDEO_VSA"
    assert quant.spec is not None and quant.spec.backend == "FASTVIDEO_VSA"
    assert quant.backend_explicit is True
    assert sparse.backend_cls is _fake_backend("BLOCK_SPARSE")
    assert sparse.backend_pref == "BLOCK_SPARSE"


def test_identical_effective_profiles_dedup_to_shared_record(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={
            "a": AttentionConfig(default=AttentionSpec(backend="SDPA")),
            "b": AttentionConfig(default=AttentionSpec(backend="SDPA")),
        },
        default=[{"start": 0, "end": None, "profile": "a"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    assert attention._schedule_profiles == ("a", "b")
    assert attention._schedule_candidates["a"] is attention._schedule_candidates["b"]
    assert attention._schedule_candidates["a"].impl is attention._schedule_candidates["b"].impl


def test_same_backend_different_params_are_distinct_candidates(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={
            "topk7": AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=7)),
            "topk9": AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=9)),
        },
        default=[{"start": 0, "end": None, "profile": "topk7"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    assert attention._schedule_candidates["topk7"] is not attention._schedule_candidates["topk9"]


def test_explicit_and_default_profiles_with_same_backend_do_not_share(attention_env):
    # The dedup key must include backend_explicit, so an explicit profile and a
    # platform-default profile that share a backend name do not collapse onto one impl.
    explicit = AttentionConfig(default=AttentionSpec(backend="PLATFORM_DEFAULT"))
    schedule = AttentionScheduleConfig(
        profiles={"explicit": explicit, "implicit": AttentionConfig()},
        default=[{"start": 0, "end": None, "profile": "explicit"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    exp = attention._schedule_candidates["explicit"]
    imp = attention._schedule_candidates["implicit"]
    assert exp.backend_explicit is True
    assert imp.backend_explicit is False
    assert exp is not imp
    assert exp.impl is not imp.impl


def test_profile_not_referenced_by_default_is_still_prepared(attention_env):
    # Preparation covers ALL configured profiles, not only those the default references.
    schedule = AttentionScheduleConfig(
        profiles={
            "used": AttentionConfig(default=AttentionSpec(backend="SDPA")),
            "unused": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE")),
        },
        default=[{"start": 1, "end": 3, "profile": "used"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    assert "unused" in attention._schedule_candidates
    assert attention._schedule_profiles == ("unused", "used")


def test_per_role_profile_candidate_resolves_role_spec(attention_env):
    # A per_role profile resolves the role-specific spec independently.
    schedule = AttentionScheduleConfig(
        profiles={"roled": AttentionConfig(per_role={"self": AttentionSpec(backend="BLOCK_SPARSE")})},
        default=[{"start": 0, "end": None, "profile": "roled"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule), role="self")

    rec = attention._schedule_candidates["roled"]
    assert rec.backend_cls is _fake_backend("BLOCK_SPARSE")
    assert rec.spec is not None and rec.spec.backend == "BLOCK_SPARSE"


def test_impl_override_applies_to_candidate(attention_env):
    # The override must subclass the candidate's resolved impl, mirroring the baseline contract.
    vsa_impl_cls = _fake_impl("FASTVIDEO_VSA")
    _Specialized = type("_Specialized", (vsa_impl_cls,), {})

    schedule = AttentionScheduleConfig(
        profiles={"quant": AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=7))},
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    attention = attention_env.build(
        _make_config(schedule=schedule),
        impl_overrides={"FASTVIDEO_VSA": _Specialized},
    )

    rec = attention._schedule_candidates["quant"]
    assert rec.impl_cls is _Specialized
    assert type(rec.impl) is _Specialized


def test_custom_attention_with_schedule_is_rejected(attention_env):
    # A custom-attention layer cannot represent candidates; reject with the
    # specific guard, not any ValueError/TypeError.
    schedule = AttentionScheduleConfig(
        profiles={"quant": AttentionConfig(default=AttentionSpec(backend="SDPA"))},
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    custom = SimpleNamespace()  # stands in for a model-owned nn.Module kernel

    with pytest.raises(ValueError, match="cannot be combined with custom_attention"):
        attention_env.build(
            _make_config(schedule=schedule),
            custom_attention=custom,
            skip_sequence_parallel=True,
        )


def test_profile_resolution_error_propagates(attention_env):
    # An unbuildable candidate must fail at construction, not be silently skipped.
    def _boom(*, role, head_size, attention_config=None, role_category=None, allow_trtllm_default=True):
        if attention_config is not None:
            spec, _ = attention_config.resolve_with_source(role=role, role_category=role_category)
            if spec is not None and spec.backend.upper() == "BLOCK_SPARSE":
                raise ImportError("optional kernel unavailable")
        return _fake_resolve(
            role=role,
            head_size=head_size,
            attention_config=attention_config,
            role_category=role_category,
            allow_trtllm_default=allow_trtllm_default,
        )

    attention_env.monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", _boom)
    schedule = AttentionScheduleConfig(
        profiles={"bad": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "bad"}],
    )
    with pytest.raises(ImportError, match="optional kernel unavailable"):
        attention_env.build(_make_config(schedule=schedule))


def test_scheduled_layer_registers_no_extra_submodules(attention_env):
    # Candidates live in a plain dict; AttentionImpl is not an nn.Module, so a
    # scheduled layer registers no submodule and its state_dict matches the baseline.
    schedule = AttentionScheduleConfig(
        profiles={"quant": AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=7))},
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    plain = attention_env.build(_make_config(schedule=None))
    scheduled = attention_env.build(_make_config(schedule=schedule))

    assert [n for n, _ in scheduled.named_modules()] == [n for n, _ in plain.named_modules()]
    assert list(scheduled.state_dict()) == list(plain.state_dict())


def test_paged_candidate_is_promoted_to_flash_attn(attention_env):
    # A marked-paged layer whose profile falls back to the platform default must
    # promote the candidate to a paged-capable backend, exactly like the baseline.
    attention_env.monkeypatch.setattr(FlashAttentionBackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    schedule = AttentionScheduleConfig(
        profiles={"implicit": AttentionConfig()},
        default=[{"start": 0, "end": None, "profile": "implicit"}],
    )
    config = _make_config(schedule=schedule, kv_mode=layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER)
    attention = attention_env.build(config, paged_kv_cache_role="primary")

    rec = attention._schedule_candidates["implicit"]
    assert rec.backend_cls is FlashAttentionBackend
    assert rec.backend_pref == "FLASH_ATTN"


def test_allgather_trtllm_candidate_is_rejected(attention_env):
    # A candidate selecting TRTLLM_ATTN under AllGather-KV SP must be rejected at
    # construction, reproducing the baseline guard.
    schedule = AttentionScheduleConfig(
        profiles={"trt": AttentionConfig(default=AttentionSpec(backend="TRTLLM_ATTN"))},
        default=[{"start": 0, "end": None, "profile": "trt"}],
    )
    config = _make_config(schedule=schedule, allgather_degree=2)
    with pytest.raises(ValueError, match="AllGather-KV sequence parallelism"):
        attention_env.build(config)


# --- Post-load startup compatibility traversal ------------------------------
#
# Construction reproduces the guards that decide WHICH backend a layer uses.
# The remaining per-candidate guarantees need the loaded model and the resolved parallel
# plan, so they are checked once after load, before serving: paged-KV representability,
# KV-cache-quantization support, SP auto-pad mask capability and calibration presence.
# The checks that run without this traversal do not read each prepared candidate, so an
# incompatible candidate would fail inside the first kernel after the layer switches to it,
# or run dense when its calibration curve is missing.


def _pipeline(layer, *, sp_plan=None, name="attn1"):
    """A minimal loaded-pipeline stand-in exposing named_modules() and an optional _sp_plan."""
    import torch.nn as nn

    transformer = nn.Module()
    setattr(transformer, name, layer)
    if sp_plan is not None:
        transformer._sp_plan = sp_plan
    root = nn.Module()
    root.transformer = transformer
    return root


def _flat_auto_pad_plan():
    """auto_pad on the plan entry itself, one level below the plan root."""
    return {"hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True)}


def _auto_pad_plan(*, auto_pad=True, nested_list=False):
    """A plan shaped the way real models declare it: module id -> param/output key -> input spec.

    ``SequenceParallelInputType`` (vllm_omni/diffusion/distributed/sp_plan.py) is itself a dict, so
    auto_pad lives two levels below the plan root; ``WanTransformer3DModel._sp_plan`` is the in-tree
    example. ``nested_list`` wraps the spec in a list, which the type alias also allows.
    """
    spec = SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=auto_pad)
    return {"rope": {0: [spec] if nested_list else spec}}


def test_startup_validation_is_noop_without_schedule(attention_env):
    # Critical invariant: no schedule configured -> no traversal, no new failure mode.
    layer = attention_env.build(_make_config(schedule=None))
    config = _make_config(schedule=None)

    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 0


def test_startup_validation_accepts_compatible_candidates(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={
            "a": AttentionConfig(default=AttentionSpec(backend="SDPA")),
            "b": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE")),
        },
        default=[{"start": 0, "end": None, "profile": "a"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 2


def test_startup_validation_rejects_layer_missing_a_prepared_profile(attention_env):
    # Validation covers ALL configured profiles, so a layer that lost one is a defect.
    schedule = AttentionScheduleConfig(
        profiles={
            "a": AttentionConfig(default=AttentionSpec(backend="SDPA")),
            "b": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE")),
        },
        default=[{"start": 0, "end": None, "profile": "a"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)
    layer._schedule_candidates.pop("b")

    with pytest.raises(ValueError, match=r"profile 'b' was not prepared"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_rejects_candidate_without_paged_support(attention_env):
    # An explicit profile is never promoted, so a non-paged backend survives construction and
    # would raise NotImplementedError inside the first paged kernel. Reject at startup instead.
    attention_env.monkeypatch.setattr(FlashAttentionBackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    schedule = AttentionScheduleConfig(
        profiles={"nopaged": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "nopaged"}],
    )
    config = _make_config(schedule=schedule, kv_mode=layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER)
    layer = attention_env.build(config, paged_kv_cache_role="primary")

    with pytest.raises(ValueError, match=r"profile 'nopaged'.*does not support Scheduler-managed paged KV"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_covers_profile_not_referenced_by_default(attention_env):
    # The unreferenced profile is still prepared and must still be validated.
    attention_env.monkeypatch.setattr(FlashAttentionBackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    schedule = AttentionScheduleConfig(
        profiles={
            "used": AttentionConfig(),
            "unused": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE")),
        },
        default=[{"start": 1, "end": 3, "profile": "used"}],
    )
    config = _make_config(schedule=schedule, kv_mode=layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER)
    layer = attention_env.build(config, paged_kv_cache_role="primary")

    assert layer._schedule_candidates["used"].backend_cls is FlashAttentionBackend
    with pytest.raises(ValueError, match=r"profile 'unused'.*does not support Scheduler-managed paged KV"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_rejects_candidate_matching_an_explicit_non_paged_baseline(attention_env):
    # An explicit baseline is not promoted either. A candidate identical to it passes the identity
    # check, so without the capability check the error would come only from KV-cache registration at
    # engine start, as a NotImplementedError that does not name the profile.
    schedule = AttentionScheduleConfig(
        profiles={"same": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "same"}],
    )
    baseline = AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))
    config = _make_config(baseline=baseline, schedule=schedule, kv_mode=layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER)
    layer = attention_env.build(config, paged_kv_cache_role="primary")

    assert layer.attn_backend is _fake_backend("BLOCK_SPARSE")
    assert layer.backend_explicit is True
    assert layer._schedule_candidates["same"].backend_cls is layer.attn_backend
    with pytest.raises(ValueError, match=r"profile 'same'.*does not support Scheduler-managed paged KV"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_rejects_paged_capable_candidate_that_differs_from_baseline(attention_env):
    # forward_paged runs the native implementation bound to the baseline and never reads the
    # candidate's spec, so even a paged-capable candidate would be silently ignored.
    attention_env.monkeypatch.setattr(FlashAttentionBackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    attention_env.monkeypatch.setattr(_fake_backend("BLOCK_SPARSE"), "supports_paged_kv", True)
    schedule = AttentionScheduleConfig(
        profiles={"paged": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "paged"}],
    )
    config = _make_config(schedule=schedule, kv_mode=layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER)
    layer = attention_env.build(config, paged_kv_cache_role="primary")

    with pytest.raises(ValueError, match=r"profile 'paged'.*native implementation bound to the baseline"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_accepts_paged_candidate_matching_baseline(attention_env):
    # The profile falls back to the platform default and is promoted exactly like the baseline.
    attention_env.monkeypatch.setattr(FlashAttentionBackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    schedule = AttentionScheduleConfig(
        profiles={"same": AttentionConfig()},
        default=[{"start": 0, "end": None, "profile": "same"}],
    )
    config = _make_config(schedule=schedule, kv_mode=layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER)
    layer = attention_env.build(config, paged_kv_cache_role="primary")

    assert layer.attn_backend is FlashAttentionBackend
    assert layer._schedule_candidates["same"].backend_cls is FlashAttentionBackend
    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1


def test_startup_validation_rejects_candidate_without_kv_cache_dtype_support(attention_env):
    # _init_kv_cache_quantization probes the baseline impl only; a candidate that cannot hold
    # the configured cache dtype would fail later, so the traversal probes every candidate.
    schedule = AttentionScheduleConfig(
        profiles={"quant": AttentionConfig(default=AttentionSpec(backend="SDPA"))},
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    config = _make_config(schedule=schedule, kv_dtype="fp8")
    attention_env.monkeypatch.setattr(_fake_impl("PLATFORM_DEFAULT"), "_kv_cache_dtypes_ok", ("fp8",))
    layer = attention_env.build(config)

    with pytest.raises(ValueError, match=r"profile 'quant'.*kv_cache_dtype='fp8'"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_accepts_candidate_supporting_kv_cache_dtype(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={"quant": AttentionConfig(default=AttentionSpec(backend="SDPA"))},
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    config = _make_config(schedule=schedule, kv_dtype="fp8")
    attention_env.monkeypatch.setattr(_fake_impl("PLATFORM_DEFAULT"), "_kv_cache_dtypes_ok", ("fp8",))
    attention_env.monkeypatch.setattr(_fake_impl("SDPA"), "_kv_cache_dtypes_ok", ("fp8",))
    layer = attention_env.build(config)

    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1


def test_startup_validation_rejects_auto_pad_incompatible_candidate(attention_env):
    # SP auto-pad needs attention_mask. The pad-time probe reads the config for role "self" only,
    # so the traversal checks every prepared candidate on a layer whose component plans auto_pad.
    schedule = AttentionScheduleConfig(
        profiles={"nomask": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "nomask"}],
    )
    attention_env.monkeypatch.setattr(
        _fake_backend("BLOCK_SPARSE"), "supports_attention_mask", staticmethod(lambda spec=None: False)
    )
    config = _make_config(schedule=schedule, sp_size=2)
    layer = attention_env.build(config)

    with pytest.raises(ValueError, match=r"profile 'nomask'.*attention_mask"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer, sp_plan=_flat_auto_pad_plan()), config)


def test_startup_validation_skips_mask_check_without_auto_pad_or_sp(attention_env):
    # Same layer and profile, but no auto_pad plan and SP=1: mask capability is not required,
    # so a mask-free candidate stays legal.
    schedule = AttentionScheduleConfig(
        profiles={"nomask": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "nomask"}],
    )
    attention_env.monkeypatch.setattr(
        _fake_backend("BLOCK_SPARSE"), "supports_attention_mask", staticmethod(lambda spec=None: False)
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1


def test_startup_validation_rejects_candidate_without_resolvable_calibration(attention_env):
    # target_sparsity needs a calibrated curve for THIS layer's expert. A curve that only
    # exists for another expert would silently leave the candidate dense.
    schedule = AttentionScheduleConfig(
        profiles={
            "sparse": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"other_expert": {"a": 1.0, "b": 2.0}}},
                )
            )
        },
        default=[{"start": 0, "end": None, "profile": "sparse"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    with pytest.raises(ValueError, match=r"profile 'sparse'.*calibration"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_preserves_ignored_layer_dense_fallback(attention_env):
    # An ignored layer legitimately stays dense; that fallback must not be reported as a defect.
    schedule = AttentionScheduleConfig(
        profiles={
            "sparse": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"transformer": {"a": 1.0, "b": 2.0, "ignore": ["*attn1"]}}},
                )
            )
        },
        default=[{"start": 0, "end": None, "profile": "sparse"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1


def test_startup_validation_accepts_candidate_with_resolvable_calibration(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={
            "sparse": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"transformer": {"a": 1.0, "b": 2.0}}},
                )
            )
        },
        default=[{"start": 0, "end": None, "profile": "sparse"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1


# --- Startup validation: nested SP plans, KV-cache dtype "float", ring attention, explicit threshold ---


def test_startup_validation_detects_auto_pad_nested_in_module_plan(attention_env):
    # Real plans nest the spec under a module id and a parameter key, so a walker that only
    # unwraps list/tuple would never see auto_pad and the mask check would never run.
    schedule = AttentionScheduleConfig(
        profiles={"nomask": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "nomask"}],
    )
    attention_env.monkeypatch.setattr(
        _fake_backend("BLOCK_SPARSE"), "supports_attention_mask", staticmethod(lambda spec=None: False)
    )
    config = _make_config(schedule=schedule, sp_size=2)
    layer = attention_env.build(config)

    with pytest.raises(ValueError, match=r"profile 'nomask'.*attention_mask"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer, sp_plan=_auto_pad_plan()), config)


def test_startup_validation_detects_auto_pad_in_nested_list_entry(attention_env):
    # The type alias also allows a list of input specs per parameter key.
    schedule = AttentionScheduleConfig(
        profiles={"nomask": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "nomask"}],
    )
    attention_env.monkeypatch.setattr(
        _fake_backend("BLOCK_SPARSE"), "supports_attention_mask", staticmethod(lambda spec=None: False)
    )
    config = _make_config(schedule=schedule, sp_size=2)
    layer = attention_env.build(config)
    model = _pipeline(layer, sp_plan=_auto_pad_plan(nested_list=True))

    with pytest.raises(ValueError, match=r"profile 'nomask'.*attention_mask"):
        layer_mod.validate_attention_schedule_candidates(model, config)


def test_startup_validation_ignores_nested_plan_without_auto_pad(attention_env):
    # The same nesting with auto_pad=False must NOT require mask support: a model that never pads
    # keeps mask-free candidates legal.
    schedule = AttentionScheduleConfig(
        profiles={"nomask": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "nomask"}],
    )
    attention_env.monkeypatch.setattr(
        _fake_backend("BLOCK_SPARSE"), "supports_attention_mask", staticmethod(lambda spec=None: False)
    )
    config = _make_config(schedule=schedule, sp_size=2)
    layer = attention_env.build(config)
    model = _pipeline(layer, sp_plan=_auto_pad_plan(auto_pad=False))

    assert layer_mod.validate_attention_schedule_candidates(model, config) == 1


def test_startup_validation_accepts_kv_cache_dtype_float(attention_env):
    # The baseline guard (Attention._init_kv_cache_quantization) exempts "float", so the traversal
    # must too; otherwise a config that loads without a schedule is rejected once a schedule is set.
    schedule = AttentionScheduleConfig(
        profiles={"quant": AttentionConfig(default=AttentionSpec(backend="SDPA"))},
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    config = _make_config(schedule=schedule, kv_dtype="float")
    layer = attention_env.build(config)  # _FakeImpl supports no dtype, including "float"

    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1


def _ring_config(schedule, **kwargs):
    """An od_config for the traversal only.

    The layer itself is built with ring_degree=1 because Attention.__init__ constructs a ring
    runner through get_sp_group(), which needs a real distributed group. The traversal's ring rule
    is what these tests exercise, and it reads ring_degree from the config it is handed.
    """
    return _make_config(schedule=schedule, ring_degree=2, **kwargs)


def test_startup_validation_rejects_skip_softmax_candidate_under_ring(attention_env):
    # _run_ring_attention rejects skip-softmax by reading the BASELINE impl, so a candidate carrying
    # it would be silently ignored after the layer switches to it. It must be rejected at startup.
    schedule = AttentionScheduleConfig(
        profiles={
            "sparse": AttentionConfig(
                default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax=SkipSoftmaxSpec(threshold=0.2))
            )
        },
        default=[{"start": 0, "end": None, "profile": "sparse"}],
    )
    layer = attention_env.build(_make_config(schedule=schedule))

    with pytest.raises(ValueError, match=r"profile 'sparse'.*[Rr]ing"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), _ring_config(schedule))


def test_startup_validation_rejects_candidate_backend_divergence_under_ring(attention_env):
    # The ring runner is constructed once from the baseline's backend preference (Attention.__init__)
    # and RingParallelAttention.run_attention refuses explicit selections it cannot honor, so a
    # candidate that resolves to a different backend cannot be represented under ring.
    schedule = AttentionScheduleConfig(
        profiles={"other": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "other"}],
    )
    layer = attention_env.build(_make_config(schedule=schedule))

    assert layer.backend_pref != layer._schedule_candidates["other"].backend_pref
    with pytest.raises(ValueError, match=r"profile 'other'.*ring"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), _ring_config(schedule))


def test_startup_validation_accepts_ring_candidate_matching_baseline(attention_env):
    # A profile that resolves to the same backend preference as the baseline, with no skip-softmax,
    # is representable by the ring runner the layer already built.
    schedule = AttentionScheduleConfig(
        profiles={"same": AttentionConfig()},
        default=[{"start": 0, "end": None, "profile": "same"}],
    )
    layer = attention_env.build(_make_config(schedule=schedule))

    assert layer._schedule_candidates["same"].backend_pref == layer.backend_pref
    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), _ring_config(schedule)) == 1


def test_startup_validation_accepts_explicit_threshold_without_calibration(attention_env):
    # An explicit threshold is the calibration-free path, so it must not require a curve. The
    # traversal returns early for it; this pins that behaviour.
    schedule = AttentionScheduleConfig(
        profiles={
            "thresh": AttentionConfig(
                default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax=SkipSoftmaxSpec(threshold=0.2))
            )
        },
        default=[{"start": 0, "end": None, "profile": "thresh"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    assert layer._schedule_candidates["thresh"].spec.skip_calibration is None
    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1


# --- Startup validation: per-component SP plans, layers without candidates, per-candidate calibration ---
#
# This section also covers a model with no attention layer, the candidate dedup key (calibration,
# impl class, unserializable kwargs), what the ring rule compares (resolved backend name, explicit
# flag, backend kwargs) and the layers that the ring and KV-cache-quantization rules skip.


def _two_component_pipeline(scheduled_layer, other_layer, *, pad_first=True, pad_second=False):
    """A pipeline with two DiT components, each with its own SP plan, like a real two-expert model.

    The registry applies SP hooks per component (_apply_sequence_parallel_if_enabled in
    vllm_omni/diffusion/registry.py), so an auto_pad declaration on one component must not impose
    mask support on the other component's layers.
    """
    import torch.nn as nn

    first = nn.Module()
    first.attn1 = scheduled_layer
    if pad_first:
        first._sp_plan = _auto_pad_plan()
    second = nn.Module()
    second.attn1 = other_layer
    if pad_second:
        second._sp_plan = _auto_pad_plan()
    root = nn.Module()
    root.transformer = first
    root.transformer_2 = second
    return root


def _mask_free_profile(name="nomask"):
    return AttentionScheduleConfig(
        profiles={name: AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": name}],
    )


def test_startup_validation_scopes_auto_pad_to_the_owning_component(attention_env):
    # auto_pad is declared per DiT component, so requiring mask support from a
    # component that never pads would reject configurations that run fine.
    attention_env.monkeypatch.setattr(
        _fake_backend("BLOCK_SPARSE"), "supports_attention_mask", staticmethod(lambda spec=None: False)
    )
    schedule = _mask_free_profile()
    config = _make_config(schedule=schedule, sp_size=2)
    padded = attention_env.build(config)
    unpadded = attention_env.build(config)
    model = _two_component_pipeline(padded, unpadded, pad_first=True, pad_second=False)

    with pytest.raises(ValueError, match=r"profile 'nomask'.*attention_mask.*transformer\.attn1") as exc:
        layer_mod.validate_attention_schedule_candidates(model, config)
    assert "transformer_2" not in str(exc.value)


def test_startup_validation_accepts_mask_free_candidate_in_unpadded_component(attention_env):
    # The same mask-free candidate, but only the OTHER component pads. That component holds no
    # Attention layer here, so nothing is required to support a mask and the layer validates.
    import torch.nn as nn

    attention_env.monkeypatch.setattr(
        _fake_backend("BLOCK_SPARSE"), "supports_attention_mask", staticmethod(lambda spec=None: False)
    )
    schedule = _mask_free_profile()
    config = _make_config(schedule=schedule, sp_size=2)
    plain = attention_env.build(config)
    model = _two_component_pipeline(plain, nn.Module(), pad_first=False, pad_second=True)

    assert layer_mod.validate_attention_schedule_candidates(model, config) == 1


def test_startup_validation_rejects_model_without_attention_layers(attention_env):
    # A schedule that validated nothing must not pass silently.
    import torch.nn as nn

    schedule = _mask_free_profile()
    config = _make_config(schedule=schedule)
    empty = nn.Module()
    empty.transformer = nn.Module()

    with pytest.raises(ValueError, match="no attention layer"):
        layer_mod.validate_attention_schedule_candidates(empty, config)


def test_startup_validation_ignores_layer_built_without_schedule_config(attention_env):
    # Attention can be constructed with no current diffusion config; such a layer
    # legitimately has no candidates and must not be reported as a missing profile.
    schedule = _mask_free_profile()
    aware = attention_env.build(_make_config(schedule=schedule))
    unaware = attention_env.build(_make_config(schedule=None))
    model = _two_component_pipeline(aware, unaware, pad_first=False, pad_second=False)

    assert layer_mod.validate_attention_schedule_candidates(model, _make_config(schedule=schedule)) == 1


def test_profiles_differing_only_in_calibration_do_not_share(attention_env):
    # The dedup key includes the spec's skip_calibration, so two profiles with identical backend
    # kwargs but different curves do not collapse onto one impl that holds only one of the curves.
    schedule = AttentionScheduleConfig(
        profiles={
            "curve_a": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(threshold=0.2),
                    skip_calibration={"by_expert": {"transformer": {"a": 1.0, "b": 2.0}}},
                )
            ),
            "curve_b": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(threshold=0.2),
                    skip_calibration={"by_expert": {"transformer": {"a": 9.0, "b": 8.0}}},
                )
            ),
        },
        default=[{"start": 0, "end": None, "profile": "curve_a"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    assert attention._schedule_candidates["curve_a"] is not attention._schedule_candidates["curve_b"]
    assert attention._schedule_candidates["curve_a"].impl is not attention._schedule_candidates["curve_b"].impl


def test_identical_profiles_still_share_after_calibration_keying(attention_env):
    # Keying on skip_calibration must not break dedup for profiles that really are identical.
    spec = dict(
        backend="TRTLLM_ATTN",
        skip_softmax=SkipSoftmaxSpec(threshold=0.2),
        skip_calibration={"by_expert": {"transformer": {"a": 1.0, "b": 2.0}}},
    )
    schedule = AttentionScheduleConfig(
        profiles={
            "a": AttentionConfig(default=AttentionSpec(**spec)),
            "b": AttentionConfig(default=AttentionSpec(**spec)),
        },
        default=[{"start": 0, "end": None, "profile": "a"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    assert attention._schedule_candidates["a"] is attention._schedule_candidates["b"]


def test_startup_validation_uses_the_calibration_stamping_would_use(attention_env):
    # Validation resolves the curve from the dict that stamping writes onto the candidate, which is
    # the candidate's own skip_calibration when it has one. Here the first profile's own dict has no
    # curve for this layer's expert, so that profile is rejected although the second profile's dict
    # would resolve one.
    schedule = AttentionScheduleConfig(
        profiles={
            "a_first": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"other_expert": {"a": 1.0, "b": 2.0}}},
                )
            ),
            "b_second": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"transformer": {"a": 3.0, "b": 4.0}}},
                )
            ),
        },
        default=[{"start": 0, "end": None, "profile": "a_first"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    with pytest.raises(ValueError, match=r"profile 'a_first'.*calibration"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_accepts_when_effective_calibration_ignores_the_layer(attention_env):
    # The accepted counterpart: the first profile's own dict ignores this layer, so staying dense is
    # legitimate for that candidate, and the second profile resolves a curve from its own dict.
    schedule = AttentionScheduleConfig(
        profiles={
            "a_first": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"transformer": {"a": 1.0, "b": 2.0, "ignore": ["*attn1"]}}},
                )
            ),
            "b_second": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"transformer": {"a": 3.0, "b": 4.0}}},
                )
            ),
        },
        default=[{"start": 0, "end": None, "profile": "a_first"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 2


def test_startup_validation_rejects_later_profile_whose_own_curve_misses_the_layer(attention_env):
    # Stamping uses the candidate's own skip_calibration. The first profile's dict covers this layer,
    # but the later profile's own dict does not, so that later profile must be rejected. It would
    # resolve a curve only if the first profile's dict were stamped onto every candidate.
    schedule = AttentionScheduleConfig(
        profiles={
            "a_first": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"transformer": {"a": 1.0, "b": 2.0}}},
                )
            ),
            "b_second": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"other_expert": {"a": 3.0, "b": 4.0}}},
                )
            ),
        },
        default=[{"start": 0, "end": None, "profile": "a_first"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    with pytest.raises(ValueError, match=r"profile 'b_second'.*calibration"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_skips_ring_rules_when_layer_skips_sequence_parallel(attention_env):
    # A layer with skip_sequence_parallel never enters the ring runner.
    schedule = AttentionScheduleConfig(
        profiles={
            "sparse": AttentionConfig(
                default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax=SkipSoftmaxSpec(threshold=0.2))
            )
        },
        default=[{"start": 0, "end": None, "profile": "sparse"}],
    )
    layer = attention_env.build(_make_config(schedule=schedule), skip_sequence_parallel=True)

    assert layer.skip_sequence_parallel is True
    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), _ring_config(schedule)) == 1


def test_startup_validation_accepts_lowercase_backend_that_resolves_like_baseline(attention_env):
    # The traversal compares resolved backend names, not the raw preference string.
    # The baseline is explicit too, so only the spelling differs.
    baseline = AttentionConfig(default=AttentionSpec(backend="PLATFORM_DEFAULT"))
    schedule = AttentionScheduleConfig(
        profiles={"lower": AttentionConfig(default=AttentionSpec(backend="platform_default"))},
        default=[{"start": 0, "end": None, "profile": "lower"}],
    )
    layer = attention_env.build(_make_config(baseline=baseline, schedule=schedule))
    record = layer._schedule_candidates["lower"]
    config = _ring_config(schedule, baseline=baseline)

    assert record.backend_pref == "platform_default"
    assert record.backend_cls.get_name() == layer.attn_backend.get_name()
    assert (record.backend_explicit, layer.backend_explicit) == (True, True)
    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1


def test_startup_validation_rejects_explicit_candidate_over_automatic_ring_baseline(attention_env):
    # The ring runner is bound to the baseline's explicit flag. An automatic baseline falls
    # back to SDPA ring where an explicit selection of the same backend must raise (ring.py), so the
    # same backend name is not enough.
    schedule = AttentionScheduleConfig(
        profiles={"pinned": AttentionConfig(default=AttentionSpec(backend="PLATFORM_DEFAULT"))},
        default=[{"start": 0, "end": None, "profile": "pinned"}],
    )
    layer = attention_env.build(_make_config(schedule=schedule))
    record = layer._schedule_candidates["pinned"]

    assert record.backend_cls is layer.attn_backend
    assert (record.backend_explicit, layer.backend_explicit) == (True, False)
    with pytest.raises(ValueError, match=r"profile 'pinned'.*explicit=True.*ring runner"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), _ring_config(schedule))


@pytest.mark.parametrize(("candidate_topk", "accepted"), [(8, False), (4, True)], ids=["different", "same"])
def test_startup_validation_compares_backend_kwargs_under_ring(attention_env, candidate_topk, accepted):
    # The ring runner never reads backend kwargs, so a quant or sparse setting that differs
    # from the baseline's would be dropped without an error. Equal kwargs stay representable.
    baseline = AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=4))
    schedule = AttentionScheduleConfig(
        profiles={
            "vsa": AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=candidate_topk))
        },
        default=[{"start": 0, "end": None, "profile": "vsa"}],
    )
    layer = attention_env.build(_make_config(baseline=baseline, schedule=schedule))
    config = _ring_config(schedule, baseline=baseline)

    assert layer._schedule_candidates["vsa"].backend_cls is layer.attn_backend
    if accepted:
        assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1
    else:
        with pytest.raises(ValueError, match=r"profile 'vsa'.*backend kwargs"):
            layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_skips_kv_probe_when_layer_disables_kv_quant(attention_env):
    # A layer that opts out of KV-cache quantization at forward time must not be rejected at startup.
    schedule = AttentionScheduleConfig(
        profiles={"quant": AttentionConfig(default=AttentionSpec(backend="SDPA"))},
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    config = _make_config(schedule=schedule, kv_dtype="fp8")
    attention_env.monkeypatch.setattr(_fake_impl("PLATFORM_DEFAULT"), "_kv_cache_dtypes_ok", ("fp8",))
    layer = attention_env.build(config, disable_kv_quant=True)

    assert layer._disable_kv_quant is True
    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1


def test_startup_validation_skips_kv_probe_for_skip_layer_index(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={"quant": AttentionConfig(default=AttentionSpec(backend="SDPA"))},
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    build_config = _make_config(schedule=schedule, kv_dtype="fp8")
    attention_env.monkeypatch.setattr(_fake_impl("PLATFORM_DEFAULT"), "_kv_cache_dtypes_ok", ("fp8",))
    layer = attention_env.build(build_config)
    layer.layer_idx = 3
    validate_config = _make_config(schedule=schedule, kv_dtype="fp8")
    validate_config.diffusion_kv_cache_skip_layer_indices = [3]

    assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), validate_config) == 1


def test_profiles_with_same_qualname_and_different_impl_classes_do_not_share(attention_env):
    # The dedup key must use the impl class object, not its __qualname__.
    made = []

    def get_impl():
        cls = type("SameName", (_FakeImpl,), {})
        made.append(cls)
        return cls

    attention_env.monkeypatch.setattr(_fake_backend("SDPA"), "get_impl_cls", staticmethod(get_impl))
    schedule = AttentionScheduleConfig(
        profiles={
            "a": AttentionConfig(default=AttentionSpec(backend="SDPA")),
            "b": AttentionConfig(default=AttentionSpec(backend="SDPA")),
        },
        default=[{"start": 0, "end": None, "profile": "a"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    assert attention._schedule_candidates["a"].impl_cls is not attention._schedule_candidates["b"].impl_cls
    assert attention._schedule_candidates["a"].impl_cls.__qualname__ == "SameName"
    assert len(made) >= 2


def test_unserializable_backend_kwargs_do_not_share_via_repr(attention_env):
    # Equal repr must not collapse two unserializable kwargs dicts.
    class _Marker:
        def __repr__(self):
            return "Marker()"

    def kwargs(self):
        del self
        return {"marker": _Marker()}

    attention_env.monkeypatch.setattr(AttentionSpec, "backend_kwargs", kwargs)
    schedule = AttentionScheduleConfig(
        profiles={
            "a": AttentionConfig(default=AttentionSpec(backend="SDPA")),
            "b": AttentionConfig(default=AttentionSpec(backend="SDPA")),
        },
        default=[{"start": 0, "end": None, "profile": "a"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    assert attention._schedule_candidates["a"] is not attention._schedule_candidates["b"]
