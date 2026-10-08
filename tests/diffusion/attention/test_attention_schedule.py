# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contracts for integer-step attention schedules; no backend construction."""

import copy
import json
from dataclasses import FrozenInstanceError, asdict

import msgspec
import pytest

from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
    parse_attention_schedule,
    resolve_attention_schedule,
    select_attention_profile,
    validate_attention_schedule,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_attention_schedule_half_open_boundaries_and_gaps():
    schedule = parse_attention_schedule([{"start": 3, "end": 6, "profile": "sparse"}])
    validate_attention_schedule(schedule, profiles={"sparse"}, total_steps=8)

    assert [select_attention_profile(schedule, i, total_steps=8) for i in range(8)] == [
        None,
        None,
        None,
        "sparse",
        "sparse",
        "sparse",
        None,
        None,
    ]
    assert select_attention_profile(schedule, 3, total_steps=8) == "sparse"


@pytest.mark.parametrize(
    "request_schedule, expected",
    [(None, "quantized"), ([], None), ([{"start": 0, "end": 8, "profile": "sparse"}], "sparse")],
)
def test_attention_schedule_inherit_disable_replace(request_schedule, expected):
    default = [{"start": 3, "end": None, "profile": "quantized"}]
    original = copy.deepcopy((request_schedule, default))

    bound = resolve_attention_schedule(request_schedule, default, profiles={"quantized", "sparse"}, total_steps=8)

    assert select_attention_profile(bound, 4, total_steps=8) == expected
    assert (request_schedule, default) == original


def test_attention_schedule_none_is_distinct_from_empty():
    assert parse_attention_schedule(None) is None
    assert parse_attention_schedule([]) == ()
    assert resolve_attention_schedule(None, (), profiles=set(), total_steps=1) == ()


def test_attention_schedule_open_end_and_adjacent_ranges():
    schedule = parse_attention_schedule(
        [{"start": 0, "end": 3, "profile": "first"}, {"start": 3, "end": None, "profile": "last"}]
    )
    validate_attention_schedule(schedule, profiles={"first", "last"}, total_steps=5)
    assert [select_attention_profile(schedule, i, total_steps=5) for i in range(5)] == [
        "first",
        "first",
        "first",
        "last",
        "last",
    ]


def test_attention_schedule_detached_immutable_and_serializable():
    raw = [{"start": 3, "end": 6, "profile": "sparse"}]
    schedule = parse_attention_schedule(raw)
    raw[0]["start"] = 0
    raw.append({"start": 6, "end": None, "profile": "other"})

    assert schedule == (AttentionScheduleRange(3, 6, "sparse"),)
    assert hash(schedule) == hash(copy.deepcopy(schedule))
    assert parse_attention_schedule(msgspec.msgpack.decode(msgspec.msgpack.encode(schedule))) == schedule
    assert parse_attention_schedule(json.loads(json.dumps([asdict(entry) for entry in schedule]))) == schedule
    with pytest.raises(FrozenInstanceError):
        schedule[0].start = 0


@pytest.mark.parametrize("bad", [True, 3, 1.5, "[]", {}, {"profiles": {}}, [None], ["sparse"]])
def test_attention_schedule_rejects_wrong_container_or_entry_type(bad):
    with pytest.raises(TypeError):
        parse_attention_schedule(bad)


@pytest.mark.parametrize(
    "entry",
    [
        {"start": True, "end": 6, "profile": "sparse"},
        {"start": 3.0, "end": 6, "profile": "sparse"},
        {"start": 3, "end": False, "profile": "sparse"},
        {"start": 3, "end": 6.0, "profile": "sparse"},
        {"start": 3, "end": 6, "profile": 7},
    ],
)
def test_attention_schedule_rejects_noninteger_steps_and_nonstring_names(entry):
    with pytest.raises(TypeError):
        parse_attention_schedule([entry])


@pytest.mark.parametrize(
    "entry",
    [
        {"start": -1, "end": 6, "profile": "sparse"},
        {"start": 3, "end": 3, "profile": "sparse"},
        {"start": 3, "end": 2, "profile": "sparse"},
        {"start": 3, "profile": "sparse"},
        {"end": 6, "profile": "sparse"},
        {"start": 3, "end": 6},
        {"start": 3, "end": 6, "profile": "sparse", "backend": "TORCH_SDPA"},
        {"start": 3, "end": 6, "profile": ""},
        {"start": 3, "end": 6, "profile": "3sparse"},
        {"start": 3, "end": 6, "profile": "a.b"},
        {"start": 3, "end": 6, "profile": "a\n"},
    ],
)
def test_attention_schedule_rejects_invalid_fields_ranges_and_names(entry):
    with pytest.raises(ValueError):
        parse_attention_schedule([entry])


@pytest.mark.parametrize("ranges", [[(3, 6), (2, 3)], [(3, 6), (5, 7)], [(3, None), (6, 7)]])
def test_attention_schedule_rejects_unsorted_overlap_and_nonterminal_open_end(ranges):
    with pytest.raises(ValueError, match="ordered|overlap"):
        parse_attention_schedule([{"start": start, "end": end, "profile": "sparse"} for start, end in ranges])


@pytest.mark.parametrize("entry", [AttentionScheduleRange(3, 9, "sparse"), AttentionScheduleRange(8, None, "sparse")])
def test_attention_schedule_rejects_bounds_against_actual_total_even_when_inherited(entry):
    with pytest.raises(ValueError, match="total_steps"):
        resolve_attention_schedule(None, (entry,), profiles={"sparse"}, total_steps=8)


@pytest.mark.parametrize("name", ["q", "Quant_2", "sparse-middle"])
def test_attention_schedule_accepts_profile_name_grammar(name):
    schedule = resolve_attention_schedule([{"start": 0, "end": 1, "profile": name}], (), profiles={name}, total_steps=1)
    assert select_attention_profile(schedule, 0, total_steps=1) == name


def test_attention_schedule_rejects_unknown_profiles():
    with pytest.raises(ValueError, match="unknown.*profile"):
        resolve_attention_schedule([{"start": 0, "end": None, "profile": "missing"}], (), profiles={"sparse"})


@pytest.mark.parametrize("total", [True, 0, -1, 8.0])
def test_attention_schedule_rejects_invalid_actual_total(total):
    with pytest.raises((TypeError, ValueError), match="total_steps"):
        validate_attention_schedule((), profiles=set(), total_steps=total)


@pytest.mark.parametrize("step", [True, 1.0, -1, 8])
def test_attention_schedule_rejects_invalid_selection_step(step):
    with pytest.raises((TypeError, ValueError), match="step_index"):
        select_attention_profile((), step, total_steps=8)


def test_attention_schedule_typed_ranges_validate_at_construction():
    with pytest.raises(ValueError, match="end"):
        AttentionScheduleRange(3, 2, "sparse")
    with pytest.raises(TypeError, match="start"):
        AttentionScheduleRange(True, 6, "sparse")
