# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Entrypoints move a request ``attention_schedule`` from extra params onto the typed sampling field.

Covered: /v1/videos/sync, the realtime video handler, /v1/images/generations, the chat
diffusion helper and diffusion TTS /v1/audio/speech. Each one lets the request value replace a
stage-default schedule, treats null as an omitted key, and reports an invalid value as a client
error before generation.
"""

import json
from argparse import Namespace
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from pytest_mock import MockerFixture
from vllm.entrypoints.openai.models.protocol import BaseModelPath

from tests.entrypoints.openai_api.test_image_server import MockGenerationResult
from tests.entrypoints.openai_api.test_serving_speech import create_mock_audio_output_for_test
from tests.entrypoints.openai_api.test_video_server import FakeAsyncOmni
from vllm_omni.diffusion.attention.schedule import AttentionScheduleRange, InvalidAttentionScheduleError
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.diffusion_request_utils import apply_normalized_diffusion_request_extra_args
from vllm_omni.entrypoints.openai.models.serving import _DiffusionServingModels
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.entrypoints.openai.serving_video_output_stream import OmniStreamingVideoOutputHandler
from vllm_omni.errors import client_error_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_STAGE_DEFAULT = [{"start": 0, "end": 2, "profile": "sparse"}]
_REPLACEMENT = [{"start": 1, "end": 3, "profile": "dense"}]
_STAGE_DEFAULT_RANGES = (AttentionScheduleRange(start=0, end=2, profile="sparse"),)
_REPLACEMENT_RANGES = (AttentionScheduleRange(start=1, end=3, profile="dense"),)

# (request value, typed field seen by the engine) when the stage default carries _STAGE_DEFAULT.
_OVERRIDES = [([], ()), (_REPLACEMENT, _REPLACEMENT_RANGES), (None, _STAGE_DEFAULT_RANGES)]
_OVERRIDE_IDS = ["disable", "replace", "null-keeps-stage-default"]

_INVALID = [
    [{"start": 0, "end": 2, "profile": "sparse"}, {"start": 1, "end": 3, "profile": "dense"}],
    [{"start": True, "end": 2, "profile": "sparse"}],
]
_INVALID_IDS = ["overlapping", "bool-step"]


# /v1/videos/sync


@pytest.fixture
def client(mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    app = FastAPI()
    app.state.api_server_count = 1
    app.include_router(router)
    app.state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=FakeAsyncOmni(),
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    with TestClient(app) as test_client:
        yield test_client


def _post_sync(client, extra_params):
    return client.post(
        "/v1/videos/sync",
        data={"prompt": "A boat on a lake.", "extra_params": json.dumps(extra_params)},
    )


def test_sync_video_moves_schedule_into_typed_field(client):
    response = _post_sync(
        client,
        {"attention_schedule": [{"start": 0, "end": 2, "profile": "sparse"}], "flow_shift": 5.0},
    )

    assert response.status_code == 200
    captured = client.app.state.openai_serving_video._engine_client.captured_sampling_params_list[0]
    assert captured.attention_schedule == (AttentionScheduleRange(start=0, end=2, profile="sparse"),)
    assert "attention_schedule" not in captured.extra_args
    assert captured.extra_args["flow_shift"] == 5.0


@pytest.mark.parametrize(("value", "expected"), _OVERRIDES, ids=_OVERRIDE_IDS)
def test_sync_video_request_schedule_replaces_stage_default(client, value, expected):
    engine = client.app.state.openai_serving_video._engine_client
    engine.default_sampling_params_list = [OmniDiffusionSamplingParams(attention_schedule=_STAGE_DEFAULT)]

    response = _post_sync(client, {"attention_schedule": value})

    assert response.status_code == 200
    captured = engine.captured_sampling_params_list[0]
    assert captured.attention_schedule == expected
    assert "attention_schedule" not in captured.extra_args


@pytest.mark.parametrize("schedule", _INVALID, ids=_INVALID_IDS)
def test_sync_video_rejects_invalid_schedule_before_generation(client, schedule):
    response = _post_sync(client, {"attention_schedule": schedule})

    assert response.status_code == 400
    assert "attention_schedule" in response.json()["detail"]
    assert client.app.state.openai_serving_video._engine_client.captured_sampling_params_list is None


# Realtime video handler


def _realtime_handler(stage_default=None):
    handler = object.__new__(OmniStreamingVideoOutputHandler)
    defaults = [] if stage_default is None else [OmniDiffusionSamplingParams(attention_schedule=stage_default)]
    handler._engine_client = SimpleNamespace(default_sampling_params_list=defaults)
    return handler


@pytest.mark.parametrize(("value", "expected"), _OVERRIDES, ids=_OVERRIDE_IDS)
async def test_realtime_video_request_schedule_replaces_stage_default(value, expected):
    request = VideoGenerationRequest(prompt="p", extra_params={"attention_schedule": value, "other": 1})

    _, gen_params, _ = await _realtime_handler(_STAGE_DEFAULT)._build_prompt_and_sampling_params(request)

    assert gen_params.attention_schedule == expected
    assert "attention_schedule" not in gen_params.extra_args
    assert gen_params.extra_args["other"] == 1


@pytest.mark.parametrize("schedule", _INVALID, ids=_INVALID_IDS)
async def test_realtime_video_rejects_invalid_schedule(schedule):
    request = VideoGenerationRequest(prompt="p", extra_params={"attention_schedule": schedule})

    with pytest.raises(HTTPException) as excinfo:
        await _realtime_handler()._build_prompt_and_sampling_params(request)

    assert excinfo.value.status_code == HTTPStatus.BAD_REQUEST.value
    assert "attention_schedule" in excinfo.value.detail


# /v1/images/generations


class _ImageEngine:
    def __init__(self):
        self.captured_sampling_params_list: list[Any] | None = None

    async def generate(self, **kwargs):
        self.captured_sampling_params_list = kwargs["sampling_params_list"]
        yield MockGenerationResult([Image.new("RGB", (16, 16), color="blue")])


@pytest.fixture
def image_client():
    engine = _ImageEngine()
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    app.state.diffusion_engine = engine
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    app.state.openai_serving_models = _DiffusionServingModels(
        [BaseModelPath(name="Qwen/Qwen-Image", model_path="Qwen/Qwen-Image")]
    )
    app.state.args = Namespace(
        default_sampling_params='{"0": {"num_inference_steps":4, "guidance_scale":7.5, "generator_device":"cpu"}}',
        max_generated_image_size=1024 * 1792,
    )
    return TestClient(app)


def _post_image(client, extra_params):
    return client.post("/v1/images/generations", json={"prompt": "A red fox.", "extra_params": extra_params})


def test_image_generation_moves_schedule_into_typed_field(image_client):
    response = _post_image(image_client, {"attention_schedule": _REPLACEMENT, "other": 1})

    assert response.status_code == 200
    captured = image_client.app.state.engine_client.captured_sampling_params_list[0]
    assert captured.attention_schedule == _REPLACEMENT_RANGES
    assert "attention_schedule" not in captured.extra_args
    assert captured.extra_args["other"] == 1


@pytest.mark.parametrize("schedule", _INVALID, ids=_INVALID_IDS)
def test_image_generation_rejects_invalid_schedule_before_generation(image_client, schedule):
    response = _post_image(image_client, {"attention_schedule": schedule})

    assert response.status_code == 400
    assert "attention_schedule" in response.json()["detail"]
    assert image_client.app.state.engine_client.captured_sampling_params_list is None


# Chat diffusion helper


@pytest.mark.parametrize(("value", "expected"), _OVERRIDES, ids=_OVERRIDE_IDS)
def test_chat_diffusion_extra_args_replace_stage_default_schedule(value, expected):
    params = OmniDiffusionSamplingParams(attention_schedule=_STAGE_DEFAULT, extra_args={"stage": 1})

    apply_normalized_diffusion_request_extra_args(params, {"attention_schedule": value, "other": 2})

    assert params.attention_schedule == expected
    assert params.extra_args == {"stage": 1, "other": 2}


@pytest.mark.parametrize("schedule", _INVALID, ids=_INVALID_IDS)
def test_chat_diffusion_extra_args_reject_invalid_schedule_as_client_error(schedule):
    params = OmniDiffusionSamplingParams()

    with pytest.raises(InvalidAttentionScheduleError, match="attention_schedule") as excinfo:
        apply_normalized_diffusion_request_extra_args(params, {"attention_schedule": schedule})

    assert client_error_metadata(excinfo.value) == (400, "invalid_attention_schedule")


# Diffusion TTS /v1/audio/speech


def _diffusion_speech_server(mocker: MockerFixture, stage_default=None):
    engine = mocker.MagicMock()
    engine.default_sampling_params_list = [OmniDiffusionSamplingParams(attention_schedule=stage_default)]

    async def generate(*args, **kwargs):
        yield create_mock_audio_output_for_test()

    engine.generate = mocker.MagicMock(side_effect=generate)
    server = OmniOpenAIServingSpeech.for_diffusion(diffusion_engine=engine, model_name="test-model")
    mocker.patch.object(
        server, "create_audio", return_value=mocker.MagicMock(audio_data=b"dummy", media_type="audio/wav")
    )
    return server, engine


@pytest.mark.parametrize(("value", "expected"), _OVERRIDES, ids=_OVERRIDE_IDS)
async def test_diffusion_speech_request_schedule_replaces_stage_default(mocker: MockerFixture, value, expected):
    server, engine = _diffusion_speech_server(mocker, _STAGE_DEFAULT)
    request = OpenAICreateSpeechRequest(input="Hello", extra_params={"attention_schedule": value, "other": 1})

    response = await server.create_speech(request)

    assert response.status_code == 200
    passed = engine.generate.call_args.kwargs["sampling_params_list"][0]
    assert passed.attention_schedule == expected
    assert passed.extra_args == {"other": 1}
    # The request works on a copy, so the stage default keeps its schedule.
    assert engine.default_sampling_params_list[0].attention_schedule == _STAGE_DEFAULT_RANGES


@pytest.mark.parametrize("schedule", _INVALID, ids=_INVALID_IDS)
async def test_diffusion_speech_rejects_invalid_schedule_before_generation(mocker: MockerFixture, schedule):
    server, engine = _diffusion_speech_server(mocker)
    request = OpenAICreateSpeechRequest(input="Hello", extra_params={"attention_schedule": schedule})

    response = await server.create_speech(request)

    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    assert "attention_schedule" in response.body.decode()
    engine.generate.assert_not_called()
