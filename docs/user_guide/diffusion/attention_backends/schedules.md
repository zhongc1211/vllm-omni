# Attention Schedules

An attention schedule switches the attention configuration at fixed denoising
step indices. For example, a schedule can keep the first steps on a dense base
configuration and run the later steps with a sparse or quantized profile.

A schedule configuration has two parts:

- **Profiles** are named attention configurations declared at server startup.
- **Ranges** assign a profile to a range of step indices. The server holds
  default ranges, and a request can replace or disable them.

This guide uses two more terms:

- A **schedule** is the list of ranges that applies to a request.
- The **base configuration** is the attention configuration that the server
  resolves without a schedule: `--diffusion-attention-config`,
  `--diffusion-attention-backend`, the environment variables
  `DIFFUSION_ATTENTION_BACKEND` and `DIFFUSION_ATTENTION_QUANT`, or the
  platform default. Steps that no range covers use the base configuration; see
  the [attention backend overview](../attention_backends.md#configuration).

Automated coverage uses toy models and stand-in attention kernels on CPU.
Wan2.2 T2V schedules have also been exercised in process on an NVIDIA B300.
Model validation remains incomplete; see
[Verification status](#verification-status).

## Pipelines that accept a schedule

A schedule needs a pipeline that reports its denoising progress (step index
and step count) to the attention layers. The following pipelines do:

| Model | Request mode | Step mode |
| --- | --- | --- |
| MiniMax-H3 | Accepted | Accepted |
| Wan2.2 (T2V, I2V, S2V, VACE) | Accepted | The model has no step mode |
| HunyuanImage-3.0 | Accepted | Accepted |

The table describes support established by the code and CPU tests. Of these
pipelines, only Wan2.2 T2V has been exercised with a schedule on real weights.
The other model and execution-mode combinations still need model validation.

With any other pipeline, a server that configures profiles fails at startup.
[Diffusion Execution Modes](../execution_modes.md) describes the two modes.

## Configure profiles and default ranges

Pass `--diffusion-attention-schedule` a JSON object with the keys `profiles`
and `default`. The flag has no environment variable. The following example
sets dense `TRTLLM_ATTN` as the base configuration, keeps steps 0 to 9 on it,
and runs step 10 through the last step with Skip-Softmax:

```bash
vllm-omni serve <model> \
  --diffusion-attention-backend TRTLLM_ATTN \
  --diffusion-attention-schedule '{
    "profiles": {
      "sparse": {
        "default": {
          "backend": "TRTLLM_ATTN",
          "skip_softmax": {"threshold": 0.05}
        }
      }
    },
    "default": [{"start": 10, "end": null, "profile": "sparse"}]
  }'
```

`TRTLLM_ATTN` has its own [requirements](trtllm.md#requirements). This example
does not load on HunyuanImage-3.0, and it does not load on Wan2.2 with
sequence parallelism; see [Compatibility limits](#compatibility-limits).

The key `default` appears at two levels. At the top level it is the list of
default ranges. Inside a profile it is that profile's default attention spec.

### Profiles

A profile is a complete attention configuration. It takes the same `default`
and `per_role` keys as `--diffusion-attention-config` and uses the same
[per-role resolution order](../attention_backends.md#per-role-configuration).

- A profile does not inherit from the base configuration. `per_role` entries
  of the base configuration are not carried into a profile. A role that the
  profile does not cover uses the platform default, not the base backend.
- `--diffusion-attention-backend`, `DIFFUSION_ATTENTION_BACKEND`,
  `DIFFUSION_ATTENTION_QUANT`, and `--fastvideo-vsa-topk` do not apply to
  profiles. A `FASTVIDEO_VSA` profile sets `fastvideo_vsa_topk` in its own
  spec, for example `{"backend": "FASTVIDEO_VSA", "fastvideo_vsa_topk": 64}`.
  A profile that uses quantization sets `quant` in its own spec.
- A profile name starts with an ASCII letter and continues with ASCII letters,
  digits, `_`, or `-`.
- Every profile is prepared and validated at startup on every attention layer
  that can take part in a schedule, including a profile that no default range
  selects. A request can select any declared profile and cannot define a new
  one.
- A `--diffusion-attention-schedule` value with no profiles and an empty
  `default` list configures nothing; the server runs as if the flag were not
  set. A non-empty `default` list without profiles fails at startup.

A profile's `default` spec applies to every attention role of the model. This
includes Wan2.2 cross-attention and the MiniMax-H3 token refiner
(`minimax_h3.token_refiner`), which runs in every transformer forward. On the
Wan2.2 T2V, I2V, and VACE transformer the cross-attention role is `cross`. The
Wan2.2 S2V transformer sets no role on its attention layers. All of them,
including cross-attention, resolve as `self`, so a `per_role.cross` entry
matches no layer there. To keep a role dense inside a range, add a `per_role`
entry to the profile:

```json
{
  "profiles": {
    "sparse": {
      "default": {
        "backend": "TRTLLM_ATTN",
        "skip_softmax": {"threshold": 0.05}
      },
      "per_role": {
        "minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}
      }
    }
  },
  "default": [{"start": 10, "end": null, "profile": "sparse"}]
}
```

### Ranges

Each range is an object with exactly three keys:

| Key | Value | Meaning |
| --- | --- | --- |
| `start` | integer `>= 0` | First step of the range. Steps are counted from 0. |
| `end` | integer `> start`, or `null` | Step after the last step of the range. `null` extends the range to the last step of the denoising sequence. The key is required. |
| `profile` | profile name | Profile used on steps `start` to `end - 1`. |

- List ranges in ascending order. Ranges must not overlap; adjacent ranges are
  allowed. A range with `"end": null` must be the last one.
- Steps before the first range, between two ranges, or after the last range
  use the base configuration.
- Ranges are never clipped. A request whose ranges do not fit the number of
  steps it runs is rejected; see
  [Validation and errors](#validation-and-errors). The server does not check
  the default ranges against a step count at startup.
- The index counts denoising steps, not model evaluations. With
  classifier-free guidance, both evaluations of a step use the same profile.
- A pipeline that runs several denoising sequences in one request applies the
  schedule to each sequence from step 0. This covers each Wan2.2 S2V clip,
  and each MiniMax-H3 output seed and continuation window. A MiniMax-H3
  `latent_refine` pass is a second sequence with its own step count, and a
  request cannot combine it with a non-empty schedule; see
  [Compatibility limits](#compatibility-limits).
- The schedule is active only inside the denoising loop. Attention calls
  outside it, for example during prompt encoding or decoding, use the base
  configuration.

### Sigma windows

A sigma schedule uses the same prepared profiles but selects by normalized
scheduler noise, not by the step index or raw model timestep.
MiniMax-H3 publishes its shifted rectified-flow video sigma directly.
Wan2.2 and HunyuanImage-3.0 use `scheduler.sigmas[i] / scheduler.sigmas[0]`;
Wan DMD uses its fixed flow timestep divided by the training timestep scale.
The independent context field `denoise_sigma` must be in `[0, 1]`.
A missing or invalid sigma fails rather than silently choosing the base backend.

Each window has exactly `low`, `high`, and `profile` keys.
Boundaries are finite numbers with `0 <= low < high <= 1`.
Windows are half-open **[low, high)**, except a window ending at **1.0**
also includes **1.0**. List windows in ascending order; adjacent windows
such as `[0, 0.3)` and `[0.3, 1]` are allowed. At sigma 0.3 the second
window applies. Overlap, descending windows, out-of-range values, and
undeclared profile names are rejected. Gaps use the base configuration.

Sigma thresholds refer to noise values on the actual shifted trajectory.
Changing the step count or flow shift changes which evaluation first crosses
a threshold; it does not change the threshold. A discrete trajectory need not
contain a sample exactly at the configured boundary.

Use the service configuration's `sigma` list for defaults and the request's
`attention_sigma_schedule` for overrides. `null` (Python `None`) inherits,
`[]` disables, and a non-empty list replaces the windows.
The effective step and sigma schedules cannot both be non-empty, including
inherited defaults. To replace a step default with sigma windows, also send
`attention_schedule: []`. The same rule applies at service startup.
Request mode and step mode both validate the resolved batch before denoising.

Range boundaries are step indices. The Skip-Softmax key
`disabled_until_timestep` is a separate control that compares the normalized
timestep; see [Timestep gating](trtllm.md#timestep-gating). When a profile
sets it, a step inside a range still runs dense while the normalized timestep
is above `disabled_until_timestep`.

### Dotted flags

`--diffusion-attention-schedule` also accepts dotted flags. Pass the `default`
list as one JSON value in both forms, and use only one form on a command line.

```bash
# Dotted flags
vllm-omni serve <model> \
  --diffusion-attention-schedule.profiles.dense.default TORCH_SDPA \
  --diffusion-attention-schedule.default \
  '[{"start":3,"end":null,"profile":"dense"}]'

# Equivalent JSON
vllm-omni serve <model> \
  --diffusion-attention-schedule \
  '{"profiles":{"dense":{"default":"TORCH_SDPA"}},"default":[{"start":3,"end":null,"profile":"dense"}]}'
```

Here `"default": "TORCH_SDPA"` inside the profile is shorthand for
`"default": {"backend": "TORCH_SDPA"}`.

### Deploy configuration

The first example, as a per-stage deploy configuration:

```yaml
stages:
  - stage_id: 0
    diffusion_attention_config:
      default:
        backend: TRTLLM_ATTN
    diffusion_attention_schedule:
      profiles:
        sparse:
          default:
            backend: TRTLLM_ATTN
            skip_softmax:
              threshold: 0.05
      default:
        - start: 10
          end: null
          profile: sparse
```

Set both keys on the diffusion stage. In the bundled two-stage deployments for
MiniMax-H3 (`minimax_h3_disaggregated.yaml`) and HunyuanImage-3.0
(`hunyuan_image_3_moe.yaml`), the diffusion stage is `stage_id: 1`.

A `--diffusion-attention-schedule` value on the command line is deep-merged
onto the stage's deploy-configuration value:

- Profiles with different names are both kept.
- A profile with the same name is merged key by key. Deploy-configuration keys
  that the command line does not set remain.
- A command-line `default` list replaces the deploy-configuration list.
  Without one, the deploy-configuration list is kept.

### Python API

```python
from vllm_omni.diffusion.data import AttentionScheduleConfig, OmniDiffusionConfig

config = OmniDiffusionConfig(
    diffusion_attention_schedule=AttentionScheduleConfig(
        profiles={
            "sparse": {
                "default": {
                    "backend": "TRTLLM_ATTN",
                    "skip_softmax": {"threshold": 0.05},
                }
            }
        },
        default=[{"start": 10, "end": None, "profile": "sparse"}],
    ),
    # Other OmniDiffusionConfig fields, including the base configuration.
)
```

## Override the schedule in a request

A request sets the field `attention_schedule` to a list of ranges in the same
format. It can name only profiles declared at startup.

| Request value | Effect |
| --- | --- |
| omitted or `null` | Use the server's default ranges. |
| `[]` | Disable the schedule. Every step uses the base configuration. |
| non-empty list | Replace the server's default ranges for this request. The two lists are not merged. |

A stage's `default_sampling_params` can also carry `attention_schedule`. On
the video endpoints, a request that omits the field or sends `null` then uses
that value in place of the server's default ranges.

`attention_schedule` is not a top-level request parameter. Put it in the
object that the entry point provides for model-specific parameters:

| Entry point | Where the field goes |
| --- | --- |
| `POST /v1/videos`, `POST /v1/videos/sync` | In the JSON-encoded `extra_params` form field |
| `POST /v1/images/generations` | In `extra_params`. The endpoint reads it only when the server runs a single diffusion stage; a multi-stage deployment ignores it without an error. |
| `POST /v1/chat/completions` with a diffusion model | In `extra_args`, at the request root or inside `extra_body` |
| Offline inference | `OmniDiffusionSamplingParams(attention_schedule=[...])` |

- On `POST /v1/chat/completions`, give the field in one place only. A request
  that repeats it is rejected. An `attention_schedule` key at the request root
  or directly inside `extra_body` is ignored. With the OpenAI Python client,
  pass `extra_body={"extra_args": {"attention_schedule": [...]}}`.
- The bundled HunyuanImage-3.0 deployment `hunyuan_image_3_moe.yaml` has two
  stages, so `POST /v1/images/generations` ignores the field there. Send it
  through `POST /v1/chat/completions`.
- `POST /v1/audio/speech` reads the field from `extra_params` only when the
  server runs a single diffusion stage. `/v1/realtime/video` reads it from
  `extra_params` of the `session.start` message. None of the pipelines listed
  above was checked against these two endpoints.
- `POST /v1/images/edits` and `POST /v1/audio/generate` do not accept the
  field.

The following example assumes a server started with `--port 8091`:

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=A boat on a lake." \
  -F "num_inference_steps=40" \
  -F "seed=42" \
  -F 'extra_params={"attention_schedule":[{"start":20,"end":null,"profile":"sparse"}]}' \
  -o boat.mp4
```

For offline inference, set the field on the sampling parameters:

```python
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

sampling_params = OmniDiffusionSamplingParams(
    num_inference_steps=40,
    seed=42,
    attention_schedule=[{"start": 20, "end": None, "profile": "sparse"}],
)
```

### Batching

Requests share a batch only when their `attention_schedule` values are equal
after a stage `default_sampling_params` value, if any, is applied. When the
stage sets none, a request that omits the field and a request that sends the
same ranges as the server's default ranges are not batched together.

In step mode, when the schedule in effect for a batch is non-empty, MiniMax-H3
and HunyuanImage-3.0 run each request in its own transformer forward, so that
each request selects a profile at its own step. This includes default ranges
that the requests inherit.

In HunyuanImage-3.0 step mode, a batch that runs without a schedule and holds
more than one request still requires `TORCH_SDPA` as the base self-attention
backend; see [Diffusion Execution Modes](../execution_modes.md#step-execution).
Such a batch occurs when the requests send `[]` or when the server has no
default ranges. A batch with a schedule skips that check.

## Validation and errors

The server fails at startup when the schedule configuration is malformed, when
the default ranges name a profile that is not declared, or when a profile
cannot run on an attention layer of the loaded model. The last case covers the
backend's platform requirements and the
[compatibility limits](#compatibility-limits) below. A platform error for a
profile can name `--diffusion-attention-backend` in its text even though the
backend came from a profile.

A request with an invalid schedule fails before its first denoising step:

| Cause | Example message |
| --- | --- |
| A range has a missing or extra key | `attention_schedule ranges require exactly start, end and profile` |
| A step is not an integer | `attention_schedule start must be an integer, got True` |
| `end` is not greater than `start` | `attention_schedule end must be greater than start` |
| Ranges overlap, are out of order, or follow a range with `"end": null` | `attention_schedule ranges must be ordered and must not overlap` |
| A profile is not declared at startup | `attention_schedule references unknown profile(s): ['sparse']` |
| A range does not fit the steps that run | `attention_schedule range AttentionScheduleRange(start=20, end=None, profile='sparse') exceeds total_steps=20` |

Other malformed values, such as a negative step, a value that is not a list,
or an invalid profile name, are rejected the same way with their own message.
A server without profiles returns the unknown-profile error for every
non-empty request schedule.

The response depends on where the error is detected. On the HTTP endpoints:

- A malformed schedule or an undeclared profile returns HTTP 400 with the
  message in the JSON error body.
- A range that does not fit is detected when the pipeline builds its timestep
  sequence. The response is HTTP 400 in both request and step mode. Request
  mode preserves the error status for single requests and batches, including
  when the stage uses `distributed_executor_backend: mp`.
- `POST /v1/videos` creates the job first, so these errors appear on the job
  record and not in the response to the `POST`.

`/v1/realtime/video` is a WebSocket endpoint and returns no HTTP status for
these errors. The server sends `video.start` first and then sends
`{"type": "error", "message": "<text>"}`.

The fit check uses the length of the timestep sequence that the pipeline
builds for the request. That length can differ from `num_inference_steps`.
The check also applies to the server's default ranges when a request inherits
them.

## Compatibility limits

| Feature | Behavior with schedule profiles |
| --- | --- |
| `--diffusion-compile-granularity full` | Rejected at startup. Use regional scope, which is the default. |
| Cache backends (`--cache-backend`) | Rejected at startup, including when the default ranges are empty. |
| MiniMax-H3 request-scoped Cache-DiT | In request mode, a request with `quality=high` whose schedule is non-empty is rejected before denoising. This includes a schedule inherited from the server's default ranges. The response is HTTP 400 with `attention_schedule cannot be combined with MiniMax H3 Cache-DiT` in the error message. Send `quality=lossless` or `"attention_schedule": []`. In step mode, `quality=high` is rejected with HTTP 400 for every request, with or without a schedule. See [Request-Scoped Quality](../cache_acceleration/cache_dit.md#request-scoped-quality-minimax-h3). |
| MiniMax-H3 `latent_refine` | In request mode, a request with `latent_refine` whose schedule is non-empty is rejected before denoising. This includes a schedule inherited from the server's default ranges and a `latent_refine` value inherited from `--additional-config`. A server that sets both non-empty default ranges and a `latent_refine` default therefore rejects every request that overrides neither. The response is HTTP 400 with `attention_schedule cannot be combined with MiniMax H3 latent_refine` in the error message. Send `"attention_schedule": []` or `"latent_refine": false`. `latent_upscale` without `latent_refine` is accepted with a schedule. In step mode, `latent_refine` is rejected with HTTP 400 for every request, with or without a schedule. See [Latent super-resolution](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#latent-super-resolution) in the MiniMax-H3 recipe. |
| Sequence parallelism on a model that pads the sequence | Applies to Ulysses, Ring, and AllGather-KV on Wan2.2 and HunyuanImage-3.0. MiniMax-H3 declares no sequence-parallel padding, so this check does not apply to it; its packed-sequence padding has its own limit, listed below the table. Every profile must select a backend with attention-mask support on every attention layer of the transformer, including cross-attention. The check runs at startup and does not depend on the request shape. The backends are listed below the table. |
| Ring sequence parallelism | On layers that take part in sequence parallelism, a profile must not set `skip_softmax`. It must also resolve to the same backend as the base configuration, selected the same way (named explicitly in both, or left to the platform default in both), with the same backend options. A schedule therefore cannot change attention on those layers. |
| AllGather-KV sequence parallelism | On layers that take part in sequence parallelism, a profile that resolves to `TRTLLM_ATTN` is rejected at startup, as for the base configuration. A `RAINFUSION_ATTN` profile with sparsity above 0 (the default is 0.8) is rejected on every layer that selects it. |
| Scheduler-managed [paged KV](../paged_kv_cache.md) | On paged KV layers a profile must resolve to the same backend as the base configuration, selected the same way, with the same backend options. |
| KV-cache quantization (`--diffusion-kv-cache-dtype`) | On each layer that quantizes its KV cache, every profile's backend must support the configured dtype on the platform. `auto` and `float` add no requirement. |
| Model-owned attention kernels | A layer that uses a model-owned kernel instead of a backend rejects schedule profiles at startup. |
| `target_sparsity` in a `TRTLLM_ATTN` profile | Requires a calibration curve for each attention layer that the checkpoint's calibration does not list under `ignore`. Without one, startup fails; use `threshold`. |

When the base configuration takes its `quant` options from
`DIFFUSION_ATTENTION_QUANT`, a profile must repeat the same `quant` options in
its own spec to pass the ring and paged KV checks in the table, because the
variable does not apply to profiles.

The backends with attention-mask support are `TORCH_SDPA`, `FLASH_ATTN`,
`CUDNN_ATTN`, `FLASH_ATTN_HUB`, `FLASH_ATTN_3_HUB`, and `FLASHINFER_ATTN` when
its kernel resolves to `fa2` or `fa3`. `TRTLLM_ATTN`, `FASTVIDEO_VSA`,
`RAINFUSION_ATTN`, `SAGE_ATTN`, and `SAGE_ATTN_3` have none. A
`FLASHINFER_ATTN` profile has none when its kernel resolves to `cute-dsl`,
which is what `auto` resolves to on Blackwell.

Model-specific limits:

- **Wan2.2**: a profile can select `FASTVIDEO_VSA` on a self-attention layer
  only when the base backend is `FASTVIDEO_VSA`, because the model builds the
  VSA gate projection only in that case. The S2V transformer has no VSA gate
  and no such check; a `FASTVIDEO_VSA` profile there loads and runs SDPA.
- **MiniMax-H3**: on the DiT blocks, a profile's backend must be able to
  exclude the padding rows of the packed sequence. A `FASTVIDEO_VSA` profile
  needs a [FastH3 VSA checkpoint](fastvideo_vsa.md), unless the base backend
  is already `FASTVIDEO_VSA`. On the gated layers of a FastH3 checkpoint,
  every profile must select `FASTVIDEO_VSA`. These checks do not apply to the
  token refiner.
- **HunyuanImage-3.0**: on image attention, a profile can select only a
  backend with attention-mask support. `TRTLLM_ATTN`, `FASTVIDEO_VSA`, and
  `RAINFUSION_ATTN` profiles are therefore rejected at startup on this model,
  including the `TRTLLM_ATTN` example on this page.

A range selects a profile. It does not guarantee that every attention call in
the range runs the sparse or quantized kernel, because the backend's own
fallbacks still apply. For example:

- Skip-Softmax stays dense, without a log line, on steps whose normalized
  timestep is above `disabled_until_timestep`. With `target_sparsity`, it also
  stays dense on the layers that the calibration lists under `ignore`.
- On a causal attention layer, `TRTLLM_ATTN` turns
  [SAGE](trtllm.md#sage-quantization) off when the layer is built and logs
  nothing. A profile that sets `quant` runs that layer without SAGE.
- On CUDA, `FASTVIDEO_VSA` falls back to SDPA when its preconditions are not
  met or its kernel raises, and logs
  `FASTVIDEO_VSA falling back to SDPA: <reason>` once. On NPU, XPU, and MUSA
  it always runs SDPA and logs nothing.
- `RAINFUSION_ATTN` stays dense on the layers in `skip_layers`, on the first
  `start_step` steps, and on the last `end_step` steps.

Schedules have no dedicated handling for CPU offload, layerwise offload, or
LoRA, and no test covers those combinations.

## Compilation

With profiles configured, each attention layer selects its implementation at
call time from the published step index. To keep that selection out of the
compiled graph, the attention call runs eagerly. The rest of a compiled block,
such as projections and normalization, stays compiled. See
[Regional Compilation](../regional_compilation.md) for the compile settings.

- The eager attention call applies to every request on a server that
  configures profiles. This includes requests that send
  `"attention_schedule": []` and servers whose default ranges are empty. A
  server without profiles does not make this eager call.
- **Wan2.2**: the attention calls inside each compiled transformer block run
  eagerly.
- **MiniMax-H3**: packed attention already runs outside the compiled block
  without a schedule, so profiles add no graph break.
- **HunyuanImage-3.0**: the model declares no repeated blocks, so regional
  compilation does not apply to it, and `full` scope is rejected with
  profiles. With profiles configured, the model runs uncompiled.
- `--enforce-eager` disables the generic compile setup. A schedule then runs
  without compiled blocks.

No profile runs during startup. Wan2.2 T2V, I2V, and VACE send one startup
warm-up request, with an empty schedule. MiniMax-H3, HunyuanImage-3.0, and
Wan2.2 S2V send no startup warm-up request, so on MiniMax-H3 and Wan2.2 S2V
the first request also compiles the blocks.

A profile can add one compilation in the middle of a request. The following
results come from tests on a toy model, each with one profile and without
sequence parallelism:

- When the profile's kernel returned a tensor whose strides differ from the
  base kernel's output, the code after the attention call was compiled once
  more at the first step that ran the profile.
- When the two outputs differed only in whether they are views, or in the
  shape of the viewed tensor, the extra compilation occurred only with dynamic
  compilation under `torch.no_grad()`. The server runs under `torch.no_grad()`
  with HSDP, and in request mode also with distributed layerwise offload.
  MiniMax-H3 calls its transformer under `torch.inference_mode()` in request
  mode, so on MiniMax-H3 this condition applies only to step mode with HSDP.
- When the profile's kernel returned the same tensor layout as the base
  kernel, requests with different ranges, with `[]`, and with the field
  omitted on a server without default ranges ran on the same compiled blocks
  without recompilation.

These tests ran on CPU with a Dynamo backend that counts graph executions and
does not run Inductor. Compile counts under Inductor on MiniMax-H3 and Wan2.2,
and which pairs of attention backends trigger the additional compilation on
those models, have not been measured.

After startup, send one warm-up request that runs the base configuration and
every profile for at least one step each before you measure latency.

## Confirm that a schedule takes effect

At startup the loader logs how many profile and layer combinations it
validated:

```text
Attention schedule: <N> prepared candidate(s) passed startup checks.
```

The startup log also prints one line for each distinct resolution. The line
starts with `Resolved diffusion attention backend '<name>' for role='<role>'`
and ends with `via <source>`. A layer with a role category, such as the
MiniMax-H3 token refiner, adds `(role_category='<category>')` before `via`.
Profiles are resolved through the same code
and the line does not contain the profile name, so a line can come from the
base configuration or from a profile. A profile that resolves to the same
backend and source as the base configuration adds no line.

No log line or metric reports which profile ran at which step. To confirm
that a schedule changes the output, compare a scheduled request against a
request with the same seed and `"attention_schedule": []` on the same server.

With `RAINFUSION_ATTN` and `end_step` in the base configuration on Wan2.2,
that comparison has a second cause of difference. Wan2.2 publishes the total
number of steps only for a request with a non-empty schedule, and
`RAINFUSION_ATTN` stays dense on every call when `end_step` is set and no
total is published. The steps that no range covers can therefore run the
sparse kernel in the scheduled request, while every step of the `[]` request
stays dense.

## Verification status

- Automated tests cover configuration parsing, startup validation, per-step
  profile selection, and compile behavior. They also cover request handling
  for `/v1/videos/sync`, `/v1/images/generations`, `/v1/audio/speech`, the
  realtime video handler, and the chat request helper. `POST /v1/videos` uses
  the same handling code as `/v1/videos/sync` and has no test with a schedule.
- CPU tests cover toy models and stand-in attention kernels. Real-weight
  Wan2.2 T2V measurements have also run on a single NVIDIA B300 with dense
  `TRTLLM_ATTN`, SAGE quantization, and Skip-Softmax. These measurements use
  the in-process benchmark, not an HTTP server.
- The latest SAGE run passed the benchmark's checks. The Skip-Softmax run
  still failed the strict comparison between unscheduled baseline and
  candidate output. Two separate runs of the unmodified baseline also
  differed despite matching recorded inputs and effective configuration.
  The cause remains unresolved, so these results do not establish a speed
  or quality guarantee.
- MiniMax-H3, HunyuanImage-3.0, NPU execution, `FASTVIDEO_VSA`, and
  `RAINFUSION_ATTN` have not been validated with real scheduled inference.
- The HTTP status codes and error bodies on this page come from reading the
  code and from handler tests, not from a running server.
- Sequence parallelism with a schedule has not been run on multiple GPUs.

Benchmark your model, shape, and schedule against an all-dense run with the
same seed before you rely on a schedule.
