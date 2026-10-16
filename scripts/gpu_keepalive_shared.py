#!/usr/bin/env python3
"""Closed-loop GPU utilization filler: co-resident with real work, not instead of it.

This is a second, independent keepalive. ``scripts/gpu_keepalive.py`` stays as
it is; nothing here imports it and nothing there changes. Pick one per job.

Why a second one
----------------
``gpu_keepalive.py`` is binary: it runs a saturating matmul on a card, or --
once a ``card_<n>.busy`` marker appears -- yields that card entirely. That makes
a card's reading the claimant's problem, and a claimant that spends its
allocation on pandas and sklearn produces a reading of ~0%. Job 16904802 was
cancelled on exactly that: four cards reading 99.95 / 91.36 / 59.25 / 76.11,
average 81.7%, killed because ONE card sat below the 60% floor. The two low
cards were the sandbox pool, yielded to solution code that never reached CUDA.

The metric the watchdog scores is occupancy over time, not FLOPs: a card counts
as busy for any interval in which at least one kernel was resident, whether that
kernel used 4 SMs or 132. So what matters is a filler that never stops, and that
keeps running while the claimant runs too, filling the gaps the claimant leaves.

What does NOT follow from that is a tiny kernel. The first version of this
script used 1024x1024 on the theory that a small kernel leaves more of each SM
to the claimant, and measured 62% occupancy at a duty cycle of 0.95 (job
17136661): a 1024-cube fp16 matmul runs in a few microseconds on an H200, which
is less than the CPU-side cost of dispatching it, so the card idled waiting to
be fed. The theory was wrong anyway -- without MPS the two processes time-slice
whole contexts rather than sharing SMs, so kernel size buys no concurrency, only
preemption granularity, and Pascal onward preempts mid-kernel. Hence the 4096
default, which is what the original keepalive used to reach 99.95%.

How it behaves
--------------
Every card, claimed or not, is driven by the same controller. Each period it
reads the card's real utilization -- the claimant's contribution included, since
that is the number being scored -- and steps its own duty cycle toward
``TARGET_UTIL``:

* card genuinely busy (vLLM decoding, a solution actually training)
  -> reading is already at target -> duty decays to ``MIN_DUTY``, so the filler
  costs the real work essentially nothing.
* card idle or CPU-bound (sandbox running sklearn, rollout between bursts)
  -> reading is low -> duty climbs to the cap and holds the card above the floor.

That is the whole point of measuring instead of assuming: interference is paid
only where there is nothing to interfere with.

The cost, stated honestly
-------------------------
Without CUDA MPS, kernels from two processes time-slice rather than run side by
side, so while the controller is still backing off (one or two periods after a
claimant starts using its card) the claimant can lose up to about half of the
filler's current duty in throughput. ``CLAIMED_MAX_DUTY`` bounds that. Set
``CLAIMED_MAX_DUTY=0`` to get the original script's yield-on-claim behaviour.

That cap applies only while the claimant is *actually computing*, which is not
the same as holding the card. A solution that claimed a card and then spent its
whole timeout in pandas is contributing nothing to the reading, and capping the
filler at 0.6 there would park the card at 60% -- the exact number that got job
16904802 cancelled. So the claimant's own contribution is estimated by backing
this process's duty out of the reading, and the low cap engages only above
``CLAIMANT_BUSY_PCT``. Protect the claimant when there is a claimant to protect;
fill the card when there is not.

Configuration (all optional, all ``RLLM_GPU_SHARE_*``)
------------------------------------------------------
``DEVICES``            ``all`` (default), ``last``, or ``0,2,3``
``BUSY_DIR``           dir where ``card_<n>.busy`` marks a claimed card
``TARGET_UTIL``        utilization to steer each card to (default 92)
``MAX_DUTY``           duty cap on a free card (default 0.95)
``CLAIMED_MAX_DUTY``   duty cap while a claimed card's claimant is computing (0.6)
``CLAIMANT_BUSY_PCT``  estimated claimant utilization that engages that cap (15)
``CLAIMANT_IDLE_PCT``  estimate at which it disengages again (8)
``MIN_DUTY``           floor the controller may decay to (default 0.0)
``DUTY_STEP``          controller step per period (default 0.15)
``HYSTERESIS``         dead band above target, in points (default 4)
``INTERVAL_SEC``       control period (default 1.0; NVML samples about this fast)
``SLICE_SEC``          sub-slice the duty is spread over (default 0.1)
``SIZE``               filler matmul dimension (default 4096; see above -- too
                       small and the CPU cannot dispatch fast enough to keep
                       the card fed, which reads as low utilization)
``CHUNK``              launches per sync; 0 (default) sizes it from the measured
                       kernel time so each sync covers ``CHUNK_TARGET_MS``
``CHUNK_TARGET_MS``    milliseconds of work between syncs (default 5)
``REPORT_SEC``         per-card log cadence (default 60)
``LOG_CSV``            optional per-period sample log

``DEVICES`` and ``BUSY_DIR`` fall back to the ``RLLM_GPU_KEEPALIVE_*`` names, so
a launcher already exporting those can switch scripts without new wiring.

``--probe`` reads each selected card's utilization once, prints it, and exits;
use it to check the NVML path works before trusting the controller to a job.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

_RUNNING = True

DEFAULTS = {
    "target_util": 92.0,
    "max_duty": 0.95,
    "claimed_max_duty": 0.6,
    "claimant_busy_pct": 15.0,
    "claimant_idle_pct": 8.0,
    "min_duty": 0.0,
    "duty_step": 0.15,
    "hysteresis": 4.0,
    "interval_s": 1.0,
    "slice_s": 0.1,
    "size": 4096,
    "chunk": 0,  # 0 = size it from the measured kernel time (see chunk_for_launch_us)
    "chunk_target_ms": 5.0,
    "report_s": 60.0,
}


def _handle_stop(signum: int, _frame: object) -> None:
    global _RUNNING
    print(f"[gpu-share] received signal {signum}; stopping", flush=True)
    _RUNNING = False


def _env(name: str, fallback: str = "") -> str:
    """Read ``RLLM_GPU_SHARE_<name>``, falling back to the older keepalive name."""

    value = os.environ.get(f"RLLM_GPU_SHARE_{name}", "")
    if not value and fallback:
        value = os.environ.get(fallback, "")
    return value.strip()


def _float_env(name: str, default: float, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        print(f"[gpu-share] invalid {name}={raw!r}; using {default}", flush=True)
        return default
    parsed = max(parsed, minimum)
    return parsed if maximum is None else min(parsed, maximum)


def _int_env(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        print(f"[gpu-share] invalid {name}={raw!r}; using {default}", flush=True)
        return default
    return max(parsed, 1)


def parse_devices(spec: str, device_count: int) -> list[int]:
    """Resolve a device spec against the cards this process can see."""

    spec = (spec or "all").strip().lower()
    if spec in {"", "all"}:
        return list(range(device_count))
    if spec == "last":
        return [device_count - 1] if device_count else []

    devices: list[int] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            device = int(raw)
        except ValueError:
            print(f"[gpu-share] ignoring invalid device {raw!r}", flush=True)
            continue
        if 0 <= device < device_count:
            devices.append(device)
        else:
            print(f"[gpu-share] ignoring device {device}; visible count={device_count}", flush=True)
    return sorted(set(devices))


def card_is_claimed(busy_dir: str, device: int) -> bool:
    """Is someone holding this card for real work?

    Unreadable busy dir reads as "free": the filler is auxiliary, and a marker
    it cannot read must never be able to stop it, only to let it keep running.
    """
    if not busy_dir:
        return False
    try:
        return os.path.exists(os.path.join(busy_dir, f"card_{device}.busy"))
    except OSError:
        return False


def next_duty(
    duty: float,
    measured_util: float | None,
    *,
    target_util: float,
    max_duty: float,
    min_duty: float,
    duty_step: float,
    hysteresis: float,
) -> float:
    """One integral-control step toward ``target_util``.

    ``measured_util`` is the card's entire reading, this process's own filler
    included, because that is the number the watchdog scores. No attempt is made
    to separate the claimant's share from ours -- not being able to tell them
    apart is the useful part: whatever the claimant already supplies is duty we
    do not have to add, and a claimant holding the card at 100% drives this
    straight down to ``min_duty``.

    ``None`` means every probe failed. An unmeasurable card then runs at
    ``max_duty`` rather than at nothing, because an unmeasurable card is exactly
    the one whose reading gets a job cancelled.
    """
    if measured_util is None:
        return max_duty
    duty = min(duty, max_duty)  # a lowered cap (card just got claimed) binds now
    if measured_util < target_util:
        return min(max_duty, duty + duty_step)
    if measured_util > target_util + hysteresis:
        return max(min_duty, duty - duty_step)
    return duty


# Backing our own duty out of a reading divides by (1 - duty), so at duty 0.95 a
# one-point NVML wobble becomes twenty points of claimant estimate. Job 17158524
# oscillated on exactly that: a claimed but CPU-only card read 96 for one period,
# the cap engaged at 0.6, the reading fell to ~57, the cap released, and the card
# averaged 71% -- under the 80 the check wants and heading for the 60 the cluster
# cancels on. Past this amplification the estimate is not noisy, it is
# meaningless, and reporting nothing is the honest answer.
MAX_ESTIMATOR_AMPLIFICATION = 5.0

# Periods of consecutive evidence before the protective cap engages. One noisy
# sample must not be able to halve the duty cycle; a claimant that really is
# computing is still detected within a couple of seconds.
CLAIMANT_BUSY_PERIODS = 2


def estimate_claimant_util(
    measured_util: float | None,
    duty: float,
    *,
    max_amplification: float = MAX_ESTIMATOR_AMPLIFICATION,
) -> float | None:
    """Back this process's own filler out of a reading, leaving the claimant's share.

    Occupancy is a union, not a sum, so inverting it assumes the two processes'
    kernels are independent in time -- rough, but it only ever has to tell "the
    claimant is doing nothing" from "the claimant is doing real work".

    ``None`` when the answer cannot be trusted: an unreadable card, or a duty
    high enough that ``1 / (1 - duty)`` amplifies the reading's own error past
    ``max_amplification``. Declining to answer there costs nothing, because a
    claimant that is really computing pins the reading high, and the controller
    drops the duty on its own until this estimate becomes well conditioned again.
    """
    if measured_util is None:
        return None
    if duty > 1.0 - 1.0 / max_amplification:
        return None
    estimate = (measured_util - 100.0 * duty) / (1.0 - duty)
    return max(0.0, min(100.0, estimate))


def claimed_cap(
    state: tuple[bool, int],
    claimant_util: float | None,
    *,
    free_cap: float,
    busy_cap: float,
    busy_pct: float,
    idle_pct: float,
    busy_periods: int = CLAIMANT_BUSY_PERIODS,
) -> tuple[float, tuple[bool, int]]:
    """Decide this period's duty cap for a claimed card, plus the carried state.

    Holding a card is not the same as using it. A solution that claimed a card
    and then spent its timeout in pandas contributes nothing to the reading, and
    the protective cap would leave that card parked at the cap -- 60%, which is
    the cancellation floor. So the cap engages on evidence of real GPU work:
    ``busy_periods`` consecutive readings above ``busy_pct`` to engage, and a
    single reading below ``idle_pct`` to release, with the band between them
    keeping a claimant near the threshold from flipping the cap every period.

    Asymmetric on purpose. Engaging costs the claimant nothing it will not get
    back a period later; staying engaged against a claimant that has gone quiet
    costs the card 30 points of utilization for as long as the mistake lasts.

    An unreadable or unusable claimant estimate reads as idle, deliberately --
    the same rule ``next_duty`` applies to an unmeasurable card, because an
    unmeasurable card is the one whose reading gets a job cancelled.
    """
    active, streak = state
    if claimant_util is None:
        return free_cap, (False, 0)
    if active:
        active = claimant_util > idle_pct
        return (busy_cap if active else free_cap), (active, streak if active else 0)
    streak = streak + 1 if claimant_util > busy_pct else 0
    active = streak >= busy_periods
    return (busy_cap if active else free_cap), (active, streak)


class UtilProbe:
    """Reads a card's utilization the way the cluster's watchdog reads it.

    NVML via torch first: it maps a torch ordinal to the right physical card even
    under a narrowed CUDA_VISIBLE_DEVICES. Then nvidia-smi keyed by GPU UUID,
    because nvidia-smi's own index is physical and ours may not be. Then nothing,
    which the controller treats as "run open-loop". Each fallback is latched on
    first failure, so a broken probe costs one attempt rather than one per period.
    """

    def __init__(self, torch) -> None:
        self._torch = torch
        self._nvml_ok = True
        self._smi_ok = True
        self._uuids: dict[int, str] = {}

    @property
    def source(self) -> str:
        if self._nvml_ok:
            return "nvml"
        if self._smi_ok:
            return "nvidia-smi"
        return "none"

    def read(self, device: int) -> float | None:
        if self._nvml_ok:
            try:
                return float(self._torch.cuda.utilization(device))
            except Exception as exc:
                self._nvml_ok = False
                print(f"[gpu-share] NVML utilization unavailable ({exc}); trying nvidia-smi", flush=True)
        if self._smi_ok:
            return self._read_smi(device)
        return None

    def _device_uuid(self, device: int) -> str | None:
        if device not in self._uuids:
            try:
                uuid = getattr(self._torch.cuda.get_device_properties(device), "uuid", None)
            except Exception:
                uuid = None
            self._uuids[device] = str(uuid) if uuid else ""
        return self._uuids[device] or None

    def _read_smi(self, device: int) -> float | None:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except Exception as exc:
            self._smi_ok = False
            print(f"[gpu-share] nvidia-smi unavailable ({exc}); running open-loop", flush=True)
            return None
        rows = []
        for line in completed.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2:
                rows.append(parts)
        return parse_smi_utilization(rows, device, self._device_uuid(device))


def _bare_uuid(value: str) -> str:
    """Strip nvidia-smi's ``GPU-`` prefix. Spelled out rather than str.removeprefix
    because this script has to run under whatever python is on PATH, which is not
    always the trainer venv's, and removeprefix wants 3.9."""

    return value[4:] if value.startswith("GPU-") else value


def parse_smi_utilization(rows: list[list[str]], device: int, uuid: str | None) -> float | None:
    """Pick this device's utilization out of parsed nvidia-smi rows.

    UUID match when we have one; positional only as a last resort, which is
    correct whenever this process started before CUDA_VISIBLE_DEVICES was
    narrowed -- the same precondition the busy-marker indices already rely on.
    """
    if uuid:
        wanted = _bare_uuid(uuid)
        for smi_uuid, util in rows:
            if _bare_uuid(smi_uuid) == wanted:
                try:
                    return float(util)
                except ValueError:
                    return None
        return None
    if 0 <= device < len(rows):
        try:
            return float(rows[device][1])
        except ValueError:
            return None
    return None


def _burst(torch, a, b, out, work_s: float, chunk: int) -> None:
    """Run filler matmuls for ``work_s``, synchronising so the queue cannot overshoot.

    Enqueue is orders of magnitude faster than execution, so an unsynchronised
    loop would leave the card working long after the deadline and make the duty
    cycle fiction -- which on a shared card means stealing from the claimant.
    Syncing every ``chunk`` launches bounds the overshoot to one chunk's runtime.

    The sync is not free: it drains the queue, and the card idles until the CPU
    wakes and refills it. ``chunk`` is therefore sized in time rather than in
    launches -- see ``chunk_for_launch_us``.
    """
    deadline = time.monotonic() + work_s
    while _RUNNING and time.monotonic() < deadline:
        for _ in range(chunk):
            torch.mm(a, b, out=out)
        torch.cuda.synchronize()


def chunk_for_launch_us(launch_us: float, target_ms: float, *, limit: int = 512) -> int:
    """How many launches to queue between synchronisations.

    Each sync drains the queue and the card idles until the CPU wakes up and
    feeds it again, so the fixed cost of a sync is paid once per chunk: the
    shorter the chunk, the larger the share of wall time the card spends idle.
    Measured on an H200 with a hard-coded chunk of 8, a duty cycle of 0.95
    delivered only 79% occupancy (job 17148049) -- 17% of the requested time went
    into those gaps.

    Sizing the chunk in TIME rather than in launches makes that share a constant
    instead of a property of whatever GPU this lands on: a bigger card runs the
    same matmul faster and simply gets more launches per chunk. The ceiling is
    duty accuracy -- a chunk overshoots its deadline by at most its own runtime,
    which is why the target is milliseconds against a 100ms slice.
    """
    if launch_us <= 0:
        return 1
    return max(1, min(limit, int(round(target_ms * 1000.0 / launch_us))))


def measure_launch_us(torch, device: int, a, b, out, iters: int = 50) -> float:
    """Mean microseconds per filler matmul, dispatch included.

    The duty cycle is only honoured if the GPU can be kept fed, and it cannot be
    if a kernel finishes faster than python can queue the next one. Measuring it
    at startup turns that failure into a number in the log instead of an
    unexplained low reading hours later.
    """
    with torch.cuda.device(device):
        torch.mm(a, b, out=out)
        torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(iters):
            torch.mm(a, b, out=out)
        torch.cuda.synchronize()
    return (time.monotonic() - start) / iters * 1e6


def _run_period(*, torch, device: int, a, b, out, duty: float, interval_s: float, slice_s: float, chunk: int) -> None:
    """Spend one control period on this card at the given duty.

    The work is chopped into sub-slices rather than run as one block. The same
    duty spread thinly scores the same to a watchdog sampling occupancy, but it
    never stalls a co-resident claimant for a contiguous stretch.
    """
    if duty <= 0:
        time.sleep(interval_s)
        return
    slices = max(1, int(round(interval_s / slice_s))) if slice_s > 0 else 1
    with torch.cuda.device(device):
        for _ in range(slices):
            if not _RUNNING:
                return
            slice_start = time.monotonic()
            work_s = slice_s * duty
            if work_s > 0:
                _burst(torch, a, b, out, work_s, chunk)
            rest_s = slice_s - (time.monotonic() - slice_start)
            if rest_s > 0:
                time.sleep(rest_s)


def run_device(
    *,
    torch,
    device: int,
    buffers,
    settings: dict,
    busy_dir: str,
    probe: UtilProbe | None,
    csv_writer=None,
) -> None:
    """Control loop for one card: measure, step the duty, spend the period."""

    a, b, out = buffers
    # Bind this thread to the card and warm cuBLAS on it. The buffers were
    # allocated on the main thread, and without this the first matmul here runs
    # with no current context ("Attempting to run cuBLAS, but there was no
    # current CUDA context"), which torch fixes for us but only after a warning
    # and a slow first launch.
    torch.cuda.set_device(device)
    with torch.cuda.device(device):
        torch.mm(a, b, out=out)
        torch.cuda.synchronize()

    duty = settings["max_duty"]
    claimed = False
    claimant_state: tuple[bool, int] = (False, 0)
    last_report = 0.0
    while _RUNNING:
        now = time.monotonic()
        is_claimed = card_is_claimed(busy_dir, device)
        if is_claimed != claimed:
            claimed = is_claimed
            state = "claimed" if claimed else "free"
            print(f"[gpu-share] device {device} now {state}", flush=True)
            if not claimed:
                claimant_state = (False, 0)
        measured = probe.read(device) if probe is not None else None
        # Estimated against the duty that produced this reading, i.e. last
        # period's, which is the one still in `duty` at this point.
        claimant = estimate_claimant_util(measured, duty) if claimed else None
        if claimed:
            cap, claimant_state = claimed_cap(
                claimant_state,
                claimant,
                free_cap=settings["max_duty"],
                busy_cap=settings["claimed_max_duty"],
                busy_pct=settings["claimant_busy_pct"],
                idle_pct=settings["claimant_idle_pct"],
            )
        else:
            cap = settings["max_duty"]
        duty = next_duty(
            duty,
            measured,
            target_util=settings["target_util"],
            max_duty=cap,
            min_duty=settings["min_duty"],
            duty_step=settings["duty_step"],
            hysteresis=settings["hysteresis"],
        )
        if csv_writer is not None:
            csv_writer(device, measured, duty, claimed, claimant)
        if now - last_report >= settings["report_s"]:
            seen = "n/a" if measured is None else f"{measured:.0f}%"
            est = "n/a" if claimant is None else f"{claimant:.0f}%"
            print(
                f"[gpu-share] device {device}: util={seen} duty={duty:.2f} cap={cap:.2f} claimed={claimed} claimant={est}",
                flush=True,
            )
            last_report = now
        _run_period(
            torch=torch,
            device=device,
            a=a,
            b=b,
            out=out,
            duty=duty,
            interval_s=settings["interval_s"],
            slice_s=settings["slice_s"],
            chunk=settings["chunk"],
        )


def load_settings() -> dict:
    max_duty = _float_env("MAX_DUTY", DEFAULTS["max_duty"], maximum=1.0)
    return {
        "target_util": _float_env("TARGET_UTIL", DEFAULTS["target_util"], maximum=100.0),
        "max_duty": max_duty,
        # Capped by max_duty so a claimed card can never be filled harder than a
        # free one, whatever the two variables are set to independently.
        "claimed_max_duty": min(_float_env("CLAIMED_MAX_DUTY", DEFAULTS["claimed_max_duty"], maximum=1.0), max_duty),
        "claimant_busy_pct": _float_env("CLAIMANT_BUSY_PCT", DEFAULTS["claimant_busy_pct"], maximum=100.0),
        "claimant_idle_pct": _float_env("CLAIMANT_IDLE_PCT", DEFAULTS["claimant_idle_pct"], maximum=100.0),
        "min_duty": min(_float_env("MIN_DUTY", DEFAULTS["min_duty"], maximum=1.0), max_duty),
        "duty_step": _float_env("DUTY_STEP", DEFAULTS["duty_step"], minimum=0.01, maximum=1.0),
        "hysteresis": _float_env("HYSTERESIS", DEFAULTS["hysteresis"], maximum=100.0),
        "interval_s": _float_env("INTERVAL_SEC", DEFAULTS["interval_s"], minimum=0.1),
        "slice_s": _float_env("SLICE_SEC", DEFAULTS["slice_s"], minimum=0.01),
        "size": _int_env("SIZE", DEFAULTS["size"]),
        # 0 means "size it from the measured kernel time" and is the default;
        # an explicit RLLM_GPU_SHARE_CHUNK overrides the measurement.
        "chunk": max(0, int(_float_env("CHUNK", DEFAULTS["chunk"]))),
        "chunk_target_ms": _float_env("CHUNK_TARGET_MS", DEFAULTS["chunk_target_ms"], minimum=0.5),
        "report_s": _float_env("REPORT_SEC", DEFAULTS["report_s"]),
    }


def _open_csv(path: str):
    """Per-period samples, so the duty profile is measured rather than assumed."""

    if not path:
        return None, None
    try:
        handle = open(path, "a", buffering=1)  # noqa: SIM115 - lives for the process
    except OSError as exc:
        print(f"[gpu-share] cannot write {path} ({exc}); continuing without a CSV", flush=True)
        return None, None
    if handle.tell() == 0:
        handle.write("timestamp,device,util_gpu_pct,duty,claimed,claimant_util_pct\n")
    lock = threading.Lock()

    def write(device: int, util: float | None, duty: float, claimed: bool, claimant: float | None) -> None:
        row = f"{time.time():.0f},{device},{'' if util is None else f'{util:.0f}'},{duty:.2f},{int(claimed)},{'' if claimant is None else f'{claimant:.0f}'}\n"
        with lock:
            try:
                handle.write(row)
            except OSError:
                pass

    return write, handle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", action="store_true", help="print each card's utilization once and exit")
    args = parser.parse_args(argv)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _handle_stop)

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent.
        print(f"[gpu-share] torch import failed; exiting: {exc}", flush=True)
        return 0

    if not torch.cuda.is_available():
        print("[gpu-share] CUDA is not available; exiting", flush=True)
        return 0

    device_count = torch.cuda.device_count()
    devices = parse_devices(_env("DEVICES", "RLLM_GPU_KEEPALIVE_DEVICES") or "all", device_count)
    if not devices:
        print("[gpu-share] no valid devices selected; exiting", flush=True)
        return 0

    probe = UtilProbe(torch)
    if args.probe:
        for device in devices:
            value = probe.read(device)
            print(f"[gpu-share] device {device}: util={'unreadable' if value is None else f'{value:.0f}%'} via {probe.source}")
        return 0

    settings = load_settings()
    busy_dir = _env("BUSY_DIR", "RLLM_GPU_KEEPALIVE_BUSY_DIR")
    csv_writer, _csv_handle = _open_csv(_env("LOG_CSV"))
    size = settings["size"]

    buffers = []
    for device in devices:
        with torch.cuda.device(device):
            a = torch.randn((size, size), device="cuda", dtype=torch.float16)
            b = torch.randn((size, size), device="cuda", dtype=torch.float16)
            out = torch.empty((size, size), device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            buffers.append((device, (a, b, out)))

    print(
        f"[gpu-share] started devices={devices} target={settings['target_util']}% "
        f"max_duty={settings['max_duty']} claimed_max_duty={settings['claimed_max_duty']} "
        f"(engages above claimant {settings['claimant_busy_pct']}%) "
        f"size={size} interval_s={settings['interval_s']} slice_s={settings['slice_s']} "
        f"busy_dir={busy_dir or '<none>'} probe={probe.source}",
        flush=True,
    )

    # One number that explains a low reading before it happens: a filler kernel
    # shorter than its own dispatch cost cannot keep the card fed, and the
    # measured occupancy lands well under the duty the controller thinks it is
    # running (62% at duty 0.95 with size=1024 on an H200).
    launch_us = measure_launch_us(torch, buffers[0][0], *buffers[0][1])
    if settings["chunk"] == 0:
        settings["chunk"] = chunk_for_launch_us(launch_us, settings["chunk_target_ms"])
    print(
        f"[gpu-share] filler kernel: {launch_us:.0f}us per launch at size={size} -> chunk={settings['chunk']} ({settings['chunk'] * launch_us / 1000.0:.1f}ms per sync)",
        flush=True,
    )
    if launch_us < 50.0:
        print(
            f"[gpu-share] WARNING: {launch_us:.0f}us is near the dispatch floor; the card will idle between launches and read below the duty cycle. Raise RLLM_GPU_SHARE_SIZE.",
            flush=True,
        )

    with torch.no_grad():
        threads = [
            threading.Thread(
                target=run_device,
                kwargs={
                    "torch": torch,
                    "device": device,
                    "buffers": device_buffers,
                    "settings": settings,
                    "busy_dir": busy_dir,
                    "probe": probe,
                    "csv_writer": csv_writer,
                },
                name=f"gpu-share-{device}",
                daemon=True,
            )
            for device, device_buffers in buffers
        ]
        for thread in threads:
            thread.start()
        while _RUNNING and any(thread.is_alive() for thread in threads):
            time.sleep(0.5)
        for thread in threads:
            thread.join(timeout=settings["interval_s"] + 1.0)

    print("[gpu-share] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
