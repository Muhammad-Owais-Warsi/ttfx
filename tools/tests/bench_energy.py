"""Energy benchmark: how many joules one run of an effect costs.

Reads the CPU package energy counters (RAPL, via /sys/class/powercap) before
and after each run. The counters cover the whole package, so the figure is
everything the machine did in that window — ttfx, and with --sink tty the
terminal emulator and compositor drawing its output too. An idle baseline is
sampled on both sides of every run and subtracted to give the net cost of the
run itself; sampling it around each run keeps drift in the background load
from being booked as the effect's.

A CPU without RAPL (anything before Sandy Bridge) falls back to the battery's
discharge rate, which covers the whole machine, screen included. That only
reads anything while running on battery, so unplug the charger first.

Next to the energy, perf counts the instructions and cycles ttfx executed
(user space only, unless perf_event_paranoid allows more). Instructions repeat
almost exactly from run to run, so they are the figure to compare two builds
by; cycles follow the energy more closely, since a stalled core still draws
power. The columns stay empty when perf is not installed.

Unlike bench_full.py, frame pacing stays ON: energy is power over wall time,
so an effect has to run at the rate it is really used at.

The counters are readable by root only. Either run this under sudo (ttfx is
still run as the invoking user), or open them up until the next reboot:

    sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj

Usage: bench_energy.py [--frame-rate N] [--repeats N] [--idle-seconds N]
                       [--sink null|tty] [effect ...]
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUST = ROOT / "target/release/ttfx"

# Same canvas knobs as bench_full.py, so the two harnesses describe the same run.
COLS = os.environ.get("TTFX_BENCH_COLS", "100")
LINES = os.environ.get("TTFX_BENCH_LINES", "30")

ENV = {**os.environ, "COLUMNS": COLS, "LINES": LINES}


class Rapl:
    """The package-level energy counters, one per CPU socket."""

    def __init__(self, base: Path) -> None:
        # Top-level zones only (intel-rapl:0, not intel-rapl:0:0): the subzones
        # are slices of the package and would be counted twice.
        self.zones = [
            (z / "energy_uj", int((z / "max_energy_range_uj").read_text()))
            for z in sorted(base.glob("intel-rapl:*"))
            if z.name.count(":") == 1 and (z / "name").read_text().startswith("package")
        ]
        if not self.zones:
            raise SystemExit(f"no RAPL package zones under {base} — nothing to measure with")
        try:
            self.read()
        except PermissionError:
            raise SystemExit(
                "the RAPL energy counters are readable by root only. Run this under sudo, or:\n"
                f"    sudo chmod a+r {base}/intel-rapl:*/energy_uj"
            )

    name = "RAPL package"

    def read(self) -> list[int]:
        return [int(f.read_text()) for f, _ in self.zones]

    def joules(self, before: list[int], after: list[int]) -> float:
        total = 0
        for (_, wrap), b, a in zip(self.zones, before, after):
            total += a - b if a >= b else a - b + wrap
        return total / 1e6


class Battery:
    """The battery's discharge rate, integrated over time.

    The rate is sampled rather than the charge level differenced: the level
    moves in steps of tens of joules, more than a short effect costs.
    """

    name = "battery discharge"
    INTERVAL = 0.25

    def __init__(self, base: Path) -> None:
        found = [b for b in sorted(base.glob("*")) if (b / "power_now").exists()]
        if not found:
            raise SystemExit(f"no RAPL zones and no battery with a power reading under {base}")
        self.bats = found
        if not any((b / "status").read_text().strip() == "Discharging" for b in found):
            raise SystemExit("this machine has no RAPL, so the battery is the meter — unplug "
                             "the charger and run again, or pass --no-energy to only count")
        self.total, self.last, self.lock = 0.0, time.monotonic(), threading.Lock()
        threading.Thread(target=self.sample, daemon=True).start()

    def sample(self) -> None:
        while True:
            time.sleep(self.INTERVAL)
            self.read()

    def read(self) -> float:
        # Every read closes the slice since the previous one, so a window's
        # edges land on the instant asked for and not on the sampler's tick.
        with self.lock:
            watts = sum(int((b / "power_now").read_text()) for b in self.bats) / 1e6
            now = time.monotonic()
            self.total += watts * (now - self.last)
            self.last = now
            return self.total

    def joules(self, before: float, after: float) -> float:
        return after - before


class NoMeter:
    """Stands in when only the counts are wanted; the energy columns stay empty."""

    name = "none"

    def read(self) -> float:
        return 0.0

    def joules(self, before: float, after: float) -> float:
        return 0.0


def meter(powercap: Path, supply: Path):
    if any(powercap.glob("intel-rapl:*")):
        return Rapl(powercap)
    return Battery(supply)


def renamed(scratch: str) -> Path:
    """A copy of the binary under a name nothing else is looking for.

    Omarchy's lock and screensaver scripts `pkill -x ttfx` to stop the
    screensaver, which also ends a benchmark run that an idle machine locks
    in the middle of.
    """
    os.chmod(scratch, 0o755)
    return Path(shutil.copy(RUST, Path(scratch) / "ttfx-bench"))


def effects() -> list[str]:
    out = subprocess.run([str(RUST), "--help"], capture_output=True, text=True).stdout
    names, grab = [], False
    for line in out.splitlines():
        if line.startswith("Commands:"):
            grab = True
            continue
        if line.startswith("Options:"):
            break
        if grab and line.startswith("  ") and line.strip():
            n = line.split()[0]
            if n != "help":
                names.append(n)
    return names


def bench_input() -> bytes:
    filler = "the quick brown fox jumps over the lazy dog"
    if os.environ.get("TTFX_BENCH_FILL") == "1":
        rows, width = max(1, int(LINES) - 4), max(20, int(COLS) - 10)
        lines = [f"benchmark line {i:03d} — {filler}" for i in range(rows)]
        lines = [(l * (width // len(l) + 1))[:width] for l in lines]
    else:
        lines = [f"benchmark line {i:03d} — {filler}" for i in range(20)]
    return "\n".join(lines).encode()


def idle_watts(rapl, seconds: float) -> float:
    """Mean power with nothing of ours running."""
    e, t = rapl.read(), time.monotonic()
    time.sleep(seconds)
    return rapl.joules(e, rapl.read()) / (time.monotonic() - t)


PERF = shutil.which("perf")


def counters(path: str) -> tuple[int, int]:
    """(instructions, cycles) out of a `perf stat -x,` report."""
    found = {}
    for line in Path(path).read_text().splitlines():
        field = line.split(",")
        if len(field) > 2 and field[0].isdigit():  # skips "<not counted>"
            found[field[2].split(":")[0]] = int(field[0])
    return found.get("instructions", 0), found.get("cycles", 0)


def run(rapl, cmd: list[str], data: bytes, sink) -> tuple[float, float, int, int]:
    """One run: (wall seconds, joules, instructions, cycles)."""
    # Under sudo, only the counter needs root — the effect runs as the real user.
    drop = {}
    if os.geteuid() == 0 and "SUDO_UID" in os.environ:
        drop = {"user": int(os.environ["SUDO_UID"]), "group": int(os.environ["SUDO_GID"])}
    with tempfile.NamedTemporaryFile() as report:
        if "user" in drop:
            os.chown(report.name, drop["user"], drop["group"])
        if PERF:
            cmd = [PERF, "stat", "-x,", "-o", report.name, "-e", "instructions,cycles", "--", *cmd]
        e0, t0 = rapl.read(), time.monotonic()
        done = subprocess.run(cmd, input=data, stdout=sink, stderr=subprocess.DEVNULL, env=ENV, **drop)
        t1, e1 = time.monotonic(), rapl.read()
        if done.returncode != 0:
            raise SystemExit(f"{' '.join(cmd)} exited with {done.returncode}")
        instr, cycles = counters(report.name) if PERF else (0, 0)
        return t1 - t0, rapl.joules(e0, e1), instr, cycles


def main() -> int:
    ap = argparse.ArgumentParser(description="Energy used to run an effect, in joules.")
    ap.add_argument("effects", nargs="*", help="effects to measure (default: all)")
    ap.add_argument("--frame-rate", default="60")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--idle-seconds", type=float, default=3,
                    help="length of each baseline sample; one is taken between every two runs")
    ap.add_argument("--sink", choices=["null", "tty"], default="null",
                    help="null measures ttfx alone; tty draws on this terminal, so the "
                         "emulator and compositor are in the figure as well")
    ap.add_argument("--no-energy", action="store_true",
                    help="skip the energy meter and report the perf counts only")
    ap.add_argument("--powercap", type=Path, default=Path("/sys/class/powercap"))
    ap.add_argument("--power-supply", type=Path, default=Path("/sys/class/power_supply"))
    args = ap.parse_args()

    if args.no_energy:
        rapl, args.idle_seconds = NoMeter(), 0
    else:
        rapl = meter(args.powercap, args.power_supply)
    known = effects()
    unknown = [e for e in args.effects if e not in known]
    if unknown:
        raise SystemExit(f"unknown effect: {', '.join(unknown)}")
    data = bench_input()
    sink = subprocess.DEVNULL if args.sink == "null" else None

    # Each run is bracketed by two baseline samples; the one after a run is also
    # the one before the next, so the cost is one sample per run.
    idles = [idle_watts(rapl, args.idle_seconds)]
    scratch = tempfile.TemporaryDirectory()
    binary = renamed(scratch.name)

    rows = []
    for e in args.effects or known:
        cmd = [str(binary), "--seed", "1", "--frame-rate", args.frame_rate, e]
        runs = []
        for _ in range(args.repeats):
            secs, joules, instr, cycles = run(rapl, cmd, data, sink)
            idles.append(idle_watts(rapl, args.idle_seconds))
            before, after = idles[-2:]
            # Half the gap between the two baselines is how far off their mean
            # can be; over the run's length that is the error bar on its net.
            runs.append((joules - (before + after) / 2 * secs, joules,
                         abs(after - before) / 2 * secs, instr, cycles))
        # Median run by net energy: the counter sees the whole machine, so the
        # best run is as likely a lucky quiet moment as a true floor.
        rows.append((e, *sorted(runs)[len(runs) // 2]))

    text = data.decode().splitlines()
    print(f"canvas: {COLS}x{LINES} · input: {len(text)} lines x {max(len(l) for l in text)} cols"
          f" · {args.frame_rate} fps · sink {args.sink} · median of {args.repeats}")
    print(f"meter: {rapl.name}")
    if not args.no_energy:
        print(f"idle: {statistics.mean(idles):.2f} W ± {statistics.pstdev(idles):.2f},"
              f" {min(idles):.2f}–{max(idles):.2f} over {len(idles)} samples"
              f" of {args.idle_seconds:g} s")
    print()

    rows.sort(key=lambda r: -r[5] if args.no_energy else -r[1])
    print(f"{'effect':<17}{'joules':>10}{'net J':>10}{'± J':>9}{'M instr':>10}{'M cycles':>10}")
    print("-" * 66)
    for e, net, joules, error, instr, cycles in rows:
        energy = f"{joules:10.2f}{net:10.2f}{error:9.2f}" if not args.no_energy else " " * 29
        count = f"{instr/1e6:10.1f}{cycles/1e6:10.1f}" if instr else ""
        print(f"{e:<17}{energy}{count}")
    print("-" * 66)
    if not args.no_energy:
        print("net = above the idle baseline sampled before and after the run; ± J is how far")
        print("that baseline moved across it. A net J inside its ± is noise, not a cost.")
        print("1 Wh = 3600 J.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
