# ASU Sol — the reference deployment

**Role:** reference
**Domain:** execution
**Owns:** a dated snapshot of what ASU Sol offers — its published hardware,
partitions and QoS, and how its probed record reads — and the Sol examples the
design was drawn against.
**Read with:** [`scheduler.md`](?doc=execution/scheduler.md) — the queue model,
the fit, the header · [`configuration.md`](?doc=configuration.md) § 5 — the
machine record · [`generator.md`](?doc=execution/generator.md) § 4.4a — where a
benchmark runs

## The short version

**The live source for Sol is its machine record**, written by `jobset probe` on
Sol and read by `molbuilder jobset machines`: every queue the account may use,
each with its wall, its caps and the machines it draws from. This page is a
snapshot of somebody else's facility — the published facts as read on
2026-08-11 (§ 2, § 3) beside the probed record of 2026-10-19 (§ 3.3, § 5.2) —
and ASU Research Computing changes Sol without telling us. **Nothing in
`molbuilder/` may hard-code a number from this page**: each job states its own
queue, wall, memory and shape
([`architecture.md`](?doc=execution/architecture.md) § 5.2). Two facts here are
load-bearing for the design (§ 1); where the published page and the record
differ, the record is what prep reads.

---

## 1. Why this page exists

Sol is the machine molbuilder is used on. Two of its facts shaped the design:

1. **A queue's limits belong to the (partition, QoS) pair**, not to the
   partition: `htc` at the `debug` QoS allows 15 minutes, at `public` 4 hours.
   The machine record models a queue — a *domain* — as that pair
   ([`scheduler.md`](?doc=execution/scheduler.md) § 4).
2. **One partition holds many machines.** GPU nodes have 48 cores and standard
   nodes 128, and `htc`, `public` and `general` each hold both, so "run it with
   GPUs" and "run it on CPU" land on different machines with different core
   budgets ([`scheduler.md`](?doc=execution/scheduler.md) R0, R11).

## 2. The hardware, by node type — as published

| node type | CPU | cores | memory | accelerator |
|---|---|--:|--:|---|
| **Standard Compute** | 2× AMD EPYC 7713 Zen3 | **128** | 512 GiB | — |
| **High Memory** | 2× AMD EPYC 7713 Zen3 | **128** | **2048 GiB** | — |
| **GPU A100** | 2× AMD EPYC 7413 Zen3 | **48** | 512 GiB | 4× NVIDIA A100 80 GiB |
| **GPU A30** | 2× AMD EPYC 7413 Zen3 | **48** | 512 GiB | 3× NVIDIA A30 24 GiB |
| **GPU MIG** | 2× AMD EPYC 7413 Zen3 | **48** | 512 GiB | 16× A100 sliced (20/10 GiB) |
| GraceHopper | NVIDIA Grace (aarch64) | 72 | 512 GiB | 1× GH200 480 GB |
| Xilinx FPGA | 2× AMD EPYC 7443 Zen3 | 48 | 256 GiB | 1× Xilinx U280 |
| NEC | 2× AMD EPYC 9274F Zen4 | 48 | 512 GiB | 1× NEC Vector Engine |
| GPU MI200 | AMD EPYC 9254 | 24 | 77 GiB | 2× AMD MI200 |

**Totals, as published:** 21,000+ cores, 113+ TB RAM, 290+ GPUs, seven 2 TB
high-memory nodes.

The rows that matter for SIESTA are the first five. The FPGA, Vector Engine,
Grace Hopper and MI200 rows are a different architecture (`aarch64`, ROCm) from
anything molbuilder's environments are built for.

The probed record differs in places — its `highmem` queue lists 11 nodes of
128 cores at 1007.3 GB, and newer GPU nodes (H100, H200, L40) appear in `htc` and
`general` (§ 5.2). The record is what the fit reads.

## 3. Partitions and QoS

### 3.1 Partitions

The wall is the probed record's, at the `public` QoS (2026-10-19); the role is
the published page's:

| partition | wall at `public` | what it is |
|---|---|---|
| **htc** | **4 hours** | RC-owned and private nodes; jobs run uninterrupted |
| **public** | 7 days | RC-owned nodes; no preemption |
| **general** | **14 days** | includes privately-owned nodes; CPU and GPU |
| **highmem** | 7 days | the high-memory nodes |
| **lightwork** | 1 day | oversubscribed; for builds and light interactive work — **no policy cap on cores in the record**, nodes of 48 and 128 cores |
| `gaudi` · `arm` · `fpga` | 7 days | accelerator and `aarch64` nodes |

### 3.2 QoS

| QoS | wall | preemptable | in the probed record |
|---|---|---|---|
| **public** | by partition | no | on every partition; 300 000 submitted jobs per user |
| **debug** | **15 minutes** | no | on every partition (`htc-debug`, `public-debug`, …); **2 submitted jobs per user** |
| **private** | by partition | **yes** — by the hardware's owners | on every partition (`htc-private`, …); 10 000 submitted jobs per user |
| **grp_&lt;lab&gt;** | by arrangement | conditional | not in this account's record |
| **long** | 14 days (published) | no | not in this account's record — granted per user |
| **class** | 24 hours (published) | no | not in this account's record |

The published page lists `debug` on `general` and `htc` only; the probed account
holds it on every partition. The record's `debug` cap of 2 submitted jobs is the
limit a benchmark meets ([`scheduler.md`](?doc=execution/scheduler.md) R14).

### 3.3 How the pair is named

Each pair is one domain; the one at `public` takes the partition's name, every
other `<partition>-<qos>` ([`scheduler.md`](?doc=execution/scheduler.md) § 4). On
Sol the record holds 24: `htc`, `htc-debug`, `htc-private`, `public`,
`public-debug`, `public-private`, `general`, `general-debug`, `general-private`,
and the same three for `highmem`, `lightwork`, `gaudi`, `arm` and `fpga`.

## 4. How resources are asked for on Sol

```bash
#SBATCH -p htc            # partition
#SBATCH -q public         # QoS
#SBATCH -t 0-4            # wall time, D-HH
#SBATCH -c 32             # cores
#SBATCH --mem=80GB        # memory for the whole job
#SBATCH --gres=gpu:1      # GPUs -- a count (Sol also takes a type, e.g. a100:1)
```

| Sol's fact | what molbuilder does |
|---|---|
| **memory defaults to 2 GB per core** when unspecified (the record's `default_mem_per_core_gb: 2.0`) | never relies on it: a job states its memory, or prep refuses ([`architecture.md`](?doc=execution/architecture.md) § 5.2) |
| **`--mem=0` means the whole node's memory** | read as no stated limit ([`scheduler.md`](?doc=execution/scheduler.md) R2) |
| **a GPU ask can name the type** — `a100:1`, `a100.40gb:1`, `a100.20gb:1`, `a30:2` | asks a count, `--gres=gpu:N`, and names no card ([`gpu.md`](?doc=execution/gpu.md) § 1.2); the monitor's log (`.monitor.log`) records the card it landed on, in its `[MACHINE]` line ([`scheduler.md`](?doc=execution/scheduler.md) R12) |
| **`-N` spreads a job over nodes** | every header asks one node, `-N 1` ([`scheduler.md`](?doc=execution/scheduler.md) § 6a) |

## 5. What this settles for the design

### 5.1 The benchmark and the run go to different queues

A benchmark runs somewhere shorter and higher-priority than the run it informs
([`generator.md`](?doc=execution/generator.md) § 4.4a). On Sol, by the record's
walls:

| | domain | why |
|---|---|---|
| **the benchmark** | `htc` (4 h, uninterrupted), or `htc-debug` (15 min, 2 jobs) for a quick look | trials are minutes; the queue is fast |
| **the real run** | `public` (7 d), `general` (14 d), `highmem` (7 d) for big cells | the calculation needs the time |

The run card's own `time` and `domain` make that split
([`job-request.md`](?doc=execution/job-request.md) § 3.2a).

### 5.2 On Sol, comparability is decided per machine

Whether a measurement on one queue describes a run on another is decided by the
machine each landed on, never by the queue
([`scheduler.md`](?doc=execution/scheduler.md) R11, R12). A Sol queue cannot
answer it, because each draws from several machines. The probed record
(2026-10-19), at the `public` QoS:

| domain | nodes | machine types | cores | devices |
|---|--:|--:|---|---|
| `htc` | 212 | 14 | 48 · 64 · 96 · 128 | a100 · a100.20gb · a100.40gb · a30 · h100 · h200 · h200.35gb · l40 · gpu |
| `general` | 100 | 13 | 48 · 64 · 96 · 128 | a100 · a100.40gb · a30 · h100 · h200 · h200.35gb · l40 · gpu |
| `public` | 166 | 4 | 48 · 128 | a100 · a100.20gb · a30 |
| `lightwork` | 3 | 2 | 48 · 128 | a100.20gb |
| `highmem` | 11 | 1 | 128 | — |
| `gaudi` · `arm` · `fpga` | 10 · 4 · 5 | 1 · 1 · 2 | 152 · 72 · 48 | hl225 · gh200 · mi200 |

So "measured on `htc`, running on `public`" says nothing about comparability:
either side could be a 48-core A100 node or a 128-core node with no device.

### 5.3 A queue's row in Sol's record

`htc`, as the probe wrote it — the fields are
[`scheduler.md`](?doc=execution/scheduler.md) § 2's (machines abridged):

```jsonc
{ "name": "htc", "partition": "htc", "qos": "public",
  "max_time": "4:00:00",
  "max_cores": 128, "max_mem_gb": 1511.2,
  "default_mem_per_core_gb": 2.0,
  "max_cpus_per_job": null, "max_cpus_per_node": null,
  "max_submit_jobs": 300000,
  "gpu": { "a100.40gb": 4, "a100": 4, "h100": 8, "a30": 3, "l40": 4,
           "h200": 1, "h200.35gb": 4, "gpu": 1, "a100.20gb": 16 },
  "node_types": [
    { "cores": 128, "nodes": 134, "mem_gb": 503.2 },
    { "cores":  64, "nodes":   4, "mem_gb": 503.5, "gpu": { "a100.40gb": 4 } },
    …
    { "cores":  48, "nodes":  51, "mem_gb": 503.5, "gpu": { "a100": 4 } },
    …  ] }
```

## 6. Sources, and when they were read

The published pages, read **2026-08-11** — ASU RC changes these without notice:

- [Sol Partitions and QoS](https://asurc.atlassian.net/wiki/spaces/RC/pages/1908867081/Sol+Partitions+and+QoS) — § 3
- [Supercomputer Hardware](https://docs.rc.asu.edu/supercomputer-hardware/) — § 2
- [Requesting Resources](https://docs.rc.asu.edu/requesting-resources/) — § 4
- [The Sol supercomputer](https://asurc.atlassian.net/wiki/spaces/RC/pages/1852637228/The+Sol+supercomputer) — § 2 totals

The machine record, probed on Sol **2026-10-19T16:34Z** (`environments/sol.json`,
schema `molbuilder/environment@3`) — § 3.1's walls, § 3.2's record column, § 3.3,
§ 5. Re-probe on Sol for the current answer; `jobset machines` prints when a
record was measured.

The PEARC23 paper ([Jennewein et al.](https://dl.acm.org/doi/fullHtml/10.1145/3569951.3597573))
describes four partitions — `general`, `htc`, `highmem`, `lightwork` — where the
wiki lists more; the wiki is newer and is what § 3 follows.
