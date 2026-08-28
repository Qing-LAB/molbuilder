# ASU Sol — the reference deployment

**Role:** reference
**Domain:** execution

**Companions:**
[`execution/architecture.md`](?doc=execution/architecture.md) § 8 — the config
file whose `scheduler.routing` this fills in;
[`execution/generator.md`](?doc=execution/generator.md) § 4 — capability,
allocation and sweep, and why per-domain hardware is needed;
[`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 5 — the config
guide a person actually edits.

> **This is a snapshot of somebody else's facility, not a contract.**
> ASU Research Computing changes Sol without telling us. **Nothing in
> `molbuilder/` may hard-code a number from this page** — they belong in
> `molbuilder.json`, and this document exists to say *what to put there and why*.
> Sources and the date it was read are at the foot; re-read before trusting it.

---

## 1. Why this document exists

Sol is the machine molbuilder is actually used on, and the design was drawn
against a generic cluster. **Two things here are load-bearing rather than
informative:**

1. **A domain's limits are a property of the (partition, QOS) *pair*** — not of
   the partition. `general` is 7 days at `public` QOS and 14 at `long`.
   `molbuilder.json`'s `routing` already models a domain as a named bundle of
   partition + QOS + limits, which turns out to be exactly right.
2. **GPU nodes have fewer cores than CPU nodes** — 48 against 128. So *"run it
   with GPUs"* and *"run it on CPU"* are not the same machine with an
   accelerator bolted on; they are different node types with different core
   budgets. A benchmark sweeping *GPU vs no GPU*
   ([`generator.md`](?doc=execution/generator.md) § 4) crosses that boundary.

---

## 2. The hardware, by node type

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

**Totals:** 21,000+ cores, 113+ TB RAM, 290+ GPUs, seven 2 TB high-memory nodes.

> **The rows that matter for SIESTA are the first five.** The FPGA, Vector
> Engine, Grace Hopper and MI200 rows are recorded so nobody proposes them as a
> target without noticing they are a different architecture (`aarch64`, ROCm)
> than anything molbuilder's environments are built for.

---

## 3. Partitions and QOS — and why the pair is the unit

### 3.1 Partitions

| partition | max wall time | what it is |
|---|---|---|
| **public** | 7 days | Research-Computing-owned CPU nodes. **No preemption.** |
| **general** | 7 days | includes privately-owned nodes; CPU, GPU and FPGA |
| **htc** | **4 hours** | RC-owned *and* private nodes; jobs run **uninterrupted** |
| **highmem** | 2 days (7 with extended QOS) | the 2 TB nodes |
| **lightwork** | 1 day | oversubscribed; **max 8 cores per node**; for builds and light interactive work |
| fpga · arm | — | the accelerator and `aarch64` nodes |

### 3.2 QOS

| QOS | max wall time | preemptable | notes |
|---|---|---|---|
| **public** | by partition | no | the default |
| **debug** | **15 minutes** | no | `general` and `htc` only; for turnaround while troubleshooting |
| **private** | by arrangement | **yes** — by the hardware's owners | the trade for using privately-owned nodes beyond `htc`'s 4 h |
| **grp_<lab>** | by arrangement | conditional | lab-owned hardware; no fairshare impact for members |
| **long** | **14 days** | no | not available by default — granted per user |
| **class** | 24 hours | no | ≤ 32 cores, 320 GB, 4 GPUs, 2 running jobs |

> **This is why a "cluster" is the wrong unit and a named domain is the right
> one.** *"7 days"* is not a fact about `general`; it is a fact about
> `general` + `public`, and the same partition at `long` gives 14. A config that
> keyed limits to the partition would be wrong for half the pairs.

---

## 4. How resources are actually asked for

```bash
#SBATCH -p htc            # partition
#SBATCH -q public         # QOS
#SBATCH -t 0-4            # wall time, D-HH
#SBATCH -c 32             # cores
#SBATCH --mem=80GB        # memory for the whole job
#SBATCH -N 2              # nodes -- MPI only
#SBATCH -G a100:1         # GPUs, by type
```

**Facts with consequences:**

| fact | consequence for molbuilder |
|---|---|
| **memory defaults to 2 GB per core** when unspecified | a job asking `-c 128` and no `--mem` gets 256 GB, not the node's 512. The wrapper should not rely on the default |
| **`--mem=0` means the whole node's memory** | already the documented value of `Resources.mem` |
| **`-G` names the GPU *type*** — `a100:1`, `a100.40gb:1`, `a100.20gb:1`, `a30:2` | the MIG slices are separate askable types, not a smaller ask of the same one. A GPU sweep axis is over *(type, count)*, not a bare count |
| **`-N` only helps MPI** | true for SIESTA, false for the PySCF path — so it is not a global default |

---

## 5. What this settles for the design

### 5.1 The benchmark/run split is real, and it has names

[`generator.md`](?doc=execution/generator.md) § 4.4a says a benchmark normally
runs somewhere shorter and higher-priority than the run it informs. **On Sol
that is not an abstraction:**

| | domain | why |
|---|---|---|
| **the benchmark** | `htc` + `public` (4 h, uninterrupted), or `debug` (15 min) for a quick look | trials are minutes; the 4 h cap is no constraint and the queue is fast |
| **the real run** | `public` or `general` (7 days), `highmem` for big cells, `long` if granted | the calculation needs the time, and asking for more waits longer |

### 5.2 On Sol, transfer is decided per MACHINE — the partition cannot answer

*Rewritten 2026-08-27 against the probe's own record. What stood here is kept
below, because how it was wrong is the useful part.*

`generator.md` § 4.4a says `choice` (the mechanism) carries to another cluster
**"provided the node type is comparable"**. **A Sol partition cannot tell you
whether that holds**, because a partition is a queue and draws from many
machines (`scheduler.md` **R0**). The probe, run 2026-08-27:

| domain | machine types | cores across them | devices |
|---|---|---|---|
| `htc` · `debug` | **14** | 48 · 64 · 96 · 128 | a100 · a100.20gb · a100.40gb · a30 · h100 · h200 · l40 |
| `general` | **13** | 48 · 64 · 96 · 128 | a100 · a100.40gb · a30 · h100 · h200 · l40 |
| `public` | **4** | 48 · 128 | a100 · a100.20gb · a30 |
| `lightwork` | 2 | 48 · 128 | a100.20gb |
| `highmem` · `gaudi` · `arm` · `fpga` | 1 each | 128 · 152 · 72 · 48 | — · hl225 · gh200 · — |

**So "measured on `htc`, running on `public`" says nothing about comparability**
— either side could be a 48-core A100 node or a 128-core node with no device,
and the two differ by 2.7× in cores. What decides it is the machine each run
actually landed on, which is `scheduler.md` **R11**, and which is why **R12**
has the monitor record it on the node rather than inferring it from the queue.

> **⛔ What this section claimed until 2026-08-27, and why it was wrong.** It
> said *"all nodes in `public` and `general` are uniformly AMD EPYC … the same
> silicon"*, and `generator.md` § 4.4a starred it as the proof that
> transferability is checkable. Two failures, and they are different:
>
> * **Uniform vendor is not uniform hardware.** AMD EPYC is true and does not
>   support the conclusion: core counts inside one queue run 48–128, and the
>   section's own counter-example — *"a GPU node has 48 cores against a standard
>   node's 128"* — was already describing two machines **in the same partition**
>   without noticing.
> * **A stated figure nobody re-measured** — *"`lightwork` caps at 8 cores"*.
>   The probe finds **48- and 128-core nodes** there and `max_cores: 128`. It
>   was used to justify a transferability exception, so the number was doing
>   real work.
>
> **⚠ And that second one is not fully settled, which is worth being exact
> about.** Two different facts can both be called a cap: the **size of a node**
> and the **most cores a QOS lets one job ask for**. The probe measures the
> first and **does not measure the second at all** — `max_cores` is derived as
> the widest node (R3's corollary), never read from a QOS limit. So:
>
> * read as *node size*, the 8 is contradicted — the nodes are 48 and 128;
> * read as a *request cap*, the probe can neither confirm nor deny it, **and if
>   such a cap exists our `max_cores: 128` is wrong for admission** — R2 would
>   admit a 128-core job that the scheduler then refuses.
>
> **This needs a person on Sol** (`scontrol show partition lightwork`, or the
> QOS table) rather than another reading of the record. Recorded here rather
> than quietly resolved in whichever direction was convenient.
>
> Kept rather than deleted because the second is the recurring one: this page's
> job is to hold **measured** site facts, and a figure that arrives by memory
> reads exactly like one that arrives by probe.

### 5.3 Decision 38's shape, now concrete

The per-domain block `scheduler.routing` is missing needs exactly the columns
this page has:

```jsonc
{ "name": "bench",                  // what you type: --domain bench
  "partition": "htc", "qos": "public",
  "max_time": "0-04:00:00",
  "max_cores": 128,                 // ← the WIDEST machine, for admission (R3)
  "max_mem_gb": 512,
  "default_mem_per_core_gb": 2,
  "node_types": [                   // ← every machine the queue draws from (R0)
    { "cores": 128, "nodes": 134, "mem_gb": 503 },
    { "cores":  48, "nodes":  51, "mem_gb": 503, "gpu": { "a100": 4 } } ] }
```

**`node_types` is the field that does the new work**, and it is a **list**
because a queue holds a mixture. Everything else bounds an allocation; this is
the only place a machine can be named, which is what `scheduler.md` R11 needs.

> **This block declared a scalar `node_type: "standard"` until 2026-08-27**, and
> called it *"the transferability key"*. R0 retired it: a queue that holds 134
> standard nodes **and** 51 A100 nodes has no one node type, so no honest value
> existed to write — and the probe, correctly, never wrote one. A cluster that
> genuinely is uniform declares a single entry above.
>
> **The key was never the declaration anyway.** A domain says what a queue could
> give you; transfer is decided by what a run actually got (R12). Declaring a
> machine here would only have described the queue more precisely — it could not
> have said which of its machines your trial landed on.

---

## 6. Sources, and when they were read

Read **2026-08-11**. ASU RC changes these without notice — re-read before
relying on a number.

- [Sol Partitions and QoS](https://asurc.atlassian.net/wiki/spaces/RC/pages/1908867081/Sol+Partitions+and+QoS) — § 3
- [Supercomputer Hardware](https://docs.rc.asu.edu/supercomputer-hardware/) — § 2
- [Requesting Resources](https://docs.rc.asu.edu/requesting-resources/) — § 4
- [The Sol supercomputer](https://asurc.atlassian.net/wiki/spaces/RC/pages/1852637228/The+Sol+supercomputer) — § 2 totals

> **One discrepancy, recorded rather than resolved.** The PEARC23 paper
> ([Jennewein et al.](https://dl.acm.org/doi/fullHtml/10.1145/3569951.3597573))
> describes **four** partitions — `general`, `htc`, `highmem`, `lightwork` —
> while the current wiki lists those plus `public`, `fpga` and `arm`. The wiki
> is newer and is what § 3 follows; the paper is a 2023 snapshot of a facility
> that has grown.
