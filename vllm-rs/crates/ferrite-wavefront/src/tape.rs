// SPDX-License-Identifier: Apache-2.0
//! The **Tape** — the scheduled, replayable form of a [`SubtileGraph`].
//!
//! The DAG in `subtile.rs` is pure dataflow: producer→consumer edges are
//! implicit in [`Operand::Sub`]. Scheduling assigns each node to one of
//! `P` workers (the persistent megakernel's co-resident threadgroups) and
//! turns every *cross-worker* edge into an explicit point-to-point
//! handshake: the producer's worker `Signal`s a one-shot flag after
//! computing it; the consumer's worker `Wait`s on that flag before
//! reading it. Same-worker edges need no flag — program order on that
//! worker already serializes them. This is the design's point-to-point
//! sync (measured 0.18 µs/hop, flat in core count), NOT a global barrier.
//!
//! Two consumers here:
//!   - [`play`] — the **host tape player**. Replays a schedule on the CPU
//!     honoring `Wait`/`Signal`, computing each node with the *same*
//!     [`eval_node`] arithmetic as [`eval_dag`]. Tier A asserts the two
//!     agree bit-for-bit, so the schedule + edge stamping are proven
//!     correct on the host before any GPU/MSL.
//!   - (later) GPU emission — the same `Schedule` becomes `P` MSL tapes.
//!
//! [`partition_roundrobin`] is a deliberately dumb partitioner used as a
//! *test fixture* to exercise the player across many worker counts. The
//! real wavefront scheduler (modulo-scheduled, p1/p2-optimized) produces
//! a better `Schedule` that the unchanged [`play`] runs.

use crate::subtile::{Operand, SubtileGraph, SubtileId, eval_node};

/// One instruction in a worker's tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapeInstr {
    /// Compute this subtile (its inputs are guaranteed ready: same-worker
    /// producers ran earlier in this tape; cross-worker producers are
    /// gated by a preceding `Wait`).
    Compute(SubtileId),
    /// Set one-shot flag `f` — the producer side of a cross-worker edge.
    Signal(u32),
    /// Block until one-shot flag `f` is set — the consumer side.
    Wait(u32),
}

/// One worker's ordered instruction stream.
#[derive(Clone, Debug, Default)]
pub struct Worker {
    pub tape: Vec<TapeInstr>,
}

/// A full schedule: `P` worker tapes plus the flag count they reference.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub workers: Vec<Worker>,
    pub num_flags: u32,
}

impl Schedule {
    /// Total number of `Compute` instructions across all workers — must
    /// equal the node count (every node is computed exactly once).
    pub fn total_computes(&self) -> usize {
        self.workers
            .iter()
            .flat_map(|w| &w.tape)
            .filter(|i| matches!(i, TapeInstr::Compute(_)))
            .count()
    }
}

/// Build a [`Schedule`] from a node→worker assignment (`worker_of[id]`).
/// Each worker computes its nodes in ascending-id (topological) order; a
/// one-shot flag is allocated per producer that has any cross-worker
/// consumer; consumers `Wait` on it, the producer `Signal`s it.
///
/// **Deadlock-free for *any* assignment:** every dependency has a smaller
/// id than its consumer, and each worker runs in ascending-id order, so a
/// producer's `Signal` always precedes — in its own worker's tape — the
/// point at which any consumer could block on it. (The player still
/// checks for deadlock defensively.) This is why the real scheduler only
/// has to choose a good `worker_of`; correctness of the resulting tape is
/// guaranteed here.
pub fn schedule_from_assignment(
    graph: &SubtileGraph,
    worker_of: &[u32],
    num_workers: usize,
) -> Schedule {
    let n = graph.nodes.len();
    assert_eq!(worker_of.len(), n, "worker_of must cover every node");

    // A producer needs a flag iff some consumer lives on another worker.
    let mut needs_flag = vec![false; n];
    for node in &graph.nodes {
        let cw = worker_of[node.id.0 as usize];
        for inp in &node.inputs {
            if let Operand::Sub(prod) = inp
                && worker_of[prod.0 as usize] != cw
            {
                needs_flag[prod.0 as usize] = true;
            }
        }
    }
    let mut flag_of = vec![u32::MAX; n];
    let mut num_flags = 0u32;
    for (i, &nf) in needs_flag.iter().enumerate() {
        if nf {
            flag_of[i] = num_flags;
            num_flags += 1;
        }
    }

    let mut workers: Vec<Worker> = (0..num_workers).map(|_| Worker::default()).collect();
    for node in &graph.nodes {
        let w = worker_of[node.id.0 as usize] as usize;
        // One Wait per distinct cross-worker producer flag.
        let mut waited: Vec<u32> = Vec::new();
        for inp in &node.inputs {
            if let Operand::Sub(prod) = inp
                && worker_of[prod.0 as usize] != worker_of[node.id.0 as usize]
            {
                let f = flag_of[prod.0 as usize];
                debug_assert_ne!(f, u32::MAX, "cross-worker producer must have a flag");
                if !waited.contains(&f) {
                    waited.push(f);
                    workers[w].tape.push(TapeInstr::Wait(f));
                }
            }
        }
        workers[w].tape.push(TapeInstr::Compute(node.id));
        if needs_flag[node.id.0 as usize] {
            workers[w]
                .tape
                .push(TapeInstr::Signal(flag_of[node.id.0 as usize]));
        }
    }

    Schedule { workers, num_flags }
}

/// Round-robin partition (node `i` → worker `i % p`) — a test fixture for
/// the player; the real wavefront scheduler in `scheduler.rs` produces a
/// balanced assignment and feeds the same [`schedule_from_assignment`].
pub fn partition_roundrobin(graph: &SubtileGraph, p: u32) -> Schedule {
    let p = p.max(1) as usize;
    let worker_of: Vec<u32> = (0..graph.nodes.len()).map(|i| (i % p) as u32).collect();
    schedule_from_assignment(graph, &worker_of, p)
}

/// Replay a schedule on the host. Returns per-node output buffers
/// (indexed by `SubtileId`), identical to [`eval_dag`] when the
/// schedule's sync is correct. Panics on deadlock (a missing `Signal`,
/// or cyclic `Wait`s) and on any node computed before a producer it reads
/// (a missing `Wait` edge) — both are the bugs host-first is meant to
/// catch.
///
/// [`eval_dag`]: crate::subtile::eval_dag
pub fn play(graph: &SubtileGraph, schedule: &Schedule, sources: &[&[f32]]) -> Vec<Vec<f32>> {
    let n = graph.nodes.len();
    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); n];
    let mut computed = vec![false; n];
    let mut flags = vec![false; schedule.num_flags as usize];
    let mut pc = vec![0usize; schedule.workers.len()];

    loop {
        let mut progressed = false;
        let mut all_done = true;
        for (wi, worker) in schedule.workers.iter().enumerate() {
            if pc[wi] >= worker.tape.len() {
                continue;
            }
            all_done = false;
            match worker.tape[pc[wi]] {
                TapeInstr::Wait(f) => {
                    if flags[f as usize] {
                        pc[wi] += 1;
                        progressed = true;
                    }
                    // else: this worker stays blocked this round.
                }
                TapeInstr::Signal(f) => {
                    flags[f as usize] = true;
                    pc[wi] += 1;
                    progressed = true;
                }
                TapeInstr::Compute(id) => {
                    let node = &graph.nodes[id.0 as usize];
                    for inp in &node.inputs {
                        if let Operand::Sub(prod) = inp {
                            assert!(
                                computed[prod.0 as usize],
                                "node {} computed before producer {} (missing Wait edge)",
                                id.0, prod.0
                            );
                        }
                    }
                    outs[id.0 as usize] = eval_node(node, graph, sources, &outs);
                    computed[id.0 as usize] = true;
                    pc[wi] += 1;
                    progressed = true;
                }
            }
        }
        if all_done {
            break;
        }
        assert!(
            progressed,
            "deadlock: no worker advanced (missing Signal or cyclic Waits)"
        );
    }

    // Every node must have been computed exactly once.
    debug_assert!(computed.iter().all(|&c| c), "some node never computed");
    outs
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subtile::{TilingPolicy, assemble_result, eval_dag, lower_gemm_standalone};

    /// Deterministic f32 fill in `[-1, 1)` (mirrors subtile.rs's test
    /// helper — kept local to avoid a cross-module test-only export).
    fn rng_fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (s >> 33) as u32;
                (bits as f32 / 2147483648.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// Tier A: scheduled tape replay reproduces direct topological eval
    /// bit-for-bit, across worker counts and tilings (incl. split-K, so
    /// cross-worker reduce edges are exercised).
    #[test]
    fn tape_replay_matches_eval_dag_bit_exact() {
        let (m, n, k) = (2u32, 80, 192);
        let a = rng_fill((m * k) as usize, 21);
        let w = rng_fill((n * k) as usize, 22);
        for &p in &[1u32, 2, 3, 4, 8, 16] {
            for policy in [
                TilingPolicy {
                    nb: 16,
                    k_chunks: 1,
                },
                TilingPolicy {
                    nb: 24,
                    k_chunks: 3,
                },
                TilingPolicy { nb: 8, k_chunks: 4 },
            ] {
                let g = lower_gemm_standalone(m, n, k, policy);
                let want = eval_dag(&g, &[&a, &w]);
                let sched = partition_roundrobin(&g, p);
                let got = play(&g, &sched, &[&a, &w]);
                assert_eq!(got, want, "per-node mismatch at p={p}, policy={policy:?}");
                assert_eq!(
                    assemble_result(&g, &got),
                    assemble_result(&g, &want),
                    "assembled mismatch at p={p}, policy={policy:?}"
                );
            }
        }
    }

    /// Every node computed exactly once; flags only where edges cross.
    #[test]
    fn partition_invariants() {
        let g = lower_gemm_standalone(
            2,
            100,
            200,
            TilingPolicy {
                nb: 32,
                k_chunks: 4,
            },
        );

        // p=1: no cross-worker edges → no flags, no Wait/Signal.
        let s1 = partition_roundrobin(&g, 1);
        assert_eq!(s1.num_flags, 0);
        assert_eq!(s1.total_computes(), g.nodes.len());
        assert!(
            s1.workers
                .iter()
                .flat_map(|w| &w.tape)
                .all(|i| matches!(i, TapeInstr::Compute(_)))
        );

        // p>1: every node still computed exactly once; some edges cross.
        let s4 = partition_roundrobin(&g, 4);
        assert_eq!(s4.total_computes(), g.nodes.len());
        assert!(
            s4.num_flags > 0,
            "split-K reduces must cross workers at p=4"
        );

        // Signals and Waits are balanced: each one-shot flag is signaled
        // exactly once and waited at least once.
        let mut signals = vec![0u32; s4.num_flags as usize];
        let mut waits = vec![0u32; s4.num_flags as usize];
        for w in &s4.workers {
            for i in &w.tape {
                match i {
                    TapeInstr::Signal(f) => signals[*f as usize] += 1,
                    TapeInstr::Wait(f) => waits[*f as usize] += 1,
                    TapeInstr::Compute(_) => {}
                }
            }
        }
        for f in 0..s4.num_flags as usize {
            assert_eq!(signals[f], 1, "flag {f} signaled exactly once");
            assert!(waits[f] >= 1, "flag {f} waited at least once");
        }
    }

    /// Each worker computes its nodes in ascending id (topological) order
    /// — the property that makes the round-robin partition deadlock-free.
    #[test]
    fn per_worker_computes_are_topological() {
        let g = lower_gemm_standalone(
            3,
            64,
            256,
            TilingPolicy {
                nb: 16,
                k_chunks: 4,
            },
        );
        let s = partition_roundrobin(&g, 4);
        for w in &s.workers {
            let mut last = None;
            for i in &w.tape {
                if let TapeInstr::Compute(id) = i {
                    if let Some(prev) = last {
                        assert!(id.0 > prev, "computes must ascend: {prev} then {}", id.0);
                    }
                    last = Some(id.0);
                }
            }
        }
    }

    /// The player handles chained `Sub` edges (silu → mul) that cross
    /// workers, not just gemm split-K reduce trees.
    #[test]
    fn tape_replay_elementwise_chain_bit_exact() {
        use crate::subtile::lower_gate_up_silu_mul_standalone;
        let (m, d) = (2u32, 96);
        let gate = rng_fill((m * d) as usize, 51);
        let up = rng_fill((m * d) as usize, 52);
        let g = lower_gate_up_silu_mul_standalone(m, d, 8);
        let want = eval_dag(&g, &[&gate, &up]);
        for &p in &[1u32, 2, 3, 5] {
            let sched = partition_roundrobin(&g, p);
            let got = play(&g, &sched, &[&gate, &up]);
            assert_eq!(got, want, "elementwise chain replay p={p}");
        }
    }

    /// The fused decode-attention graph (rope_q / rope_k → attn) replays
    /// bit-exact — the new token rides a cross-worker `Sub` edge, never
    /// the cache.
    #[test]
    fn tape_replay_fused_attention_bit_exact() {
        use crate::subtile::lower_decode_attention_standalone;
        let (hq, hkv, hd, l) = (4u32, 2u32, 8u32, 5u32);
        let (qdim, kvdim) = ((hq * hd) as usize, (hkv * hd) as usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let data: Vec<Vec<f32>> = vec![
            rng_fill(qdim, 81),
            rng_fill(kvdim, 82),
            rng_fill(kvdim, 83),
            rng_fill(hd as usize, 84),
            rng_fill(hd as usize, 85),
            rng_fill(l as usize * kvdim, 86),
            rng_fill(l as usize * kvdim, 87),
        ];
        let srcs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let g = lower_decode_attention_standalone(1, hq, hkv, hd, l, scale);
        let want = eval_dag(&g, &srcs);
        for &p in &[1u32, 2, 3] {
            let sched = partition_roundrobin(&g, p);
            let got = play(&g, &sched, &srcs);
            assert_eq!(got, want, "fused attention replay p={p}");
        }
    }
}
