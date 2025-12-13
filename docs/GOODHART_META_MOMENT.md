# The Day We Proved Goodhart's Law (On Ourselves)

> "When a measure becomes a target, it ceases to be a good measure." — Goodhart

**Date:** 2025-12-12

## What Happened

We set out to build a project demonstrating Goodhart's Law through agent simulations.
We accidentally demonstrated it through our own development process.

## The Irony

| What We Did | The Goodhart Trap |
|-------------|-------------------|
| Used **behavior cloning** (copy the expert) | "Expert mimicry" became our proxy for "good agent" |
| OmniscientSeeker avoids poison perfectly | Training data had **zero poison examples** |
| CNN learned to go toward food | CNN had **no idea** what to do with poison |
| Added synthetic poison samples as a patch | Band-aid on a fundamentally misaligned objective |
| Still got worse survival than hardcoded | **Proxy optimization failed to generalize** |

## The Numbers

```
HARDCODED EXPERTS:
  OmniscientSeeker: 97% survival, 0 poison deaths
  ProxySeeker:      43% survival, 3.7 poison deaths

BEHAVIOR-CLONED CNNs:
  Learned GT CNN:   23% survival, 1.7 poison deaths  ← WORSE than hardcoded!
  Learned Proxy:    20% survival, 1.7 poison deaths

The student (CNN) failed to surpass the teacher (heuristic)
because we optimized for COPYING, not for SURVIVING.
```

## The Lesson

**Behavior cloning optimizes for "do what the expert did"**
...which is a PROXY for "be good at the task."

When we discovered gaps (poison avoidance), we patched them with synthetic data.
But the fundamental misalignment remained: we were teaching imitation, not survival.

## The Pivot

The solution? **True Reinforcement Learning.**

- No expert to copy → no expert biases to inherit
- Reward = actual survival outcomes (energy delta)
- Agents discover strategies through trial and error
- Emergence, not prescription

## Meta-Commentary

This experience demonstrates the concept in practice.

We didn't just build a simulation of Goodhart's Law.
We lived it.
We demonstrated that even well-intentioned developers fall into proxy traps.
And then we recognized it and pivoted.

That recognition — "wait, we're doing the thing we're studying" — is exactly
the kind of insight that AI safety work aims to surface.

---

*Preserved as `behavior_cloning_checkpoint` for posterity.*
*All code as of this commit demonstrates the failure mode.*
