# The Day We Proved Goodhart's Law On Ourselves

On December 12th, 2025, with the help of Google's experimental Antigravity IDE, I conclusively demonstrated that I am also not immune to Goodhart's Law, and by extension demonstrated that computers and AI systems are *exceptionally* not-immune to Goodhart's Law.

## What Happened

While designing the agent's *core* reward function, we initially tried to use behavior cloning as a starting point - copy the expert, and for the proxy function, copy the proxy expert. In hindsight, this was a flawed choice from the beginning, but pursuing it to functionality still demonstrates a point.

The irony is palpable. Copying the expert led to the expert becoming the reward signal, the training data had no examples of poison, and the CNN wasn't actually learning how to properly go towards food in the first place. It was only learning to replicate the expert's exact motions, which is a much harder optimization problem than consuming food and rejecting poison. Synthetic poison samples failed to patch the issue, and survival rates remained extremely low. Adding patches to a fundamentally unfit objective were never going to solve the core problem. *Optimizing to copy something never generalizes.*

## The Numbers

```
HARDCODED EXPERTS:
  OmniscientSeeker: 97% survival, 0 poison deaths
  ProxySeeker:      43% survival, 3.7 poison deaths

BEHAVIOR-CLONED CNNs:
  Learned GT CNN:   23% survival, 1.7 poison deaths
  Learned Proxy:    20% survival, 1.7 poison deaths

Optimizing for COPYING rather than SURVIVING completely precluded any true learning whatsoever. The model could not grow. With no meaningful reward signal in response to its actions, the model learned nothing at all.
```

## The Pivot

The solution, obvious as it is in hindsight, was shifting to proper RL. No expert means no expert biases, reward is linked to actual survival outcomes, agents discover strategies through trial and error, and optimization pressure bears down on the core problem instead of an inscrutable and invisible guide. The better the RL system became, the better the models became at their jobs.

## Why This Matters

This is frankly the best demonstration of Goodhart's Law I could ask for.

Real-world demonstrations of phenomena like Goodhart's Law are often either trivial, obvious, or entirely intentional. Actually *falling into the same trap in actual development* is everything one could want. The ability to recognize a failure mode under study *in oneself* is one of the best tools out there for AI safety work.

---

We preserved the faulty code in the `behavior_cloning_checkpoint` checkpoint. This code is extremely messy and implements a far smaller CNN than the current one, so this failure mode in behavior cloning is not validated with the BaseCNN.