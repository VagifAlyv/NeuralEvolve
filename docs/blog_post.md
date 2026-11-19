# I let evolution design neural networks for my iPhone

*Draft. Numbers in [brackets] come from the experiments.*

Phones have a special chip for neural networks, Apple's Neural Engine. It's very fast, but picky: run
an operation it doesn't like, and that part of the network quietly moves to the slower GPU or CPU. So
the usual rule of thumb, "less math = faster", stops working.

I wanted to know: **can a genetic algorithm discover what the chip likes, just by trying networks?**

## The recipe

Every network is described by a **genome**: 91 numbers. They say how big the input picture is, how
many blocks each part of the network has, how wide it is, which filter sizes it uses, whether it has an
"attention" add-on (squeeze-excite), and how much each part is compressed.

Evolution then does what it does in nature:

1. start with 50 random networks,
2. score each on **accuracy**, **speed on the Neural Engine** and **size**,
3. let the best ones "have children" (mix their genomes, add small random mutations),
4. repeat.

There's no single best network: a more accurate one is usually slower. So the algorithm (NSGA-II)
keeps a **Pareto front**, a menu of best trade-offs from "most accurate" to "fastest".

## The trick that made it affordable

Training every candidate from scratch would take months. Instead I trained **one giant network that
contains all the others**. Any candidate borrows its weights from it, so scoring takes seconds instead
of hours. Evolution only designs the networks; normal training (SGD) learns the weights.

## What I found

- **[Figure: MACs vs measured latency.]** The amount of math predicts real speed only roughly
  (Spearman [ρ]).
- **[Figure: Pareto fronts.]** NSGA-II found a better front than random search at the same budget
  ([hypervolume numbers]).
- **[Surprising finding from RQ7, e.g. "the front avoids squeeze-excite in late stages".]**
- **[Final front vs MobileNetV2: X% more accurate at the same latency / Y× faster at the same accuracy.]**

## What I'd do next

[2–3 sentences.]

*Code, notebook and technical report: [repository link].*
