# 5-minute pitch

1. **Problem (30 s).** Phone networks must be accurate, fast and small. Apple's Neural Engine is fast
   but picky, so "less math" doesn't mean "faster".
2. **Idea (45 s).** A genetic algorithm (NSGA-II) designs the networks. Genome = 91 integers
   (resolution; per stage depth, width, compression bits; per block kernel, expansion, SE). Evolution
   designs; SGD trains the weights.
3. **Making it affordable (45 s).** Weight-sharing supernet trained once; MLP predictors for accuracy
   and latency trained on real Core ML measurements.
4. **Experiments (1 min).** RQ1 GA vs baselines at equal budget; RQ2 MACs vs real latency; RQ3 joint
   compression; RQ4 GA ablations; RQ5 supernet trust; RQ6 Mac vs iPhone; RQ7 what evolution learned.
5. **Results (1.5 min).** The Pareto front figure; one table; one surprising finding.
6. **Limits and next steps (30 s).**

**Be ready to explain in the code:** `genome.decode`, `genome.repair`, `supernet.SuperNet.extract`,
`evaluate.SubnetEvaluator`, `search.nsga2` (operators), `search.hypervolume`.
