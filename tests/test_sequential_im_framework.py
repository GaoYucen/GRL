from grl.algorithms import adaptive_selective_greedy, full_oracle_greedy, learned_greedy, selective_greedy


class StubOracle:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []
    def score(self, seeds, candidates, step=0):
        self.calls.append((tuple(seeds), tuple(candidates), step))
        return {v: self.scores.get((step, v), self.scores.get(v, 0.0)) for v in candidates}


def test_sequential_framework_paths():
    pool = [0, 1, 2, 3]
    exact = StubOracle({0: 1.0, 1: 4.0, 2: 3.0, 3: 2.0, (1, 2): 5.0})
    full = full_oracle_greedy(pool, 2, exact)
    assert len(full.selected_seeds) == 2

    learned = StubOracle({0: 1.0, 1: 5.0, 2: 4.0, 3: 2.0})
    pred = learned_greedy(pool, 2, learned)
    assert pred.selected_seeds[0] == 1

    learned2 = StubOracle({0: 1.0, 1: 5.0, 2: 4.0, 3: 2.0})
    exact2 = StubOracle({1: 2.0, 2: 6.0, 0: 0.0, 3: 0.0})
    selective = selective_greedy(pool, 1, learned2, exact2, top_m=2)
    assert selective.selected_seeds == [2]
    assert len(exact2.calls[0][1]) == 2


def test_adaptive_expands_and_can_fallback():
    pool = [0, 1, 2, 3, 4]
    learned = StubOracle({0: 10.0, 1: 9.0, 2: 8.0, 3: 7.0, 4: 6.0})
    exact = StubOracle({0: 1.0, 1: 2.0, 2: 20.0, 3: 3.0, 4: 4.0})
    result = adaptive_selective_greedy(
        pool, 1, learned, exact,
        initial_m=2, batch_m=1, residual_beta=2.0, min_rounds=2, max_m=None,
    )
    assert result.selected_seeds == [2]
    assert result.steps[0]["verified"] >= 3
    assert len(exact.calls) >= 2
