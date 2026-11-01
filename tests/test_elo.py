from fightrl.elo import expected_score, fit_ratings


def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1700, 1500) > 0.7
    assert expected_score(1300, 1500) < 0.3


def test_fit_ratings_orders_players():
    results = [("strong", "weak", 1.0) for _ in range(10)]
    results += [("weak", "strong", 0.0) for _ in range(2)]
    ratings = fit_ratings(results, ["strong", "weak"], passes=6)
    assert ratings["strong"] > ratings["weak"] + 100
    assert abs(sum(ratings.values()) - 3000) < 1e-6


def test_fit_ratings_draws_are_equal():
    results = [("a", "b", 0.5) for _ in range(5)]
    ratings = fit_ratings(results, ["a", "b"])
    assert ratings["a"] == ratings["b"] == 1500
