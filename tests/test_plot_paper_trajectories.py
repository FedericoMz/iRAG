from irag.tools.plot_paper_trajectories import spread_endpoint_positions


def test_spread_endpoint_positions_separates_nearby_values() -> None:
    endpoints = {
        "final": 0.055,
        "controller_free": 0.0552,
        "human": 0.115,
        "fea": 0.80,
    }

    positions = spread_endpoint_positions(endpoints)
    ordered = sorted(positions.values())

    assert all(
        right - left >= 0.045 - 1e-12
        for left, right in zip(ordered, ordered[1:])
    )
    assert min(ordered) >= 0.015
    assert max(ordered) <= 0.985


def test_spread_endpoint_positions_preserves_already_separated_values() -> None:
    endpoints = {"low": 0.1, "middle": 0.3, "high": 0.8}

    assert spread_endpoint_positions(endpoints) == endpoints
