"""Shared fixtures. Data is loaded once per session (loaders hit disk)."""
import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend for figure tests

from worldcup import clean, data  # noqa: E402


@pytest.fixture(scope="session")
def results():
    return data.load_results()


@pytest.fixture(scope="session")
def played():
    return data.load_results(played_only=True)


@pytest.fixture(scope="session")
def goals():
    return data.load_goalscorers()


@pytest.fixture(scope="session")
def elo():
    return data.load_elo()


@pytest.fixture(scope="session")
def latest():
    return data.latest_elo()


@pytest.fixture(scope="session")
def editions():
    return data.load_worldcup_editions()


@pytest.fixture(scope="session")
def players():
    return clean.build_players()


@pytest.fixture(scope="session")
def teams(players):
    return clean.build_teams(players)
