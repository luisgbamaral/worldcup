"""Static lookups for venue / confederation features.

Small, hand-maintained reference data (inline dicts): team → confederation for
the 48 qualified squads plus common historical sides, and 2026 host/venue info
used by the quasi-home gradient. Keyed on canonical (squad-list) team spelling.
"""
from __future__ import annotations

# team -> confederation (canonical spelling). Covers the 48 squads + extras.
CONFEDERATION = {
    # UEFA
    "Czechia": "UEFA", "Switzerland": "UEFA", "Bosnia And Herzegovina": "UEFA",
    "Scotland": "UEFA", "Türkiye": "UEFA", "Germany": "UEFA", "Netherlands": "UEFA",
    "Sweden": "UEFA", "Belgium": "UEFA", "Spain": "UEFA", "France": "UEFA",
    "Norway": "UEFA", "Austria": "UEFA", "Portugal": "UEFA", "Croatia": "UEFA",
    "England": "UEFA", "Italy": "UEFA", "Denmark": "UEFA", "Poland": "UEFA",
    "Serbia": "UEFA", "Wales": "UEFA", "Ukraine": "UEFA", "Romania": "UEFA",
    "Greece": "UEFA", "Russia": "UEFA", "Hungary": "UEFA", "Republic of Ireland": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    "Peru": "CONMEBOL", "Chile": "CONMEBOL", "Bolivia": "CONMEBOL", "Venezuela": "CONMEBOL",
    # CONCACAF
    "Mexico": "CONCACAF", "Canada": "CONCACAF", "Haiti": "CONCACAF", "USA": "CONCACAF",
    "Curaçao": "CONCACAF", "Panama": "CONCACAF", "Costa Rica": "CONCACAF",
    "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    # CAF
    "South Africa": "CAF", "Morocco": "CAF", "Côte D'Ivoire": "CAF", "Tunisia": "CAF",
    "Egypt": "CAF", "Cabo Verde": "CAF", "Senegal": "CAF", "Algeria": "CAF",
    "Congo DR": "CAF", "Ghana": "CAF", "Nigeria": "CAF", "Cameroon": "CAF",
    "Mali": "CAF", "Ivory Coast": "CAF",
    # AFC
    "Korea Republic": "AFC", "Qatar": "AFC", "Australia": "AFC", "Japan": "AFC",
    "IR Iran": "AFC", "Saudi Arabia": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC", "South Korea": "AFC", "Iran": "AFC", "China PR": "AFC",
    # OFC
    "New Zealand": "OFC",
}

# 2026 World Cup hosts (CONCACAF).
HOSTS_2026 = {"USA", "Mexico", "Canada"}
HOST_CONFED_2026 = "CONCACAF"

# Approximate elevation (m) by venue keyword; default sea level.
_ALTITUDE = {
    "Mexico City": 2240, "Zapopan": 1566, "Guadalupe": 1566,  # Guadalajara metro
    "Atlanta": 320, "Kansas City": 270, "Arlington": 160, "Houston": 30,
}

# Country a 2026 venue sits in (only hosts qualify as "home country").
_MEXICO_KW = ("Mexico City", "Zapopan", "Guadalupe", "Mexico")
_CANADA_KW = ("Vancouver", "Toronto", "Canada")


def confederation(team: str) -> str | None:
    return CONFEDERATION.get(team)


def venue_country(venue: str | None) -> str | None:
    if not venue:
        return None
    if any(k in venue for k in _MEXICO_KW):
        return "Mexico"
    if any(k in venue for k in _CANADA_KW):
        return "Canada"
    return "USA"  # all remaining 2026 venues are in the United States


def venue_altitude(venue: str | None) -> float:
    if not venue:
        return 0.0
    for kw, alt in _ALTITUDE.items():
        if kw in venue:
            return float(alt)
    return 0.0


def host_proximity(team: str | None) -> float:
    """1.0 for a host nation, 0.5 for a host-confederation side, else 0.0."""
    if team in HOSTS_2026:
        return 1.0
    return 0.5 if CONFEDERATION.get(team) == HOST_CONFED_2026 else 0.0
