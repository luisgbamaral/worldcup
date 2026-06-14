# Raw data sources

Immutable source data — **never edited by hand**. All cleaning happens in code
(`src/worldcup/data.py`).

| Folder | Source | Contents |
|--------|--------|----------|
| `martj42/` | [martj42/international_results](https://github.com/martj42/international_results) | International match results (1872–2026), goalscorers, shootouts, former team names |
| `worldcup.json-master/` | [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) | Per-edition World Cup data (matches, squads, stadiums, groups) 1930–2026 |
| `statsbomb/` | [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Competition catalogue (event data reference) |
| `squadlist/` | WC 2026 squads | Players and coaches per team (won't change) |
| `fixtures/` | WC 2026 schedule | Group-stage calendar with results as they happen |

`scripts/update_raw.py` refreshes the GitHub sources daily (no auth).
**Elo ratings are not a raw source** — they are computed from results in
`src/worldcup/elo.py`. See each source's repository for licensing terms.
