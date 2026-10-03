# Current Dáil debate corpus data card

Measured on 3 October 2026 from the [Oireachtas debates API](https://api.oireachtas.ie/v1/debates) and its linked Official Report XML. The API was queried month by month from 1 January 2014 through 3 October 2026. Records whose chamber URI ended in `/house/dail` were retained. The latest returned Dáil date was **1 October 2026**. Parliamentary questions published separately after July 2012 are outside this corpus.

| Year | Dáil XML records | Parsed speeches |
| --- | ---: | ---: |
| 2014 | 124 | 47,373 |
| 2015 | 118 | 47,958 |
| 2016 | 92 | 33,142 |
| 2017 | 105 | 44,227 |
| 2018 | 107 | 45,024 |
| 2019 | 104 | 42,438 |
| 2020 | 85 | 27,111 |
| 2021 | 97 | 33,390 |
| 2022 | 106 | 39,441 |
| 2023 | 101 | 37,374 |
| 2024 | 85 | 30,641 |
| 2025 | 97 | 39,240 |
| 2026 through 1 October | 77 | 29,467 |
| **Total** | **1,298** | **496,826** |

The earliest listed Dáil debate is 15 January 2014. Cached XML totals **787,736,635 bytes**. Twenty-two listed records contain no `speech` elements. These are retained in the manifest with zero passages and their source hashes. Days absent from the API are not automatically classified as missing debates because the Dáil does not sit every calendar day. The manifest in ignored `data/oireachtas/manifest.json` records each source URL, retrieval time, update timestamp, SHA-256, byte count, and parsed speech count. It is generated locally and is not committed with the raw XML.

The complete local SQLite FTS5 database is **1,861,763,072 bytes**. Separately built 2014 and 2026 shards measured **147,181,568** and **125,972,480 bytes**. These are SQLite measurements, not D1 storage or latency measurements; D1 suitability still requires an actual import and account-level quota check.

Speech IDs use the official XML section and speech `eId` values plus date. Parsed text preserves Unicode and Irish fadas. Language is recorded only when the XML explicitly marks it; otherwise it remains unknown. Speakers are resolved from the XML `TLCPerson` references. Source links lead to the Official Report page for the sitting date. The page is a direct debate link, though not a deep link to one speech.

The [Oireachtas open-data licence](https://www.oireachtas.ie/en/open-data/license/) identifies Creative Commons Attribution 4.0 terms. Attribution and links accompany results. Debate remarks can be inaccurate, disputed, or taken out of context; the Q&A tool reports what the record says, not whether a claim is true. The retrieval index contains Dáil debate speeches only. It does not include separately published written parliamentary questions, committee debates, other chambers, or audio and video.

Rebuild with `python -m scripts.download_debates` and `python -m dail_llm.qa.index`. Run the downloader with `--refresh --start YYYY-MM-DD --end YYYY-MM-DD` to recheck source hashes for a selected period. Review sampled Official Report pages and the latest API response before describing a future refresh as current.
