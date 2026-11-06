# Weak-match signals on held-out queries (#143)

Dataset `eval/datasets/weak_match.yaml` v1: 200 queries (100 on the menu), dev 101, test 99. Commit `a2487e8`. Cut-offs are chosen on dev for the most off-menu queries flagged with at most 5% of on-menu queries flagged; test is reported once with 95% bootstrap intervals.

| Signal | Cut-off | Test precision | Test recall | Test false-weak | Dev recall | Dev false-weak |
|---|---|---|---|---|---|---|
| cosine_no_menu_word (production) | 0.550 | 100% (13 flagged) | 26% [14%, 38%] | 0% [0%, 0%] | 30% | 2% |
| cosine_no_menu_word (calibrated) | 0.636 | 92% (26 flagged) | 48% [34%, 62%] | 4% [0%, 10%] | 60% | 4% |
| cosine (calibrated) | 0.537 | 100% (14 flagged) | 28% [16%, 40%] | 0% [0%, 0%] | 24% | 4% |
| margin (calibrated) | 0.020 | 100% (4 flagged) | 8% [2%, 16%] | 0% [0%, 0%] | 6% | 4% |
| cross_encoder (calibrated) | -11.127 | 78% (9 flagged) | 14% [6%, 24%] | 4% [0%, 10%] | 18% | 2% |

Share of each test group flagged:

| Signal | category | named | near_miss | off_topic | vague |
|---|---|---|---|---|---|
| cosine_no_menu_word (production) | 0% | 0% | 6% | 73% | 0% |
| cosine_no_menu_word (calibrated) | 0% | 0% | 37% | 73% | 13% |
| cosine (calibrated) | 0% | 0% | 3% | 87% | 0% |
| margin (calibrated) | 0% | 0% | 3% | 20% | 0% |
| cross_encoder (calibrated) | 0% | 0% | 0% | 47% | 13% |

What each signal sees, by group (all splits; min / median / max):

| Group | Queries | Best cosine | Margin | Cross-encoder | Query word on the menu |
|---|---|---|---|---|---|
| category | 35 | 0.60 / 0.73 / 0.83 | 0.04 / 0.11 / 0.21 | -2.81 / 4.72 / 9.00 | 33 |
| named | 35 | 0.53 / 0.76 / 0.84 | 0.06 / 0.19 / 0.27 | 2.32 / 8.42 / 10.04 | 35 |
| near_miss | 70 | 0.46 / 0.62 / 0.72 | 0.02 / 0.05 / 0.14 | -11.18 / -7.69 / 0.50 | 32 |
| off_topic | 30 | 0.43 / 0.50 / 0.63 | 0.01 / 0.03 / 0.05 | -11.36 / -11.13 / -8.50 | 5 |
| vague | 30 | 0.52 / 0.64 / 0.72 | 0.01 / 0.03 / 0.13 | -11.27 / -4.35 / 4.12 | 23 |

Signals:

- `cosine_no_menu_word`: best cosine below the cut-off and no query word on the menu (the #135 rule).
- `cosine`: best cosine below the cut-off.
- `margin`: best cosine minus the tenth below the cut-off.
- `cross_encoder`: the reranker's best score over the ten below the cut-off.
