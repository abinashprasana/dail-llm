# CPU pilot results

The pilot completed both offline studies and the artifact audit. It establishes a working measurement pipeline; it does not establish that the new selection policies improve language modelling.

Configuration: `experiments/pilot.toml`, seed 42, 100 training steps per decoder, batch size eight, and 5,000 entries per memory. The recorded execution time was 1,244.4 seconds, with a peak process working set of 0.86 GiB. Timing comes from one run on a shared workstation and should not be treated as a stable hardware benchmark. Input hashes below identify the results independently of the local directory name.

The earlier attempt was stopped after discovering that the corpus uses “The Labour Party.” The corrected mapping recognizes that label. The earlier run is explicitly incomplete and its partial checkpoint is not used here.

## Data and checks

Preparation read 4,443,713 rows from the original archive. It selected 494,973 training characters, 24,974 validation characters, 24,978 earlier-test characters, and 24,929 later-test characters after excluding 21 duplicate speeches. No malformed rows were recorded in this archive scan. The small character caps favour debate groups that fit within the remaining budget.

All 53 local tests passed, including the real served-checkpoint generation and attention test. The suite covers deterministic CPU resume, independent likelihood calculations, unknown characters, padding and speech boundaries, chronological duplicate exclusion, source removal, historical role boundaries, matched pairs, masked-input controls, artifact mismatches, and local adversarial API checks. One existing Starlette/httpx deprecation warning remains. The CI workflow includes the synthetic research study; remote CI was not launched during this local run.

The completed pilot passed 50 artifact checks. These cover every retained speech's target positions, partition ownership, checkpoint selection, training-only memory eligibility, and equal memory counts. Sampled stored vectors were recomputed from their speech windows. Sampled likelihoods were checked with float64 log probabilities, and retrieval distances were compared with brute force. These checks do not verify historical membership records or judge generated prose.

## Results

Scores are supported-target bits per character; lower is better. Earlier-test coverage was 99.99198%. These scores use new data preparation and evaluation conventions and are not comparable to the published checkpoint's historical scores.

| System | Earlier test | Later test |
|---|---:|---:|
| Witten–Bell character five-gram | 1.8572 | 2.2734 |
| Unconditioned decoder | 3.5646 | 3.6495 |
| Party-conditioned decoder | 3.8543 | 3.9168 |
| Role-conditioned decoder | 3.8665 | 3.9215 |
| Uniform memory | 3.5449 | Reserved for historical study |
| Speech-balanced memory | 3.5377 | Reserved for historical study |
| Context-diverse memory | 3.5412 | Reserved for historical study |

All three memories selected 32 neighbours, weight 0.25, and temperature 10 on validation data. Relative to equal-sized uniform memory, the speech-balanced difference was −0.0072 BPC, with a paired debate-bootstrap 95% interval of [−0.0175, 0.0001]. The context-diverse difference was −0.0036 BPC, with an interval of [−0.0130, 0.0033]. Both intervals include zero; they are exploratory estimates from one training seed and 18 earlier-test debates.

The rare-word slice gives another reason to avoid an aggregate improvement claim. Its character BPC was 4.0785 with uniform memory, 4.0828 with speech balancing, and 4.0834 with context diversity. The selected earlier test contained no matched recorded full member-name spans. All systems had zero exact accuracy on the supported rare-word spans. The five-gram baseline substantially outperformed the decoders at this training budget.

The historical matching procedure found only one pair, leaving 46 earlier and 34 later passages unmatched. Continuing-speaker and matched-passage results are available, but this coverage cannot support a historical role-change conclusion. Some recorded party labels are outdated: the selected archive lists Eamon Gilmore under The Workers' Party in 2008. Such affiliations remain preserved and unmapped for the role experiment. Dated membership verification remains necessary before drawing political conclusions.

The 36 generated continuations had zero measured training eight-gram overlap. That does not demonstrate readability, originality, or factual accuracy. The blinded review pack contains those 36 continuations, 77 recurring-phrase probes, and one member-name probe. Human review is pending; no ratings were invented.

For the fixed prefix `The Minister for`, excluding speech `4011093` removed all six of its context-diverse memory entries. After a new neighbour search, the probability of a space changed from 0.14642 to 0.13134. This is a measured effect through retrieval, not training-data attribution.

## Reproduction and evidence

Follow [the research command guide](research.md) with the pilot profile and a new run directory. The longer three-seed profile is checked in but was not launched. The [Research interface](research-ui.md) presents these results separately from the served weights and published evaluation artifacts, which remain unchanged.

The run directory contains the full per-speech measurements, heuristic slices, copying and latency measurements, bootstrap comparisons, matched-passage records, input perturbations, source inspections, and unscored review pack. `report.json` records evaluation hashes and the hash of `verification.json`.

Key input identifiers:

```text
Archive SHA-256:
761b9025c8e5bae185439eb11c53b2190fd2946d5e5537030bfa03bc0140047f
Prepared records SHA-256:
b603d985dd69cee4b75d5aaa60a362b062eaafe949df59ef1d105f16f94fbe1d
Unconditioned checkpoint SHA-256:
ee4d22bb8e3354e7d9b79ae39c0a533709b82438206cbaa9019aca991e860943
```
