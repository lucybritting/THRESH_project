# data/

What the pipeline reads. Nothing here is written by it — every output goes to `output/`.

## raw_files/

The MIMIC-IV hosp tables, as downloaded: `labevents.csv.gz`, `d_labitems.csv.gz`,
`admissions.csv.gz`, `patients.csv.gz`. `labevents.csv.gz` is far too large to hold in
memory and is only ever streamed, in chunks of `CHUNK_SIZE` rows.

## cohorts_100/ · cohorts_50/ · cohorts_test/

The cohorts, one gzipped CSV each, picked by `main.py --cohort_folder`:

| folder | flag | holds |
|---|---|---|
| `cohorts_100/` | `all` | the 100 representative cohorts |
| `cohorts_50/` | `subset` | one cohort per code1, a subset of the above |
| `cohorts_test/` | `test` | 3 cohorts, for a quick end-to-end run |

**File names carry meaning and are not free text**: `cohort_<code1>-<code2>.csv.gz`, an
ICD-10 code for the disease the cohort is built around, a hyphen, an ICD-10 code for the
target. The cross-cohort analyses parse both codes out of the filename, because they are
recorded nowhere else. See [Cohort naming](../README.md#cohort-naming) in the main README
for what a non-conforming name costs.

The pipeline reads five columns — `subject_id`, `hadm_id`, `age`, `gender`, `label` — and
ignores everything else the files happen to carry (admission times, length of stay, race,
insurance, and per-cohort extras). `label` is the binary target; `subject_id` matters
because the folds are grouped by it, so a patient with several admissions never appears in
both train and test.

`cohorts_test/` holds two chemotherapy cohorts from the previous work — aplasia at 45 days
(`cohort_C80-D61_aplasia`) and neutropenic fever at 30 days (`cohort_C80-D70_NF`),
described in https://www.medrxiv.org/content/10.64898/2025.12.12.25342142v1 — plus
`cohort_F17-C22`. The two were renamed from `mimic_cohort_aplasia_45_days` and
`mimic_cohort_NF_30_days` to fit the naming schema above.

## top_features/

`all_mimic_top100_features_hadm.pkl`, the curated top-100 lab list. Read **only** when
`TOP100LABS` is true in `config/params.py`; with the default `False` the feature set is
every itemid in `d_labitems` minus the blacklist. Either way the blacklist is subtracted.
