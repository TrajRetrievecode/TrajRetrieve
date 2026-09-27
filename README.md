# TrajRetrieve

https://github.com/TrajRetrievecode/TrajRetrieve

Code for TrajRetrieve, a segment-level structured retrieval method for agent
execution trajectories. It splits trajectories into execution
segments, extracts structured fields from each segment, and retrieves useful
segments for an agent that gets stuck. The code covers AppWorld, ALFWorld, and
SWE-Bench Pro.

## Files

| Path | What it does |
|---|---|
| `trajretrieve/models.py` | Data structures for trajectories, segments, fields, and queries |
| `trajretrieve/config.py` | Loads the YAML configuration |
| `trajretrieve/segmenter/base.py` | Splits a trajectory into execution segments |
| `trajretrieve/schema/action_category.py` | Maps each environment's actions to shared categories |
| `trajretrieve/schema/fields.py` | Extracts the structured fields of a segment |
| `trajretrieve/schema/field_selection.py` | Chooses which fields go into the candidate text |
| `trajretrieve/index/dense_index.py` | Dense vector index |
| `trajretrieve/index/bm25_index.py` | BM25 keyword index |
| `trajretrieve/retrieval/templates.py` | Turns queries and segments into text |
| `trajretrieve/retrieval/encoders.py` | Text encoder and reranker |
| `trajretrieve/retrieval/retriever.py` | Hybrid retrieval, source filtering, and reranking |
| `adapters/<env>/loader.py` | Reads the raw trajectories of one environment |
| `adapters/<env>/annotate.py` | Generates the LLM fields of each segment, prompt included |
| `scripts/build_index.py` | Builds a segment index from trajectories |
| `scripts/evaluate.py` | Evaluates retrieval and runs field selection |
| `configs/` | Configuration template and the settings used in the paper |
| `prompts/relevance_rubric.txt` | Rubric used for the relevance labels |
| `samples/` | A few example segments, queries, and relevance labels |
| `splits/` | Checksums of `samples/` and a validation split of the sample queries |

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# build an index
python scripts/build_index.py --env appworld --trajectories /path/to/trajectories --out runs/appworld_index

# evaluate retrieval
python scripts/evaluate.py --index runs/appworld_index --queries /path/to/queries.jsonl --gold /path/to/qrels.jsonl
```

To generate the LLM fields, copy `configs/config.template.yaml` to
`configs/config.yaml`, fill in your API endpoint and key, and add `--annotate`
when building the index.

## Data

Segments and their fields are built with `scripts/build_index.py`
(`--env appworld`, `alfworld`, or `swe`). Relevance labels follow
`prompts/relevance_rubric.txt`. The records in `samples/` remain subject to the
terms of their source datasets.
