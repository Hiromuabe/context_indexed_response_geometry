# Data directory

No dataset copy, model trajectory, hidden state, or candidate tensor is
redistributed with this repository.

The trajectory generator downloads the configured public dataset through
Hugging Face Datasets and writes immutable JSONL and metadata files below this
directory.
Dataset access and redistribution remain subject to the upstream dataset terms.

Generated data and metadata are ignored by Git because run manifests can record
local checkpoint paths.
