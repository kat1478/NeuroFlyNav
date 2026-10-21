# Local provenance capture

The provenance command writes a small JSON manifest for a local command. It uses Python's
standard library and the active Conda environment; it does not install the package or add
dependencies.

## Run it

From the repository root, activate the project environment and invoke the source package with
a command-local import path:

```bash
conda activate neuroflynav
PYTHONPATH=src python -m neuroflynav.provenance --command "python -m pytest" > /tmp/neuroflynav-run-provenance.json
```

The required `--command` value is recorded exactly as supplied. The shell redirection shown
above writes the manifest outside the repository. For project records, keep manifests in private
`ops/evidence/` and follow the storage and disclosure boundaries in the research and data
policies. Never include credentials, holdout details, or sensitive paths in `--command`.

The command exits nonzero and writes an explanation to stderr if Git metadata, `environment.yml`,
or installed metadata for pytest, Ruff, or mypy is unavailable. It does not write partial JSON.

## Manifest schema 1.0

The top-level object has these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest format version, currently `1.0`. |
| `captured_at_utc` | UTC capture time in ISO 8601 form ending in `Z`. |
| `command` | Exact non-empty string passed to `--command`; it is descriptive and is not executed. |
| `git.branch` | Current branch name, or JSON `null` when HEAD is detached. |
| `git.commit` | Full commit identifier for `HEAD`. |
| `git.dirty` | Boolean from Git's porcelain status output. Changed paths and contents are never included in the manifest. |
| `environment.environment_yml_sha256` | SHA-256 of the repository's `environment.yml` bytes. |
| `python.implementation` | Python implementation reported by the interpreter. |
| `python.version` | Python version reported by the interpreter. |
| `platform.system` | Operating system name. |
| `platform.release` | Operating system release string. |
| `platform.machine` | Machine architecture string. |
| `tools.pytest`, `tools.ruff`, `tools.mypy` | Installed distribution versions read from Python package metadata. |

The command, Git identity, environment checksum, Python/platform values, and tool versions
identify aspects of the capture environment. The `environment.yml` checksum identifies only
the declared environment specification; it does not lock every resolved transitive package or
guarantee identical resolution in the future. Platform values can vary across machines.

## Infrastructure smoke check

The following command checks that the package imports from the source tree without an install:

```bash
PYTHONPATH=src python -c 'import neuroflynav; print(neuroflynav.__name__)'
```

This is an import/procedure check only. It does not run a scientific model or establish scientific
reproducibility.
