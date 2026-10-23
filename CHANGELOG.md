# Changelog

## 1.0.0 (2026-10-23)

First public release.

- One tool registry behind an MCP server: GROMACS preparation, runs and analysis; MiMiC
  QM/MM (QM region, CPMD input, coupled launch, diagnosis); MiMiCPy force matching; jobs
  on a local backend or SLURM.
- Validation gates with stable finding ids; warn-level findings acknowledged by id, on
  the record.
- An append-only provenance ledger per study, and replay from the ledger alone.
- Numeric grounding: every value parsed from engine output.
- Curated input sets and protocols with a source for every value; skills for the agent.
- Plans, approval and the loop: unattended studies with sentinels, an envelope and an
  inbox.
- Myrmidon's own agent on any OpenAI-compatible endpoint.
