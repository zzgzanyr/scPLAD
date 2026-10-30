# K562 mean-response bundle

`mean_predictor_go_reactome_ensemble.pt` is a self-contained PyTorch bundle
for the K562 condition-mean correction used by the current manuscript.

- Format: `scplad.mean_predictor.v1`
- Model seeds: `20261015`, `20261016`, `20261017`
- Genes: 5,000 in stored order
- Prior features: GO256 + Reactome256 in stored order
- Default correction coefficient: 0.75
- SHA-256: `d8d88efc6e8c45245bad5e6063439cbef3c83e6de6b7b29df51a691deea74b13`

The three model states are intentionally packaged in one file so the released
predictor matches the ensemble used for the reported correction. Loading and
application are documented in
[`docs/MEAN_CORRECTION_AND_FCS.md`](../../docs/MEAN_CORRECTION_AND_FCS.md).
