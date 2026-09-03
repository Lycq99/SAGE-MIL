# Multi-cohort reproduction

SAGE-MIL is evaluated on CAMELYON16, TCGA-NSCLC, TCGA-RCC, and TCGA-BRCA with PLIP and CONCH features.

## Splits

- CAMELYON16: the 270-slide development set is split once into 75% training and 25% validation with split seed 2022; the 129-slide official test set is unchanged.
- TCGA: patient-level 60/15/25 train/validation/test partitions.
- `Solid Tissue Normal` samples are reserved for normal-tissue reference construction and are not included in subtype-classification cohorts.

## Seeds

```text
model seeds: 2021, 2022, 2023, 2024
split seed: 2022
```

## Configurations

| Dataset | CONCH | PLIP |
|---|---|---|
| CAMELYON16 | `camelyon16_conch.yaml` | `camelyon16_plip.yaml` |
| TCGA-NSCLC | `tcga_nsclc_conch.yaml` | `tcga_nsclc_plip.yaml` |
| TCGA-RCC | `tcga_rcc_conch.yaml` | `tcga_rcc_plip.yaml` |
| TCGA-BRCA | `tcga_brca_conch.yaml` | `tcga_brca_plip.yaml` |

All configuration files use the shared study settings. Manifest, feature, anchor, and reference paths are repository-relative placeholders and should be set to the corresponding local data before training.

## Reference dictionary

The reported configuration uses `M=100` fixed reference prototypes. For CAMELYON16-CONCH, the stored reference tensor has shape `100 × 512`.

## RCC

RCC uses the multiclass true-class margin

```text
z_y - max_{c != y} z_c
```

and one-vs-rest ROC-AUC with macro F1 in the supplied configuration.
