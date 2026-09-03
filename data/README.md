# Data manifests

Raw WSIs and extracted feature tensors are not distributed. Training and evaluation use CSV manifests with one row per WSI.

Typical fields are:

```text
patient_id,slide_id,feature_path,label,split,source_project
```

CAMELYON16 keeps the official test set unchanged and divides the 270-slide development set once into 75% training and 25% validation using split seed 2022. TCGA cohorts use patient-level 60%/15%/25% training, validation, and test partitions so that all slides from the same patient stay in one split.

Each generated split manifest is accompanied by a `.meta.json` file containing its SHA-256 hash, row count, class counts, split counts, and source project when available.

Normal-tissue slides used for reference construction are stored in a separate training-side manifest and are excluded from subtype-classification cohorts.
