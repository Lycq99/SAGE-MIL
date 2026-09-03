# Data preparation

Classification manifests contain one row per WSI with patient, slide, feature, label, split, and source-project information where available.

CAMELYON16 keeps the official 129-slide test set unchanged. The 270-slide development set is divided once into 75% training and 25% validation with split seed 2022. TCGA cohorts use patient-level 60%/15%/25% training, validation, and test partitions.

Generated split manifests are accompanied by a metadata JSON file containing the manifest SHA-256 hash, class counts, split counts, and source project. Normal-tissue slides used for reference construction are kept outside the subtype-classification cohorts and restricted to the training side.
