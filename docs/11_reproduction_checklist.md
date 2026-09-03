# Reproduction checklist

Use this checklist when setting up a new machine or rerunning an experiment.

- [ ] Install Python 3.9.23 and a CUDA-enabled PyTorch 2.5.1 environment.
- [ ] Prepare the WSI manifest and verify patient/slide identifiers.
- [ ] Generate the patient-level split with split seed 2022 where applicable.
- [ ] Extract PLIP or CONCH features with the same preprocessing settings.
- [ ] Encode the semantic prompts from `configs/semantic_prompts.yaml`.
- [ ] Build the normal-tissue reference dictionary from training-side normal tissue only.
- [ ] Check the reference-dictionary prototype count for the target experiment.
- [ ] Check all paths in the selected experiment YAML.
- [ ] Run the four model seeds: 2021, 2022, 2023, 2024.
- [ ] Use the minimum-validation-loss checkpoint for test evaluation.
- [ ] Regenerate classification and calibration metrics from saved predictions.
- [ ] Run the retrospective replacement analysis from a fixed checkpoint.

Do not commit raw WSIs, restricted pretrained weights, credentials, or patient-sensitive metadata.
