# Insight Before Sight

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24338376.svg)](https://doi.org/10.5281/zenodo.24338376)

Code, trained models, and results of the paper

S. Al-Shareeda, "Insight Before Sight: V2X Pedestrian Intention for RL- and LLM-Driven Autonomous Vehicles,"
submitted to *IEEE Transactions on Intelligent Transportation Systems*, 2026.

A roadside unit (RSU) sees a pedestrian behind parked cars before the vehicle does. It estimates the probability that
the pedestrian walks onto the road without stopping and sends it over V2X. Two decision policies use this probability
under the same safety floor: a recurrent policy trained by imitation and proximal policy optimization (PPO), and an open
large language model (LLM). This repository holds everything needed to check the numbers and figures of the paper.

## Contents

| Folder | What it holds |
|---|---|
| `notebooks/rl_agent.ipynb` | Environment, RSU predictor, reference rules, imitation start, PPO training, evaluation, and robustness tests. Run on Google Colab; saved with the outputs of the paper's run. |
| `notebooks/llm_agent.ipynb` | LLM agent (Qwen2.5-7B-Instruct, 4-bit) and its prompt variants: zero-shot, A, B, C, C′, D, and C+, plus hesitation and channel faults. Needs a GPU. Saved with its outputs. |
| `notebooks/rl_cell_control_run.py` | Colab cell of the control runs (imitation start without V2X). |
| `reproduce/` | CPU scripts that rerun the tests of the paper from the trained models, and build its result figures. |
| `reproduce/models/` | The trained RSU predictor and the 17 trained policies of the final set (threshold 0.05). |
| `reproduce/models_control/` | The five control runs and their imitation networks. |
| `reproduce/models_s22/` | The four networks of supplementary S2.2. |
| `reproduce/data/` | Results of the Colab runs: every test episode of the RL and LLM agents, every LLM decision, training logs, and the prompt of variant C+. |
| `reproduce/out/` | Results of the CPU tests T0 to T9 and of supplementary S2.1 and S2.2. |
| `reproduce/figs/` | Result figures built by `make_figures.py`, with the values and counts behind the braking figures. |

## Quick start (CPU)

Python 3.10 or later.

```bash
pip install -r requirements.txt
cd reproduce
python rerun_tests.py T0      # about 8 min: all RL runs and rules on the 2,000 test scenarios
python make_figures.py        # result figures, from data/ and out/
```

Every script runs from `reproduce/` and skips a test whose output already exists. Delete a file in `out/` to run that
test again. A stopped test resumes from `out/cache/`.

The checkpoints are PyTorch files that store their settings with the weights, so they load with
`torch.load(..., weights_only=False)`. Load them only from this repository or another source you trust.

## Tests and where the paper uses them

All tests use the 2,000 test scenarios (seeds 10,000 to 11,999) at hesitation level h = 0.2, unless stated.

| Part | Command | Content | Paper |
|---|---|---|---|
| T0 | `python rerun_tests.py T0` | Nominal channel, all RL runs and rules. Equals the Colab results in `data/episodes_v6.csv`, scenario by scenario. | Table III |
| T1 | `python rerun_tests.py T1` | Channel faults: 10% and 30% loss, +300 and +500 ms delay, RSU failure. | Fig. 7, Sec. IV-D |
| T2 | `python rerun_tests.py T2` | Bursty message loss (Gilbert-Elliott model). | Sec. IV-D, S2.4 |
| T3 | `python rerun_tests.py T3` | Sensitivity to the pedestrian cue (beta = 0.85 and 1), with retrained predictors. | Sec. IV-D, S2.5 |
| T4 | `python rerun_tests.py T4` | Second scenario set (seeds 40,000 to 41,999), nominal and RSU failure. | S2.3 |
| T5 | `python rerun_tests.py T5` | Replay of scenario 10,798. | Fig. 9 |
| T6 | `python rerun_tests.py T6` | Defensive speed without V2X. | Table III |
| T7 | `python rerun_tests.py T7` | Decision logs of the RL runs and the rule. | Fig. 5 |
| T8 | `python rerun_tests.py T8` | Test windows of the RSU predictor. | Fig. 4 |
| T9 | `python rerun_tests.py T9` | Control runs (imitation start, no V2X) at each hesitation level. | Table III, Sec. IV-B |
| S2.1 | `python s21_rule_variants.py` | Rule variants on 3,000 validation scenarios (seeds 50,000 to 52,999). | Table S1 |
| S2.2 | `python s22_privileged_critic.py 1 2` | Fine-tuning with a privileged critic (scheme 1 or 2, run 2 or 4). | S2.2 |

`python train_control.py 0 1 2 3 4` retrains the control runs into `models_control/`.

The LLM results (variants zero-shot to C+, hesitation, and faults) need a GPU and come from `notebooks/llm_agent.ipynb`.
Their episodes and decisions are in `reproduce/data/`, named by the key of each prompt variant.

## Figures

`python make_figures.py <name>` builds one figure; without a name it builds all.

| Name | Output | Paper |
|---|---|---|
| `fig5` | `fig_pred_a_calibration.pdf`, `fig_pred_b_auc.pdf` | Fig. 4 |
| `fig6` | `fig_braking_a_sidewalk.pdf`, `fig_braking_b_road.pdf` | Fig. 5 |
| `fig8` | `fig_robustness.pdf` | Fig. 7 |
| `fig10` | `fig_run_tracked.pdf` | Fig. 9 |
| `fig4` | `fig_train_a_collisions.pdf`, `fig_train_b_validation.pdf`, `fig_train_legend.pdf` | Fig. S4 |
| `fig7` | `fig_hesitation.pdf` | Fig. 6 |
| `figS3` | `figS3_hesitation_time.pdf` | Fig. S3 |
| `figS1` | `figS1_training_return.pdf` | Fig. S1 |

Figs. 1 to 3 and Fig. 8 are drawings and are not built from data.

## Running the notebooks

Both notebooks mount Google Drive and keep their checkpoints and results under
`MyDrive/Colab Notebooks/checkpoints_v6`, set by `ckpt_dir` in Cell B of the RL notebook and Cell P of the LLM
notebook. Change it there to use another folder. Each run gets a subfolder named by the keys of its settings, so a
changed setting never overwrites an earlier result.

On a Colab CPU runtime, the RL notebook trains the predictor and all runs in about 70 min (Cell B), evaluates them in
about 45 min (Cell C), and runs the robustness tests in about 12 min (Cell D). The LLM notebook needs a GPU. The paper's
LLM runs used one NVIDIA A100 GPU, and its response times were measured on one NVIDIA T4 GPU.

`notebooks/rl_cell_control_run.py` is a cell to paste after Cell D of the RL notebook. `reproduce/train_control.py`
runs the same training on a CPU.

## License

The code is released under the MIT License (`LICENSE`). The trained models and the results are released under the
Creative Commons Attribution 4.0 International License (`LICENSE-DATA.md`).

## Citation

Please cite the paper if you use this code or these results. The archived release is
https://doi.org/10.5281/zenodo.24338376, and `CITATION.cff` gives its reference.
