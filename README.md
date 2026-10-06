# Solution for Assignment 1 of the course Big Data Analytics Techniques and Applications Fall 2026 at NYCU

**Reference:** https://github.com/nycubigdata-2026fall/BigData2026-Assignment1

## Changes to the provided code

| Question | Where | Change |
|---|---|---|
| Q1 | `finetune_q1-q3.ipynb`, config cell | `MANUAL_RUNS` = ranks 16 / 64 / 256 with `component='default'` (plus the Q3 runs); `RUN_FRESH_BASELINE = True` |
| Q2 | `train_one_task` | Wall-clock timer between the tagged start/end of training (after `torch.cuda.synchronize()`) |
| Q3 | `pissa_from_linear_exact` | New components: `start25` (`start = len(S)//4`), `start50` (`len(S)//2`), `bottom` (`len(S) - rank`) |
| Q4 | `finetune_q4.ipynb` | Full SVD of `q_proj` in layers 0/7/15, cumulative energy, minimum rank for >= 50% energy, 6 charts |

## File Description

- ```finetune_q1-q3.ipynb```: File for Question 1 to Question 3.
- ```finetune_q4.ipynb```: File for Question 4.
- ```runs/```: Executed copies of the two notebooks with all cell outputs saved. These are the runs behind the results in the report. Each ```finetune_q1-q3_*.ipynb``` executed the full notebook for a single configuration, selected with the ```A1_MANUAL_RUNS``` / ```A1_RUN_FRESH_BASELINE``` environment variables. The ```MANUAL_RUNS``` cell still lists every run, but the printed outputs show the configuration that actually ran.
  - ```finetune_q1-q3_fresh_baseline.ipynb```: Original Llama-3.2-1B, evaluation only (Q1a).
  - ```finetune_q1-q3_default_r16.ipynb```, ```finetune_q1-q3_default_r64.ipynb```, ```finetune_q1-q3_default_r256.ipynb```: Top singular values with target rank 16 / 64 / 256 (Q1b, Q2). ```default_r16``` is also the "Top" run for Q3.
  - ```finetune_q1-q3_start25_r16.ipynb```, ```finetune_q1-q3_start50_r16.ipynb```, ```finetune_q1-q3_bottom_r16.ipynb```: Rank 16 starting at the 25% position, starting at the 50% position, and the bottom 16 singular values (Q3).
  - ```finetune_q4_executed.ipynb```: SVD analysis of ```q_proj``` in layers 0, 7 and 15, with the minimum ranks for 50% energy and the 6 charts (Q4).
