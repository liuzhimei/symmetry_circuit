# Symmetry Circuits in Transformers (Mechanistic Interpretability Mini-Project)

**Goal:** Test whether tiny transformers trained on conservation-law tasks learn internal linear features/circuits for conserved quantities (e.g., `s=x+y`) and show invariance/equivariance under symmetry-preserving transformations.

**Tasks:** REG-SUM, CLS-VALID, REG-MODK. See `src/gen_data.py`.

**Pipeline:** Train tiny transformer → probes (R^2 for `s=x+y`) → projection ablation → activation patching → invariance checks.

**Workflow:**

1. Install requirements

```
pip install -r requirements.txt
```

2. Generate data (e.g. for REG-SUM task)

```
python -m src.gen_data --task REG-SUM --n 20000 --seed 1 --out_dir data
```

3. Train the model

```
python -m src.train_transformer --task REG-SUM --epochs 10
```

4. Run probes (linear feature analysis)

```
python -m src.run_probes --task REG-SUM
```

5. Run patching (causal ablation)

```
python -m src.run_patching --task REG-SUM --data_dir data
```

**Interpretability tools**

- `run_probes.py`: Do hidden states linearly encode the conserved quantity (the sum x+y)?
- `run_patching.py`: If we remove that feature direction, does performance drop (causal evidence)?

**Code details**

- `run_probes.py`

1. Loads the validation set (e.g. REG-SUM_val.tsv).
2. Runs the trained transformer to get hidden states at every token position.
3. Builds a target vector = true sums s = x+y (parsed from the input string).
4. Trains Ridge regression to predict s from hidden states:
   - once on the pooled representation (last token’s hidden state),
   - and separately for each position along the sequence.
5. Reports R² (explained variance) for pooled and per‑position—high R² means the sum is linearly present.

- `run_patching.py`

1. Loads the trained model and a validation batch.
2. Uses Ridge to learn a probe vector v in the pooled hidden space that predicts the sum x+y.
3. Projects out that direction from the pooled hidden states:
   \begin{equation}
   h' = h - (h \cdot v)v
   \end{equation}
4. Trains a simple ridge readout from pooled states to the task target (before and after projection) and compares MSE.
5. If MSE rises after removing the sum direction, that’s causal evidence the direction is used for the task.
