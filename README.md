# Symmetry Circuits in Tiny Transformers

**Goal:** Test whether tiny transformers trained on conservation-law tasks learn internal linear features/circuits for conserved quantities (e.g., `s=x+y`) and show invariance/equivariance under symmetry-preserving transformations.

**Tasks:** REG-SUM, REG-MODK. See `src/gen_data.py`.

**Pipeline:** Train tiny transformer → probes (R^2 for `s=x+y`) → projection ablation → activation patching → invariance checks.

**Main code:** See `main.ipynb`.
