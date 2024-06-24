# Experiments

We provide commands per experiment below.

Seeds for the three runs reported in the paper are `{0, 1, 2}`. You are expected to set the seed manually in the commands below. For the reported results, averages are taken over the values produced at the last epoch of each run. Please execute these commands in the repo root.



**Planar diffusion**
```bash
# With se2 bi-invariants
export PYTHONPATH=. && python experiments/fitting/fit_diff_plane.py seed={SEED}
```

**Navier-Stokes**
```bash
# With periodic bi-invariants
export PYTHONPATH=. && python experiments/fitting/fit_navier_stokes.py seed={SEED}
# With absolute position bi-invariants
export PYTHONPATH=. && python experiments/fitting/fit_navier_stokes.py seed={SEED} nef.invariant_type=abs_pos
# With periodic bi-invariants and autodecoding instead of meta-sgd
export PYTHONPATH=. && python experiments/fitting/fit_navier_stokes_nonmaml.py seed={SEED}
```

**Diffusion on the sphere**
```bash
# With so3 bi-invariants
export PYTHONPATH=. && python experiments/fitting/fit_diff_sphere.py seed={SEED}
# With abs pos bi-invariants
export PYTHONPATH=. && python experiments/fitting/fit_diff_sphere.py seed={SEED} nef.invariant_type=abs_pos
```

**Global shallow water equations**
Super-resolution is , the last logged wandb value is the super resolution run.
```bash
export PYTHONPATH=. && python experiments/fitting/fit_shallow_water.py seed={SEED}
```

**Internally heated convection**
```bash
export PYTHONPATH=. && python fitting/fit_ihc.py seed={SEED}
```
