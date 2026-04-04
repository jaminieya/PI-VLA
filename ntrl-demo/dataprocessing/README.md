# `ntrl-demo/dataprocessing`

Scripts that build datasets for arm / trajectory NTField training (`points.npy`, `tau_obs.npy`, etc.).

## `generate_straightline_collision_dataset.py`

Generates **`points.npy`** `(N, 12)` and **`tau_obs.npy`** `(N,)` **without RRTConnect / OMPL**. Each sample is a pair \((q_\text{start}, q_\text{goal})\) in radians with **\(\tau_\text{obs} = \|q_\text{goal} - q_\text{start}\|_2\)**, kept only if a **straight-line interpolation** in joint space is **collision-free** (FCL via `hanwen_grasping.robot_arm_configuration`).

**Example (from `PI-VLA/ntrl-demo`):**

```bash
python dataprocessing/generate_straightline_collision_dataset.py \
  --output_dir datasets/arm/UR5_straightline \
  --num_pairs 100000
```

Resolve **`robot_arm_configuration`** by ensuring `hanwen_grasping` is importable; the script adds it to `sys.path` and `chdir`s into `hanwen_grasping` for asset paths.

**Train:**

```bash
cd ntrl-demo && python train/train_arm_trajectory.py --data_path ./datasets/arm/UR5_straightline
```

---

### Sampling modes

- **`--sample_mode gaussian` (default)**  
  Draws `q_s` and `q_g` independently as **Gaussian** around a **nominal** pose (default UR5 home: `[0.7, -2.0, 2.5, -0.3, 0.7, 0.0]` rad), with **`--sigma`** (default `0.85` rad). Most feasible workspace lies near that pose, so **acceptance rates are usable** (on the order of a few percent of proposals).

- **`--sample_mode uniform`**  
  Uniform in **\([-\texttt{norm\_limit} \cdot \text{SCALE}, +\texttt{norm\_limit} \cdot \text{SCALE}]\)** per joint with `SCALE = π / 0.5`. For default `norm_limit=0.5` that is about **\([-π, π]\)** per joint. **Straight-line** segments in that huge cube almost always hit **self-collision or the table**, so **almost all proposals are rejected**. Expect failure within the default try budget unless you raise **`--max_tries_factor`** massively or shrink the box.

So: **uniform wide box + straight-line + collision** yields a **tiny** fraction of accepts; **Gaussian-around-home** is the practical default.

---

### What typical runs look like (interpreting the log)

1. **Gaussian run**  
   Lines like `accepted 10000 / 100000 (tries=268222)` mean: **10k** accepted pairs so far after **268k** random proposals. The ratio **accepted / tries** is the empirical acceptance rate (often **~3–5%** with default `sigma` and scene).

2. **Uniform run that fails**  
   If the log shows `joint box [-3.1416, 3.1416] rad per dim` and then `Only collected X pairs (need 100000)`, the sampler hit **`max_tries_factor * num_pairs`** proposals with **too few** collision-free segments. Prefer **`--sample_mode gaussian`** or loosen the scenario (see `--max_tries_factor`, **`--tau_max`**, **`--sigma`**).

3. **`tau_obs range`**  
   - **`tau_obs`** is **joint-space distance** \(\|q_g - q_s\|_2\) in **radians**.  
   - **`--tau_max`** (default `2.0`) **rejects** pairs with distance **greater** than that, so the saved histogram is **capped** at `tau_max`.  
   - **`--tau_min`** (default `0.01`) drops very short segments.

4. **Spurious `Link base had 0 children...` after “Saved …”**  
   That text comes from the **URDF / trac\_ik** link tree when the planner object is torn down; it can appear **after** the script prints success. It does **not** mean the `.npy` files were not written.

---

### `--output_dir` and the double `ntrl-demo` path

`--output_dir` is resolved with **`os.path.abspath` from the process current working directory**.

If your shell is already in `PI-VLA/ntrl-demo` and you pass:

```text
--output_dir ntrl-demo/datasets/arm/UR5_straightline
```

the output becomes:

```text
PI-VLA/ntrl-demo/ntrl-demo/datasets/arm/UR5_straightline
```

**Avoid that** by either:

- running from `ntrl-demo` with `--output_dir datasets/arm/UR5_straightline`, or  
- running from `PI-VLA` with `--output_dir ntrl-demo/datasets/arm/UR5_straightline`.

---

## Other scripts

- **`prepare_trajectory_dataset.py`** — builds `points.npy` / `tau_obs.npy` from **HDF5 trajectories** (e.g. RRT-collected episodes).  
- **`trajectory_sampler.py`** — shared **`SCALE`** and pair sampling helpers.

For straight-line data **without** RRT labels, use **`generate_straightline_collision_dataset.py`** (this README section).
