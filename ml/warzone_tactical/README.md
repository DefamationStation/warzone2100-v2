# Warzone Tactical Unit AI

This workspace implements Contract V1 for learned ground-droid control.

The project is engine-first. Training stays blocked until the engine test and simulator fidelity gates pass.

## Current order

1. Build the game with `WZ_ML_EXPERIMENT=ON`.
2. Run the scripted headless 2v2 acceptance test.
3. Export the resolved engine stats.
4. Export and load the fixed WZML model fixture.
5. Collect 10,000 C++ observation goldens.
6. Pass movement, turret, combat, and mask fidelity tests.
7. Run the CUDA throughput gate.
8. Start staged PPO work only after steps 1 to 7 pass.

Steps 1 to 7 and the formal Stage 1 promotion now pass on the first compiled Windows test system. The Stage 1
trainer keeps movement at zero and trains only target selection and fire. Its
targets require one, two, and four successful hits and occupy three range bands.
The clear-time gate checks the mean and p90, not only the median. Stage 1 proves
valid targeting and fire. It does not prove complex learning because competent
deterministic policies produce almost equal trajectories on this task.
Run the formal promotion seeds with:

```powershell
uv run --project ml\warzone_tactical python -m warzone_tactical.stage1 `
  --resolved-stats ml\warzone_tactical\runs\review4-rocket-spike-100\seed-0\resolved-stats.json `
  --output ml\warzone_tactical\runs\stage1-promotion-rocket `
  --training-seeds 4 17 31 --promotion --episodes 10000
```

Run the Stage 2 continuous range-control screen with:

```powershell
uv run --project ml\warzone_tactical python -m warzone_tactical.stage2 `
  --resolved-stats ml\warzone_tactical\runs\review4-rocket-spike-100\seed-0\resolved-stats.json `
  --stage1-output ml\warzone_tactical\runs\stage1-promotion-rocket `
  --output ml\warzone_tactical\runs\stage2-review6-seed4 `
  --training-seeds 4 --episodes 1000
```

Stage 2 is implemented but is not promoted. Review 5 showed that the fixed
continuous gate can separate policies, but seed 4 stayed below the 70 percent
time-in-band gate. Review 6 found that this result was close to the one-third
do-nothing floor. The Stage 2 report now includes that floor. The range-dwell
objective is the main reward, progress is shaping only, and no scripted drive
target enters training. The default run uses 1,500 PPO iterations. Run seed 4
first. Run seeds 17 and 31 only if seed 4 reaches about 60 percent time in band.

The Review 6 screen reached 60.33 percent for seed 4, 32.41 percent for seed 17,
and 60.06 percent for seed 31. The three-seed mean was 50.93 percent. Seed 31
also lost eight percentage points on Stage 1 retention. Under Contract V1, seed
31 failed the retention gate. Stage 2 is not promoted. This screen was made
before the KL stop was added. The KL stop has had a smoke test only.
The 10,000-spawn Q15 check showed distinct, ordered near, in-band, and far
position values. The failed seed is not caused by a degenerate position signal.
Training telemetry showed large KL and clip spikes. The next run stops a PPO
update when approximate KL is above 0.015.

Contract V1 now starts the 80/20 earlier-stage rehearsal mix at Stage 2. Each
Stage 2 and later report must show all earlier-stage retention results. A stage
cannot pass if an earlier gate falls by more than five percentage points.
Screening uses 1,000 fixed episodes. Promotion uses 10,000 fixed episodes.

The Review 7 performance refactor removed CUDA-to-host episode counters from
the rollout loop. A fixed seed produced identical values in all 50 iterations.
The run took 68.76 seconds before the change and 70.94 seconds after it. Thus,
the change is neutral for results but did not make training faster. A new sweep
measured about 39 ms for each simulator step from 512 through 8,192 arenas.
The simulator kernel sequence is still the main cost. Three concurrent jobs did
not complete one iteration in 90 seconds and were stopped. Use sequential seed
runs on this test system.

Normal builds do not contain an active ML controller. The CMake option is off by default. Network games do not use the experiment.

## Engine build

```powershell
cmake -S . -B build-ml -DWZ_ML_EXPERIMENT=ON
cmake --build build-ml --config RelWithDebInfo --target warzone2100
```

## One headless battle

Use the example specification as a template. Use an absolute output path in `report_path`.

```powershell
build-ml\src\RelWithDebInfo\warzone2100.exe `
  --autogame --headless --nosound --skirmish=ml-v1.json `
  --ml-policy=scripted --ml-players=0,1 `
  --ml-combat-test=ml\warzone_tactical\contracts\combat_test_v1.example.json `
  --ml-trace=ml-actions.jsonl `
  --ml-resolved-stats=resolved-stats.json `
  --gamestate-crc-trace=game-crc.txt
```

The fixture clears economy objects, makes a flat land area at the map center, creates two identical Viper tracked Mini-Rocket Pod droids on each team, and forces full visibility. The Mini-Rocket Pod is a direct-fire anti-tank weapon with no splash or magazine reload rule. It writes `team_0_win`, `team_1_win`, `draw`, or `unfinished`. It does not treat a timeout as a draw.

## The 100-battle spike gate

```powershell
uv run --project ml\warzone_tactical python -m warzone_tactical.engine_harness `
  --executable build-ml\src\RelWithDebInfo\warzone2100.exe `
  --repository . `
  --output ml\warzone_tactical\runs\engine-spike `
  --battles 100
```

The harness also repeats seed 0 twice. It requires equal quantized action traces and equal game-state CRC traces.

## Native model fixture

First, get `resolved-stats.json` from the engine. Then run:

```powershell
uv run --project ml\warzone_tactical python -m warzone_tactical.wzml `
  --fixture `
  --resolved-stats resolved-stats.json `
  --output ml\warzone_tactical\artifacts\fixture-v1
```

Start a battle with:

```text
--ml-policy=native:<absolute path to policy.manifest.json>
```

The loader checks the WZML file hash and the resolved engine stats hash. It refuses a mismatch.

## Tests and throughput

```powershell
uv run --project ml\warzone_tactical pytest ml\warzone_tactical\tests

uv run --project ml\warzone_tactical python -m warzone_tactical.performance `
  --resolved-stats resolved-stats.json `
  --output throughput.json
```

The throughput report tests 512, 1,024, 2,048, 4,096, and 8,192 arenas. It selects the fastest size below 85 percent of GPU memory. The first gate is 300,000 unit decisions per second.

Run the complete compiled-engine revalidation with one command:

```powershell
powershell -ExecutionPolicy Bypass -File ml\warzone_tactical\revalidate.ps1
```

This command builds the engine, runs 100 battles, keeps CRC repeat separate,
checks 10,000 observation goldens, checks fixture and trained action parity,
runs forced-hit fidelity, and tests the resolved short and long hit rates.

The measured RTX 4090 Laptop report is in `runs/throughput-v1-rocket-warm.json`. It uses the
resolved engine stats and passes the first throughput gate at 769,121 unit
decisions per second with 8,192 arenas. Each size has a discarded warmup
backward pass.

## Accepted V1 results

- The 100-battle engine gate produced 51 team-0 wins, 46 team-1 wins, 3 same-update draws, and no unfinished matches.
- Same-seed quantized actions and CRC traces are identical. These remain separate gates.
- C++ and Python match exactly for 10,000 Q15 observations, 3,612 fixture-model actions, and 3,604 promoted seed-4 actions.
- The forced-hit fidelity trace matches all 400 rows exactly, including projectile damage timing.
- Stage 1 seeds 4, 17, and 31 clear 99.75, 99.70, and 99.70 percent of 10,000 fixed episodes. All mean and p90 clear-time gates pass.
- Stage 2 uses the exact Rocket-Pod range band `512 < distance <= 1088`. Its main gate is the continuous fraction of episode time in this band. The Stage 3 kill limit is 60 seconds.
- The Stage 1 model is not a full battle model. Its drive heads are not trained until Stage 2, so an unfinished native engine battle is expected at this point.

## Known Contract V1 limits

- `V1-COMBAT-001`: The simulator does not model the engine `bumpTime` gate or moving-target lead for misses. This is accepted only for `Rocket-Pod` V1 with no splash damage. Model it before mixed weapons or splash damage.
- `V1-COMBAT-003`: Projectile damage must never appear before it appears in the engine trace. The center-distance simulator may lag an engine target-surface collision by at most one 100 ms update. This has its own fidelity gate.
- The engine ML path uses the observation-batch target position for turret aim and fire. Zero mismatches verify this fix. They do not show that live sequential ordering was equal. Stock droids still use live positions, so the incumbent benchmark has a known stock-versus-ML asymmetry.
- Re-audit the temporary target-position override before V1 uses an instant-hit weapon.
- The independent Python observation builder matches 10,000 compiled C++ golden states exactly in Q15.
- Unit slots use the fixed V1 roster. Before expansion, Contract V2 must define mid-game spawns, more than eight enemies, and the case where all assigned enemy slots are dead.
- A projectile-buffer overflow increments `projectile_drop_count`. A promotion report must show zero drops.

## Post-compile acceptance gate

Do these checks before Stage 1 training:

1. Build the engine with ML control on and off.
2. Run the full 100-battle headless integration sequence.
3. Run a normal non-ML game. Confirm that turret movement, firing, and movement are unchanged.
4. Compare the first 100 matched engine and simulator updates. Report the maximum target-distance error and the number of turret-alignment and shot-result mismatches for each side.
5. If the combat-order differences are small, add a measured bound to the Contract V1 divergence list. If they are not small, make the ML engine path use tick-start position snapshots for combat.
6. Generate 10,000 engine golden states and require exact Q15 observation parity.
7. Run the Stage 1 sanity control, then the formal promotion with training seeds 4, 17, and 31 only after all engine gates pass.

Before 4v4 work, replace repeated droid-list scans with one droid-ID lookup map per action batch.

## Frozen files

- `contracts/contract_v1.json` defines the action, observation, simulator, and result contracts.
- `benchmarks/benchmark_v1.json` defines fixed screening, promotion, training, and engine seed ranges.
- `warzone_tactical/curriculum.py` defines the eight stage gates and the 80/20 rehearsal mix.
- `warzone_tactical/evaluation.py` keeps policy leagues separate from promotion gates.

Do not change a Contract V1 benchmark after the first accepted 1v1 anchors are added. Make Contract V2 for a breaking change.
