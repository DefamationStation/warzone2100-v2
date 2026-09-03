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

Stage 2 is implemented but is not promoted. The task is to enter and hold the
range band `512 < distance <= 1088`. The gate needs a mean of 70 percent time in
band and 65 percent for every seed. The do-nothing floor is 33 percent, because
one spawn in three starts inside the band. The scripted controller reaches 93.11
percent.

Two environment faults blocked all early work. Both are corrected.

1. All arenas shared one episode clock. Each PPO update therefore saw one
   episode phase only. Explained variance fell to a large negative value at
   every multiple of 37.5 iterations, which is one episode. A random start
   phase for each arena corrected it. The mean rose from 35.48 to 50.01 percent.
   The same fault was present in Stage 1. A shared `environment.py` now supplies
   the clock, so Stages 3 to 8 inherit the correction.
2. A stale critic over-valued fast states, so the advantage gave speed a
   negative value and the policy drove itself to zero speed. The correlation
   between commanded speed and advantage reached -0.41 in a failing seed while
   a passing seed stayed positive. Two independent corrections both work: a
   higher entropy coefficient of 0.01, or four extra critic-only epochs for each
   iteration. Both remove the floor. Four extra critic epochs keep the original
   entropy of 0.002 and reach a higher peak, so that is the recommended setting.

Explained variance alone does not show either fault. Log the speed-advantage
correlation with `--diagnostics-every`.

Use 12 seeds, not 3. The result has two modes, and 3 seeds cannot separate two
conditions. A 4-seed partial once gave a gain of 6.3 points where the full 12
seeds gave a loss of 8.15 points. The sign changed.

All 12 seeds still improve at iteration 1,500 at about 0.9 points for each 100
iterations. The 1,500-iteration default was chosen for an earlier fault and is
too short. Every published Stage 2 number comes from an unfinished run.

Stage 2 training options:

```text
--entropy-coefficient        exploration pressure, default 0.002
--final-entropy-coefficient  end value for an entropy schedule
--entropy-anneal-iterations  length of that schedule
--critic-extra-epochs        extra critic-only passes for each iteration
--preactivation-penalty      weight on the squared drive pre-activations
--diagnostics-every          log the learning-signal diagnostics every N iterations
--heading-output-scale       action scale for the heading head, use 128
```

Do not reduce the entropy after the start when entropy is the only correction.
An anneal from 0.01 to 0.002 returned one seed to the floor.

## Speed of test runs

A three-seed test took 2 hours at the start. One seed of 1,500 iterations now
takes about 4.3 minutes.

Two corrections gave all of the gain. Compiling `build_observation` gave 2.17
times the speed, because the episode-phase correction had moved that call out of
the compiled region. Reusing the rollout outputs for telemetry, converting Q15
once for each minibatch, allocating the rollout buffers once, and caching the
wrapped-normal offsets gave the rest. Warm iteration time fell from 381.9 to
145.1 milliseconds. All three parity metrics stayed identical to the last bit.

Three measured claims that are false on this system:

- Host synchronization does not control the environment step time. Removing it
  made the run 3 percent slower.
- KL checks do not control the PPO update time. They use 2.69 ms of 103.59 ms.
- Mixed precision does not help. The GEMM operations use about 1 percent of the
  iteration time.

Run 3 seed groups at the same time. Do not run 6. Give each group its own
`TORCHINDUCTOR_CACHE_DIR`. Two processes that compile the same new kernel at the
same time both fail on Windows with `FileExistsError`.

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
