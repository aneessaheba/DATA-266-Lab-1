# Task 3 run sheet

Everything needed to start my CycleGAN run on the GPU lab machine, in order.
Written before the run so nothing has to be worked out on the day.

## 1. Get the code

```
git clone https://github.com/aneessaheba/DATA-266-Lab-1.git
cd DATA-266-Lab-1
git checkout anees_saheba/task3-gan
```

## 2. Put the data in place

The notebook reads from these two folders. Both are ignored by git, so the
images have to be copied onto the machine separately.

```
task3_gan/data/monet_jpg/    300 Monet paintings
task3_gan/data/photo_jpg/    7038 photographs
```

Check the counts before starting, because a missing folder only shows up
several cells in:

```
ls task3_gan/data/monet_jpg | wc -l     # expect 300
ls task3_gan/data/photo_jpg | wc -l     # expect 7038
```

## 3. Time one epoch before committing to the full run

This is the step worth not skipping. It takes about seven minutes and it
tells me whether the slot is long enough.

```
DATA266_NUM_EPOCHS=1 jupyter nbconvert --to notebook --execute \
  task3_gan/anees_saheba/src/cyclegan_monet_photo.ipynb \
  --output /tmp/timing.ipynb --ExecutePreprocessor.timeout=86400
```

Read the seconds field on the EPOCH 1/1 line of the log, then multiply.
My teammate measured 361 seconds an epoch on an RTX 5090 and my generator
benchmarks 1.07 times her cost, so the estimate is about 385 seconds, which
is 5.3 hours for 50 epochs. If the machine is slower than that, drop the
epoch count rather than the resolution, because 256 is the size Kaggle
scores at.

| Epochs | At 385 s | At 500 s |
|---|---|---|
| 30 | 3.2 h | 4.2 h |
| 40 | 4.3 h | 5.6 h |
| 50 | 5.3 h | 6.9 h |

## 4. Start the real run, detached

Detached on purpose. The run outlives a dropped ssh connection or a closed
terminal, which a plain foreground run does not.

```
nohup jupyter nbconvert --to notebook --execute \
  task3_gan/anees_saheba/src/cyclegan_monet_photo.ipynb \
  --output run_task3_full.ipynb \
  --ExecutePreprocessor.timeout=86400 \
  > task3_console.log 2>&1 &
disown
```

Watch it with:

```
tail -f reproducibility/raw_logs/task3_cyclegan_*.log
```

## 5. What a healthy run looks like

Taken from my teammate's completed run, so these are real numbers to compare
against rather than guesses.

- Generator loss settles into the 5 to 10 band within the first epoch
- Discriminator loss stays roughly between 0.1 and 0.5
- `nan_inf=0` on every epoch line. Anything else means stop and look
- Cycle losses fall and keep falling. They are the signal that the mapping is
  still invertible
- A milestone checkpoint appears every 10 epochs

If the discriminator loss collapses to near zero and the generator loss climbs
without coming back, the discriminator has won and the run is not recoverable
by waiting. Stop it and lower the discriminator learning rate.

## 6. After it finishes

```
task3_gan/anees_saheba/checkpoints/      keep the final generators file
task3_gan/anees_saheba/outputs/          figures, metrics, translations
reproducibility/raw_logs/                the raw log, never edited
```

Copy the whole member folder and the raw log off the lab machine. The full
image dump and the milestone checkpoints stay off the repository; only the
final generators file, the figures, the metrics and a handful of sample
images get committed.

## 7. My settings and why they differ from my teammate's

| | Mine | Yashashree |
|---|---|---|
| Seed | 5330 | 4349 |
| Residual blocks | 6 | 9 |
| Base filters | 80 | 64 |
| Parameters per generator | 12,239,363 | 11,378,179 |
| Total parameters | 30,008,200 | 28,285,832 |
| Identity weight | 2.5 | 5.0 |
| Cycle weight | 10.0 | 10.0 |
| Image size | 256 | 256 |
| Epochs | 50 | 50 |

The parameter budgets are within six percent of each other on purpose, so any
difference in the results is about where the capacity sits and how the
objective is weighted rather than about one model simply being bigger.
