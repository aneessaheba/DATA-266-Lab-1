# Task 1 Sequence Model Failure Analysis

**Member:** anees_saheba
**Model:** 4 layer character level GPT, d_model 256, 4 heads, context 128, seed 5330
**Checkpoint:** `checkpoints/best_model.pt` (epoch 10, validation loss 0.7538)
**Source of snippets:** `outputs/generated_samples.txt` and `outputs/generated_samples.json`

All three snippets below are copied directly from my own generated output. No
text has been edited.

## Case 1: Repetition

**Prompt:** `One day, Tom`
**Decoding:** greedy (argmax, no sampling)

**Generated snippet:**

```
One day, Tom went to the park with his mom. He saw a big storm on the ground.
He wanted to see what was inside. He wanted to see what was inside.
```

**Failure type:** Repetition

**Observation:**

The sentence "He wanted to see what was inside." is produced twice, word for
word. This sample has the worst repetition of all twelve I generated: its
repeated 4gram rate is 0.316, against 0.104 for the most diverse sample.

Greedy decoding always takes the single most probable next character, so the
process is deterministic. Once the model has written a sentence and a full
stop, the hidden state at the start of the next sentence is close to the state
it was in before, which makes the same continuation most probable again. With
no sampling there is nothing that can break out of the loop. The model has no
memory of having already written that sentence, only the 128 characters of
context, so it has no signal that it is repeating.

This is why greedy decoding is the wrong choice for open ended generation even
though it maximises probability at every single step. Across my samples greedy
averages a repeated 4gram rate of 0.249 while temperature 1.2 averages 0.117.

## Case 2: Loss of coherence

**Prompt:** `Once upon a time`
**Decoding:** greedy

**Generated snippet:**

```
They saw a big boy who was very sad and angry. They wanted to show her mommy
how to be careful when they saw the boy was so happy to have a new friend
```

**Failure type:** Loss of coherence, specifically a contradiction of an
established fact

**Observation:**

The boy is introduced as "very sad and angry" and then described as "so happy"
within the same sentence, with nothing in between that would explain the
change. There is a second problem in the same snippet: "They wanted to show her
mommy" refers to the mother as if she were absent, but the previous sentence
already placed her in the scene.

The grammar is correct throughout. What fails is state tracking. The model
predicts the next character from the previous 128 characters only, and it has
no representation of a character in the story or of facts already asserted
about them. "so happy to have a new friend" is a very common phrase in
TinyStories, so it is locally probable, and local probability is the only thing
the model optimises. Nothing in the training objective penalises contradicting
something written 40 characters earlier.

This failure is harder to see than repetition because every individual sentence
reads correctly. It only appears when the snippet is read as a whole.

## Case 3: Hallucinated words and broken grammar

**Prompt:** `Once upon a time`
**Decoding:** temperature sampling, T = 1.2

**Generated snippet:**

```
She loved playing outside and majwming running around. One day, Anna went to
the park so much adventure. That day she was even surprised eating. Carefully,
here no hourse. She saw her mom radious and ate the pastels.
```

**Failure type:** Hallucination and broken grammar

**Observation:**

Three strings here are not English words at all: "majwming", "hourse" and
"radious". They are character sequences that follow English spelling patterns
closely enough to look like words at a glance. Grammar also breaks down, in
"went to the park so much adventure" and "Carefully, here no hourse", neither
of which is a well formed sentence.

This failure is specific to character level modelling. A word level model can
only ever emit words from its vocabulary, so it cannot invent "hourse". My
model predicts one character at a time from a vocabulary of 91, so any letter
sequence is reachable. At T = 1.2 the logits are divided by 1.2 before the
softmax, which flattens the distribution and raises the probability of
characters the model considers unlikely. A single unlikely character early in a
word pushes the remaining context off the distribution the model learned, and
the rest of the word is then generated from an unfamiliar state.

The same setting that causes this is also what makes this sample the most
diverse of the twelve, with a distinct 3 of 0.802 and a repeated 4gram rate of
0.104. Raising the temperature trades repetition for incoherence.

## What the three cases show together

The three failures are not independent. Cases 1 and 3 are the two ends of one
decoding tradeoff, measured across my own samples:

| Decoding | repeated 4gram rate | distinct 3 |
|---|---|---|
| greedy | 0.249 | 0.658 |
| T = 0.5 | 0.186 | 0.697 |
| T = 0.8 | 0.132 | 0.763 |
| T = 1.2 | 0.117 | 0.791 |

Lowering temperature produces repetition and raising it produces invented words.
No setting removes both, because the two failures have opposite causes.

Case 2 is not on that axis. It appears at every temperature and comes from the
context window rather than from decoding. With 128 characters of context the
model cannot attend to a fact stated earlier than roughly two sentences back,
so it cannot stay consistent over a longer span. Increasing the context length
would be the change to test, and it is the first thing I would try with more
compute.
