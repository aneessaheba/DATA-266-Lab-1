
# Task 1: Sequence Model Failure Analysis

## Failure Case 1: Broken Grammar and Semantic Inconsistency

Generated text snippet:

"She had a big smile on her house and a special wind."

Failure type:

Broken grammar and semantic inconsistency.

Observation:

The model produces a grammatically unusual and semantically incorrect
description. A smile is normally associated with a person, not a house,
and "a special wind" is not connected clearly to the sentence. This shows
that the model learned common words and sentence patterns but did not fully
learn how to maintain meaningful relationships between objects and actions.

## Failure Case 2: Loss of Coherence and Abrupt Topic Change

Generated text snippet:

"Every day, she saw a big tree with lots of money. The pink was a big
pretty girl with a playground beautiful pla"

Failure type:

Loss of coherence and incomplete generation.

Observation:

The text begins with a girl observing a tree, but the sentence suddenly
introduces money, a pink girl, and a playground without a logical transition.
The final phrase is also incomplete. This indicates that the model can
generate locally plausible phrases but struggles to maintain a consistent
storyline over a longer sequence.

## Failure Case 3: Repetition and Semantic Inconsistency

Generated text snippet:

"They saw a big tree with a big rock. The tree was filled with a tall green
ball on the ground. Lily was so happy that she saw that she wanted to go off
to the ground."

Failure type:

Repetition and semantic inconsistency.

Observation:

The model repeats words such as "big," "tree," and "ground." It also creates
an unusual description in which a tree is filled with a ball and produces an
unclear sentence about going "off to the ground." This suggests that the
model has learned frequent character and word patterns but does not always
preserve realistic object relationships or grammatical structure.

## Overall Observation

The model achieves low validation loss and approximately 77.6% next-character
accuracy, but the generated samples still contain grammar errors, repetition,
and coherence problems. This difference occurs because next-character accuracy
can be high even when the model makes errors that damage the meaning of a
longer story. Increasing the training data, model capacity, context quality,
or training duration could be tested as possible improvements.
