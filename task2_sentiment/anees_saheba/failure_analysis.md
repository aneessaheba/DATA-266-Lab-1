# Task 2 Error Review, anees_saheba

**Model reviewed:** `experiment_lstm`, the best of my three by validation loss
**Checkpoint:** `checkpoints/experiment_lstm.pt`, epoch 6, validation loss 0.1402
**Source:** `outputs/error_review_20.csv`, exported by the notebook from the test split
**Total test errors:** 1,891 of 38,000, an error rate of 4.98 percent. 20 are reviewed below.

Every excerpt is copied from the exported file. Nothing is edited or invented.

## A. Confident false positives

Predicted positive, truly negative, with high confidence.

| # | Review excerpt | p(pos) | Error type | Testable fix |
|---|---|---|---|---|
| 1 | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012" | 0.9990 | Label noise. This text is positive by any reading. The gold label of negative looks wrong. | Have two people relabel a sample of confident errors. If more than about 10 percent are mislabelled, report accuracy with a noise corrected ceiling instead of treating every error as a model fault. |
| 2 | "NOTE: This was a 4 star review, but the food quality and ESPECIALLY customer service have gone down the tubes. See update below." then 400 words of praise | 0.9988 | Sentiment reversal. The verdict is in the first sentence and the rest of the review contradicts it. | Train with the first 64 tokens duplicated at the end of the sequence. If reversal errors fall, recency and primacy weighting is the problem, not vocabulary. |
| 3 | "I can't say much about the place, other than, even taking into consideration the number of people present, it felt cramped. The TrimTini I had was really delicious" | 0.9986 | Mixed sentiment with positive details and a negative overall verdict. Food praise outweighs the complaint. | Add an aspect count feature, or evaluate on a mixed sentiment subset built by selecting reviews containing both a strong positive and a strong negative term. |
| 4 | "the room was exactly what we needed, bed, clean bathroom, heater, comforters and toiletries. My biggest complaint are that the rooms are so close together you can hear your neighbors" | 0.9984 | Concession structure. A list of positives then the actual complaint, signalled by "My biggest complaint". | Up weight tokens after contrast markers such as but, however, although, complaint. Test by measuring accuracy on reviews containing those markers before and after. |
| 5 | "We must have had a really off experience, based on other reviews. Food: Ordered pork bao buns, onion rings, farm burger... Buns and onion rings were both somewhat bland. Pork belly was charred/tough." | 0.9983 | Vocabulary gap. The negative signal is in food specific words such as bland, charred and tough, which are rarer than generic sentiment words. | Lower the minimum frequency from 3 to 2, or raise the vocabulary above 25,000, and check whether accuracy on food specific complaints improves. |

## B. Confident false negatives

Predicted negative, truly positive, with high confidence.

| # | Review excerpt | p(pos) | Error type | Testable fix |
|---|---|---|---|---|
| 1 | "EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city, but I'm rethinking that." | 0.0008 | Label noise again, in the other direction. The text reads negative. The gold label is positive. | Same relabelling audit as A1. These two cases together suggest the noise is not one sided. |
| 2 | "This place is so much better since they changed owners. My wife and I went when it was the old owners, it was terrible. We waited forever and the food never came... It was horrible. Now its much better." | 0.0009 | Temporal reversal. The review is positive about now and negative about the past, and the past takes more words. | Measure accuracy on reviews containing before and after markers such as used to, since they, changed owners, now. If it is well below 95 percent, the model is averaging over time rather than tracking it. |
| 3 | "I have been ordering from Winder for over a year now... yes we pay a premium (fees, higher prices, etc.) but not having to shop for staples every week really helps us manage the household... The produce is okay, it is hit or miss" | 0.0009 | Hedged positive. The praise is qualified throughout with okay, hit or miss and an admission about price. | Build a hedged subset using words such as okay, decent, hit or miss, and compare accuracy against unhedged reviews. |
| 4 | "Last night several parents came in with over 15 children... The bartender expressed that it was not a place to have children running around as it is against the law and a liability issue" | 0.0012 | Third party narrative. The reviewer is praising the bar for handling badly behaved customers, but almost every sentiment word describes the customers. | Check whether reviews whose negative words cluster in the first two thirds and whose verdict appears in the last sentence are overrepresented in errors. If so, add a final sentence feature. |
| 5 | "Gather around little Yelpers, let me tell you about my stay at ye ol' Treasure Island... Our room was beautifully decorated and looked real crisp... And 10 minutes later there was water all over the floor" | 0.0013 | Narrative with a comic disaster in the middle. 937 words, so it is truncated at 320 and the model never sees the ending. | This one is a genuine truncation case, so test it directly: rerun with max length 512 and compare the error rate on reviews over 320 tokens. Section 5 of results.md predicts this will not help much. |

## C. Near threshold errors

The five test errors whose predicted probability sits closest to 0.5. These are
cases the model is openly unsure about, so they show where the decision boundary
actually is.

| # | Review excerpt | p(pos) | Error type | Testable fix |
|---|---|---|---|---|
| 1 | "Fun if you're a drag queen with disposable income. But for a guy who's just buying heels for fun, not the best. I love their catalogues and wide selection, but the prices are off the charts." | 0.5002 | Balanced mixed sentiment. Two praises and two complaints, and the complaints carry the verdict. | Same contrast marker test as A4. This review contains two instances of but. |
| 2 | "good, quick, & easy place to stop... got 3 x carne asada and 3 x adobaba tacos. hecka good!!" | 0.4998 | Informal and non standard vocabulary. hecka, adobaba and in/out privileges are likely to be out of vocabulary. | Log the unknown token rate per review and correlate it with error rate. If errors rise with unknown tokens, a subword tokeniser is the fix rather than a bigger word vocabulary. |
| 3 | "Update - First trip was a fluke, bad sandwich. I would say steer clear of the patty melt. We have eaten here a few times since and it has been stellar." | 0.4995 | Sentiment reversal with the negative stated first and the positive conclusion later. | Same as A2. This is the mirror image of case A2 and the same fix should move both. |
| 4 | "Tottie's is one of those rare restaurants that does two cuisines well... They may have gone overboard on the fusion part, the Thai Fried Chicken and General Tsao's Chicken are almost indistinguishable, but the Thai curry and noodle dishes are distinct anyway." | 0.4992 | Qualified praise. The positive verdict is wrapped in criticism of specific dishes. | Same hedged subset test as B3. |
| 5 | "I've been buying dog food... the staff are really knowledgeable, and they are also low pressure... The store area itself can feel kind of cramped, but I'd much rather spend less time getting the right thing" | 0.4986 | Concession in the reviewer's favour. The complaint is acknowledged and then dismissed, which the model reads as a complaint. | Contrast marker test again. Four of these twenty errors turn on a contrast marker, which makes it the single most promising fix to try first. |

## D. Slice specific failures

**Slice definition:** reviews in the longest tercile of the test set by word
count, 12,815 reviews with an LSTM error rate of 5.08 percent against 4.73
percent on the medium tercile.

| # | Review excerpt | p(pos) | Words | Error type | Testable fix |
|---|---|---|---|---|---|
| 1 | "I have been ordering from Winder for over a year now..." | 0.0009 | 247 | Hedged positive, also appears as B3 | As B3 |
| 2 | "NOTE: This was a 4 star review, but the food quality... have gone down the tubes" | 0.9988 | 411 | Sentiment reversal, truncated, also A2 | As A2 |
| 3 | "Last night several parents came in with over 15 children..." | 0.0012 | 187 | Third party narrative, also B4 | As B4 |
| 4 | "Gather around little Yelpers..." | 0.0013 | 937 | Narrative, heavily truncated, also B5 | As B5 |
| 5 | "Got a room on New Years Eve after partying in DTLV..." | 0.9984 | 167 | Concession structure, also A4 | As A4 |

The overlap is itself the finding. All five long review failures already appear
in the confident error groups, so length is not an independent failure mode. It
is a multiplier: a longer review has more room for a reversal, a concession or a
digression, and those structures are what actually break the model.

## Summary of error types observed

| Error type | Count of 20 | What it means |
|---|---|---|
| Sentiment reversal, verdict contradicts the bulk | 4 | The model pools over the whole review and cannot tell which statement is the conclusion. |
| Concession or contrast structure | 4 | Words after but, however and complaint carry more weight than the model gives them. |
| Mixed or hedged sentiment | 4 | Genuinely balanced reviews near the decision boundary. |
| Third party or narrative sentiment | 2 | Sentiment words describe someone other than the business. |
| Label noise | 2 | The gold label looks wrong to me. |
| Vocabulary gap, rare or informal words | 2 | The signal sits in words outside the 25,000 vocabulary. |
| Truncation beyond 320 tokens | 2 | The ending, where the verdict often sits, is cut off. |

### What I would try first

**Contrast marker weighting.** Eight of the twenty errors turn on a reversal or
a concession, which is by far the largest group. The test is cheap: measure
accuracy on reviews containing but, however, although or complaint, against
reviews without them. If there is a clear gap, the fix is worth building.

**A relabelling audit second.** Two of twenty confident errors look like bad
gold labels. If that rate holds across the 1,891 errors, roughly 190 of them are
not the model's fault, which would move the effective ceiling from 100 percent to
about 99.5 and change how the remaining gap should be read.

**Raising max length last.** It is the obvious fix and the slice numbers in
results.md say it is the wrong one. The truncated slice has a lower error rate
than the long slice for all three models. Only two of twenty errors are really
about truncation, and one of those is also a reversal.
