# Task 2 Error Review, yashashree_shinde

**Model reviewed:** `experiment_bigru`, the best of my three by macro F1
**Test accuracy:** 0.9469, so about 2,019 errors in 38,000 test reviews
**Source:** `task2_error_review.csv`, exported by my notebook from the test split

Every excerpt below is copied from that file. The text is post preprocessing,
so it is lowercased with stopwords removed, which is why it reads oddly. The
negation words are deliberately kept.

## A. Confident false positives

Predicted positive, truly negative, with high confidence.

| # | Review excerpt | p(pos) | Error type | Testable fix |
|---|---|---|---|---|
| 1 | "about average far steakhouses go rib eye tasty little over cooked didn't complain because flavor still fantastic ... i'd recommend ruth chris lg's flemings morton's" | 0.9998 | Damning with faint praise. The review says the food was fine but recommends four competitors instead. | Build a subset of reviews that name a competitor and measure accuracy on it against reviews that do not. |
| 2 | "everyday noodles lot going fast friendly service attractive decorations impeccably clean ... only one issue bland plain food" | 0.9998 | Aspect imbalance. Many positive aspects and one negative, but the negative one is the food, which decides a restaurant review. | Weight tokens that follow "only" and "issue". Test on reviews containing those markers. |
| 3 | "love paris don't love vegas version ... hotel casino not good one reason only smokers" | 0.9997 | Negation over a long span. "don't love" and "not good" are present but outweighed by the surrounding praise of the real Paris. | Measure accuracy on the contains_negation slice against reviews without negation. The gap shows whether negation scope is the problem. |
| 4 | "all delicious thai flavors dumbed down muted" | 0.9996 | Qualified praise, where the final clause reverses the first. | Up weight the last 20 percent of each review and compare accuracy before and after. |
| 5 | "gorditas filling good not great good quick stop much better options when looking full flavored spicy authentic gordita" | 0.9995 | Explicit comparison against a better alternative, with "good not great" the actual verdict. | Same competitor and comparison subset as case 1. |

## B. Confident false negatives

Predicted negative, truly positive, with high confidence.

| # | Review excerpt | p(pos) | Error type | Testable fix |
|---|---|---|---|---|
| 1 | "take star due cons n npros rooms big somewhat comfortable ... casino not overly packed ... food court nice" | 0.0002 | Structured pros and cons review. The word "cons" and the deductions dominate even though the verdict is positive. | Detect reviews containing both "pros" and "cons" and measure accuracy on that subset. |
| 2 | "place much better since changed owners ... old owners terrible waited forever food never came ... it horrible n nnow much better" | 0.0003 | Temporal reversal. The review is positive about now and negative about the past, and the past takes more words. | Build a subset using markers such as "used to", "since", "changed owners", "now", and measure accuracy on it. |
| 3 | "wonderful market extensive variety products rated market 4 only rate customer service 1" | 0.0004 | Split rating across two aspects. The reviewer scores the product high and the service low in the same sentence. | Measure accuracy on reviews containing two explicit numeric ratings. |
| 4 | "sometimes just like go out drink dance ... getting ear drums assaulted what only described nasty bass drops" | 0.0005 | Positive sentiment expressed in negative vocabulary. "assaulted" and "nasty" describe things the reviewer enjoys. | Inspect which tokens carry the largest negative weight and check how many are intensifiers rather than sentiment words. |
| 5 | "went yesterday found place just closed another victim good chef bad location ... hint future restauranteurs" | 0.0006 | Sympathetic review of a failed business. The sentiment is about the closure, not about the quality. | Hard to fix with features. Worth reporting as an inherent limit rather than an actionable bug. |

## C. Near threshold errors

The five errors whose predicted probability sits closest to 0.5.

| # | Review excerpt | p(pos) | Error type | Testable fix |
|---|---|---|---|---|
| 1 | "punk rock bar nhappy hour m f noon 5 all drinks only 2!" | 0.4998 | Almost no sentiment vocabulary. 13 tokens, mostly facts. | Measure accuracy against review length. If short reviews are worst, the model needs a prior rather than more features. |
| 2 | "cannot request level spiciness pad thai when ordering online wanted no spicy wasnt bad anyway" | 0.4997 | Double negative. "wasnt bad" is mildly positive and the model reads the negation tokens as negative. | Same negation slice test as A3. This review has three negation tokens in 19 words. |
| 3 | "far favorite theater world better ultrastar ... only thing really stay top roving packs rude noisy teens" | 0.4997 | Strong praise followed by a long complaint about other customers, not the business. | Check whether reviews complaining about other patrons are overrepresented in errors. |
| 4 | "can't complain about food everything pretty good ... didn't come basmati rice" | 0.4994 | "can't complain" is a positive idiom built from negative words. | Add a small list of fixed idioms such as "can't complain" and "not bad" and measure the effect. |
| 5 | "super cheesey sirens pirate show ti silly cheesey ... free show ok just wish hadnt wasted good camcorder tape lol" | 0.5006 | Genuinely mixed. The reviewer enjoys it and regrets it in the same breath. | None obvious. This is near the limit of what a binary label can represent. |

## D. Slice specific failures

**Slice used:** long reviews, which in my preprocessing means reviews that
reach the 256 token maximum length and are therefore truncated.

| # | Review excerpt | p(pos) | Tokens | Error type | Testable fix |
|---|---|---|---|---|---|
| 1 | "port authority formerly known patransit ... i've great experience using bus pittsburgh" | 0.4161 | 256 | Truncated before the verdict. The review is a long account that reaches its conclusion late. | Rerun with max length 512 and compare accuracy on reviews at the 256 cap. |
| 2 | "beautiful place nice ambiance ok drinks music better waitress dressed provocatively ... felt uncomfortable" | 0.7312 | 256 | Positive about the venue, negative about the experience, and truncated. | Same length test as D1. |
| 3 | "what era this? ... mirrored ceilings velvet upholstered chairs cheetah print fabric like 80's movie scarface" | 0.9022 | 256 | Sarcasm. Descriptive words read as positive while the tone is mocking. | Hard to fix with bag of words features. Report as a limitation. |
| 4 | "only biltmore apple store make typical shopper feel like total douche just walking inside exposed cultish behaviors apple fanboys" | 0.6176 | 256 | Sustained sarcasm again, with no plainly negative sentiment words. | Same as D3. |
| 5 | "let's meet chandler pita jungle! texts fiddle r dee ... what hell!?! that's not near mall!!!" | 0.6482 | 256 | Stylised narrative writing that barely resembles a review. | Measure accuracy against the fraction of out of vocabulary tokens per review. |

**All five are at exactly 256 tokens**, which is my maximum length. That is the
clearest single finding in this review: my long review failures are not
scattered, they all sit at the truncation boundary.

## Summary of error types observed

| Error type | Count of 20 | What it means |
|---|---|---|
| Negation over a long span | 5 | The model sees the negation token but not what it applies to. |
| Mixed or aspect split sentiment | 5 | Genuinely balanced reviews where one aspect carries the verdict. |
| Truncation at 256 tokens | 4 | The conclusion is cut off before the model sees it. |
| Sarcasm or irony | 3 | Positive vocabulary, negative intent, or the reverse. |
| Temporal or structural reversal | 2 | The review contradicts its own earlier half. |
| Too little signal | 1 | Short factual reviews with almost no sentiment vocabulary. |

**16 of the 20 errors are in a slice flagged as containing negation.** That
looks alarming until it is checked against the base rate, which is the point of
having slice metrics.

### The negation finding does not survive checking

My first reading of this sample was that negation was the main problem. My own
slice metrics in `task2_slice_metrics.json` say otherwise:

| Slice | Reviews | Macro F1 | Error rate |
|---|---|---|---|
| contains_negation | 23,262 | 0.9430 | 0.0532 |
| short_reviews | 12,899 | 0.9434 | 0.0549 |
| medium_reviews | 14,843 | 0.9508 | 0.0492 |
| long_reviews | 10,258 | 0.9413 | 0.0565 |
| whole test set | 38,000 | 0.9469 | 0.0531 |

Reviews containing negation fail at 0.0532 against 0.0531 for the test set as a
whole. There is no effect at all. The reason 16 of my 20 sampled errors contain
negation is that **61.2 percent of all test reviews contain negation**, so at
the base rate about 12 of 20 would be expected anyway. A binomial test on 16 of
20 against a base rate of 0.612 gives p = 0.063, which is not significant at
n = 20.

This is worth recording rather than quietly dropping. A hand inspection of 20
errors is a good way to generate hypotheses and a bad way to test them, because
an error sample tells you nothing about the base rate it was drawn from.

### What the slice metrics actually show

**Long reviews are the worst slice**, 0.0565 against 0.0492 for medium, a 15
percent relative increase in error rate. That matches section D, where all five
sampled long review failures sit at exactly 256 tokens, my maximum length.

### What I would try first

**Raise the maximum length to 512.** This is the one hypothesis that both the
slice metrics and the error sample support. Long reviews fail more often, and
every long review failure I sampled is at the truncation boundary. Rerunning at
512 and comparing the error rate on reviews over 256 tokens settles it.

**Test mixed and aspect split reviews second.** Five of the twenty are reviews
that praise one aspect and condemn another. A subset built from reviews
containing both strong positive and strong negative terms would measure
whether this is a real weakness or another base rate illusion.

**Report sarcasm as a limit rather than a bug.** Three of the twenty are
sarcastic, and no feature available to a model of this kind would catch them.
