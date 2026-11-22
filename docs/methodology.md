# Methodology

## Preprocessing (`preprocess.py`)
1. Strip HTML tags (`<br />`) and decode entities.
2. Lower-case, remove URLs, split punctuation into separate tokens.
3. Expand `n't` to ` not`.
4. **Negation marking**: the next 3 words after *not, no, never, nothing,
   without, ...* get a `NOT_` prefix, stopping at punctuation or *but*.

No stop-word removal or stemming: stop-word lists usually contain *not* and
*no*, which carry a lot of sentiment.
