You are conducting a deep verification of a scientific hypothesis. Decompose the hypothesis into its core assumptions and evaluate each independently.

Goal: {{ goal }}

Hypothesis under verification:
<HYPOTHESIS_TEXT id="{{ hypothesis_id }}">
{{ hypothesis_text }}
</HYPOTHESIS_TEXT_END id="{{ hypothesis_id }}">

Procedure:
1. List every assumption the hypothesis rests on. Each assumption must be a single, testable claim — not the hypothesis itself restated.
2. For each assumption, classify it as `plausible`, `uncertain`, or `implausible`, and write a one-paragraph rationale grounded in the literature you can find (use the available search tools). If the literature is silent, say so.
3. Identify the single weakest assumption — the one whose failure would collapse the hypothesis fastest.
4. Suggest one concrete experiment that would either confirm or kill the weakest assumption.

When complete, call the `record_review` tool with `kind="verification"`. Populate `assumptions[]` with one entry per assumption you analyzed, set the overall `verdict` based on the most consequential finding (treat `disproved` as reserved for cases where an assumption is contradicted by strong literature evidence), and use `notes` to flag the weakest assumption and proposed experiment.
