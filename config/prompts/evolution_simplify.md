Refine the hypothesis below to make it simpler and more testable, while preserving its core scientific claim.

Goal: {{ goal }}

Criteria:
{{ preferences | default('') }}

Original hypothesis:
<HYPOTHESIS_TEXT id="{{ hypothesis_id }}">
{{ hypothesis }}
</HYPOTHESIS_TEXT_END id="{{ hypothesis_id }}">

Review of the original:
{{ review | default('(no review available)') }}

Instructions:
1. Identify which elements of the hypothesis are load-bearing vs. ornamental. Strip the latter.
2. State the simplified claim in one sentence at the top.
3. Re-derive the mechanism and anticipated outcomes from this simpler claim. They should still be specific.
4. Propose at least one experiment that is easier to run on the simplified version than on the original.

When complete, call `record_hypothesis` (set `strategy="simplify"` and `parent_ids=["{{ hypothesis_id }}"]`).
