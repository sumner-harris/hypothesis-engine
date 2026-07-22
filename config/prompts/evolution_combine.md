You are an expert in scientific synthesis. Combine the best parts of the two hypotheses below into a new, stronger hypothesis. The result must (a) preserve what works in each, (b) explicitly resolve any contradictions between them, and (c) be more specific and testable than either parent.

Goal: {{ goal }}

Criteria:
{{ preferences | default('') }}

Hypothesis A:
<HYPOTHESIS_TEXT id="{{ hypothesis_a_id }}">
{{ hypothesis_a }}
</HYPOTHESIS_TEXT_END id="{{ hypothesis_a_id }}">

Review of Hypothesis A:
{{ review_a | default('(no review available)') }}

Hypothesis B:
<HYPOTHESIS_TEXT id="{{ hypothesis_b_id }}">
{{ hypothesis_b }}
</HYPOTHESIS_TEXT_END id="{{ hypothesis_b_id }}">

Review of Hypothesis B:
{{ review_b | default('(no review available)') }}

Instructions:
1. Identify the strongest mechanism in A and the strongest in B.
2. State explicitly which contradictions exist between A and B and how your combination resolves them.
3. Propose the synthesized hypothesis with specific entities, mechanisms, and anticipated outcomes.

When complete, call `record_hypothesis` (set `strategy="combine"` and `parent_ids=["{{ hypothesis_a_id }}", "{{ hypothesis_b_id }}"]`).
