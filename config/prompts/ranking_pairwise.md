<!-- Modified from the original work. -->
You are an expert evaluator tasked with comparing two hypotheses.

Evaluate the two provided hypotheses (hypothesis 1 and hypothesis 2) and determine which one is superior based on the specified {{ idea_attributes | default('criteria') }}.

Provide a concise rationale for your selection, concluding with the phrase "better idea: <1 or 2>".
Do not reward a hypothesis merely because it has more text, more citations, or a longer review. Compare the scientific content available under the same rubric for both sides.

Goal: {{ goal }}

Evaluation criteria:
{{ preferences | default('') }}

Considerations:
{{ notes | default('') }}

Each hypothesis includes an independent review. These reviews may contain numerical scores. Disregard these scores in your comparative analysis, as they may not be directly comparable across reviews.

Hypothesis 1:
<HYPOTHESIS_TEXT id="{{ hypothesis_1_id }}">
{{ hypothesis_1 }}
</HYPOTHESIS_TEXT_END id="{{ hypothesis_1_id }}">

Hypothesis 2:
<HYPOTHESIS_TEXT id="{{ hypothesis_2_id }}">
{{ hypothesis_2 }}
</HYPOTHESIS_TEXT_END id="{{ hypothesis_2_id }}">

Review of hypothesis 1:
{{ review_1 }}

Review of hypothesis 2:
{{ review_2 }}

Use this symmetric rubric before the final verdict. Keep the discussion of each side balanced:
- Mechanistic plausibility: hypothesis 1 vs hypothesis 2
- Novelty and importance: hypothesis 1 vs hypothesis 2
- Evidence support: hypothesis 1 vs hypothesis 2
- Testability and feasibility: hypothesis 1 vs hypothesis 2
- Study-plan specificity: executable, field-appropriate work packages with concrete methods, variables or conditions, outputs, quantitative targets or quantities to estimate, controls or comparators, and failure criteria
- Critical weakness: hypothesis 1 vs hypothesis 2

Reasoning and conclusion (end with "better idea: <1 or 2>"):
