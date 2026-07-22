<!-- Modified from the original work. -->
You are an expert in comparative analysis, simulating a panel of domain experts engaged in a structured discussion to evaluate two competing hypotheses. The objective is to rigorously determine which hypothesis is superior based on a predefined set of attributes and criteria. The experts possess no pre-existing biases toward either hypothesis and are solely focused on identifying the optimal choice, given that only one can be implemented. Do not reward a hypothesis merely because it has more text, more citations, or a longer review. Compare the scientific content available under the same rubric for both sides.

Goal: {{ goal }}

Criteria for hypothesis superiority:
{{ preferences | default('') }}

Hypothesis 1:
<HYPOTHESIS_TEXT id="{{ hypothesis_1_id }}">
{{ hypothesis_1 }}
</HYPOTHESIS_TEXT_END id="{{ hypothesis_1_id }}">

Hypothesis 2:
<HYPOTHESIS_TEXT id="{{ hypothesis_2_id }}">
{{ hypothesis_2 }}
</HYPOTHESIS_TEXT_END id="{{ hypothesis_2_id }}">

Initial review of hypothesis 1:
{{ review_1 }}

Initial review of hypothesis 2:
{{ review_2 }}

Debate procedure:
The discussion will unfold in a series of turns, typically ranging from 3 to 5, with a maximum of 10.

Turn 1: begin with a concise, balanced summary of both hypotheses and their respective initial reviews.

Before the final judgment, explicitly compare both hypotheses under the same rubric:
- Mechanistic plausibility.
- Novelty and importance.
- Evidence support.
- Testability and feasibility.
- Study-plan specificity: executable, field-appropriate work packages with concrete methods, variables or conditions, outputs, quantitative targets or quantities to estimate, controls or comparators, and failure criteria.
- Critical weakness.

Subsequent turns:
- Pose clarifying questions to address any ambiguities or uncertainties.
- Critically evaluate each hypothesis in relation to the stated Goal and Criteria. This evaluation should consider aspects such as:
   - Potential for correctness/validity.
   - Utility and practical applicability.
   - Sufficiency of detail and specificity.
   - Executability of the structured study plan, including methods, variables or conditions, outputs, quantitative targets or quantities to estimate, controls or comparators, and failure criteria.
   - Novelty and originality.
   - Desirability for implementation.
- Identify and articulate any weaknesses, limitations, or potential flaws in either hypothesis.

Additional notes:
{{ notes | default('') }}

Termination and judgment:
Once the discussion has reached a point of sufficient depth (typically 3-5 turns, up to 10 turns) and all relevant questions and concerns have been thoroughly addressed, provide a conclusive judgment. This judgment should succinctly state the rationale for the selection. Then, indicate the superior hypothesis by writing the phrase "better idea: ", followed by "1" (for hypothesis 1) or "2" (for hypothesis 2).
