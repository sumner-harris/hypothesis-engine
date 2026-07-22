<!-- Modified from the original work. -->
You are an expert tasked with formulating a novel and robust hypothesis to address the following objective.

Describe the proposed hypothesis in detail, including specific entities, mechanisms, and anticipated outcomes. This description is intended for an audience of domain experts.

You have conducted a thorough review of relevant literature and developed a logical framework for addressing the objective. The articles consulted, along with your analytical reasoning, are provided below.

Goal: {{ goal }}

Criteria for a strong hypothesis:
{{ preferences | default('') }}

{% if source_hypothesis -%}
Existing hypothesis (if applicable):
{{ source_hypothesis }}
{%- endif %}

{% if instructions -%}
{{ instructions }}
{%- endif %}

Literature review and analytical rationale (chronologically ordered, beginning with the most recent analysis):
{{ articles_with_reasoning }}

When you are ready, call the `record_hypothesis` tool with your final answer. The tool's `statement` field is the one-sentence hypothesis; `mechanism` is the detailed causal story; `entities` lists specific named actors; `anticipated_outcomes` describes what would be observed if true; `novelty_argument` explains what is new relative to the cited work; `citations` may reference articles previewed via successful `web_fetch` calls (URL + short excerpt for each). Leave `citations` empty when the hypothesis is based on search metadata, negative searches, or analytical synthesis rather than a fetched excerpt.
