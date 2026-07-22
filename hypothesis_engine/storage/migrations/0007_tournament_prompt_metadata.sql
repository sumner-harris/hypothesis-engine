ALTER TABLE tournament_matches ADD COLUMN prompt1_hyp_id TEXT REFERENCES hypotheses(id);
ALTER TABLE tournament_matches ADD COLUMN prompt2_hyp_id TEXT REFERENCES hypotheses(id);
ALTER TABLE tournament_matches ADD COLUMN prompt1_side TEXT;
ALTER TABLE tournament_matches ADD COLUMN prompt2_side TEXT;
ALTER TABLE tournament_matches ADD COLUMN winner_prompt_position INTEGER;
ALTER TABLE tournament_matches ADD COLUMN prompt1_chars INTEGER;
ALTER TABLE tournament_matches ADD COLUMN prompt2_chars INTEGER;
ALTER TABLE tournament_matches ADD COLUMN prompt_order_key TEXT;
