-- Persist structured citations on hypothesis rows.
ALTER TABLE hypotheses ADD COLUMN citations TEXT NOT NULL DEFAULT '[]';
