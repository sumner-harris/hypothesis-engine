-- Prevent concurrent ranking workers from judging the same unordered pair.
CREATE TABLE IF NOT EXISTS tournament_pair_reservations (
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    pair_key        TEXT NOT NULL,
    task_id         TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    reserved_at     INTEGER NOT NULL,
    PRIMARY KEY(session_id, pair_key)
);
CREATE INDEX IF NOT EXISTS tpr_task ON tournament_pair_reservations(task_id);
