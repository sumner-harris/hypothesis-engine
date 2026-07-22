-- Migration 0002: add reviews.feasibility column.
-- Older DBs created from a schema.sql that predated the four-score model are
-- missing this column; the canonical schema.sql already has it. SQLite's
-- ALTER TABLE … ADD COLUMN is idempotent in the migration runner because the
-- whole migration is wrapped in a single transaction keyed off
-- schema_migrations.version.

ALTER TABLE reviews ADD COLUMN feasibility REAL;
