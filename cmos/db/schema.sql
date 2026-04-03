-- CMOS SQLite Schema
-- Version: 2.1
-- Minimal seed schema for MCP-based project initialization

PRAGMA foreign_keys = ON;

-- Project-level metadata
-- Standard keys:
--   project_id: UUID or slug uniquely identifying this CMOS project
--   project_name: Human-readable project name
--   tracelab_project_id: UUID of linked TraceLab project (for cross-referencing)
--   created_at: ISO timestamp when project was initialized
--   schema_version: Schema version identifier
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Initialize standard metadata keys (no-op if already exists)
INSERT OR IGNORE INTO metadata (key, value) VALUES ('project_id', '');
INSERT OR IGNORE INTO metadata (key, value) VALUES ('project_name', '');
INSERT OR IGNORE INTO metadata (key, value) VALUES ('tracelab_project_id', '');
INSERT OR IGNORE INTO metadata (key, value) VALUES ('created_at', datetime('now'));
INSERT OR IGNORE INTO metadata (key, value) VALUES ('schema_version', '2.1');

CREATE TABLE IF NOT EXISTS sprints (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  focus TEXT,
  status TEXT,
  start_date TEXT,
  end_date TEXT,
  total_missions INTEGER,
  completed_missions INTEGER
);

CREATE TABLE IF NOT EXISTS missions (
  id TEXT PRIMARY KEY,
  sprint_id TEXT REFERENCES sprints(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  completed_at TEXT,
  notes TEXT,

  -- Full mission specification fields
  objective TEXT,
  context TEXT,
  success_criteria TEXT,  -- JSON array
  deliverables TEXT,      -- JSON array
  reference_docs TEXT,    -- JSON array
  domain_fields TEXT,     -- JSON object

  -- Timestamp fields for KPI calculations
  created_at TEXT,        -- When mission was added
  started_at TEXT,        -- When mission first moved to In Progress
  updated_at TEXT,        -- Last state change timestamp

  -- Legacy metadata field
  metadata TEXT
);

CREATE TABLE IF NOT EXISTS mission_dependencies (
  from_id TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
  to_id TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
  type TEXT NOT NULL,
  PRIMARY KEY (from_id, to_id)
);

CREATE TABLE IF NOT EXISTS contexts (
  id TEXT PRIMARY KEY,
  source_path TEXT NOT NULL,
  content TEXT NOT NULL,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS context_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  context_id TEXT NOT NULL,
  session_id TEXT,
  source TEXT,
  content_hash TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_context_snapshots_ctx ON context_snapshots (context_id, created_at);
CREATE INDEX IF NOT EXISTS idx_context_snapshots_hash ON context_snapshots (context_id, content_hash);

CREATE TABLE IF NOT EXISTS session_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT,
  agent TEXT,
  mission TEXT,
  action TEXT,
  status TEXT,
  summary TEXT,
  next_hint TEXT,
  raw_event TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mission TEXT,
  source_path TEXT NOT NULL,
  ts TEXT,
  payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  title TEXT NOT NULL,
  sprint_id TEXT REFERENCES sprints(id) ON DELETE SET NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  agent TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  summary TEXT,
  captures TEXT DEFAULT '[]',
  next_steps TEXT,
  metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_type ON sessions (type);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions (started_at DESC);

CREATE TABLE IF NOT EXISTS prompt_mappings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt TEXT NOT NULL,
  behavior TEXT NOT NULL
);

-- Strategic decisions index for queryable project memory
CREATE TABLE IF NOT EXISTS strategic_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  context_id TEXT NOT NULL DEFAULT 'master_context',
  decision_text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  sprint_id TEXT,
  snapshot_id INTEGER,
  project_domain TEXT,
  session_id TEXT,  -- Reference to session where decision was captured
  mission_id TEXT,  -- Reference to mission that produced this decision
  source_chunk_ids TEXT,  -- JSON array of TraceLab chunk UUIDs for decision provenance
  category TEXT,          -- architectural | process | tooling | design | business
  superseded_by INTEGER,  -- FK to newer decision that replaces this one
  status TEXT NOT NULL DEFAULT 'active',  -- active | superseded | archived
  evidence TEXT,          -- JSON array of TraceLab evidence references [{type, id}]
  FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE,
  FOREIGN KEY (sprint_id) REFERENCES sprints(id) ON DELETE SET NULL,
  FOREIGN KEY (snapshot_id) REFERENCES context_snapshots(id) ON DELETE SET NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
  FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE SET NULL,
  FOREIGN KEY (superseded_by) REFERENCES strategic_decisions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_strategic_decisions_created ON strategic_decisions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_strategic_decisions_sprint ON strategic_decisions (sprint_id);
CREATE INDEX IF NOT EXISTS idx_strategic_decisions_domain ON strategic_decisions (project_domain);
CREATE INDEX IF NOT EXISTS idx_strategic_decisions_session ON strategic_decisions (session_id);
CREATE INDEX IF NOT EXISTS idx_strategic_decisions_mission ON strategic_decisions (mission_id);
CREATE INDEX IF NOT EXISTS idx_strategic_decisions_status ON strategic_decisions (status);
CREATE INDEX IF NOT EXISTS idx_strategic_decisions_category ON strategic_decisions (category);

-- Structured learnings for queryable project knowledge
CREATE TABLE IF NOT EXISTS learnings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  category TEXT,            -- technical | process | agent-behavior | tooling
  status TEXT NOT NULL DEFAULT 'active',  -- active | archived | superseded
  sprint_id TEXT,
  session_id TEXT,
  mission_id TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (sprint_id) REFERENCES sprints(id) ON DELETE SET NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
  FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_learnings_status ON learnings (status);
CREATE INDEX IF NOT EXISTS idx_learnings_sprint ON learnings (sprint_id);
CREATE INDEX IF NOT EXISTS idx_learnings_category ON learnings (category);

-- FTS5 full-text search index for strategic decisions
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
  decision_text,
  content='strategic_decisions',
  content_rowid='id'
);

-- Auto-sync triggers for FTS5 index
CREATE TRIGGER IF NOT EXISTS decisions_fts_insert AFTER INSERT ON strategic_decisions BEGIN
  INSERT INTO decisions_fts(rowid, decision_text) VALUES (new.id, new.decision_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_fts_delete AFTER DELETE ON strategic_decisions BEGIN
  INSERT INTO decisions_fts(decisions_fts, rowid, decision_text) VALUES('delete', old.id, old.decision_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_fts_update AFTER UPDATE OF decision_text ON strategic_decisions BEGIN
  INSERT INTO decisions_fts(decisions_fts, rowid, decision_text) VALUES('delete', old.id, old.decision_text);
  INSERT INTO decisions_fts(rowid, decision_text) VALUES (new.id, new.decision_text);
END;

-- Project identity view for easy access to project-level metadata
CREATE VIEW IF NOT EXISTS project_identity AS
SELECT
  (SELECT value FROM metadata WHERE key = 'project_id') AS project_id,
  (SELECT value FROM metadata WHERE key = 'project_name') AS project_name,
  (SELECT value FROM metadata WHERE key = 'tracelab_project_id') AS tracelab_project_id,
  (SELECT value FROM metadata WHERE key = 'created_at') AS created_at,
  (SELECT value FROM metadata WHERE key = 'schema_version') AS schema_version;

CREATE VIEW IF NOT EXISTS active_missions AS
SELECT m.id,
       m.name,
       m.status,
       m.completed_at,
       m.notes,
       s.id AS sprint_id,
       s.title AS sprint_title
  FROM missions m
  LEFT JOIN sprints s ON s.id = m.sprint_id
 WHERE m.status IN ('Current', 'In Progress');

CREATE VIEW IF NOT EXISTS mission_details AS
SELECT m.id,
       m.name,
       m.status,
       s.id AS sprint_id,
       s.title AS sprint_title,
       m.objective,
       m.context,
       m.success_criteria,
       m.deliverables,
       m.reference_docs,
       m.domain_fields,
       m.completed_at,
       m.notes
  FROM missions m
  LEFT JOIN sprints s ON s.id = m.sprint_id;

CREATE VIEW IF NOT EXISTS sprint_summary AS
SELECT
  s.id AS sprint_id,
  s.title,
  s.status,
  s.focus,
  s.start_date,
  s.end_date,
  COUNT(m.id) AS total_missions,
  COUNT(CASE WHEN m.status = 'Completed' THEN 1 END) AS completed_missions,
  COUNT(CASE WHEN m.status = 'Blocked' THEN 1 END) AS blocked_missions,
  COUNT(CASE WHEN m.status IN ('Current', 'In Progress') THEN 1 END) AS active_missions,
  (
    SELECT COUNT(DISTINCT sd.id)
    FROM strategic_decisions sd
    WHERE sd.sprint_id = s.id
  ) AS decisions_count
FROM sprints s
LEFT JOIN missions m ON m.sprint_id = s.id
GROUP BY s.id, s.title, s.status, s.focus, s.start_date, s.end_date;
