-- FinOps ledger: RANGE-partitioned by request timestamp.
-- Monthly partitions are created idempotently at gateway startup
-- (see app.db.postgres.ensure_partitions). The DEFAULT partition
-- absorbs rows that land outside those windows.

CREATE TABLE IF NOT EXISTS finops_ledger (
    request_id          TEXT            NOT NULL,
    ts                  TIMESTAMPTZ     NOT NULL,
    tenant_id           TEXT            NOT NULL,
    conversation_id     TEXT,
    cache_status        TEXT            NOT NULL,
    similarity          DOUBLE PRECISION,
    routed_tier         TEXT,
    routed_model        TEXT,
    provider            TEXT,
    failover_from       TEXT,
    tokens_in           INTEGER,
    tokens_out          INTEGER,
    cost_actual_usd     NUMERIC(14, 8),
    cost_saved_usd      NUMERIC(14, 8),
    guardrail_ms        DOUBLE PRECISION,
    l1_ms               DOUBLE PRECISION,
    embed_ms            DOUBLE PRECISION,
    l2_ms               DOUBLE PRECISION,
    router_ms           DOUBLE PRECISION,
    gateway_overhead_ms DOUBLE PRECISION,
    ttft_ms             DOUBLE PRECISION,
    total_ms            DOUBLE PRECISION,
    status              TEXT            NOT NULL
) PARTITION BY RANGE (ts);

CREATE TABLE IF NOT EXISTS finops_ledger_default
    PARTITION OF finops_ledger DEFAULT;

CREATE INDEX IF NOT EXISTS finops_ledger_tenant_ts_idx
    ON finops_ledger (tenant_id, ts);
