-- =============================================================================
-- Lumiq Initial Schema
-- Migration: 001_initial_schema.sql
-- Target:    Supabase (PostgreSQL 15)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- SECTION 1: Extensions
-- -----------------------------------------------------------------------------

-- pgcrypto provides gen_random_uuid() — preferred over uuid-ossp in Supabase PG15
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- -----------------------------------------------------------------------------
-- SECTION 2: Tables
-- (order matters: parent tables before child tables)
-- -----------------------------------------------------------------------------

-- Users table (mirrors auth.users, populated via trigger on signup)
CREATE TABLE IF NOT EXISTS public.users (
    id          UUID        PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email       TEXT        NOT NULL UNIQUE,
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chat table
-- NOTE: dataset_id removed — datasets have a FK to chat, not the reverse.
--       Keeping dataset_id here would create a circular dependency and is never
--       written or read by the backend.
CREATE TABLE IF NOT EXISTS public.chat (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    title       TEXT        NOT NULL DEFAULT 'New Chat',
    status      TEXT        NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'archived')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Message table
-- NOTE: sequence_number has no DEFAULT — the backend always computes and
--       supplies it explicitly (fetches MAX then increments). A default of 1
--       would silently mask missing inserts and cause ordering bugs.
CREATE TABLE IF NOT EXISTS public.message (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id         UUID        NOT NULL REFERENCES public.chat(id) ON DELETE CASCADE,
    role            TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT        NOT NULL,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    sequence_number INTEGER     NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Enforce ordering integrity: duplicate sequence positions within a chat are
    -- a data bug. The backend already computes MAX+1, but this is a hard guard.
    CONSTRAINT uq_message_chat_sequence UNIQUE (chat_id, sequence_number)
);

-- Dataset table
-- Explicit ::jsonb casts for JSONB defaults (required by strict Supabase validator)
CREATE TABLE IF NOT EXISTS public.dataset (
    id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id              UUID        NOT NULL REFERENCES public.chat(id) ON DELETE CASCADE,
    user_id              UUID        NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    filename             TEXT        NOT NULL,
    storage_path         TEXT        NOT NULL,
    schema_info          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    sample_rows          JSONB       NOT NULL DEFAULT '[]'::jsonb,
    summary_stats        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    chroma_collection_id TEXT        NOT NULL,
    row_count            INTEGER     NOT NULL DEFAULT 0,
    column_count         INTEGER     NOT NULL DEFAULT 0,
    uploaded_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Execution log table
CREATE TABLE IF NOT EXISTS public.execution_log (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id          UUID        NOT NULL REFERENCES public.chat(id) ON DELETE CASCADE,
    message_id       UUID        REFERENCES public.message(id),
    user_id          UUID        NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    user_query       TEXT        NOT NULL,
    generated_code   TEXT        NOT NULL,
    execution_output TEXT,
    execution_status TEXT        NOT NULL
                     CHECK (execution_status IN ('success', 'error', 'timeout', 'rejected')),
    error_message    TEXT,
    retry_count      INTEGER     NOT NULL DEFAULT 0,
    execution_time_ms FLOAT,
    executed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- -----------------------------------------------------------------------------
-- SECTION 3: Indexes
-- Minimal, purposeful indexes on high-frequency FK/filter columns only.
-- Design notes:
--   • idx_chat_user_updated: composite (user_id, updated_at DESC) covers both
--     the WHERE user_id = ? filter AND the ORDER BY updated_at DESC sort in a
--     single index scan — replaces two separate single-column indexes.
--   • idx_chat_active_user: partial index for the common UI query that lists
--     only active chats; smaller index, faster scans.
--   • idx_message_chat_sequence: composite leading on chat_id makes a separate
--     idx_message_chat_id redundant — PostgreSQL can use this for equality on
--     chat_id alone via index prefix scan.
--   • idx_dataset_user_id / idx_execution_log_user_id: required for RLS policy
--     evaluation on user_id = auth.uid() and for JOIN paths.
-- -----------------------------------------------------------------------------

-- chat: covering index for "get user's chats ordered by last activity"
CREATE INDEX IF NOT EXISTS idx_chat_user_updated
    ON public.chat(user_id, updated_at DESC);

-- chat: partial index for active-only chat listings (common UI filter)
CREATE INDEX IF NOT EXISTS idx_chat_active_user
    ON public.chat(user_id)
    WHERE status = 'active';

-- message: composite covers both chat_id equality lookups AND ordered reads
-- (no separate idx_message_chat_id needed — chat_id is the leading column here)
CREATE INDEX IF NOT EXISTS idx_message_chat_sequence
    ON public.message(chat_id, sequence_number ASC);

-- dataset: chat_id FK + user_id for RLS evaluation
CREATE INDEX IF NOT EXISTS idx_dataset_chat_id
    ON public.dataset(chat_id);
CREATE INDEX IF NOT EXISTS idx_dataset_user_id
    ON public.dataset(user_id);

-- execution_log: chat_id FK + user_id for RLS evaluation
CREATE INDEX IF NOT EXISTS idx_execution_log_chat_id
    ON public.execution_log(chat_id);
CREATE INDEX IF NOT EXISTS idx_execution_log_user_id
    ON public.execution_log(user_id);
CREATE INDEX IF NOT EXISTS idx_execution_log_message_id
ON public.execution_log(message_id);


-- -----------------------------------------------------------------------------
-- SECTION 4: Row Level Security (RLS)
-- All tables default-deny; policies grant only owner access.
-- auth.uid() is the correct Supabase JWT claim accessor.
-- -----------------------------------------------------------------------------

ALTER TABLE public.users          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.message        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.dataset        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.execution_log  ENABLE ROW LEVEL SECURITY;

-- Drop policies before re-creating to ensure idempotency on repeat runs
DO $$ BEGIN
    DROP POLICY IF EXISTS "Users can read own data"  ON public.users;
    DROP POLICY IF EXISTS "Users own chats"          ON public.chat;
    DROP POLICY IF EXISTS "Users own messages"       ON public.message;
    DROP POLICY IF EXISTS "Users own datasets"       ON public.dataset;
    DROP POLICY IF EXISTS "Users own logs"           ON public.execution_log;
END $$;

-- users: each row is the authenticated user themselves
CREATE POLICY "Users can read own data"
    ON public.users
    FOR ALL
    USING (auth.uid() = id);

-- chat: owned by the authenticated user
CREATE POLICY "Users own chats"
    ON public.chat
    FOR ALL
    USING (auth.uid() = user_id);

-- message: accessible if the parent chat belongs to the authenticated user
-- Using EXISTS instead of IN for better planner performance on large datasets
CREATE POLICY "Users own messages"
    ON public.message
    FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM public.chat
            WHERE public.chat.id = chat_id
              AND public.chat.user_id = auth.uid()
        )
    );

-- dataset: directly owns user_id column — simpler and avoids subquery
CREATE POLICY "Users own datasets"
    ON public.dataset
    FOR ALL
    USING (auth.uid() = user_id);

-- execution_log: directly owns user_id column
CREATE POLICY "Users own logs"
    ON public.execution_log
    FOR ALL
    USING (auth.uid() = user_id);


-- -----------------------------------------------------------------------------
-- SECTION 5: Triggers
-- -----------------------------------------------------------------------------

-- 5a. Sync auth.users → public.users on new signup
--     SECURITY DEFINER: runs as the function owner (postgres), not the caller.
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.users (id, email)
    VALUES (NEW.id, NEW.email)
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

-- Idempotent trigger creation: drop before re-creating (PG15 supports
-- CREATE OR REPLACE TRIGGER, but DROP + CREATE is universally safe)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();


-- 5b. Auto-update updated_at on chat row updates
--     Only chat requires this because the backend also manually touches
--     updated_at in add_message(); the trigger acts as a safety net.
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
-- search_path hardening: prevents search-path injection attacks on SECURITY
-- DEFINER functions. set_updated_at is not SECURITY DEFINER itself, but
-- explicit path is a good hygiene baseline for all trigger functions.
SET search_path = public
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_chat_set_updated_at ON public.chat;
CREATE TRIGGER trg_chat_set_updated_at
    BEFORE UPDATE ON public.chat
    FOR EACH ROW
    EXECUTE FUNCTION public.set_updated_at();

-- =============================================================================
-- End of migration 001_initial_schema.sql
-- =============================================================================
