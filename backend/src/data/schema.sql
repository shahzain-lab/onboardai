-- ============================================================================
-- DATABASE SCHEMA FOR TASK MANAGEMENT
-- ============================================================================

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) UNIQUE NOT NULL,  -- Slack/Discord user ID
    email VARCHAR(255),
    name VARCHAR(255),
    role VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',  -- pending, in_progress, completed, blocked
    priority VARCHAR(20) DEFAULT 'medium',  -- low, medium, high, urgent
    source VARCHAR(100),  -- standup, meeting, onboarding, manual
    source_id VARCHAR(255),  -- conversation_id or meeting_id
    due_date TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes for quick lookups
    CONSTRAINT status_check CHECK (status IN ('pending', 'in_progress', 'completed', 'blocked')),
    CONSTRAINT priority_check CHECK (priority IN ('low', 'medium', 'high', 'urgent'))
);

-- Standups table
CREATE TABLE IF NOT EXISTS standups (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    standup_date DATE NOT NULL,
    yesterday_tasks JSONB,  -- Array of task descriptions
    today_tasks JSONB,      -- Array of task descriptions
    blockers JSONB,         -- Array of blocker descriptions
    summary TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(user_id, standup_date)
);

-- Meetings table
CREATE TABLE IF NOT EXISTS meetings (
    id SERIAL PRIMARY KEY,
    meeting_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500),
    transcript TEXT,
    summary TEXT,
    action_items JSONB,  -- Array of action item objects
    participants JSONB,  -- Array of user_ids
    meeting_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Onboarding table
CREATE TABLE IF NOT EXISTS onboarding (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role VARCHAR(100),
    start_date DATE,
    manager_id VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',  -- active, completed, on_hold
    progress_percentage INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT progress_check CHECK (progress_percentage >= 0 AND progress_percentage <= 100)
);

-- Conversations table (for tracking agent conversations)
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(100) REFERENCES users(user_id) ON DELETE SET NULL,
    workflow_type VARCHAR(50),
    agent_name VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',  -- active, completed, failed
    messages JSONB,  -- Array of message objects
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_tasks_user_status ON tasks(user_id, status);

CREATE INDEX IF NOT EXISTS idx_standups_user_date ON standups(user_id, standup_date DESC);

CREATE INDEX IF NOT EXISTS idx_meetings_date ON meetings(meeting_date DESC);
CREATE INDEX IF NOT EXISTS idx_meetings_participants ON meetings USING GIN (participants);

CREATE INDEX IF NOT EXISTS idx_onboarding_user_id ON onboarding(user_id);
CREATE INDEX IF NOT EXISTS idx_onboarding_status ON onboarding(status);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_workflow ON conversations(workflow_type);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS FOR AUTO-UPDATE
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_meetings_updated_at BEFORE UPDATE ON meetings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_onboarding_updated_at BEFORE UPDATE ON onboarding
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- USEFUL QUERIES (STORED PROCEDURES)
-- ============================================================================

-- Get next task for user
CREATE OR REPLACE FUNCTION get_next_task(p_user_id VARCHAR)
RETURNS TABLE (
    task_id INTEGER,
    title VARCHAR,
    description TEXT,
    status VARCHAR,
    priority VARCHAR,
    due_date TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        id,
        tasks.title,
        tasks.description,
        tasks.status,
        tasks.priority,
        tasks.due_date
    FROM tasks
    WHERE user_id = p_user_id
      AND status IN ('pending', 'in_progress')
    ORDER BY 
        CASE priority
            WHEN 'urgent' THEN 1
            WHEN 'high' THEN 2
            WHEN 'medium' THEN 3
            WHEN 'low' THEN 4
        END,
        due_date NULLS LAST,
        created_at
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Get user task summary
CREATE OR REPLACE FUNCTION get_user_task_summary(p_user_id VARCHAR)
RETURNS TABLE (
    total_tasks BIGINT,
    pending_tasks BIGINT,
    in_progress_tasks BIGINT,
    completed_tasks BIGINT,
    blocked_tasks BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_tasks,
        COUNT(*) FILTER (WHERE status = 'pending') as pending_tasks,
        COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress_tasks,
        COUNT(*) FILTER (WHERE status = 'completed') as completed_tasks,
        COUNT(*) FILTER (WHERE status = 'blocked') as blocked_tasks
    FROM tasks
    WHERE user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE DATA FOR TESTING
-- ============================================================================

-- Insert sample user
INSERT INTO users (user_id, email, name, role) 
VALUES ('U062M4SSY3V', 'test@example.com', 'Test User', 'Developer')
ON CONFLICT (user_id) DO NOTHING;

-- Insert sample tasks
INSERT INTO tasks (user_id, title, description, status, priority, source) 
VALUES 
    ('U062M4SSY3V', 'Complete authentication module', 'Implement OAuth2 flow', 'in_progress', 'high', 'standup'),
    ('U062M4SSY3V', 'Write unit tests', 'Add tests for new features', 'pending', 'medium', 'meeting'),
    ('U062M4SSY3V', 'Code review PR #123', 'Review teammates pull request', 'pending', 'urgent', 'manual'),
    ('U062M4SSY3V', 'Update documentation', 'Document new API endpoints', 'pending', 'low', 'standup')
ON CONFLICT DO NOTHING;

-- ============================================================================
-- VIEWS FOR EASY QUERYING
-- ============================================================================

-- Active tasks view
CREATE OR REPLACE VIEW active_tasks AS
SELECT 
    t.id,
    t.user_id,
    u.name as user_name,
    t.title,
    t.status,
    t.priority,
    t.due_date,
    t.created_at
FROM tasks t
JOIN users u ON t.user_id = u.user_id
WHERE t.status IN ('pending', 'in_progress')
ORDER BY 
    CASE t.priority
        WHEN 'urgent' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END,
    t.due_date NULLS LAST;

-- Recent standups view
CREATE OR REPLACE VIEW recent_standups AS
SELECT 
    s.id,
    s.user_id,
    u.name as user_name,
    s.standup_date,
    jsonb_array_length(s.yesterday_tasks) as yesterday_count,
    jsonb_array_length(s.today_tasks) as today_count,
    jsonb_array_length(s.blockers) as blocker_count,
    s.created_at
FROM standups s
JOIN users u ON s.user_id = u.user_id
ORDER BY s.standup_date DESC, s.created_at DESC;