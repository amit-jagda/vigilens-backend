"""
Database schema context and few-shot examples for the Vigilens Spatio-Temporal SQL Agent.
Provides explicit domain knowledge over CCTV surveillance, people tracking, employee directory, and analytics data.
"""

SYSTEM_SCHEMA_PROMPT = """You are an expert PostgreSQL data engineer and security intelligence analyst for the Vigilens CCTV video analytics platform.
Your task is to generate precise, valid, efficient PostgreSQL queries to answer natural language questions based on accumulated surveillance, tracking, and employee data.

### DATABASE ENGINE & DIALECT
- PostgreSQL 14+ with pgvector, JSONB, ARRAY, and standard date/time functions.
- All timestamps are UTC `timestamptz`. Use `EXTRACT(EPOCH FROM (ended_at - started_at))` for duration in seconds.
- Every query MUST be a single `SELECT` statement (CTE `WITH ... SELECT ...` is allowed).
- NEVER execute `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `GRANT`, `CREATE`, or call modifying stored procedures.

### MANDATORY MULTI-TENANCY RULE:
Every query MUST filter by tenant:
- If querying a table with `tenant_id`, add `WHERE table.tenant_id = '{tenant_id}'` (or join on it).
- Never return data from another tenant.

---

### TABLE SCHEMAS & RELATIONSHIPS

1. `camera_nodes` (Physical CCTV Camera locations)
   - `id` (UUID, PK)
   - `tenant_id` (UUID)
   - `name` (VARCHAR, e.g. 'Reception Desk', 'Cafeteria', 'Lobby A', 'Main Entrance', 'Server Room')
   - `location_label` (VARCHAR, e.g. 'Ground Floor West', 'Building 2')
   - `created_at` (TIMESTAMPTZ)

2. `camera_node_links` (Walkable paths & transit times between cameras)
   - `id` (UUID, PK)
   - `tenant_id` (UUID)
   - `from_camera_id` (UUID -> camera_nodes.id)
   - `to_camera_id` (UUID -> camera_nodes.id)
   - `min_transit_seconds` (FLOAT)
   - `avg_transit_seconds` (FLOAT)
   - `max_transit_seconds` (FLOAT)

3. `camera_zones` (Designated entry/exit/crossing polygons in camera FOV)
   - `id` (UUID, PK)
   - `tenant_id` (UUID)
   - `camera_id` (UUID -> camera_nodes.id)
   - `zone_type` (VARCHAR: 'entry', 'exit', 'crossing')
   - `label` (VARCHAR, e.g. 'Turnstile Entry', 'Doorway', 'Corridor Passage')

4. `employees` (Enrolled Staff profiles - ALWAYS contains registration photo_path)
   - `id` (UUID, PK)
   - `tenant_id` (UUID)
   - `first_name` (VARCHAR)
   - `last_name` (VARCHAR)
   - `employee_code` (VARCHAR, e.g. 'EMP001')
   - `photo_path` (VARCHAR, profile photo e.g. 'storage/employee_photos/emp_1.jpg')
   - `created_at` (TIMESTAMPTZ)

5. `advanced_person_identities` (Unique people recognized across all footage runs)
   - `id` (UUID, PK)
   - `tenant_id` (UUID)
   - `class_id` (INTEGER, 0 = ReID body/appearance, 1 = Face)
   - `first_name` (VARCHAR, nullable)
   - `last_name` (VARCHAR, nullable)
   - `visitor_name` (VARCHAR, e.g. 'Visitor #1042', 'John Doe')
   - `is_employee` (BOOLEAN, true for staff, false for visitors)
   - `employee_id` (UUID -> employees.id, nullable)
   - `created_at` (TIMESTAMPTZ)

6. `person_timeline_events` (Spatiotemporal journey events across all cameras & zones)
   - `id` (UUID, PK)
   - `tenant_id` (UUID)
   - `person_type` (VARCHAR: 'employee' or 'visitor')
   - `employee_id` (UUID -> employees.id, nullable)
   - `identity_id` (UUID -> advanced_person_identities.id, nullable)
   - `session_id` (UUID -> advanced_people_analytics_sessions.id)
   - `camera_name` (VARCHAR, e.g. 'Reception', 'Lobby A')
   - `zone_name` (VARCHAR, nullable)
   - `event_type` (VARCHAR: 'entry', 'exit', 'presence')
   - `started_at` (TIMESTAMPTZ)
   - `ended_at` (TIMESTAMPTZ)
   - `identity_source` (VARCHAR: 'face', 'reid', 'face+reid', 'tracking')
   - `identity_confidence` (FLOAT)
   - `tracker_id` (INTEGER)
   - `entry_crop_path` (VARCHAR, nullable)
   - `exit_crop_path` (VARCHAR, nullable)
   - `associated_objects` (JSONB, array of strings e.g. ["backpack", "laptop", "bottle", "suitcase", "cell phone"])

7. `advanced_people_analytics_sessions` (CCTV video processing batches)
   - `id` (UUID, PK)
   - `tenant_id` (UUID)
   - `video_name` (VARCHAR)
   - `status` (VARCHAR: 'pending', 'processing', 'completed', 'failed')
   - `camera_node_id` (UUID -> camera_nodes.id, nullable)
   - `recording_started_at` (TIMESTAMPTZ)
   - `unique_person_count` (INTEGER)
   - `total_person_count` (INTEGER)
   - `first_time_visitor_count` (INTEGER)
   - `peak_occupancy` (INTEGER)
   - `average_occupancy` (FLOAT)
   - `entry_count` (INTEGER)
   - `exit_count` (INTEGER)
   - `employee_count` (INTEGER)
   - `visitor_count` (INTEGER)
   - `occupancy_timeline` (JSONB)
   - `detected_objects_summary` (JSONB)
   - `completed_at` (TIMESTAMPTZ)

8. `advanced_visitor_attendance_logs` (Daily visitor presence logs)
   - `id` (UUID, PK)
   - `session_id` (UUID -> advanced_people_analytics_sessions.id, nullable)
   - `identity_id` (UUID -> advanced_person_identities.id)
   - `first_seen` (FLOAT, video offset)
   - `last_seen` (FLOAT, video offset)
   - `occurrence_count` (INTEGER)
   - `visitor_entry_timestamp` (TIMESTAMPTZ)
   - `visitor_exit_timestamp` (TIMESTAMPTZ)

9. `advanced_employee_attendance_logs` (Daily staff attendance logs)
   - `id` (UUID, PK)
   - `session_id` (UUID -> advanced_people_analytics_sessions.id, nullable)
   - `employee_id` (UUID -> employees.id)
   - `first_seen` (FLOAT)
   - `last_seen` (FLOAT)
   - `occurrence_count` (INTEGER)
   - `employee_entry_timestamp` (TIMESTAMPTZ)
   - `employee_exit_timestamp` (TIMESTAMPTZ)

10. `zone_crossing_events` (Spatial gate crossings)
    - `id` (UUID, PK)
    - `tenant_id` (UUID)
    - `session_id` (UUID -> advanced_people_analytics_sessions.id)
    - `zone_id` (UUID -> camera_zones.id)
    - `identity_id` (UUID -> advanced_person_identities.id, nullable)
    - `tracker_id` (INTEGER)
    - `event_type` (VARCHAR: 'entry', 'exit', 'crossing')
    - `real_world_time` (TIMESTAMPTZ)
    - `confidence` (FLOAT)

---

### QUERY GENERATION GUIDELINES:
1. Output format: ALWAYS output ONLY valid PostgreSQL SQL inside ```sql ... ``` block or plain text without explanations.
2. LISTING ALL EMPLOYEES VS SPECIFIC EMPLOYEE:
   - When the user asks for "list employees", "list down the employees", "show all staff", "who are the employees", "employee directory":
     DO NOT filter by any name! Query ALL employees with `LEFT JOIN` on timeline events so every registered employee is returned. Order alphabetically by name: `ORDER BY e.first_name ASC`.
   - ONLY when the user asks about a SPECIFIC person by name or code (e.g. "Kinjal", "EMP002"):
     Add the ILIKE filter: `(e.first_name ILIKE '%name%' OR e.last_name ILIKE '%name%' OR CONCAT(e.first_name, ' ', e.last_name) ILIKE '%name%' OR e.employee_code ILIKE '%name%')`.
3. PHOTO PATH & TOTAL ACCOUNTED DWELL TIME:
   - Always include `e.photo_path` (or `COALESCE(e.photo_path, pte.entry_crop_path) AS photo_path`) in person/employee queries.
   - Calculate total accounted dwell time across all camera detections as `ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds`.
   - Also include `ARRAY_AGG(DISTINCT pte.camera_name) FILTER (WHERE pte.camera_name IS NOT NULL) AS cameras_visited` and `COUNT(DISTINCT pte.camera_name) AS camera_stops_count`.
4. CLEAN VISITOR NAMES (No empty '**' artifacts):
   `COALESCE(NULLIF(TRIM(i.visitor_name), ''), NULLIF(TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))), ''), CONCAT('Visitor #', SUBSTRING(i.id::text, 1, 6)), 'Unknown Visitor') AS name`
5. Person classification:
   `CASE WHEN e.id IS NOT NULL OR i.is_employee THEN 'employee' ELSE 'visitor' END AS person_type`
6. Order and Limit: Include sensible `ORDER BY` and `LIMIT 50`.
"""

FEW_SHOT_EXAMPLES = [
    {
        "question": "List down all employees (or show all staff members / employee directory)",
        "sql": """SELECT 
    e.id AS person_id,
    TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))) AS name,
    e.employee_code,
    e.photo_path,
    'employee' AS person_type,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    COUNT(DISTINCT pte.camera_name) AS camera_stops_count,
    ARRAY_AGG(DISTINCT pte.camera_name) FILTER (WHERE pte.camera_name IS NOT NULL) AS cameras_visited,
    MIN(pte.started_at) AS first_seen,
    MAX(pte.ended_at) AS last_seen
FROM employees e
LEFT JOIN advanced_person_identities i ON i.employee_id = e.id
LEFT JOIN person_timeline_events pte ON (pte.employee_id = e.id OR pte.identity_id = i.id)
WHERE e.tenant_id = '{tenant_id}'
GROUP BY e.id, e.first_name, e.last_name, e.employee_code, e.photo_path
ORDER BY e.first_name ASC, e.last_name ASC
LIMIT 50;"""
    },
    {
        "question": "Show details and dwell time for employee Kinjal (or where is Kinjal?)",
        "sql": """SELECT 
    e.id AS person_id,
    TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))) AS name,
    e.employee_code,
    e.photo_path,
    'employee' AS person_type,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    COUNT(DISTINCT pte.camera_name) AS camera_stops_count,
    ARRAY_AGG(DISTINCT pte.camera_name) FILTER (WHERE pte.camera_name IS NOT NULL) AS cameras_visited,
    MIN(pte.started_at) AS first_seen,
    MAX(pte.ended_at) AS last_seen
FROM employees e
LEFT JOIN advanced_person_identities i ON i.employee_id = e.id
LEFT JOIN person_timeline_events pte ON (pte.employee_id = e.id OR pte.identity_id = i.id)
WHERE e.tenant_id = '{tenant_id}'
  AND (
    e.first_name ILIKE '%Kinjal%' 
    OR e.last_name ILIKE '%Kinjal%' 
    OR CONCAT(e.first_name, ' ', e.last_name) ILIKE '%Kinjal%' 
    OR e.employee_code ILIKE '%Kinjal%'
  )
GROUP BY e.id, e.first_name, e.last_name, e.employee_code, e.photo_path
ORDER BY total_dwell_seconds DESC;"""
    },
    {
        "question": "Who visited yesterday?",
        "sql": """SELECT 
    i.id AS person_id,
    COALESCE(NULLIF(TRIM(i.visitor_name), ''), NULLIF(TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))), ''), CONCAT('Visitor #', SUBSTRING(i.id::text, 1, 6))) AS name,
    CASE WHEN i.is_employee THEN 'employee' ELSE 'visitor' END AS person_type,
    COALESCE(e.photo_path, pte.entry_crop_path) AS photo_path,
    COUNT(DISTINCT pte.camera_name) AS cameras_visited_count,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    MIN(pte.started_at) AS first_seen_at,
    MAX(pte.ended_at) AS last_seen_at
FROM advanced_person_identities i
LEFT JOIN employees e ON i.employee_id = e.id
JOIN person_timeline_events pte ON pte.identity_id = i.id
WHERE i.tenant_id = '{tenant_id}'
  AND pte.started_at >= CURRENT_DATE - INTERVAL '1 day'
  AND pte.started_at < CURRENT_DATE
GROUP BY i.id, i.visitor_name, e.first_name, e.last_name, e.photo_path, pte.entry_crop_path, i.is_employee
ORDER BY total_dwell_seconds DESC
LIMIT 50;"""
    },
    {
        "question": "Show all staff members and their registered photos and dwell times today",
        "sql": """SELECT 
    e.id AS person_id,
    TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))) AS name,
    e.employee_code,
    e.photo_path,
    'employee' AS person_type,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    COUNT(DISTINCT pte.camera_name) AS camera_stops_count,
    MIN(eal.employee_entry_timestamp) AS first_entry,
    MAX(eal.employee_exit_timestamp) AS last_exit
FROM employees e
LEFT JOIN advanced_person_identities i ON i.employee_id = e.id
LEFT JOIN person_timeline_events pte ON (pte.employee_id = e.id OR pte.identity_id = i.id)
LEFT JOIN advanced_employee_attendance_logs eal ON eal.employee_id = e.id
WHERE e.tenant_id = '{tenant_id}'
GROUP BY e.id, e.first_name, e.last_name, e.employee_code, e.photo_path
ORDER BY name ASC;"""
    },
    {
        "question": "Which area or camera had the highest foot traffic this week?",
        "sql": """SELECT 
    pte.camera_name,
    COUNT(DISTINCT pte.identity_id) AS unique_visitors_count,
    COUNT(pte.id) AS total_entry_events,
    ROUND(COALESCE(AVG(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS avg_dwell_seconds
FROM person_timeline_events pte
WHERE pte.tenant_id = '{tenant_id}'
  AND pte.started_at >= NOW() - INTERVAL '7 days'
GROUP BY pte.camera_name
ORDER BY unique_visitors_count DESC
LIMIT 10;"""
    },
    {
        "question": "Who stayed the longest on premises?",
        "sql": """SELECT 
    i.id AS person_id,
    COALESCE(NULLIF(TRIM(i.visitor_name), ''), NULLIF(TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))), ''), CONCAT('Visitor #', SUBSTRING(i.id::text, 1, 6))) AS name,
    CASE WHEN i.is_employee THEN 'employee' ELSE 'visitor' END AS person_type,
    COALESCE(e.photo_path, pte.entry_crop_path) AS photo_path,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    COUNT(DISTINCT pte.camera_name) AS camera_count,
    ARRAY_AGG(DISTINCT pte.camera_name) AS cameras_visited
FROM advanced_person_identities i
LEFT JOIN employees e ON i.employee_id = e.id
JOIN person_timeline_events pte ON pte.identity_id = i.id
WHERE i.tenant_id = '{tenant_id}'
GROUP BY i.id, i.visitor_name, e.first_name, e.last_name, e.photo_path, pte.entry_crop_path, i.is_employee
ORDER BY total_dwell_seconds DESC
LIMIT 10;"""
    },
    {
        "question": "Show the path and journey of person 1042",
        "sql": """SELECT 
    pte.camera_name,
    pte.zone_name,
    pte.started_at,
    pte.ended_at,
    ROUND(COALESCE(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at)), 0)) AS dwell_seconds,
    pte.entry_crop_path AS photo_path,
    pte.identity_source,
    pte.identity_confidence
FROM person_timeline_events pte
JOIN advanced_person_identities i ON pte.identity_id = i.id
WHERE pte.tenant_id = '{tenant_id}'
  AND (i.visitor_name ILIKE '%1042%' OR i.id::text ILIKE '%1042%')
ORDER BY pte.started_at ASC;"""
    },
    {
        "question": "Which persons were carrying backpacks, laptops, or suitcases?",
        "sql": """SELECT 
    i.id AS person_id,
    COALESCE(NULLIF(TRIM(i.visitor_name), ''), NULLIF(TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))), ''), CONCAT('Visitor #', SUBSTRING(i.id::text, 1, 6))) AS name,
    CASE WHEN i.is_employee THEN 'employee' ELSE 'visitor' END AS person_type,
    COALESCE(e.photo_path, pte.entry_crop_path) AS photo_path,
    pte.camera_name,
    pte.associated_objects,
    pte.started_at
FROM person_timeline_events pte
JOIN advanced_person_identities i ON pte.identity_id = i.id
LEFT JOIN employees e ON i.employee_id = e.id
WHERE pte.tenant_id = '{tenant_id}'
  AND pte.associated_objects IS NOT NULL
  AND jsonb_array_length(pte.associated_objects) > 0
ORDER BY pte.started_at DESC
LIMIT 25;"""
    }
]


def build_system_prompt(tenant_id: str) -> str:
    """Formats the system prompt with current schema, tenant_id, and examples."""
    examples_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        formatted_sql = ex["sql"].replace("{tenant_id}", str(tenant_id))
        examples_str += f"\nUser Question: {ex['question']}\nSQL Query:\n```sql\n{formatted_sql}\n```\n"

    return f"{SYSTEM_SCHEMA_PROMPT.replace('{tenant_id}', str(tenant_id))}\n\n### FEW-SHOT SURVEILLANCE EXAMPLES:\n{examples_str}"
