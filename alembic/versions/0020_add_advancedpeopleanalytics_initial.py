"""add_advancedpeopleanalytics_initial

Revision ID: 0020
Revises: 0019
Create Date: 2026-08-27 14:30:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '0020'
down_revision = '0019'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. camera_nodes
    op.create_table(
        'camera_nodes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('location_label', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 2. camera_zones
    op.create_table(
        'camera_zones',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('camera_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('camera_nodes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('zone_type', sa.String(20), nullable=False),
        sa.Column('label', sa.String(255), nullable=False),
        sa.Column('polygon', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 3. camera_zone_links
    op.create_table(
        'camera_zone_links',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('from_zone_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('camera_zones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('to_zone_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('camera_zones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('avg_transit_seconds', sa.Float(), server_default='30.0', nullable=False),
        sa.Column('max_transit_seconds', sa.Float(), server_default='300.0', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 4. advanced_uploaded_videos
    op.create_table(
        'advanced_uploaded_videos',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('original_name', sa.String(255), nullable=False),
        sa.Column('saved_path', sa.String(512), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 5. advanced_people_analytics_sessions
    op.create_table(
        'advanced_people_analytics_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('video_name', sa.String(255), nullable=False),
        sa.Column('video_path', sa.String(512), nullable=False),
        sa.Column('output_video_path', sa.String(512), nullable=True),
        sa.Column('status', sa.String(20), server_default='pending', nullable=False),
        sa.Column('session_type', sa.String(50), nullable=True),
        sa.Column('camera_node_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('camera_nodes.id', ondelete='SET NULL'), nullable=True),
        sa.Column('recording_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('line_start', postgresql.ARRAY(sa.Integer()), nullable=True),
        sa.Column('line_end', postgresql.ARRAY(sa.Integer()), nullable=True),
        sa.Column('similarity_threshold', sa.Float(), server_default='0.85', nullable=False),
        sa.Column('confidence_threshold', sa.Float(), server_default='0.3', nullable=False),
        sa.Column('track_employees', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('register_new_visitors', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('track_repeat_visitors', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('line_crossing_analysis', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('track_occupancy', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('unique_person_count', sa.Integer(), nullable=True),
        sa.Column('total_person_count', sa.Integer(), nullable=True),
        sa.Column('first_time_visitor_count', sa.Integer(), nullable=True),
        sa.Column('peak_occupancy', sa.Integer(), nullable=True),
        sa.Column('average_occupancy', sa.Float(), nullable=True),
        sa.Column('entry_count', sa.Integer(), nullable=True),
        sa.Column('exit_count', sa.Integer(), nullable=True),
        sa.Column('employee_count', sa.Integer(), nullable=True),
        sa.Column('visitor_count', sa.Integer(), nullable=True),
        sa.Column('occupancy_timeline', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 6. advanced_person_identities
    op.create_table(
        'advanced_person_identities',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('class_id', sa.Integer(), server_default='0', nullable=False),
        sa.Column('first_name', sa.String(100), nullable=True),
        sa.Column('last_name', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 7. advanced_person_embeddings
    op.create_table(
        'advanced_person_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('identity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_person_identities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('embedding', Vector(512), nullable=False),
        sa.Column('bbox', postgresql.ARRAY(sa.Integer()), nullable=True),
        sa.Column('timestamp', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 8. advanced_person_occurrences
    op.create_table(
        'advanced_person_occurrences',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_people_analytics_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('identity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_person_identities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('tracker_id', sa.Integer(), nullable=False),
        sa.Column('first_seen', sa.Float(), nullable=False),
        sa.Column('last_seen', sa.Float(), nullable=False),
        sa.Column('crop_path', sa.String(512), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 9. advanced_line_crossing_logs
    op.create_table(
        'advanced_line_crossing_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_people_analytics_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('identity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_person_identities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('tracker_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('direction', sa.String(10), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 10. advanced_employee_attendance_logs
    op.create_table(
        'advanced_employee_attendance_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_people_analytics_sessions.id', ondelete='CASCADE'), nullable=True),
        sa.Column('employee_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('employees.id', ondelete='CASCADE'), nullable=False),
        sa.Column('first_seen', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('last_seen', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('occurrence_count', sa.Integer(), server_default='1', nullable=False),
        sa.Column('employee_entry_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('employee_exit_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 11. advanced_visitor_attendance_logs
    op.create_table(
        'advanced_visitor_attendance_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_people_analytics_sessions.id', ondelete='CASCADE'), nullable=True),
        sa.Column('identity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_person_identities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('first_seen', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('last_seen', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('occurrence_count', sa.Integer(), server_default='1', nullable=False),
        sa.Column('visitor_entry_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('visitor_exit_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 12. advanced_employee_session_detections
    op.create_table(
        'advanced_employee_session_detections',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_people_analytics_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('employee_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('employees.id', ondelete='CASCADE'), nullable=False),
        sa.Column('first_seen', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('last_seen', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('occurrence_count', sa.Integer(), server_default='1', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 13. zone_crossing_events
    op.create_table(
        'zone_crossing_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_people_analytics_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('zone_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('camera_zones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('identity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_person_identities.id', ondelete='SET NULL'), nullable=True),
        sa.Column('tracker_id', sa.Integer(), nullable=False),
        sa.Column('event_type', sa.String(20), nullable=False),
        sa.Column('real_world_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('reid_embedding', Vector(512), nullable=True),
        sa.Column('face_confirmed', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('confidence', sa.Float(), server_default='1.0', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 14. cross_camera_identity_links
    op.create_table(
        'cross_camera_identity_links',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('from_session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_people_analytics_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('to_session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_people_analytics_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('from_identity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_person_identities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('to_identity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_person_identities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('reid_score', sa.Float(), nullable=False),
        sa.Column('time_score', sa.Float(), nullable=False),
        sa.Column('final_score', sa.Float(), nullable=False),
        sa.Column('method', sa.String(50), server_default='spatial_reid_fusion', nullable=False),
        sa.Column('is_confirmed', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

    # 15. person_timeline_events
    op.create_table(
        'person_timeline_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('person_type', sa.String(20), nullable=False),
        sa.Column('employee_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('employees.id', ondelete='SET NULL'), nullable=True),
        sa.Column('identity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_person_identities.id', ondelete='SET NULL'), nullable=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('advanced_people_analytics_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('camera_name', sa.String(255), nullable=False),
        sa.Column('zone_name', sa.String(255), nullable=True),
        sa.Column('event_type', sa.String(20), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('identity_source', sa.String(50), nullable=False),
        sa.Column('identity_confidence', sa.Float(), server_default='1.0', nullable=False),
        sa.Column('tracker_id', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )


def downgrade() -> None:
    op.drop_table('person_timeline_events')
    op.drop_table('cross_camera_identity_links')
    op.drop_table('zone_crossing_events')
    op.drop_table('advanced_employee_session_detections')
    op.drop_table('advanced_visitor_attendance_logs')
    op.drop_table('advanced_employee_attendance_logs')
    op.drop_table('advanced_line_crossing_logs')
    op.drop_table('advanced_person_occurrences')
    op.drop_table('advanced_person_embeddings')
    op.drop_table('advanced_person_identities')
    op.drop_table('advanced_people_analytics_sessions')
    op.drop_table('advanced_uploaded_videos')
    op.drop_table('camera_zone_links')
    op.drop_table('camera_zones')
    op.drop_table('camera_nodes')
