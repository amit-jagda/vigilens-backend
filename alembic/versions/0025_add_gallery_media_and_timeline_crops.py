"""add_gallery_media_and_timeline_crops

Revision ID: 0025
Revises: 0024
Create Date: 2026-09-10 15:30:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0025'
down_revision = '0024'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'advanced_people_analytics_sessions',
        sa.Column('gallery_media_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('gallery_media.id', ondelete='SET NULL'), nullable=True)
    )
    op.add_column(
        'person_timeline_events',
        sa.Column('entry_crop_path', sa.String(512), nullable=True)
    )
    op.add_column(
        'person_timeline_events',
        sa.Column('exit_crop_path', sa.String(512), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('person_timeline_events', 'exit_crop_path')
    op.drop_column('person_timeline_events', 'entry_crop_path')
    op.drop_column('advanced_people_analytics_sessions', 'gallery_media_id')
