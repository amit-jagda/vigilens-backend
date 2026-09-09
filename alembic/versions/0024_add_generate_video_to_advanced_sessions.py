"""add_generate_video_to_advanced_sessions

Revision ID: 0024
Revises: 0023
Create Date: 2026-09-08 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0024'
down_revision = '0023'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'advanced_people_analytics_sessions',
        sa.Column('generate_video', sa.Boolean(), server_default='false', nullable=False)
    )


def downgrade() -> None:
    op.drop_column('advanced_people_analytics_sessions', 'generate_video')
