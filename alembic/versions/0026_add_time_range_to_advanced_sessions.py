"""add_time_range_to_advanced_sessions

Revision ID: 0026
Revises: 0025
Create Date: 2026-09-11 11:55:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0026'
down_revision = '0025'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'advanced_people_analytics_sessions',
        sa.Column('start_time_sec', sa.Float(), nullable=True)
    )
    op.add_column(
        'advanced_people_analytics_sessions',
        sa.Column('end_time_sec', sa.Float(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('advanced_people_analytics_sessions', 'end_time_sec')
    op.drop_column('advanced_people_analytics_sessions', 'start_time_sec')
