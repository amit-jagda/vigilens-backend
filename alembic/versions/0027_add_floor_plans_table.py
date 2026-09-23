"""add_floor_plans_table

Revision ID: 0027
Revises: 0026
Create Date: 2026-09-21 07:55:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0027'
down_revision = '0026'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'floor_plans',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('image_filepath', sa.String(512), nullable=True),
        sa.Column('canvas_width_px', sa.Integer(), server_default='1920', nullable=False),
        sa.Column('canvas_height_px', sa.Integer(), server_default='1080', nullable=False),
        sa.Column('scale_meters_per_px', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    op.create_index('ix_floor_plans_tenant_id', 'floor_plans', ['tenant_id'])


def downgrade() -> None:
    op.drop_index('ix_floor_plans_tenant_id', table_name='floor_plans')
    op.drop_table('floor_plans')
