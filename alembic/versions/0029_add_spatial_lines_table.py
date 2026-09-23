"""add_spatial_lines_table

Revision ID: 0029
Revises: 0028
Create Date: 2026-09-21 07:57:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0029'
down_revision = '0028'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'spatial_lines',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('floor_plan_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('floor_plans.id', ondelete='CASCADE'), nullable=False),
        sa.Column('x1', sa.Float(), nullable=False),
        sa.Column('y1', sa.Float(), nullable=False),
        sa.Column('x2', sa.Float(), nullable=False),
        sa.Column('y2', sa.Float(), nullable=False),
        sa.Column('line_type', sa.String(50), server_default='wall', nullable=False),
        sa.Column('label', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    op.create_index('ix_spatial_lines_tenant_id', 'spatial_lines', ['tenant_id'])
    op.create_index('ix_spatial_lines_floor_plan_id', 'spatial_lines', ['floor_plan_id'])


def downgrade() -> None:
    op.drop_index('ix_spatial_lines_floor_plan_id', table_name='spatial_lines')
    op.drop_index('ix_spatial_lines_tenant_id', table_name='spatial_lines')
    op.drop_table('spatial_lines')
