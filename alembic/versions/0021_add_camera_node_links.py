"""add_camera_node_links

Revision ID: 0021
Revises: 0020
Create Date: 2026-08-31 14:30:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0021'
down_revision = '0020'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'camera_node_links',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('from_camera_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('camera_nodes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('to_camera_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('camera_nodes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('min_transit_seconds', sa.Float(), server_default='5.0', nullable=False),
        sa.Column('avg_transit_seconds', sa.Float(), server_default='30.0', nullable=False),
        sa.Column('max_transit_seconds', sa.Float(), server_default='300.0', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_delete', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('update_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )


def downgrade() -> None:
    op.drop_table('camera_node_links')
