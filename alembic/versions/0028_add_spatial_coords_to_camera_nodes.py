"""add_spatial_coords_to_camera_nodes

Revision ID: 0028
Revises: 0027
Create Date: 2026-09-21 07:56:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0028'
down_revision = '0027'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'camera_nodes',
        sa.Column('floor_plan_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('floor_plans.id', ondelete='SET NULL'), nullable=True)
    )
    op.add_column(
        'camera_nodes',
        sa.Column('x_coord', sa.Float(), nullable=True)
    )
    op.add_column(
        'camera_nodes',
        sa.Column('y_coord', sa.Float(), nullable=True)
    )
    op.add_column(
        'camera_nodes',
        sa.Column('fov_angle', sa.Float(), server_default='0.0', nullable=True)
    )
    op.create_index('ix_camera_nodes_floor_plan_id', 'camera_nodes', ['floor_plan_id'])


def downgrade() -> None:
    op.drop_index('ix_camera_nodes_floor_plan_id', table_name='camera_nodes')
    op.drop_column('camera_nodes', 'fov_angle')
    op.drop_column('camera_nodes', 'y_coord')
    op.drop_column('camera_nodes', 'x_coord')
    op.drop_column('camera_nodes', 'floor_plan_id')
