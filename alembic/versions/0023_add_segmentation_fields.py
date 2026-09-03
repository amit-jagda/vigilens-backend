"""add_segmentation_fields

Revision ID: 0023
Revises: 0022
Create Date: 2026-09-01 02:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0023'
down_revision = '0022'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add segmentation and metadata columns to advanced_person_embeddings
    op.add_column(
        'advanced_person_embeddings',
        sa.Column('embedding_type', sa.String(20), server_default='appearance', nullable=False)
    )
    op.add_column(
        'advanced_person_embeddings',
        sa.Column('is_segmented', sa.Boolean(), server_default='false', nullable=False)
    )
    op.add_column(
        'advanced_person_embeddings',
        sa.Column('mask_coverage', sa.Float(), nullable=True)
    )
    op.add_column(
        'advanced_person_embeddings',
        sa.Column('recorded_date', sa.Date(), nullable=True)
    )
    op.add_column(
        'advanced_person_embeddings',
        sa.Column('face_anchored', sa.Boolean(), server_default='false', nullable=False)
    )

    # Add mask_coverage column to zone_crossing_events
    op.add_column(
        'zone_crossing_events',
        sa.Column('mask_coverage', sa.Float(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('zone_crossing_events', 'mask_coverage')
    op.drop_column('advanced_person_embeddings', 'face_anchored')
    op.drop_column('advanced_person_embeddings', 'recorded_date')
    op.drop_column('advanced_person_embeddings', 'mask_coverage')
    op.drop_column('advanced_person_embeddings', 'is_segmented')
    op.drop_column('advanced_person_embeddings', 'embedding_type')
