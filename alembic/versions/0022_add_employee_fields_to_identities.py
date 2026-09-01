"""add_employee_fields_to_identities

Revision ID: 0022
Revises: 0021
Create Date: 2026-09-01 01:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0022'
down_revision = '0021'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_employee, employee_id, visitor_name to advanced_person_identities if not already present
    op.add_column(
        'advanced_person_identities',
        sa.Column('is_employee', sa.Boolean(), server_default='false', nullable=False)
    )
    op.add_column(
        'advanced_person_identities',
        sa.Column('employee_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('employees.id', ondelete='SET NULL'), nullable=True)
    )
    op.add_column(
        'advanced_person_identities',
        sa.Column('visitor_name', sa.String(200), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('advanced_person_identities', 'visitor_name')
    op.drop_column('advanced_person_identities', 'employee_id')
    op.drop_column('advanced_person_identities', 'is_employee')
