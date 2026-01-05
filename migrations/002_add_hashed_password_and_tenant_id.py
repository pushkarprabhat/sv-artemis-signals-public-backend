"""
Alembic migration: Add hashed_password and tenant_id to users table
"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('users', sa.Column('hashed_password', sa.String(128), nullable=False, server_default=''))
    op.add_column('users', sa.Column('tenant_id', sa.String(64), nullable=False, server_default='default_tenant'))

def downgrade():
    op.drop_column('users', 'hashed_password')
    op.drop_column('users', 'tenant_id')
