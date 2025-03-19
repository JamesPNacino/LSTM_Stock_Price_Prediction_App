"""created table

Revision ID: 3d0b12018ffa
Revises: 
Create Date: 2025-03-19 12:20:39.159721

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3d0b12018ffa'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('neural_network',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ticker', sa.Text(), nullable=True),
    sa.Column('pattern', sa.Text(), nullable=True),
    sa.Column('days_out_parameter', sa.Integer(), nullable=True),
    sa.Column('percent_increase_parameter', sa.Integer(), nullable=True),
    sa.Column('total_observations_train_and_val', sa.Integer(), nullable=True),
    sa.Column('accuracy_test', sa.Integer(), nullable=True),
    sa.Column('most_frequent_class_test_pct', sa.Integer(), nullable=True),
    sa.Column('data_start_date', sa.Text(), nullable=True),
    sa.Column('data_end_date', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('neural_network')
    # ### end Alembic commands ###
