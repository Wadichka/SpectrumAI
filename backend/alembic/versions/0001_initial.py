"""initial

Создаёт все таблицы основной БД по ER-диаграмме главы 5 (§5.9.2)
и заполняет справочник ``functional_group`` 25 целевыми группами из
таблицы 5.3 главы 5 (§5.4.1, §5.9.2 — «заполняется при инициализации базы»).

Revision ID: 0001
Revises:
Create Date: 2026-05-14
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# 25 функциональных групп из таблицы 5.3 главы 5 (§5.4.1).
# Поле name — английский snake_case из листинга 5.1 главы; description — русский
# текст из таблицы; characteristic_bands — диапазоны характеристических полос (см⁻¹).
FUNCTIONAL_GROUPS_SEED: list[dict[str, str]] = [
    {"code": "FG01", "name": "alcohol_OH", "smarts_pattern": "[OX2H][CX4]",
     "characteristic_bands": "3200-3600, 1050",
     "description": "Алифатический спирт, валентные и деформационные колебания O–H"},
    {"code": "FG02", "name": "phenol_OH", "smarts_pattern": "[OX2H][c]",
     "characteristic_bands": "3200-3550, 1200",
     "description": "Гидроксил, связанный с ароматическим кольцом"},
    {"code": "FG03", "name": "carbonyl", "smarts_pattern": "[CX3]=[OX1]",
     "characteristic_bands": "1650-1820",
     "description": "Общая полоса валентного колебания C=O"},
    {"code": "FG04", "name": "aldehyde", "smarts_pattern": "[CX3H1](=O)[#6]",
     "characteristic_bands": "2720, 2820, 1720",
     "description": "Дублет валентных C–H альдегида"},
    {"code": "FG05", "name": "ketone", "smarts_pattern": "[#6][CX3](=O)[#6]",
     "characteristic_bands": "1705-1720",
     "description": "Кетонная карбонильная группа"},
    {"code": "FG06", "name": "carboxylic_acid", "smarts_pattern": "[CX3](=O)[OX2H]",
     "characteristic_bands": "2500-3300 (шир.), 1700",
     "description": "Димерная O–H и C=O"},
    {"code": "FG07", "name": "ester", "smarts_pattern": "[#6][CX3](=O)[OX2][#6]",
     "characteristic_bands": "1735-1750, 1200",
     "description": "C=O и асимметричные C–O"},
    {"code": "FG08", "name": "amide_primary", "smarts_pattern": "[CX3](=O)[NX3H2]",
     "characteristic_bands": "3350, 3180, 1650",
     "description": "Две N–H, амид I"},
    {"code": "FG09", "name": "amide_secondary", "smarts_pattern": "[CX3](=O)[NX3H1]",
     "characteristic_bands": "3300, 1650, 1550",
     "description": "Амид I и амид II"},
    {"code": "FG10", "name": "amide_tertiary", "smarts_pattern": "[CX3](=O)[NX3H0]",
     "characteristic_bands": "1640-1680",
     "description": "Только амид I"},
    {"code": "FG11", "name": "amine_primary", "smarts_pattern": "[NX3;H2;!$(NC=O)]",
     "characteristic_bands": "3300-3500 (дублет), 1600",
     "description": "Две N–H"},
    {"code": "FG12", "name": "amine_secondary", "smarts_pattern": "[NX3;H1;!$(NC=O)]",
     "characteristic_bands": "3300-3400",
     "description": "Одна N–H"},
    {"code": "FG13", "name": "amine_tertiary", "smarts_pattern": "[NX3;H0;!$(NC=O)]",
     "characteristic_bands": "1020-1250",
     "description": "Только C–N"},
    {"code": "FG14", "name": "nitrile", "smarts_pattern": "[CX2]#[NX1]",
     "characteristic_bands": "2200-2260",
     "description": "Валентное C≡N"},
    {"code": "FG15", "name": "nitro",
     "smarts_pattern": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
     "characteristic_bands": "1500-1570, 1300-1380",
     "description": "Асимметричное и симметричное N=O"},
    {"code": "FG16", "name": "ether", "smarts_pattern": "[OD2]([#6])[#6]",
     "characteristic_bands": "1050-1150",
     "description": "Валентное C–O–C"},
    {"code": "FG17", "name": "alkene", "smarts_pattern": "[CX3]=[CX3]",
     "characteristic_bands": "1620-1680, 3000-3100",
     "description": "C=C и винильные C–H"},
    {"code": "FG18", "name": "alkyne", "smarts_pattern": "[CX2]#[CX2]",
     "characteristic_bands": "2100-2260, 3300",
     "description": "C≡C и терминальный ≡C–H"},
    {"code": "FG19", "name": "aromatic_ring", "smarts_pattern": "c1ccccc1",
     "characteristic_bands": "1450-1600, 3000-3100",
     "description": "Скелетные колебания кольца"},
    {"code": "FG20", "name": "ch2_group", "smarts_pattern": "[CH2]",
     "characteristic_bands": "2850, 2925, 1465",
     "description": "Метиленовые валентные C–H"},
    {"code": "FG21", "name": "ch3_group", "smarts_pattern": "[CH3]",
     "characteristic_bands": "2870, 2960, 1375",
     "description": "Метильные валентные и деформационные C–H"},
    {"code": "FG22", "name": "c_f_bond", "smarts_pattern": "[CX4]F",
     "characteristic_bands": "1000-1400",
     "description": "Сильная характеристическая полоса"},
    {"code": "FG23", "name": "c_cl_bond", "smarts_pattern": "[CX4]Cl",
     "characteristic_bands": "600-800",
     "description": "Тяжёлый галоген, низкочастотная область"},
    {"code": "FG24", "name": "sulfoxide_sulfone",
     "smarts_pattern": "[#6][SX3](=O)[#6],[#6][SX4](=O)(=O)[#6]",
     "characteristic_bands": "1020-1060, 1300-1370",
     "description": "S=O связи"},
    {"code": "FG25", "name": "thiol_thioether", "smarts_pattern": "[SX2H],[SX2]([#6])[#6]",
     "characteristic_bands": "2550-2600, 600-700",
     "description": "S–H и C–S связи"},
]


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('cache_entry',
    sa.Column('key', sa.String(length=64), nullable=False),
    sa.Column('result_json', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('key')
    )
    with op.batch_alter_table('cache_entry', schema=None) as batch_op:
        batch_op.create_index('ix_cache_entry_expires_at', ['expires_at'], unique=False)

    op.create_table('compound',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=255), nullable=True),
    sa.Column('iupac_name', sa.String(length=512), nullable=True),
    sa.Column('cas_number', sa.String(length=15), nullable=True),
    sa.Column('smiles_canonical', sa.Text(), nullable=False),
    sa.Column('inchi', sa.Text(), nullable=False),
    sa.Column('inchi_key', sa.String(length=27), nullable=False),
    sa.Column('molecular_formula', sa.String(length=64), nullable=True),
    sa.Column('molecular_weight', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('inchi_key')
    )
    with op.batch_alter_table('compound', schema=None) as batch_op:
        batch_op.create_index('ix_compound_cas', ['cas_number'], unique=False)
        batch_op.create_index('ix_compound_inchi_key', ['inchi_key'], unique=True)
        batch_op.create_index('ix_compound_name', ['name'], unique=False)

    op.create_table('functional_group',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('code', sa.String(length=8), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('smarts_pattern', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('characteristic_bands', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('code'),
    sa.UniqueConstraint('name')
    )
    op.create_table('model_version',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model_type', sa.String(length=32), nullable=False),
    sa.Column('version', sa.String(length=32), nullable=False),
    sa.Column('file_path', sa.Text(), nullable=False),
    sa.Column('metrics_json', sa.Text(), nullable=True),
    sa.Column('trained_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('model_type', 'version', name='uq_model_type_version')
    )
    op.create_table('user',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('username', sa.String(length=64), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('password_hash', sa.String(length=60), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('username')
    )
    op.create_table('compound_functional_group',
    sa.Column('compound_id', sa.Integer(), nullable=False),
    sa.Column('functional_group_id', sa.Integer(), nullable=False),
    sa.Column('count', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['compound_id'], ['compound.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['functional_group_id'], ['functional_group.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('compound_id', 'functional_group_id')
    )
    op.create_table('identification_request',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('input_spectrum_path', sa.Text(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('processing_time_ms', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('identification_request', schema=None) as batch_op:
        batch_op.create_index('ix_request_timestamp', ['timestamp'], unique=False)
        batch_op.create_index('ix_request_user', ['user_id'], unique=False)

    op.create_table('spectrum',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('compound_id', sa.Integer(), nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('phase', sa.String(length=16), nullable=True),
    sa.Column('technique', sa.String(length=32), nullable=True),
    sa.Column('wavenumber_min', sa.Float(), nullable=False),
    sa.Column('wavenumber_max', sa.Float(), nullable=False),
    sa.Column('n_points', sa.Integer(), nullable=False),
    sa.Column('file_path', sa.Text(), nullable=False),
    sa.Column('quality_score', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['compound_id'], ['compound.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('spectrum', schema=None) as batch_op:
        batch_op.create_index('ix_spectrum_compound', ['compound_id'], unique=False)
        batch_op.create_index('ix_spectrum_source', ['source'], unique=False)

    op.create_table('identification_result',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('request_id', sa.Integer(), nullable=False),
    sa.Column('compound_id', sa.Integer(), nullable=False),
    sa.Column('rank', sa.Integer(), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.Column('method', sa.String(length=32), nullable=False),
    sa.Column('compound_name_cached', sa.String(length=255), nullable=True),
    sa.ForeignKeyConstraint(['compound_id'], ['compound.id'], ),
    sa.ForeignKeyConstraint(['request_id'], ['identification_request.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('identification_result', schema=None) as batch_op:
        batch_op.create_index('ix_result_request', ['request_id'], unique=False)

    op.create_table('predicted_functional_group',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('request_id', sa.Integer(), nullable=False),
    sa.Column('functional_group_id', sa.Integer(), nullable=False),
    sa.Column('probability', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['functional_group_id'], ['functional_group.id'], ),
    sa.ForeignKeyConstraint(['request_id'], ['identification_request.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('spectrum_embedding',
    sa.Column('spectrum_id', sa.Integer(), nullable=False),
    sa.Column('embedding_vector', sa.LargeBinary(), nullable=False),
    sa.Column('model_version', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['spectrum_id'], ['spectrum.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('spectrum_id')
    )
    # ### end Alembic commands ###

    # Seed справочника функциональных групп (§5.4.1, табл. 5.3).
    functional_group_table = sa.table(
        "functional_group",
        sa.column("code", sa.String),
        sa.column("name", sa.String),
        sa.column("smarts_pattern", sa.Text),
        sa.column("description", sa.Text),
        sa.column("characteristic_bands", sa.Text),
    )
    op.bulk_insert(functional_group_table, FUNCTIONAL_GROUPS_SEED)


def downgrade() -> None:
    """Downgrade schema."""
    # Удаление таблиц автоматически снимает seed-данные functional_group.
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('spectrum_embedding')
    op.drop_table('predicted_functional_group')
    with op.batch_alter_table('identification_result', schema=None) as batch_op:
        batch_op.drop_index('ix_result_request')

    op.drop_table('identification_result')
    with op.batch_alter_table('spectrum', schema=None) as batch_op:
        batch_op.drop_index('ix_spectrum_source')
        batch_op.drop_index('ix_spectrum_compound')

    op.drop_table('spectrum')
    with op.batch_alter_table('identification_request', schema=None) as batch_op:
        batch_op.drop_index('ix_request_user')
        batch_op.drop_index('ix_request_timestamp')

    op.drop_table('identification_request')
    op.drop_table('compound_functional_group')
    op.drop_table('user')
    op.drop_table('model_version')
    op.drop_table('functional_group')
    with op.batch_alter_table('compound', schema=None) as batch_op:
        batch_op.drop_index('ix_compound_name')
        batch_op.drop_index('ix_compound_inchi_key')
        batch_op.drop_index('ix_compound_cas')

    op.drop_table('compound')
    with op.batch_alter_table('cache_entry', schema=None) as batch_op:
        batch_op.drop_index('ix_cache_entry_expires_at')

    op.drop_table('cache_entry')
    # ### end Alembic commands ###
