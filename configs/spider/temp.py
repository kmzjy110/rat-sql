({'raw_question': 'Which professionals have operated a treatment that costs less than the average? Give me theor first names and last names.', 'questi
on': ['which', 'professional', 'have', 'operate', 'a', 'treatment', 'that', 'cost', 'less', 'than', 'the', 'average', '?', 'give', 'i', 'theor', 'first', 'name', 'a
nd', 'last', 'name', '.'], 'question_for_copying': ['which', 'professionals', 'have', 'operated', 'a', 'treatment', 'that', 'costs', 'less', 'than', 'the', 'average
', '?', 'give', 'me', 'theor', 'first', 'names', 'and', 'last', 'names', '.'], 'db_id': 'dog_kennels', 'sc_link': {'q_col_match': {'16,11': 'CEM', '17,11': 'CEM', '
16,35': 'CEM', '17,35': 'CEM', '19,12': 'CEM', '20,12': 'CEM', '19,40': 'CEM', '20,40': 'CEM', '1,33': 'CPM', '1,46': 'CPM', '5,8': 'CPM', '5,9': 'CPM', '5,44': 'CP
M', '5,47': 'CPM', '7,49': 'CPM', '17,25': 'CEM', '20,25': 'CEM'}, 'q_tab_match': {'1,6': 'TEM', '5,7': 'TEM', '5,3': 'TPM'}}, 'cv_link': {'num_date_match': {}, 'ce
ll_match': {}}, 'columns': [['<type: text>', '*'], ['<type: text>', 'breed', 'code'], ['<type: text>', 'breed', 'name'], ['<type: number>', 'charge', 'id'], ['<type
: text>', 'charge', 'type'], ['<type: number>', 'charge', 'amount'], ['<type: text>', 'size', 'code'], ['<type: text>', 'size', 'description'], ['<type: text>', 'tr
eatment', 'type', 'code'], ['<type: text>', 'treatment', 'type', 'description'], ['<type: number>', 'owner', 'id'], ['<type: text>', 'first', 'name'], ['<type: text
>', 'last', 'name'], ['<type: text>', 'street'], ['<type: text>', 'city'], ['<type: text>', 'state'], ['<type: text>', 'zip', 'code'], ['<type: text>', 'email', 'ad
dress'], ['<type: text>', 'home', 'phone'], ['<type: text>', 'cell', 'number'], ['<type: number>', 'dog', 'id'], ['<type: number>', 'owner', 'id'], ['<type: text>',
 'abandon', 'yes', 'or', 'no'], ['<type: text>', 'breed', 'code'], ['<type: text>', 'size', 'code'], ['<type: text>', 'name'], ['<type: text>', 'age'], ['<type: tim
e>', 'date', 'of', 'birth'], ['<type: text>', 'gender'], ['<type: text>', 'weight'], ['<type: time>', 'date', 'arrive'], ['<type: time>', 'date', 'adopt'], ['<type:
 time>', 'date', 'depart'], ['<type: number>', 'professional', 'id'], ['<type: text>', 'role', 'code'], ['<type: text>', 'first', 'name'], ['<type: text>', 'street'
], ['<type: text>', 'city'], ['<type: text>', 'state'], ['<type: text>', 'zip', 'code'], ['<type: text>', 'last', 'name'], ['<type: text>', 'email', 'address'], ['<
type: text>', 'home', 'phone'], ['<type: text>', 'cell', 'number'], ['<type: number>', 'treatment', 'id'], ['<type: number>', 'dog', 'id'], ['<type: number>', 'prof
essional', 'id'], ['<type: text>', 'treatment', 'type', 'code'], ['<type: time>', 'date', 'of', 'treatment'], ['<type: number>', 'cost', 'of', 'treatment']], 'table
s': [['breed'], ['charge'], ['size'], ['treatment', 'type'], ['owner'], ['dog'], ['professional'], ['treatment']], 'table_bounds': [1, 3, 6, 8, 10, 20, 33, 44, 50],
 'column_to_table': {'0': None, '1': 0, '2': 0, '3': 1, '4': 1, '5': 1, '6': 2, '7': 2, '8': 3, '9': 3, '10': 4, '11': 4, '12': 4, '13': 4, '14': 4, '15': 4, '16':
4, '17': 4, '18': 4, '19': 4, '20': 5, '21': 5, '22': 5, '23': 5, '24': 5, '25': 5, '26': 5, '27': 5, '28': 5, '29': 5, '30': 5, '31': 5, '32': 5, '33': 6, '34': 6,
 '35': 6, '36': 6, '37': 6, '38': 6, '39': 6, '40': 6, '41': 6, '42': 6, '43': 6, '44': 7, '45': 7, '46': 7, '47': 7, '48': 7, '49': 7}, 'table_to_columns': {'0': [
1, 2], '1': [3, 4, 5], '2': [6, 7], '3': [8, 9], '4': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], '5': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], '6': [33,
 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], '7': [44, 45, 46, 47, 48, 49]}, 'foreign_keys': {'21': 10, '23': 1, '24': 6, '45': 20, '46': 33, '47': 8}, 'foreign_keys_t
ables': {'5': [0, 2, 4], '7': [3, 5, 6]}, 'primary_keys': [1, 3, 6, 8, 10, 20, 33, 44]}, NL2CodeDecoderPreprocItem(tree={'_type': 'sql', 'select': {'_type': 'select
', 'is_distinct': True, 'aggs': [{'_type': 'agg', 'agg_id': {'_type': 'NoneAggOp'}, 'val_unit': {'_type': 'Column', 'col_unit1': {'_type': 'col_unit', 'agg_id': {'_
type': 'NoneAggOp'}, 'is_distinct': False, 'col_id': 35}}}, {'_type': 'agg', 'agg_id': {'_type': 'NoneAggOp'}, 'val_unit': {'_type': 'Column', 'col_unit1': {'_type'
: 'col_unit', 'agg_id': {'_type': 'NoneAggOp'}, 'is_distinct': False, 'col_id': 40}}}]}, 'from': {'_type': 'from', 'table_units': [{'_type': 'Table', 'table_id': 6}
, {'_type': 'Table', 'table_id': 7}]}, 'sql_where': {'_type': 'sql_where', 'where': {'_type': 'Lt', 'val_unit': {'_type': 'Column', 'col_unit1': {'_type': 'col_unit
', 'agg_id': {'_type': 'NoneAggOp'}, 'is_distinct': False, 'col_id': 49}}, 'val1': {'_type': 'ValSql', 's': {'_type': 'sql', 'select': {'_type': 'select', 'is_disti
nct': False, 'aggs': [{'_type': 'agg', 'agg_id': {'_type': 'Avg'}, 'val_unit': {'_type': 'Column', 'col_unit1': {'_type': 'col_unit', 'agg_id': {'_type': 'NoneAggOp
'}, 'is_distinct': False, 'col_id': 49}}}]}, 'from': {'_type': 'from', 'table_units': [{'_type': 'Table', 'table_id': 7}]}, 'sql_where': {'_type': 'sql_where'}, 'sq
l_groupby': {'_type': 'sql_groupby'}, 'sql_orderby': {'_type': 'sql_orderby', 'limit': False}, 'sql_ieu': {'_type': 'sql_ieu'}}}}}, 'sql_groupby': {'_type': 'sql_gr
oupby'}, 'sql_orderby': {'_type': 'sql_orderby', 'limit': False}, 'sql_ieu': {'_type': 'sql_ieu'}}, orig_code={'from': {'table_units': [['table_unit', 6], ['table_u
nit', 7]], 'conds': []}, 'select': [True, [[0, [0, [0, 35, False], None]], [0, [0, [0, 40, False], None]]]], 'where': [[False, 4, [0, [0, 49, False], None], {'from'
: {'table_units': [['table_unit', 7]], 'conds': []}, 'select': [False, [[5, [0, [0, 49, False], None]]]], 'where': [], 'groupBy': [], 'having': [], 'orderBy': [], '
limit': None, 'intersect': None, 'union': None, 'except': None}, None]], 'groupBy': [], 'having': [], 'orderBy': [], 'limit': None, 'intersect': None, 'union': None
, 'except': None}))