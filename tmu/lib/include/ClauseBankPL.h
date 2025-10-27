

void cbpl_type_i_feedback(
    unsigned int *ta_state,
    unsigned int *feedback_to_ta,
    int number_of_clauses,
    int number_of_literals,
    int number_of_state_bits,
    float update_p,
    float s,
    unsigned int boost_true_positive_feedback,
    unsigned int reuse_random_feedback,
    unsigned int max_included_literals,
    unsigned int *clause_active,
    unsigned int *clause_patches,
    unsigned int *clause_outputs
);

void cbpl_type_ii_feedback(
    unsigned int *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_state_bits,
    int number_of_patches,
    float update_p,
    unsigned int *clause_active,
    unsigned int *clause_patches,
    unsigned int *clause_outputs
);

void cbpl_calculate_clause_outputs_predict(
    unsigned int *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_state_bits,
    int number_of_patches,
    unsigned int *clause_output,
    unsigned int *Xi
);

void cbpl_calculate_clause_outputs_update(
    unsigned int *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_state_bits,
    int number_of_patches,
    unsigned int *clause_output,
    unsigned int *Xi
);

void cbpl_included_literals(
    unsigned int *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_state_bits,
    unsigned int *actions
);

void cbpl_calculate_literal_frequency(
    unsigned int *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_state_bits,
    unsigned int *clause_active,
    unsigned int *literal_count
);

int cbpl_number_of_include_actions(
    unsigned int *ta_state,
    int clause,
    int number_of_literals,
    int number_of_state_bits
);

void cbpl_initialize_incremental_clause_calculation(
    unsigned int *ta_state,
    unsigned int *literal_clause_map,
    unsigned int *literal_clause_map_pos,
    unsigned int *false_literals_per_clause,
    int number_of_clauses,
    int number_of_literals,
    int number_of_state_bits,
    unsigned int *previous_Xi
);

void cbpl_calculate_clause_outputs_incremental_batch(
    unsigned int * literal_clause_map,
    unsigned int *literal_clause_map_pos,
    unsigned int *false_literals_per_clause,
    int number_of_clauses,
    int number_of_literals,
    int number_of_patches,
    unsigned int *clause_output,
    unsigned int *previous_Xi,
    unsigned int *Xi,
    int batch_size
);

void cbpl_calculate_clause_outputs_incremental(
    unsigned int * literal_clause_map,
    unsigned int *literal_clause_map_pos,
    unsigned int *false_literals_per_clause,
    int number_of_clauses,
    int number_of_literals,
    unsigned int *previous_Xi,
    unsigned int *Xi
);

void cbpl_get_literals(
    const unsigned int *ta_state,
    unsigned int number_of_clauses,
    unsigned int number_of_literals,
    unsigned int number_of_state_bits,
    unsigned int *result
);

void cbpl_get_model(
	const unsigned int *ta_state,
	unsigned int number_of_clauses,
	unsigned int number_of_literals,
	unsigned int number_of_state_bits,
	unsigned int *model
);

void cbpl_transform_example(
	unsigned int *X,
	unsigned int *encoded_X,
	int dim_y,
	int dim_x
);

void cbpl_pack_weights(
	int *weights,
	unsigned int *packed_weights,
	int num_weights,
	int bits_per_weight
);