

#ifdef _MSC_VER
#  include <intrin.h>
#  define __builtin_popcount __popcnt
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fast_rand.h"

#include "ClauseBankPL.h"

static inline void cbpl_initialize_random_streams(unsigned int *feedback_to_ta, int number_of_literals, int number_of_ta_chunks, float s)
{
	// Initialize all bits to zero	
	memset(feedback_to_ta, 0, number_of_ta_chunks*sizeof(unsigned int));

	int n = number_of_literals;
	float p = 1.0 / s;

	int active = normal(n * p, n * p * (1 - p));
	active = active >= n ? n : active;
	active = active < 0 ? 0 : active;
	while (active--) {
		int f = fast_rand() % (number_of_literals);
		while (feedback_to_ta[f / 32] & (1 << (f % 32))) {
			f = fast_rand() % (number_of_literals);
	    }
		feedback_to_ta[f / 32] |= 1 << (f % 32);
	}
}

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void cbpl_inc(unsigned int *ta_state, unsigned int active, int number_of_state_bits)
{
	unsigned int carry, carry_next;

	carry = active;
	for (int b = 0; b < number_of_state_bits; ++b) {

		carry_next = ta_state[b] & carry; // Sets carry bits (overflow) passing on to next bit
		ta_state[b] = ta_state[b] ^ carry; // Performs increments with XOR
		carry = carry_next;
	}

	if (carry > 0) {
		for (int b = 0; b < number_of_state_bits; ++b) {
			ta_state[b] |= carry;
		}
	} 	
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void cbpl_dec(
        unsigned int *ta_state,
        unsigned int active,
        int number_of_state_bits
){
	unsigned int carry, carry_next;
    unsigned int ta_val;

	carry = active;
	for (int b = 0; b < number_of_state_bits; ++b) {

        ta_val = ta_state[b];
        carry_next = (~ta_val) & carry; // Sets carry bits (overflow) passing on to next bit
        ta_state[b] = ta_val ^ carry; // Performs increments with XOR
        carry = carry_next;

	}

	if (carry > 0) {
		for (int b = 0; b < number_of_state_bits; ++b) {
			ta_state[b] &= ~carry;
		}
	}
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
// Note: No longer needed
static inline void cbpl_calculate_clause_output_feedback(unsigned int *ta_state, unsigned int *output_one_patches, unsigned int *clause_output, unsigned int *clause_patch, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *Xi)
{
	int output_one_patches_count = 0;
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & (Xi[patch*number_of_ta_chunks + k])) == ta_state[pos];

			if (!output) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & (Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1]) & filter) ==
			(ta_state[pos] & filter);

		if (output) {
			output_one_patches[output_one_patches_count] = patch;
			output_one_patches_count++;
		}
	}

	if (output_one_patches_count > 0) {
		*clause_output = 1;

		int patch_id = fast_rand() % output_one_patches_count;
 		*clause_patch = output_one_patches[patch_id];
	} else {
		*clause_output = 0;
	}
}


static inline unsigned int cbpl_calculate_clause_output_update(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *Xi)
{
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & (Xi[patch*number_of_ta_chunks + k] )) == ta_state[pos];

			if (!output) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & (Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1]) & filter) ==
			(ta_state[pos] & filter);

		if (output) {
			return(1);
		}
	}

	return(0);
}

static inline void cbpl_calculate_clause_output_patchwise(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *output, unsigned int *Xi)
{
	for (int patch = 0; patch < number_of_patches; ++patch) {
		output[patch] = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output[patch] = output[patch] && (ta_state[pos] & Xi[patch*number_of_ta_chunks + k]) == ta_state[pos];

			if (!output[patch]) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output[patch] = output[patch] &&
			(ta_state[pos] & Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] & filter) ==
			(ta_state[pos] & filter);
	}

	return;
}

static inline unsigned int cbpl_calculate_clause_output_predict(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *Xi)
{
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		unsigned int all_exclude = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & Xi[patch*number_of_ta_chunks + k]) == ta_state[pos];

			if (!output) {
				break;
			}
			all_exclude = all_exclude && (ta_state[pos] == 0);
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] & filter) ==
			(ta_state[pos] & filter);

		all_exclude = all_exclude && ((ta_state[pos] & filter) == 0);

		if (output && all_exclude == 0) {
			return(1);
		}
	}

	return(0);
}

// Necessary change is managing Xi as consecutive (already handled in PL)
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
)
{
    // Large mask/filter
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	if (reuse_random_feedback && s > 1.0) {
		cbpl_initialize_random_streams(feedback_to_ta, number_of_literals, number_of_ta_chunks, s);
	}
	// for (int i = 0; i < number_of_clauses / 32; ++i) {
	// 	printf("Clause outputs chunk %d: 0x%08x\n", i, clause_outputs[i]);
	// }

	for (int j = 0; j < number_of_clauses; ++j) {
		if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}
		
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
		
		unsigned int clause_output = clause_outputs[j];
		// printf("Clause %d output: %d\n", j, clause_output);
		// unsigned int clause_patch;

		// cbpl_calculate_clause_output_feedback(&ta_state[clause_pos], output_one_patches, &clause_output, &clause_patch, number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, Xi);

		if (!reuse_random_feedback && s > 1.0) {
			cbpl_initialize_random_streams(feedback_to_ta, number_of_literals, number_of_ta_chunks, s);
		}

		if (clause_output && cbpl_number_of_include_actions(ta_state, j, number_of_literals, number_of_state_bits) <= max_included_literals) {
			// Type Ia Feedback
			// printf("Type Ia feedback on Clause %d\n", j);
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;

				if (boost_true_positive_feedback == 1) {
	 				cbpl_inc(&ta_state[clause_pos + ta_pos], clause_patches[j*number_of_ta_chunks + k], number_of_state_bits);
				} else {
					cbpl_inc(&ta_state[clause_pos + ta_pos], clause_patches[j*number_of_ta_chunks + k] & (~feedback_to_ta[k]), number_of_state_bits);
				}

				if (s > 1.0) {
		 			cbpl_dec(&ta_state[clause_pos + ta_pos], (~clause_patches[j*number_of_ta_chunks + k]) & feedback_to_ta[k], number_of_state_bits);
		 		} else {
		 			cbpl_dec(&ta_state[clause_pos + ta_pos], (~clause_patches[j*number_of_ta_chunks + k]), number_of_state_bits);
		 		}
			}
		} else {
			// Type Ib Feedback
			// printf("Type Ib feedback on Clause %d\n", j);
				
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;

				if (s > 1.0) {
					cbpl_dec(&ta_state[clause_pos + ta_pos],  feedback_to_ta[k], number_of_state_bits);
				} else {
					cbpl_dec(&ta_state[clause_pos + ta_pos], 1, number_of_state_bits);
				}
			}
		}
	}
}

// Same here, handle Xi as consecutive
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
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		unsigned int clause_output = clause_outputs[j];
		// printf("Clause %d output: %u\n", j, clause_output);
		// unsigned int clause_patch;
		// cbpl_calculate_clause_output_feedback(&ta_state[clause_pos], output_one_patches, &clause_output, &clause_patch, number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, Xi);

		if (clause_output) {				
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;
				cbpl_inc(&ta_state[clause_pos + ta_pos],  (~clause_patches[j*number_of_ta_chunks + k]), number_of_state_bits);
			}
		}
	}
}


void cbpl_calculate_clause_outputs_predict(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        unsigned int *clause_output,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
		clause_output[j] = cbpl_calculate_clause_output_predict(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, Xi);
	}
}


// This function retrieves the count of literals from the given Tsetlin Automaton state.
// ta_state: an array representing the state of the Tsetlin Automaton.
// number_of_clauses: the total number of clauses in the TA.
// number_of_literals: the total number of literals in the TA.
// number_of_state_bits: the number of bits used to represent each state in the TA.
// result: an array to store the count of each literal.
void cbpl_get_literals(
    const unsigned int *ta_state,
    unsigned int number_of_clauses,
    unsigned int number_of_literals,
    unsigned int number_of_state_bits,
    unsigned int *result
){
    // Calculate the number of chunks required to represent all literals.
    unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

    // Iterate through all the clauses.
    for (unsigned int j = 0; j < number_of_clauses; j++) {
        // Iterate through all the literals.
        for (unsigned int k = 0; k < number_of_literals; k++) {

            // Determine which chunk the literal is in and its position within the chunk.
            unsigned int ta_chunk = k / 32;
            unsigned int chunk_pos = k % 32;

            // Calculate the position of the literal in the TA state array.
            unsigned int pos = j * number_of_ta_chunks * number_of_state_bits + ta_chunk * number_of_state_bits + number_of_state_bits-1;

            // Check if the literal is present (bit is set) in the TA state array.
            if (ta_state[pos] & (1 << chunk_pos)) {
                // Increment the count of the literal in the result array.
                unsigned int result_pos = j * number_of_literals + k;
                result[result_pos] = 1;
            }
        }
    }
}

void cbpl_get_model(
	const unsigned int *ta_state,
	unsigned int number_of_clauses,
	unsigned int number_of_literals,
	unsigned int number_of_state_bits,
	unsigned int *model
)
{
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;
	
	// Calculate number of 32-bit chunks needed per clause (after padding)
	unsigned int padded_literals = number_of_literals;
	if (number_of_literals % 32 != 0) {
		padded_literals = ((number_of_literals / 32) + 1) * 32;
	}
	unsigned int chunks_per_clause = padded_literals / 32;
	unsigned int padding = padded_literals - number_of_literals;

	// Pack each clause's literals into 32-bit integers directly from ta_state
	unsigned int model_idx = 0;
	for (unsigned int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_base = j * number_of_ta_chunks * number_of_state_bits;		
		for (unsigned int chunk = 0; chunk < chunks_per_clause; chunk++) {
			unsigned int packed_value = 0;
			
			// For each of the 32 bits in this chunk
			for (int i = 0; i < 32; i++) {
				// Position in the padded array
				int padded_pos = chunk * 32 + i;
				
				// Check if this is padding or actual data
				if (padded_pos < padding) {
					// This is padding, bit = 0 (already initialized)
					continue;
				}
				
				int reversed_pos = padded_pos - padding;
				
				int original_pos = number_of_literals - 1 - reversed_pos;
				
				unsigned int ta_chunk = original_pos / 32;
				unsigned int chunk_pos = original_pos % 32;
				
				// Read directly from ta_state
				unsigned int ta_pos = clause_base + ta_chunk * number_of_state_bits + number_of_state_bits - 1;

				if (ta_state[ta_pos] & (1 << chunk_pos)) {
					packed_value |= (1 << (31 - i));
				}
			}
			
			model[model_idx++] = packed_value;
		}
	}
}

void cbpl_transform_example(
	unsigned int *X,
	unsigned int *encoded_X,
	int dim_y,
	int dim_x
)
{
	// X is a 2D image with dimensions dim_x (rows) x dim_y (columns)
	// Each row needs to be reversed and packed into 32-bit integers
	// The packed result is stored in encoded_X
	
	// Calculate padding needed for each row
	unsigned int padded_dim_y = dim_y;
	if (dim_y % 32 != 0) {
		padded_dim_y = ((dim_y / 32) + 1) * 32;
	}
	unsigned int chunks_per_row = padded_dim_y / 32;
	unsigned int padding = padded_dim_y - dim_y;
	
	unsigned int encoded_idx = 0;
	
	// Process each row of the image
	for (int row = 0; row < dim_x; row++) {
		unsigned int row_offset = row * dim_y;
		
		// Pack this row into chunks
		// Python logic: row[::-1] reverses the row, then pack_bits_32 pads and packs
		for (unsigned int chunk = 0; chunk < chunks_per_row; chunk++) {
			unsigned int packed_value = 0;
			
			// For each of the 32 bits in this chunk
			for (int i = 0; i < 32; i++) {
				// Position in the padded array
				int padded_pos = chunk * 32 + i;
				
				// Check if this is padding or actual data
				if (padded_pos < padding) {
					// This is padding, bit = 0 (already initialized)
					continue;
				}
				
				// Position in the reversed array (after removing padding offset)
				int reversed_pos = padded_pos - padding;
				
				// Original position (before reversal)
				int original_pos = dim_y - 1 - reversed_pos;
				
				// Read the bit from X
				// X is stored as individual bits (0 or 1) in the array
				unsigned int bit_value = X[row_offset + original_pos];
				
				// Place it at the correct position in packed_value
				// Bit at position i in the padded array should go to bit (31-i)
				if (bit_value) {
					packed_value |= (1 << (31 - i));
				}
			}
			
			encoded_X[encoded_idx++] = packed_value;
		}
	}
}

void cbpl_pack_weights(
	int *weights,
	unsigned int *packed_weights,
	int num_rows,
	int num_cols,
	int bits_per_weight
)
{
	// Pack weights with transpose: input is [num_rows, num_cols], transpose to [num_cols, num_rows]
	// Then pack and reverse
	// This avoids creating a transposed copy in Python
	
	int num_weights_per_chunk = 32 / bits_per_weight;
	unsigned int weight_mask = (1 << bits_per_weight) - 1;
	
	unsigned int chunk = 0;
	int packed_idx = 0;
	int weight_count = 0;
	int total_weights = num_rows * num_cols;
	
	for (int row = 0; row < num_rows; row++) {
		for (int col = 0; col < num_cols; col++) {
			// Reset chunk when starting a new one
			if ((weight_count % num_weights_per_chunk == 0) || ((weight_count % num_cols) == 0)) {
				chunk = 0;
			}
			
			// Access weight at [row, col] in original array (row-major order)
			int weight_value = weights[row * num_cols + col];
			unsigned int masked_weight = weight_value & weight_mask;
			
			// Calculate bit position within the chunk
			int bit_position = (weight_count % num_weights_per_chunk) * bits_per_weight;
			
			// OR the masked weight into the chunk
			chunk |= masked_weight << bit_position;
			
			// Check if chunk is full or if the clause is fully written
			if ((weight_count % num_weights_per_chunk == num_weights_per_chunk - 1) || ((weight_count % num_cols) == (num_cols - 1))) {
				packed_weights[packed_idx++] = chunk;
			}
			
			weight_count++;
		}
	}
	
	// Reverse the array in-place
	int total_chunks = packed_idx;
	for (int i = 0; i < total_chunks / 2; i++) {
		unsigned int temp = packed_weights[i];
		packed_weights[i] = packed_weights[total_chunks - 1 - i];
		packed_weights[total_chunks - 1 - i] = temp;
	}
}

void cbpl_calculate_clause_outputs_update(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        unsigned int *clause_output,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}

	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
		clause_output[j] = cbpl_calculate_clause_output_update(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, Xi);
	}
}

void cbpl_calculate_literal_frequency(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        unsigned int *clause_active,
        unsigned int *literal_count
)
{
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int k = 0; k < number_of_literals; k++) {
		literal_count[k] = 0;
	}
	
	for (int j = 0; j < number_of_clauses; j++) {
		if ((!clause_active[j])) {
			continue;
		}

		for (int k = 0; k < number_of_literals; k++) {
			unsigned int ta_chunk = k / 32;
			unsigned int chunk_pos = k % 32;
			unsigned int pos = j * number_of_ta_chunks * number_of_state_bits + ta_chunk * number_of_state_bits + number_of_state_bits-1;
			if ((ta_state[pos] & (1 << chunk_pos)) > 0) {
				literal_count[k] += 1;
			}
		}
	}
}

void cbpl_included_literals(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        unsigned int *actions
)
{
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int k = 0; k < number_of_ta_chunks; k++) {
		actions[k] = 0;
	}
	
	for (int j = 0; j < number_of_clauses; j++) {	
		for (int k = 0; k < number_of_ta_chunks; k++) {
			unsigned int pos = j * number_of_ta_chunks * number_of_state_bits + k * number_of_state_bits + number_of_state_bits-1;
			actions[k] |= ta_state[pos];
		}
	}
}

int cbpl_number_of_include_actions(
        unsigned int *ta_state,
        int clause,
        int number_of_literals,
        int number_of_state_bits
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;
	
	unsigned int clause_pos = clause*number_of_ta_chunks*number_of_state_bits;

	int number_of_include_actions = 0;
	for (int k = 0; k < number_of_ta_chunks-1; ++k) {
		unsigned int ta_pos = k*number_of_state_bits + number_of_state_bits-1;
		number_of_include_actions += __builtin_popcount(ta_state[clause_pos + ta_pos]);
	}
	unsigned int ta_pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
	number_of_include_actions += __builtin_popcount(ta_state[clause_pos + ta_pos] & filter);

	return(number_of_include_actions);
}


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
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	unsigned int *current_Xi = Xi;
	for (int b = 0; b < batch_size; ++b) {
		for (int j = 0; j < number_of_clauses; ++j) {
			clause_output[b*number_of_clauses + j] = 0;
		}

		for (int patch = 0; patch < number_of_patches; ++patch) {
			cbpl_calculate_clause_outputs_incremental(literal_clause_map, literal_clause_map_pos, false_literals_per_clause, number_of_clauses, number_of_literals, previous_Xi, current_Xi);
			for (int j = 0; j < number_of_clauses; ++j) {
				if (false_literals_per_clause[j] == 0) {
					clause_output[b*number_of_clauses + j] = 1;
				}
			}
			current_Xi += number_of_ta_chunks;
		}
	}
}

void cbpl_initialize_incremental_clause_calculation(
        unsigned int *ta_state,
        unsigned int *literal_clause_map,
        unsigned int *literal_clause_map_pos,
        unsigned int *false_literals_per_clause,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        unsigned int *previous_Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	// Initialize all literals as false for the previous example per patch
	for (int k = 0; k < number_of_ta_chunks; ++k) {
		previous_Xi[k] = 0;
	}

	// Initialize all false literal counters to 0 per patch
	for (int j = 0; j < number_of_clauses; ++j) {
		false_literals_per_clause[j] = 0;
	}

	// Build the literal clause map, and update the false literal counters
	// Start filling out the map from position 0
	unsigned int pos = 0;
	for (int k = 0; k < number_of_literals; ++k) {
		unsigned int ta_chunk = k / 32;
		unsigned int chunk_pos = k % 32;

		// For each literal, find out which clauses includes it
		for (int j = 0; j < number_of_clauses; ++j) {	
			// Obtain the clause ta chunk containing the literal decision (exclude/include)
			unsigned int clause_ta_chunk = j * number_of_ta_chunks * number_of_state_bits + ta_chunk * number_of_state_bits + number_of_state_bits - 1;
			if (ta_state[clause_ta_chunk] & (1 << chunk_pos)) {
				// Literal k included in clause j
				literal_clause_map[pos] = j;

				++false_literals_per_clause[j];
				++pos;
			}
		}
		literal_clause_map_pos[k] = pos;
	}

	// Make empty clauses false
	for (int j = 0; j < number_of_clauses; ++j) {
		if (false_literals_per_clause[j] == 0) {
			false_literals_per_clause[j] = 1;
		}
	}
}

void cbpl_calculate_clause_outputs_incremental(
        unsigned int * literal_clause_map,
        unsigned int *literal_clause_map_pos,
        unsigned int *false_literals_per_clause,
        int number_of_clauses,
        int number_of_literals,
        unsigned int *previous_Xi,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	// Look up each in literal clause map
	unsigned int start_pos = 0;
	for (int k = 0; k < number_of_literals; ++k) {
		unsigned int ta_chunk = k / 32;
		unsigned int chunk_pos = k % 32;

		// Check which literals have changed
		if ((Xi[ta_chunk] & (1 << chunk_pos)) && !(previous_Xi[ta_chunk] & (1 << chunk_pos))) {
			// If the literal now is True, decrement the false literal counter of all clauses including the literal
			for (int j = 0; j < literal_clause_map_pos[k] - start_pos; ++j) {
				--false_literals_per_clause[literal_clause_map[start_pos + j]];
			}
		} else if (!(Xi[ta_chunk] & (1 << chunk_pos)) && (previous_Xi[ta_chunk] & (1 << chunk_pos))) {
			// If the literal now is False, increment the false counter of all clauses including literal
			for (int j = 0; j < literal_clause_map_pos[k] - start_pos; ++j) {
				++false_literals_per_clause[literal_clause_map[start_pos + j]];
			}
		}

		start_pos = literal_clause_map_pos[k];
	}

	// Copy current Xi to previous_Xi
	for (int k = 0; k < number_of_ta_chunks; ++k) {
		previous_Xi[k] = Xi[k];
	}
}