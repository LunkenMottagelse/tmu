
from xml.parsers.expat import model
from tmu.tmulib import ffi, lib
import tmu.tools
from tmu.clause_bank.base_clause_bank import BaseClauseBank

import numpy as np
import math
import logging

_LOGGER = logging.getLogger(__name__)


class ClauseBankPL(BaseClauseBank):
    clause_bank: np.ndarray
    incremental_clause_evaluation_initialized: bool
    co_p = None  # _cffi_backend._CDataBase
    cob_p = None  # _cffi_backend._CDataBase
    ptr_clause_and_target = None  # _cffi_backend._CDataBase
    cop_p = None  # _cffi_backend._CDataBase
    ptr_feedback_to_ta = None  # _cffi_backend._CDataBase
    ptr_output_one_patches = None  # _cffi_backend._CDataBase
    ptr_literal_clause_count = None  # _cffi_backend._CDataBase
    ptr_actions = None  # _cffi_backend._CDataBase

    def __init__(
            self,
            seed: int,
            d: float,
            number_of_state_bits_ind: int,
            number_of_state_bits_ta: int,
            batch_size: int,
            incremental: bool,
            **kwargs
    ):
        super().__init__(seed=seed, **kwargs)

        self.d = d
        assert isinstance(number_of_state_bits_ta, int)
        self.number_of_state_bits_ta = number_of_state_bits_ta
        self.number_of_state_bits_ind = int(number_of_state_bits_ind)
        self.batch_size = batch_size
        self.incremental = incremental
        self.number_of_classes = int(kwargs.get("number_of_classes", 10))
        self.get_weights_callback = kwargs.get("get_weights_callback", None)

        self.clause_output = np.empty(self.number_of_clauses, dtype=np.uint32, order="c")
        self.clause_output_batch = np.empty(self.number_of_clauses * batch_size, dtype=np.uint32, order="c")
        self.clause_and_target = np.zeros(self.number_of_clauses * self.number_of_ta_chunks, dtype=np.uint32, order="c")
        self.clause_output_patchwise = np.empty(self.number_of_clauses * self.number_of_patches, dtype=np.uint32, order="c")
        self.feedback_to_ta = np.empty(self.number_of_ta_chunks, dtype=np.uint32, order="c")
        self.output_one_patches = np.empty(self.number_of_patches, dtype=np.uint32, order="c")
        self.literal_clause_count = np.empty(self.number_of_literals, dtype=np.uint32, order="c")
        self.model = np.empty(self.number_of_clauses * self.number_of_ta_chunks, dtype=np.uint32, order="c")
        self.transformed_example = np.empty(self.dim[0] * math.ceil(self.dim[1] / 32.0), dtype=np.uint32, order="c")

        self.type_ia_feedback_counter = np.zeros(self.number_of_clauses, dtype=np.uint32, order="c")

        # Incremental Clause Evaluation
        self.literal_clause_map = np.empty(
            (int(self.number_of_literals * self.number_of_clauses)),
            dtype=np.uint32,
            order="c"
        )
        self.literal_clause_map_pos = np.empty(
            (int(self.number_of_literals)),
            dtype=np.uint32,
            order="c"
        )
        self.false_literals_per_clause = np.empty(
            int(self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
            order="c"
        )
        self.previous_xi = np.empty(
            int(self.number_of_ta_chunks) * int(self.number_of_patches),
            dtype=np.uint32,
            order="c"
        )

        self.initialize_clauses()

        # Finally, map numpy arrays to CFFI compatible pointers.
        self._cffi_init()

        # Set pcg32 seed
        if self.seed is not None:
            assert isinstance(self.seed, int), "Seed must be a integer"

            lib.pcg32_seed(self.seed)
            lib.xorshift128p_seed(self.seed)
        
        # Program PL
        from pynq import Overlay
        from pynq import allocate

        self.ol = Overlay("/home/xilinx/modded_tmu/bitfiles/TM_Inference.bit") # Hardcoded path to bitfiles
        self.img_decision_ol = self.ol.axi_dma_0
        self.ie_ol = self.ol.axi_dma_1
        self.weight_ol = self.ol.axi_dma_2

        packed_image_size = self.dim[1] * math.ceil(self.dim[2] / 32.0)
        self.bits_per_weight = 9  # FIXME: Hardcoded for now, but could be parameterized
        packed_weight_size = math.ceil(self.number_of_classes * self.number_of_clauses / math.floor(32 / self.bits_per_weight))
        packed_clauses_size = math.ceil(self.number_of_clauses / 32.0)
        self.packed_patch_size = math.ceil(self.number_of_literals / 32.0)

        self.image_buffer = allocate(shape=(packed_image_size,), dtype=np.uint32, cacheable=1)
        self.weight_buffer = allocate(shape=(packed_weight_size,), dtype=np.uint32, cacheable=1)
        self.ie_buffer = allocate(shape=(self.number_of_clauses * self.number_of_ta_chunks,), dtype=np.uint32, cacheable=1)
        self.decision_buffer = allocate(shape=(self.number_of_classes + packed_clauses_size + self.packed_patch_size*self.number_of_clauses,), dtype=np.uint32, cacheable=1)

    def _cffi_init(self):
        self.co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)  # clause_output
        self.cob_p = ffi.cast("unsigned int *", self.clause_output_batch.ctypes.data)  # clause_output_batch
        self.ptr_clause_and_target = ffi.cast("unsigned int *", self.clause_and_target.ctypes.data)  # clause_and_target
        self.cop_p = ffi.cast("unsigned int *", self.clause_output_patchwise.ctypes.data)  # clause_output_patchwise
        self.ptr_feedback_to_ta = ffi.cast("unsigned int *", self.feedback_to_ta.ctypes.data)  # feedback_to_ta
        self.ptr_output_one_patches = ffi.cast("unsigned int *", self.output_one_patches.ctypes.data)  # output_one_patches
        self.ptr_literal_clause_count = ffi.cast("unsigned int *", self.literal_clause_count.ctypes.data)  # literal_clause_count
        self.tiafc_p = ffi.cast("unsigned int *", self.type_ia_feedback_counter.ctypes.data)  # literal_clause_count

        # Clause Initialization
        self.ptr_ta_state = ffi.cast("unsigned int *", self.clause_bank.ctypes.data)
        self.ptr_ta_state_ind = ffi.cast("unsigned int *", self.clause_bank_ind.ctypes.data)


        # Action Initialization
        self.ptr_actions = ffi.cast("unsigned int *", self.actions.ctypes.data)

        # Incremental Clause Evaluation Initialization
        self.lcm_p = ffi.cast("unsigned int *", self.literal_clause_map.ctypes.data)
        self.lcmp_p = ffi.cast("unsigned int *", self.literal_clause_map_pos.ctypes.data)
        self.flpc_p = ffi.cast("unsigned int *", self.false_literals_per_clause.ctypes.data)
        self.previous_xi_p = ffi.cast("unsigned int *", self.previous_xi.ctypes.data)

        self.model_p = ffi.cast("unsigned int *", self.model.ctypes.data)
        self.transformed_example_p = ffi.cast("unsigned int *", self.transformed_example.ctypes.data)

    def initialize_clauses(self):
        self.clause_bank = np.empty(
            shape=(self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
            dtype=np.uint32,
            order="c"
        )

        # np.uint32(~0) will be deprecated in numpy>=2.0, changed to np.array(~0).astype(np.uint32)
        self.clause_bank[:, :, 0: self.number_of_state_bits_ta - 1] = np.array(~0).astype(np.uint32)
        self.clause_bank[:, :, self.number_of_state_bits_ta - 1] = 0
        self.clause_bank = np.ascontiguousarray(self.clause_bank.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ta)))

        self.actions = np.ascontiguousarray(np.zeros(self.number_of_ta_chunks, dtype=np.uint32))

        self.clause_bank_ind = np.empty(
            (self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ind), dtype=np.uint32)

        # np.uint32(~0) will be deprecated in numpy>=2.0, changed to np.array(~0).astype(np.uint32)
        self.clause_bank_ind[:, :, :] = np.array(~0).astype(np.uint32)

        self.clause_bank_ind = np.ascontiguousarray(self.clause_bank_ind.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ind)))



        self.incremental_clause_evaluation_initialized = False

    def calculate_clause_outputs_predict(self, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        if not self.incremental:
            lib.cbpl_calculate_clause_outputs_predict(
                self.ptr_ta_state,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                self.co_p,
                xi_p
            )
            return self.clause_output

        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        if not self.incremental_clause_evaluation_initialized:

            lib.cbpl_initialize_incremental_clause_calculation(
                self.ptr_ta_state,
                self.lcm_p,
                self.lcmp_p,
                self.flpc_p,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.previous_xi_p
            )

            self.incremental_clause_evaluation_initialized = True

        if e % self.batch_size == 0:
            lib.cbpl_calculate_clause_outputs_incremental_batch(
                self.lcm_p,
                self.lcmp_p,
                self.flpc_p,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_patches,
                self.cob_p,
                self.previous_xi_p,
                xi_p,
                np.minimum(self.batch_size, encoded_X.shape[0] - e)
            )

        return self.clause_output_batch.reshape((self.batch_size, self.number_of_clauses))[e % self.batch_size, :]

    def weight_packing_bits_32(self, bits_per_weight, model):
        # Pack weights into 32-bit chunks
        packed_weights = []
        num_weights_per_chunk = 32 // bits_per_weight
        for i in range(0, model.shape[0]):
            if i % num_weights_per_chunk == 0:
                chunk = 0
            chunk |= (model[i] & ((1 << bits_per_weight) - 1)) << (i % num_weights_per_chunk * bits_per_weight)
            
            #  [--------------------Chunk is full-------------------------]    [-----Last element------]
            if ((i % num_weights_per_chunk) == (num_weights_per_chunk) - 1) or (i == model.shape[0] - 1):
                packed_weights.append(chunk)
        return np.array(packed_weights, dtype=np.uint32)

    # Pack bits into 32-bit ints. assuming bits are in big-endian order (MSB first element)
    def pack_bits_32(self, bits):
        # Pad with zeros to make length multiple of 32
        padded_bits = bits.copy()
        while len(padded_bits) % 32 != 0:
            padded_bits.insert(0, 0) # pad from the MSB
        
        # Convert to numpy array
        bit_array = np.array(padded_bits, dtype=np.uint8)
        
        # Reshape into groups of 32 bits
        bit_groups = bit_array.reshape(-1, 32)
        
        # Convert each group of 32 bits to a 32-bit integer
        powers_of_2 = 2 ** np.arange(31, -1, -1)  # [2^31, 2^30, ..., 2^1, 2^0]
        packed_32bit = np.dot(bit_groups, powers_of_2)
        
        return packed_32bit.astype(np.uint32)

    def pack_image(self, image):
        # Testing image transpose
        packed_image = np.zeros((image.shape[1], (image.shape[0] + 31) // 32), dtype=np.uint32)
        # image = np.transpose(image)  # Transpose to match expected layout
        for i, row in enumerate(image):
            packed_row = self.pack_bits_32(row[::-1].tolist()) # Invert row for little-endian bit order
            packed_image[i] = packed_row
        return np.concatenate(packed_image).astype(np.uint32)

    def calculate_clause_outputs_update_fpga(self, X_train, e):
        # Get weights using callback function
        if self.get_weights_callback is None:
            raise RuntimeError("get_weights_callback is not set. Please provide a callback function when creating ClauseBankPL.")
        
        # Get all weights from all weight banks using the callback
        weights = self.get_weights_callback()
        weights = np.transpose(weights)  # Flip weights to be [classes, clauses]

        weights_packed = self.weight_packing_bits_32(self.bits_per_weight, weights.flatten())
        self.weight_buffer[:] = weights_packed[::-1]

        self.get_model() # Pointer business
        self.ie_buffer[:] = self.model
        self.image_buffer[:] = self.transform_example(X_train[e])

        # 1: ship to PL
        pl_timer = tmu.tools.BenchmarkTimer()
        with pl_timer:
            self.ie_ol.sendchannel.transfer(self.ie_buffer)
            self.weight_ol.sendchannel.transfer(self.weight_buffer)
            self.img_decision_ol.sendchannel.transfer(self.image_buffer)
            self.img_decision_ol.recvchannel.transfer(self.decision_buffer)

            # 2: capture output

            self.img_decision_ol.sendchannel.wait()
            self.ie_ol.sendchannel.wait()
            self.weight_ol.sendchannel.wait()
            self.img_decision_ol.recvchannel.wait()

        # 3: store output in correct location.
        # class sums -> First NClasses transfers contain class sums
        # clause_output -> Then N transfers contain clause outputs
        # selected_patches -> Then remaining transfers contain selected patches
        class_sums = self.decision_buffer[:self.number_of_classes]  # First NClasses transfers contain class sums
        
        # _LOGGER.info("Class sums from PL v")
        # _LOGGER.info(class_sums)
        
        # NOTE: these are stored densely
        clause_output_pl = self.decision_buffer[self.number_of_classes:self.number_of_classes + math.ceil(self.number_of_clauses / 32)] # Then N transfers contain clause outputs
        for i in range(self.number_of_clauses):
            if clause_output_pl[(i // 32)] >> (i % 32) & 1:
                self.clause_output[i] = 1
        patches = np.array(self.decision_buffer[self.number_of_classes + math.ceil(self.number_of_clauses / 32):])  # Then remaining transfers contain selected patches
        patches = patches.reshape((self.number_of_clauses, self.packed_patch_size))
        # _LOGGER.info("Clause output from PL v")
        # _LOGGER.info(self.clause_output)

        # Then flush
        # self.ie_buffer.flush()
        # self.weight_buffer.flush()
        # self.image_buffer.flush()
        # self.decision_buffer.flush()
        # _LOGGER.info(self.clause_output)
        # _LOGGER.info("Clause output from classic ^")
        
        # _LOGGER.info(f"PL time: {pl_timer.elapsed():.2f} s, Classic time: {classic_timer.elapsed():.2f} s")

        # if not np.array_equal(self.clause_output, expanded_clause_output):
        #     _LOGGER.warning("="*60)
        #     _LOGGER.warning(f"Mismatch between PL and classic TM clause outputs in example {e}!")
        #     _LOGGER.warning(f"PL output: {expanded_clause_output}")
        #     _LOGGER.warning(f"Classic output: {self.clause_output}")

        #     # Find the clauses that differ
        #     for i in range(self.number_of_clauses):
        #         if self.clause_output[i] != expanded_clause_output[i]:
        #             _LOGGER.warning(f"Clause {i} differs: PL={expanded_clause_output[i]}, Classic={self.clause_output[i]}")
        #             # Log the content of the clause
        #             literals = self.get_literals()[i]
        #             _LOGGER.warning(f"Literals for clause {i}: {literals}")
        #     _LOGGER.warning("-"*60)
        #     _LOGGER.warning("Packed Model:")
        #     _LOGGER.warning(np.vectorize(lambda x: f"0x{x:08x}")(model_packed))
        #     _LOGGER.warning("-"*60)
        #     _LOGGER.warning("Packed Image:")
        #     _LOGGER.warning(np.vectorize(lambda x: f"0x{x:08x}")(image_packed))
        #     _LOGGER.warning("-"*60)
        #     _LOGGER.warning("Packed Weights:")
        #     _LOGGER.warning(np.vectorize(lambda x: f"0x{x:08x}")(weights_packed))
        #     _LOGGER.warning("="*60)
        # else:
        #     _LOGGER.info(f"PL and classic TM clause outputs match for example {e}.")



        return self.clause_output, class_sums, patches

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        # Classic method
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        lib.cbpl_calculate_clause_outputs_update(
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            self.co_p,
            xi_p
        )

        return self.clause_output

    def type_i_feedback(
        self,
        update_p,
        clause_active,
        clause_patches,
        clause_outputs
    ):
        # encoded_X is wrong here, must be the randomly selected patches from PL. 
        ptr_cp = ffi.cast("unsigned int *", clause_patches.ctypes.data)
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        ptr_clause_outputs = ffi.cast("unsigned int *", clause_outputs.ctypes.data)
        lib.cbpl_type_i_feedback(
            self.ptr_ta_state,
            self.ptr_feedback_to_ta,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            update_p,
            self.s,
            self.boost_true_positive_feedback,
            self.reuse_random_feedback,
            self.max_included_literals,
            ptr_clause_active,
            ptr_cp,
            ptr_clause_outputs
        )

        self.incremental_clause_evaluation_initialized = False

    def type_ii_feedback(
        self,
        update_p,
        clause_active,
        clause_patches,
        clause_outputs,
    ):
        ptr_cp = ffi.cast("unsigned int *", clause_patches.ctypes.data)
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        ptr_clause_outputs = ffi.cast("unsigned int *", clause_outputs.ctypes.data)

        lib.cbpl_type_ii_feedback(
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            update_p,
            ptr_clause_active,
            ptr_cp,
            ptr_clause_outputs
        )

        self.incremental_clause_evaluation_initialized = False

    def calculate_literal_clause_frequency(
            self,
            clause_active
    ):
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.cbpl_calculate_literal_frequency(
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            ptr_clause_active,
            self.ptr_literal_clause_count
        )
        return self.literal_clause_count

    def included_literals(self):
        lib.cbpl_included_literals(
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.ptr_actions
        )
        return self.actions

    def get_literals(self, independent=False):

        result = np.zeros((self.number_of_clauses, self.number_of_literals), dtype=np.uint32, order="c")
        result_p = ffi.cast("unsigned int *", result.ctypes.data)
        lib.cbpl_get_literals(
            self.ptr_ta_state_ind if independent else self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            result_p
        )
        return result
    
    def get_model(self, independent=False):
        lib.cbpl_get_model(
            self.ptr_ta_state_ind if independent else self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.model_p
        )
        return

    def calculate_independent_literal_clause_frequency(self, clause_active):
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.cbpl_calculate_literal_frequency(
            self.ptr_ta_state_ind,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            ca_p,
            self.ptr_literal_clause_count
        )
        return self.literal_clause_count

    def number_of_include_actions(
            self,
            clause
    ):
        return lib.cbpl_number_of_include_actions(
            self.ptr_ta_state,
            clause,
            self.number_of_literals,
            self.number_of_state_bits_ta
        )

    def transform_example(
            self,
            X
    ):
        X_p = ffi.cast("unsigned int *", X.ctypes.data)
        lib.cbpl_transform_example(
            X_p,
            self.transformed_example_p,
            self.dim[0],
            self.dim[1]
        )
        return self.transformed_example

    def prepare_X_autoencoder(
            self,
            X_csr,
            X_csc,
            active_output
    ):
        X = np.ascontiguousarray(np.empty(int(self.number_of_ta_chunks), dtype=np.uint32))
        return X_csr, X_csc, active_output, X

    def produce_autoencoder_example(
            self,
            encoded_X,
            target,
            target_true_p,
            accumulation
    ):
        (X_csr, X_csc, active_output, X) = encoded_X

        target_value = self.rng.random() <= target_true_p

        lib.tmu_produce_autoencoder_example(ffi.cast("unsigned int *", active_output.ctypes.data), active_output.shape[0],
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indptr).ctypes.data),
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indices).ctypes.data),
                                             int(X_csr.shape[0]),
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indptr).ctypes.data),
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indices).ctypes.data),
                                             int(X_csc.shape[1]),
                                             ffi.cast("unsigned int *", X.ctypes.data),
                                             int(target),
                                             int(target_value),
                                             int(accumulation))

        return X.reshape((1, -1)), target_value

