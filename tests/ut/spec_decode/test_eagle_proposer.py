import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CacheConfig, CompilationMode, CUDAGraphMode, VllmConfig, set_current_vllm_config

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.eagle_proposer import (
    AscendEagleProposer,
    SpecDecodeBaseProposer,
    _MTPDraftDiagnosticContext,
)
from vllm_ascend.spec_decode.mtp_draft_diagnostics import (
    referenced_block_ids,
)


class TestEagleProposerInitialization(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.cache_config = MagicMock(spec=CacheConfig)
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.model_config = MagicMock()
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.compilation_config = MagicMock()
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    def test_initialization_eagle_graph(self):
        self.vllm_config.speculative_config.method = "eagle"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 4096
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = False
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 4096)
            self.assertTrue(proposer.use_cuda_graph)

            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.input_ids.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.positions.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 4096))
            self.assertEqual(proposer.arange.shape, (expected_max_num_tokens,))

    def test_staged_mtp_metadata_arenas_have_stable_distinct_addresses(
        self,
    ):
        proposer = SpecDecodeBaseProposer.__new__(
            SpecDecodeBaseProposer
        )
        proposer.use_staged_mtp_draft_graph = True
        proposer._staged_mtp_max_request_capacity = 4
        proposer._staged_mtp_arena_capacity = 0
        proposer._staged_mtp_metadata_arenas = []
        proposer.num_speculative_tokens = 2
        proposer.device = torch.device("cpu")
        proposer.runner = MagicMock(pin_memory=False)
        proposer.arange = torch.arange(16, dtype=torch.int32)
        proposer.arange_cpu = torch.arange(16, dtype=torch.int32)
        source = SimpleNamespace(
            query_start_loc=torch.arange(3, dtype=torch.int32),
            query_start_loc_cpu=torch.arange(3, dtype=torch.int32),
            seq_lens=torch.tensor([8, 9], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([8, 9], dtype=torch.int32),
            num_computed_tokens_cpu=torch.tensor(
                [7, 8], dtype=torch.int32
            ),
            block_table_tensor=torch.tensor(
                [[10, 11], [12, 13], [20, 21], [22, 23]],
                dtype=torch.int32,
            ),
            positions=torch.tensor([7, 8], dtype=torch.int32),
        )

        step0 = proposer._bind_staged_mtp_metadata_arena(
            source,
            draft_step=0,
            capacity=4,
            actual_reqs=2,
        )
        step0_ptr = step0.block_table_tensor.data_ptr()
        step1 = proposer._bind_staged_mtp_metadata_arena(
            step0,
            draft_step=1,
            capacity=4,
            actual_reqs=2,
        )
        step0_again = proposer._bind_staged_mtp_metadata_arena(
            source,
            draft_step=0,
            capacity=4,
            actual_reqs=2,
        )

        self.assertEqual(
            step0_again.block_table_tensor.data_ptr(),
            step0_ptr,
        )
        self.assertNotEqual(
            step0.block_table_tensor.data_ptr(),
            step1.block_table_tensor.data_ptr(),
        )
        self.assertTrue(
            torch.equal(
                step0.block_table_tensor,
                source.block_table_tensor,
            )
        )
        self.assertTrue(
            torch.equal(
                step0.query_start_loc,
                torch.arange(5, dtype=torch.int32),
            )
        )

    def test_mtp_input_transforms_preserve_sparse_decode_metadata(self):
        proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
        proposer.runner = SimpleNamespace(
            actual_seq_lengths_q=torch.tensor([2]),
            attn_state=AscendAttentionState.SpecDecoding,
            decode_token_per_req=torch.tensor([2]),
        )
        proposer.token_arange_np = np.arange(2)
        proposer.arange = torch.arange(2)
        proposer.pcp_size = 1

        def make_metadata():
            return AscendCommonAttentionMetadata(
                query_start_loc=torch.tensor([0, 2]),
                query_start_loc_cpu=torch.tensor([0, 2]),
                seq_lens=torch.tensor([120_002]),
                seq_lens_cpu=torch.tensor([120_002]),
                num_computed_tokens_cpu=torch.tensor([120_000]),
                num_reqs=1,
                num_actual_tokens=2,
                num_input_tokens=2,
                max_query_len=2,
                block_table_tensor=torch.tensor([[1, 2]]),
                slot_mapping=torch.tensor([128, 129]),
                indexer_block_table_tensor=torch.tensor([[11, 12]]),
                indexer_slot_mapping=torch.tensor([1408, 1409]),
                prompt_lens_cpu=np.array([120_000], dtype=np.int32),
                request_ids=["req"],
                cold_compact_resumes=(True,),
                resident_state_indices=torch.tensor([3]),
                resident_state_generations=torch.tensor([7]),
                resident_state_indices_cpu=np.array([3], dtype=np.int32),
                resident_state_generations_cpu=np.array([7], dtype=np.int64),
                actual_seq_lengths_q=torch.tensor([2]),
                positions=torch.tensor([120_000, 120_001]),
                attn_state=AscendAttentionState.SpecDecoding,
                decode_token_per_req=torch.tensor([2]),
                max_seq_len=120_002,
            )

        source = make_metadata()
        compact, _ = proposer.prepare_inputs(source, [[1, 2]], [1])

        padded_source = make_metadata()
        with patch(
            "vllm_ascend.spec_decode.eagle_proposer.HAS_TRITON",
            False,
        ):
            padded, *_ = proposer.prepare_inputs_padded(
                padded_source,
                SimpleNamespace(cu_num_draft_tokens=torch.tensor([1])),
                torch.tensor([2]),
            )

        for transformed, original in (
            (compact, source),
            (padded, padded_source),
        ):
            self.assertIs(transformed.prompt_lens_cpu, original.prompt_lens_cpu)
            self.assertIs(transformed.request_ids, original.request_ids)
            self.assertIs(
                transformed.cold_compact_resumes,
                original.cold_compact_resumes,
            )
            self.assertIs(
                transformed.resident_state_indices,
                original.resident_state_indices,
            )
            self.assertIs(
                transformed.resident_state_generations,
                original.resident_state_generations,
            )

    def test_initialization_eagle3_enforce_eager(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.NONE
        self.vllm_config.compilation_config.pass_config = MagicMock()
        self.vllm_config.compilation_config.pass_config.enable_sp = False
        self.vllm_config.model_config.enforce_eager = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertFalse(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_eagle3_full_graph_async(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertTrue(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_mtp_full_graph_async(self):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertFalse(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_staged_mtp_graph_does_not_override_draft_eager_by_default(
        self,
    ):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.speculative_config.enforce_eager = True
        self.vllm_config.scheduler_config.async_scheduling = False
        self.runner._use_aclgraph.return_value = True
        self.runner.cudagraph_batch_sizes = [2, 4]

        with (
            patch(
                "vllm_ascend.spec_decode.eagle_proposer.envs_ascend."
                "VLLM_ASCEND_SFA_STAGED_MTP_DRAFT_GRAPH",
                False,
            ),
            patch(
                "vllm_ascend.spec_decode.eagle_proposer."
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch(
                "vllm_ascend.spec_decode.eagle_proposer."
                "staged_sfa_graph_capture_sizes",
                return_value=(2, 4),
            ) as capture_sizes,
            set_current_vllm_config(self.vllm_config),
        ):
            proposer = AscendEagleProposer(
                vllm_config=self.vllm_config,
                device=self.device,
                runner=self.runner,
            )

        self.assertFalse(proposer.use_cuda_graph)
        self.assertFalse(proposer.use_staged_mtp_draft_graph)
        capture_sizes.assert_not_called()

    def test_staged_mtp_graph_default_off_keeps_regular_draft_graph(self):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = False
        self.runner._use_aclgraph.return_value = True
        self.runner.cudagraph_batch_sizes = [2, 4]

        with (
            patch(
                "vllm_ascend.spec_decode.eagle_proposer.envs_ascend."
                "VLLM_ASCEND_SFA_STAGED_MTP_DRAFT_GRAPH",
                False,
            ),
            patch(
                "vllm_ascend.spec_decode.eagle_proposer."
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            set_current_vllm_config(self.vllm_config),
        ):
            proposer = AscendEagleProposer(
                vllm_config=self.vllm_config,
                device=self.device,
                runner=self.runner,
            )

        self.assertTrue(proposer.use_cuda_graph)
        self.assertFalse(proposer.use_staged_mtp_draft_graph)

    def test_staged_mtp_graph_requires_explicit_opt_in(self):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.speculative_config.enforce_eager = True
        self.vllm_config.scheduler_config.async_scheduling = False
        self.runner._use_aclgraph.return_value = True
        self.runner.cudagraph_batch_sizes = [3, 6]

        with (
            patch(
                "vllm_ascend.spec_decode.eagle_proposer.envs_ascend."
                "VLLM_ASCEND_SFA_STAGED_MTP_DRAFT_GRAPH",
                True,
            ),
            patch(
                "vllm_ascend.spec_decode.eagle_proposer."
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch(
                "vllm_ascend.spec_decode.eagle_proposer."
                "staged_sfa_graph_capture_sizes",
                return_value=(3, 6),
            ),
            set_current_vllm_config(self.vllm_config),
        ):
            proposer = AscendEagleProposer(
                vllm_config=self.vllm_config,
                device=self.device,
                runner=self.runner,
            )

        self.assertTrue(proposer.use_cuda_graph)
        self.assertTrue(proposer.use_staged_mtp_draft_graph)

class TestMTPDraftDiagnostics(TestBase):
    @staticmethod
    def _proposer(output_dir: Path):
        proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.attn_layer_names = []
        proposer._draft_attn_layers = {}
        proposer._mtp_draft_diag_context = _MTPDraftDiagnosticContext(
            proposal_id=7,
            rank=3,
            output_dir=output_dir,
        )
        return proposer

    @staticmethod
    def _model_kwargs():
        return {
            "input_ids": torch.tensor([11], dtype=torch.int32),
            "positions": torch.tensor([5], dtype=torch.int64),
            "hidden_states": torch.tensor([[2.0, 4.0]]),
            "inputs_embeds": None,
        }

    @staticmethod
    def _fake_forward(**model_kwargs):
        return model_kwargs["hidden_states"] + model_kwargs["positions"].reshape(-1, 1)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.atomic_torch_save")
    @patch("vllm_ascend.spec_decode.eagle_proposer.torch.npu.synchronize")
    def test_layer_dump_order(
        self,
        synchronize,
        save,
        get_context,
    ):
        events = []
        synchronize.side_effect = lambda: events.append("sync")
        save.side_effect = lambda payload, path: events.append(path.name)
        get_context.return_value = SimpleNamespace(
            virtual_engine=0,
            no_compile_layers={},
        )
        with TemporaryDirectory() as temp_dir:
            proposer = self._proposer(Path(temp_dir))

            def model(**model_kwargs):
                events.append("model")
                return self._fake_forward(**model_kwargs)

            proposer.model = model
            proposer._run_mtp_draft_layer_with_diagnostics(
                self._model_kwargs(),
                draft_step=0,
                per_layer_attn_metadata=None,
                runtime_inputs={"num_tokens": 1},
            )

        self.assertEqual(
            events,
            [
                "sync",
                "step_0_input.pt",
                "model",
                "sync",
                "step_0_output.pt",
            ],
        )

    @patch("vllm_ascend.spec_decode.eagle_proposer.atomic_torch_save")
    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.torch.npu.synchronize",
        side_effect=RuntimeError("previous device error"),
    )
    def test_layer_pre_sync_failure_does_not_run_model(
        self,
        synchronize,
        save,
    ):
        del synchronize
        with TemporaryDirectory() as temp_dir:
            proposer = self._proposer(Path(temp_dir))
            proposer.model = MagicMock()
            with self.assertRaisesRegex(RuntimeError, "previous device error"):
                proposer._run_mtp_draft_layer_with_diagnostics(
                    self._model_kwargs(),
                    draft_step=0,
                    per_layer_attn_metadata=None,
                    runtime_inputs={"num_tokens": 1},
                )

        proposer.model.assert_not_called()
        self.assertEqual(save.call_count, 1)
        self.assertEqual(
            save.call_args.args[0]["phase"],
            "layer_pre_sync_failed",
        )

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.torch.npu.synchronize",
        side_effect=[None, RuntimeError("draft device error")],
    )
    def test_layer_post_sync_failure_preserves_input(
        self,
        synchronize,
        get_context,
    ):
        del synchronize
        get_context.return_value = SimpleNamespace(
            virtual_engine=0,
            no_compile_layers={},
        )
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            proposer = self._proposer(output_dir)
            proposer.model = MagicMock(side_effect=self._fake_forward)

            with self.assertRaisesRegex(RuntimeError, "draft device error"):
                proposer._run_mtp_draft_layer_with_diagnostics(
                    self._model_kwargs(),
                    draft_step=0,
                    per_layer_attn_metadata=None,
                    runtime_inputs={"num_tokens": 1},
                )

            self.assertTrue((output_dir / "step_0_input.pt").is_file())
            self.assertFalse((output_dir / "step_0_output.pt").exists())
            failure = torch.load(
                output_dir / "step_0_failure.pt",
                weights_only=True,
            )
            self.assertEqual(failure["phase"], "layer_post_sync_failed")
            proposer.model.assert_called_once()

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.torch.npu.synchronize")
    def test_saved_pt_round_trip_preserves_draft_io(
        self,
        synchronize,
        get_context,
    ):
        del synchronize
        get_context.return_value = SimpleNamespace(
            virtual_engine=0,
            no_compile_layers={},
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens=1,
            num_actual_tokens=1,
            is_draft_model=True,
        )
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            proposer = self._proposer(output_dir)
            proposer.model = self._fake_forward
            proposer._run_mtp_draft_layer_with_diagnostics(
                self._model_kwargs(),
                draft_step=0,
                per_layer_attn_metadata=None,
                runtime_inputs={"num_tokens": 1},
            )

            saved_input = torch.load(
                output_dir / "step_0_input.pt",
                weights_only=True,
            )
            saved_output = torch.load(
                output_dir / "step_0_output.pt",
                weights_only=True,
            )
            replayed = self._fake_forward(**saved_input["model_kwargs"])
            self.assertTrue(torch.equal(replayed, saved_output["output"]))
            self.assertEqual(
                saved_input["proposal_id"],
                saved_output["proposal_id"],
            )

    def test_cache_dump_ignores_padded_block_table_entries(self):
        block_table = torch.tensor(
            [[10, 11, 99], [20, 98, 97]],
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([9, 4], dtype=torch.int32)
        slot_mapping = torch.tensor([24], dtype=torch.int32)

        block_ids = referenced_block_ids(
            block_table,
            (slot_mapping,),
            block_size=8,
            seq_lens=seq_lens,
        )

        self.assertEqual(block_ids, [3, 10, 11, 20])

    @patch("vllm_ascend.spec_decode.eagle_proposer.torch.npu.synchronize")
    def test_diagnostic_scope_is_noop_when_debug_is_disabled(
        self,
        synchronize,
    ):
        proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer._mtp_draft_diag_context = None

        with (
            patch.dict(
                os.environ,
                {"VLLM_ASCEND_MTP_DRAFT_DEBUG": "0"},
            ),
            proposer.mtp_draft_diagnostic_scope(),
        ):
            self.assertIsNone(proposer._mtp_draft_diag_context)

        synchronize.assert_not_called()

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_world_group")
    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.torch.npu.synchronize",
        side_effect=RuntimeError("target device error"),
    )
    def test_target_boundary_failure_never_enters_scope(
        self,
        synchronize,
        get_world_group,
    ):
        del synchronize
        get_world_group.return_value.rank = 2
        proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer._mtp_draft_diag_proposal_id = 0
        proposer._mtp_draft_diag_context = None
        entered = False

        with (
            TemporaryDirectory() as temp_dir,
            patch(
                "vllm_ascend.spec_decode.eagle_proposer.MTP_DRAFT_DIAG_ROOT",
                Path(temp_dir),
            ),
            patch.dict(
                os.environ,
                {"VLLM_ASCEND_MTP_DRAFT_DEBUG": "1"},
            ),
            self.assertRaisesRegex(RuntimeError, "target device error"),
            proposer.mtp_draft_diagnostic_scope(),
        ):
            entered = True

        self.assertFalse(entered)


@unittest.skip("Skip due to the changes in #7153, fix me later")
class TestEagleProposerLoadModel(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.method = "eagle"
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.parallel_drafting = False

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 1
        mock_target_layer1 = MagicMock()
        mock_target_layer2 = MagicMock()
        mock_draft_layer1 = MagicMock()
        mock_draft_layer3 = MagicMock()
        mock_get_layers.side_effect = [
            {"layer1": mock_target_layer1, "layer2": mock_target_layer2},
            {},
            {},
            {"layer1": mock_draft_layer1, "layer3": mock_draft_layer3},
        ]

        weight = torch.zeros(0)

        mock_model = MagicMock()
        mock_model.supports_multimodal = False
        mock_model.lm_head = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_model.model.embed_tokens = MagicMock()
        mock_model.model.embed_tokens.weight = weight

        mock_get_model.return_value = MagicMock()
        mock_get_model.return_value.model.embed_tokens.weight = weight

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            mock_get_model.assert_called_once()
            self.assertEqual(self.proposer.attn_layer_names, ["layer3"])
            self.assertIs(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp_gt1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 2
        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]

        mock_model = MagicMock()
        original_embed = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)

            self.assertIsNot(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)
            self.assertEqual(self.proposer.attn_layer_names, ["layer2"])

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    @patch("vllm_ascend.spec_decode.eagle_proposer.supports_multimodal")
    def test_load_model_multimodal(self, mock_supports_multi, mock_pp_group, mock_get_model, mock_get_layers):
        mock_model = MagicMock()
        mock_model.get_language_model.return_value.lm_head = MagicMock()
        mock_supports_multi.return_value = True
        original_embed = MagicMock()
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]
        mock_pp_group.return_value.world_size = 2

        self.proposer.model = MagicMock()

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            self.assertEqual(mock_model.get_language_model.call_count, 2)
            self.assertIs(self.proposer.model.lm_head, mock_model.get_language_model.return_value.lm_head)


class TestEagleProposerDummyRun(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.num_speculative_tokens = 4
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1
        self.runner.pin_memory = False
        self.runner._sync_metadata_across_dp.return_value = (8, torch.tensor([8]), False)

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.model_config.use_mla = False
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(4)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Mock parallel state functions
        self.mock_tp_world_size = patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1
        )
        self.mock_tp_world_size.start()

        mock_dp_group = MagicMock()
        mock_dp_group.world_size = 1
        self.mock_dp_group = patch("vllm_ascend.ascend_forward_context.get_dp_group", return_value=mock_dp_group)
        self.mock_dp_group.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.model = MagicMock()
        self.proposer._runnable = MagicMock()
        self.proposer.update_stream = MagicMock()

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        self.mock_tp_world_size.stop()
        self.mock_dp_group.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # cpu does not support parallel-group, let alone `sp`
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context",
           **{"return_value.flash_comm_v1_enabled": False})
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_basic(self, mock_context, mock_get_context, mock_get_context_2):
        num_tokens = 32
        with_prefill = False

        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=num_tokens, with_prefill=with_prefill)

            self.assertTrue(self.proposer._runnable.call_count == 1)

    # cpu does not support parallel-group, let alone `sp`
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context",
           **{"return_value.flash_comm_v1_enabled": False})
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_with_prefill(self, mock_context, mock_get_context, mock_get_context_2):
        mock_context.return_value.__enter__.return_value = None
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, with_prefill=True, num_reqs=4)
            self.assertTrue(self.proposer._runnable.call_count == 1)

    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_capture(self, mock_context, mock_get_context,
                                        mock_update_full_graph_params, mock_get_context_2):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = True
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=True, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            mock_update_full_graph_params.assert_not_called()
            self.proposer.use_cuda_graph = last_use_cuda_graph
    
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_run(self, mock_context, mock_get_context,
                                    mock_update_full_graph_params, mock_get_context_2):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = False
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        self.proposer.draft_attn_groups = [MagicMock()]
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=False, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            self.assertTrue(mock_update_full_graph_params.call_count == 1)
            self.proposer.use_cuda_graph = last_use_cuda_graph


class TestEagleProposerHelperMethods(TestBase):
    # TODO: Can add some tests about prepare_next_token_ids in future.

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.scheduler_config = MagicMock(max_num_seqs=3)
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = [0, 1, 2]
        self.runner.arange_np = np.arange(10)
        self.runner.input_batch.num_reqs = 3
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # TODO: This is equivalent to disable_padded_drafter_batch=True.
    # We need to add a test_prepare_inputs_padded in future.
    def test_prepare_inputs(self):
        self.proposer.token_arange_np = np.arange(10)
        mock_attn = MagicMock()
        mock_attn.slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
        num_rejected = torch.tensor([1, 0, 1], device=self.device)
        mock_return_attn = MagicMock()

        with (
            set_current_vllm_config(self.vllm_config),
            patch.object(self.proposer, "prepare_inputs", return_value=(mock_return_attn, torch.tensor([1, 2, 4]))),
        ):
            return_attn, indices = self.proposer.prepare_inputs(mock_attn, num_rejected)
            self.assertEqual(indices.tolist(), [1, 2, 4])
