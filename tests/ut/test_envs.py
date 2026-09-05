#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import inspect
import os
from unittest.mock import patch

import vllm_ascend.envs as envs_ascend
from tests.ut.base import TestBase


class TestEnvVariables(TestBase):

    def setUp(self):
        self.env_vars = list(envs_ascend.env_variables.keys())

    def test_env_vars_behavior(self):
        for var_name in self.env_vars:
            with self.subTest(var=var_name):
                original_val = os.environ.get(var_name)
                var_handler = envs_ascend.env_variables[var_name]

                try:
                    if var_name in os.environ:
                        del os.environ[var_name]
                    self.assertEqual(getattr(envs_ascend, var_name),
                                     var_handler())

                    handler_source = inspect.getsource(var_handler)
                    if var_name in (
                        "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE",
                        "VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE",
                    ):
                        test_vals = ["true", "false"]
                    elif 'bool(int(' in handler_source:
                        test_vals = ["0", "1"]
                    elif 'int(' in handler_source:
                        test_vals = ["123", "456"]
                    elif 'float(' in handler_source:
                        test_vals = ["1.25", "2.5"]
                    else:
                        test_vals = [f"test_{var_name}", f"custom_{var_name}"]

                    for test_val in test_vals:
                        os.environ[var_name] = test_val
                        self.assertEqual(getattr(envs_ascend, var_name),
                                         var_handler())

                finally:
                    if original_val is None:
                        os.environ.pop(var_name, None)
                    else:
                        os.environ[var_name] = original_val

    def test_dir_and_getattr(self):
        self.assertEqual(sorted(envs_ascend.__dir__()), sorted(self.env_vars))
        for var_name in self.env_vars:
            with self.subTest(var=var_name):
                getattr(envs_ascend, var_name)

    def test_decode_window_save_commit_delay_windows(self):
        name = "VLLM_ASCEND_LMCACHE_DECODE_WINDOW_SAVE_COMMIT_DELAY_WINDOWS"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(name, None)
            self.assertEqual(getattr(envs_ascend, name), 0)
        with patch.dict(os.environ, {name: "3"}):
            self.assertEqual(getattr(envs_ascend, name), 3)
        with patch.dict(os.environ, {name: "-1"}):
            self.assertEqual(getattr(envs_ascend, name), 0)

    def test_dsa_resident_cache_defaults_on_and_can_be_disabled(self):
        name = "VLLM_ASCEND_DSA_RESIDENT_CACHE"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(name, None)
            self.assertTrue(getattr(envs_ascend, name))
        with patch.dict(os.environ, {name: "0"}):
            self.assertFalse(getattr(envs_ascend, name))

    def test_dsa_resident_shards_per_row_defaults_to_four(self):
        name = "VLLM_ASCEND_DSA_RESIDENT_SHARDS_PER_ROW"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(name, None)
            self.assertEqual(getattr(envs_ascend, name), 4)
        with patch.dict(os.environ, {name: "2"}):
            self.assertEqual(getattr(envs_ascend, name), 2)

    def test_staged_mtp_draft_graph_defaults_off_and_can_be_enabled(self):
        name = "VLLM_ASCEND_SFA_STAGED_MTP_DRAFT_GRAPH"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(name, None)
            self.assertFalse(getattr(envs_ascend, name))
        with patch.dict(os.environ, {name: "1"}):
            self.assertTrue(getattr(envs_ascend, name))

    def test_mtp_acceptance_diag_is_independent_of_decode_window_diag(self):
        name = "VLLM_ASCEND_MTP_ACCEPT_DIAG"
        with patch.dict(os.environ, {"VLLM_ASCEND_MTP_DW_DIAG": "1"}):
            os.environ.pop(name, None)
            self.assertFalse(getattr(envs_ascend, name))
        with patch.dict(os.environ, {name: "1", "VLLM_ASCEND_MTP_DW_DIAG": "0"}):
            self.assertTrue(getattr(envs_ascend, name))
            self.assertFalse(envs_ascend.VLLM_ASCEND_MTP_DW_DIAG)

    def test_layerwise_prefill_p_node_uses_strict_boolean_spelling(self):
        name = "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(name, None)
            self.assertFalse(getattr(envs_ascend, name))
        with patch.dict(os.environ, {name: " TrUe "}):
            self.assertTrue(getattr(envs_ascend, name))
        with (
            patch.dict(os.environ, {name: "1"}),
            self.assertRaisesRegex(ValueError, "must be 'true' or 'false'"),
        ):
            getattr(envs_ascend, name)

    def test_sparse_decode_d_node_uses_strict_boolean_spelling(self):
        name = "VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(name, None)
            self.assertFalse(getattr(envs_ascend, name))
        with patch.dict(os.environ, {name: " TrUe "}):
            self.assertTrue(getattr(envs_ascend, name))
        with (
            patch.dict(os.environ, {name: "1"}),
            self.assertRaisesRegex(ValueError, "must be 'true' or 'false'"),
        ):
            getattr(envs_ascend, name)
