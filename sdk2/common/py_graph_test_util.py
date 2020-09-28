import logging
import os
import shutil

import numpy as np
import tensorflow as tf

from sdk2.common import py_test_util
from sdk2.graph import full_graph_pipeline, lgf_graph
from sdk2.graph.export_graph import tf_graph_exporter
from sdk2.graph.import_graph import tf_saved_model_importer
from sdk2.graph.run_graph import graph_runner
from sdk2.graph.transform_graph import utils
from sdk2.graph.transform_graph.graph_transformers import fold_phasify_constants
from sdk2.graph.transform_graph.node_transformers.generic_transforms import (
    matmul_transform, opu_op_transform)
from sdk2.proto import dtypes_pb2, graph_types_pb2, inference_pb2, lgf_pb2
from sdk2.proto.configs import generate_hw_specs, generate_sim_params, generate_sw_config


class GraphTestCase(py_test_util.PythonTestCase):
    """Common functions used by tests involving graphs."""

    @staticmethod
    def relative_error(expected, measured):
        numerator = np.mean(np.abs(expected - measured))
        denominator = np.mean(np.abs(expected)) + np.finfo(np.float32).eps
        error = numerator / denominator
        logging.info("***ERROR: {0}".format(error))
        return error

    def inference_out_pb_to_array_list(self, out_pb, output_order=None):
        if output_order is None:
            assert len(out_pb.results) == 1
            return [
                utils.tensor_pb_to_array(
                    out_pb.results[0].data,
                    utils.dtype_pb_to_np_dtype(out_pb.results[0].data.dtype))
            ]

        array_dict = {}
        for named_tensor in out_pb.results:
            array_dict[named_tensor.edge_info.name + ":" +
                       str(named_tensor.edge_info.port)] = utils.tensor_pb_to_array(
                           named_tensor.data,
                           utils.dtype_pb_to_np_dtype(named_tensor.data.dtype))

        return [array_dict[name] for name in output_order]

    def inference_on_graph(self,
                           graph_path,
                           test_data_dict,
                           hw_specs,
                           sw_config,
                           sim_params,
                           graph_coll=None):
        lgf_graph_pb = lgf_graph.LightGraph.read_lgf_pb(graph_path)
        light_graph = lgf_graph.LightGraph.lgf_pb_to_graph(lgf_graph_pb)
        test_data_list = [
            test_data_dict[k.name + ":" + str(k.port)]
            for k in light_graph.input_edges()
        ]
        test_data = utils.create_inference_inputs(light_graph.input_edges(),
                                                  test_data_list)

        runner = graph_runner.GraphRunner(light_graph,
                                          hw_specs,
                                          sw_config,
                                          sim_params,
                                          graph_coll=graph_coll)
        return runner.run_single_batch(test_data)

    def get_modified_graph(self, saved_model_dir, calibration_data_dict, hw_specs,
                           sw_config, sim_params):
        light_graph = tf_saved_model_importer.ImportTFSavedModel(
            saved_model_dir, sw_config).as_light_graph()
        calibration_data_list = [
            calibration_data_dict[k.name + ":" + str(k.port)]
            for k in light_graph.input_edges()
        ]
        calibration_data = inference_pb2.BatchedInferenceInput()
        calibration_data.batches.add().CopyFrom(
            utils.create_inference_inputs(light_graph.input_edges(),
                                          calibration_data_list))
        output_path = os.path.join(self.tmp_dir, "output_lgf.pb")

        full_graph_pipeline.main(saved_model_dir, graph_types_pb2.TFSavedModel,
                                 output_path, graph_types_pb2.LGFProtobuf,
                                 calibration_data, hw_specs, sw_config, sim_params)
        return output_path

    def measure_modified_output(self,
                                saved_model_dir,
                                calibration_data_dict,
                                test_data_dict,
                                hw_specs=None,
                                sw_config=None,
                                sim_params=None,
                                output_path=None,
                                output_order=None):
        if hw_specs is None:
            hw_specs = generate_hw_specs.generate_mosaic()
        if sw_config is None:
            sw_config = generate_sw_config.generate_standard_sw_config(
                graph_types_pb2.TFSavedModel)
        if sim_params is None:
            sim_params = generate_sim_params.generate_mosaic()

        self._extra_preprocessing(hw_specs, sw_config, sim_params)

        if output_path is None:
            output_path = self.get_modified_graph(saved_model_dir, calibration_data_dict,
                                                  hw_specs, sw_config, sim_params)

        out_pb = self.inference_on_graph(output_path, test_data_dict, hw_specs,
                                         sw_config, sim_params)
        return self.inference_out_pb_to_array_list(out_pb, output_order), output_path

    def test_modifier(func):

        def wrapper(self, *args, **kwargs):
            args = tuple([self] + list(args))
            return self._preprocess_args(func, *args, **kwargs)

        return wrapper

    def _preprocess_args(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    @test_modifier
    def _test_general_matmul_helper(self,
                                    threshold,
                                    input_tensor_shape=[None, 30],
                                    weights_shape=[30, 30],
                                    calib_data_shape=[10, 30],
                                    hw_specs=None,
                                    sw_config=None,
                                    sim_params=None,
                                    skip_extra_tests=False,
                                    test_data=None,
                                    transpose_weights=False):
        """A helper function for MatMul-related unit tests.

        The parameters M, N and P determine shapes of weights and inputs in the matmul
        operation. If transpose_weights is True, the last two dims of weights will be
        transposed before matmul. If with_extra_dims is True, an extra dim will be
        prepended to both inputs and weights and the calculation result will also have
        that extra dim.
        """
        for _ in range(3):
            calibration_data = np.random.uniform(-5, 10, size=calib_data_shape).astype(
                np.float32)
            test_data = calibration_data if test_data is None else test_data
            original_graph = tf.Graph()
            saved_model_dir = os.path.join(self.tmp_dir, "original")
            if os.path.exists(saved_model_dir):
                shutil.rmtree(saved_model_dir)

            with original_graph.as_default():
                inp_tensor = tf.placeholder(tf.float32,
                                            shape=input_tensor_shape,
                                            name="input")
                w = tf.get_variable("weights",
                                    shape=weights_shape,
                                    dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(-10, 15))
                mm = tf.matmul(inp_tensor,
                               w,
                               transpose_b=transpose_weights,
                               name="matmul_of_interest")
                res = tf.identity(mm, name="output")
                with tf.Session(graph=original_graph) as sess:
                    sess.run(tf.global_variables_initializer())
                    expected = sess.run(res, feed_dict={inp_tensor: test_data})

                    tf_graph_exporter.ExportTFSavedModel.save_model(
                        saved_model_dir, sess, [inp_tensor], [res])

            measured, output_path = self.measure_modified_output(
                saved_model_dir, {inp_tensor.name: calibration_data},
                {inp_tensor.name: test_data},
                hw_specs=hw_specs,
                sw_config=sw_config,
                sim_params=sim_params)
            error = self.relative_error(expected, measured[0])
            self.assertTrue(error <= threshold)
            self._extra_checks(measured, saved_model_dir,
                               {inp_tensor.name: calibration_data},
                               {inp_tensor.name: test_data}, hw_specs, sw_config,
                               sim_params, output_path)

            if skip_extra_tests:
                break

    def _extra_checks(self, measured, saved_model_dir, calibration_data_dict,
                      test_data_dict, hw_specs, sw_config, sim_params, output_path):
        pass

    def _extra_preprocessing(self, hw_specs, sw_config, sim_params):
        pass

    def _get_conv2d_choices(self,
                            input_channels_choices=None,
                            strides_choices=None,
                            padding_choices=None,
                            kernel_choices=None):
        input_channels_choices = input_channels_choices or [1, 4, 7, 64, 256]
        strides_choices = strides_choices or [[1, 1, 1, 1], [1, 2, 2, 1], [1, 3, 3, 1]]
        padding_choices = padding_choices or ["SAME", "VALID"]
        kernel_choices = kernel_choices or [(3, 3), (4, 4)]
        for input_channels in input_channels_choices:
            output_channels_choices = set(
                [1, max(input_channels // 2, 1), input_channels, input_channels * 2])
            for output_channels in output_channels_choices:
                for strides in strides_choices:
                    for padding in padding_choices:
                        for kernel_height, kernel_width in kernel_choices:
                            yield ((input_channels, output_channels, strides, padding,
                                    kernel_height, kernel_width))

    @test_modifier
    def _test_oconv2d_helper(self,
                             threshold,
                             hw_specs=None,
                             sw_config=None,
                             sim_params=None,
                             skip_extra_tests=False,
                             B=5,
                             N=7,
                             input_channels_choices=None,
                             test_data=None):
        all_choices = self._get_conv2d_choices(
            input_channels_choices=input_channels_choices)
        for (input_channels, output_channels, strides, padding, kernel_height,
             kernel_width) in all_choices:
            calibration_data = np.random.uniform(
                -5, 10, size=(B, N * N * input_channels)).astype(np.float32)
            test_data_to_use = calibration_data if test_data is None else test_data
            original_graph = tf.Graph()
            saved_model_dir = os.path.join(self.tmp_dir, "original")
            if os.path.exists(saved_model_dir):
                shutil.rmtree(saved_model_dir)

            with original_graph.as_default():
                inp_tensor = tf.placeholder(tf.float32,
                                            shape=[None, N * N * input_channels],
                                            name="input")
                x = tf.reshape(inp_tensor, [-1, N, N, input_channels])

                w = tf.get_variable(
                    "W", [kernel_height, kernel_width, input_channels, output_channels],
                    initializer=tf.random_normal_initializer(-15, 15))

                c = tf.nn.conv2d(x, w, strides, padding)
                res = tf.identity(c, name="output")
                with tf.Session(graph=original_graph) as sess:
                    sess.run(tf.global_variables_initializer())
                    expected = sess.run(res, feed_dict={inp_tensor: test_data_to_use})
                    tf_graph_exporter.ExportTFSavedModel.save_model(
                        saved_model_dir, sess, [inp_tensor], [res])

            if sw_config is None:
                sw_config = generate_sw_config.generate_standard_sw_config(
                    graph_types_pb2.TFSavedModel)
            sw_config.disable_block_sparsity = True

            measured, output_path = self.measure_modified_output(
                saved_model_dir, {inp_tensor.name: calibration_data},
                {inp_tensor.name: test_data_to_use},
                hw_specs=hw_specs,
                sw_config=sw_config,
                sim_params=sim_params)
            error = self.relative_error(expected, measured[0])
            self.assertTrue(error <= threshold)
            self._extra_checks(measured, saved_model_dir,
                               {inp_tensor.name: calibration_data},
                               {inp_tensor.name: test_data_to_use}, hw_specs, sw_config,
                               sim_params, output_path)

            if skip_extra_tests:
                break

    @test_modifier
    def _test_depthwise_conv2d_helper(self,
                                      threshold,
                                      hw_specs=None,
                                      sw_config=None,
                                      sim_params=None,
                                      skip_extra_tests=False,
                                      B=4,
                                      N=7,
                                      input_channels_choices=[4],
                                      kernel_choices=None,
                                      test_data=None):
        channel_multiplier = 2
        all_choices = self._get_conv2d_choices(
            input_channels_choices=input_channels_choices, kernel_choices=kernel_choices)
        for (input_channels, _, strides, padding, kernel_height,
             kernel_width) in all_choices:
            calibration_data = np.random.uniform(
                -5, 10, size=(B, N * N * input_channels)).astype(np.float32)
            test_data_to_use = calibration_data if test_data is None else test_data
            original_graph = tf.Graph()
            saved_model_dir = os.path.join(self.tmp_dir, "original")
            if os.path.exists(saved_model_dir):
                shutil.rmtree(saved_model_dir)

            with original_graph.as_default():
                inp_tensor = tf.placeholder(tf.float32,
                                            shape=[None, N * N * input_channels],
                                            name="input")
                x = tf.reshape(inp_tensor, [-1, N, N, input_channels])

                w = tf.get_variable(
                    "W",
                    [kernel_height, kernel_width, input_channels, channel_multiplier],
                    initializer=tf.random_normal_initializer(-15, 15))

                c = tf.nn.depthwise_conv2d(x, w, strides, padding)
                res = tf.identity(c, name="output")
                with tf.Session(graph=original_graph) as sess:
                    sess.run(tf.global_variables_initializer())
                    expected = sess.run(res, feed_dict={inp_tensor: test_data_to_use})
                    tf_graph_exporter.ExportTFSavedModel.save_model(
                        saved_model_dir, sess, [inp_tensor], [res])

            measured, output_path = self.measure_modified_output(
                saved_model_dir, {inp_tensor.name: calibration_data},
                {inp_tensor.name: test_data_to_use},
                hw_specs=hw_specs,
                sw_config=sw_config,
                sim_params=sim_params)
            error = self.relative_error(expected, measured[0])
            self.assertTrue(error <= threshold)
            self._extra_checks(measured, saved_model_dir,
                               {inp_tensor.name: calibration_data},
                               {inp_tensor.name: test_data_to_use}, hw_specs, sw_config,
                               sim_params, output_path)

            if skip_extra_tests:
                break


class SimpleGraphs(py_test_util.PythonTestCase):

    @staticmethod
    def sv_max_graph(
            inp_shape=(1, 100), inp_dtype_t=dtypes_pb2.DT_QINT, inp_dtype_p=8,
            num_nodes=1):
        inp1 = lgf_pb2.EdgeInfo()
        inp1.name = "inp_ten1"
        inp1.port = 0
        inp1.dtype.t = inp_dtype_t
        inp1.dtype.p = inp_dtype_p
        inp1.shape.d.extend(inp_shape)

        nodes = []
        last_edge = inp1
        for i in range(num_nodes):
            outp = lgf_pb2.EdgeInfo()
            outp.CopyFrom(inp1)
            outp.name = "out_ten_{0}".format(i)

            n = lgf_pb2.LNF()
            n.name = outp.name
            n.sv_max.SetInParent()
            n.inputs.add().CopyFrom(last_edge)
            n.outputs.add().CopyFrom(outp)
            n.sv_max.scalar = 64
            n.supported = True
            last_edge = n.outputs[0]
            nodes.append(n)

        return lgf_graph.LightGraph(nodes, input_edges=[inp1], output_edges=[last_edge])

    # TODO: Get rid of this, use opu_op_transform code.
    @staticmethod
    def _setup_opu_node(opu_node, input_edge, weights_edge, output_edge, spec,
                        sw_config):
        matmul = opu_op_transform.OPUOpTransform.get_matmul_from_opu_node(opu_node)

        # Turn of adc so we get floating point result
        matmul.turn_off_adc = True
        matmul.hist_keys_before_adc.SetInParent()
        matmul.hist_keys_after_adc.SetInParent()
        matmul.dequant_method = lgf_pb2.DQ_STANDARD

        # Make sure quantize does not do anything
        matmul.quant_precision = spec.input_precision
        matmul.using_quant_bias = False
        matmul.phasify_is_folded = True

        # Create constant nodes that are inputs to opu node
        quant_params = np.array([1, 0])
        quant_params_node = opu_op_transform.OPUOpTransform.create_const_node(
            quant_params, "quant_params_scales", sw_config.float_type,
            lgf_pb2.ConstNode.GRAPH_CONST)

        num_x, num_y, _, _, _ = weights_edge.shape.d
        dequant_scales = np.ones(shape=(num_x, num_y, 1, 1))
        dequant_scales_node = opu_op_transform.OPUOpTransform.create_const_node(
            dequant_scales, "dequant_scales", sw_config.float_type,
            lgf_pb2.ConstNode.DEQUANT_SCALE)

        adc_scales = np.ones(shape=(num_x, num_y, 1, 1))
        adc_scales_node = opu_op_transform.OPUOpTransform.create_const_node(
            adc_scales, "adc_scales", sw_config.float_type, lgf_pb2.ConstNode.ADC_SCALE)

        # Inputs and outputs for OPU node
        for _ in range(opu_op_transform.OPUOpTransform.NUM_INPUTS):
            opu_node.inputs.add()

        opu_node.inputs[lgf_pb2.MatMulNode.INPUT_INDEX].CopyFrom(input_edge)
        opu_node.inputs[lgf_pb2.MatMulNode.QUANT_PARAMS_INDEX].CopyFrom(
            quant_params_node.outputs[0])
        opu_node.inputs[lgf_pb2.MatMulNode.PHASES_INDEX].CopyFrom(weights_edge)
        opu_node.inputs[lgf_pb2.MatMulNode.DEQUANT_SCALES_INDEX].CopyFrom(
            dequant_scales_node.outputs[0])
        opu_node.inputs[lgf_pb2.MatMulNode.ADC_SCALES_INDEX].CopyFrom(
            adc_scales_node.outputs[0])

        opu_node.outputs.add().CopyFrom(output_edge)

        return [quant_params_node, dequant_scales_node, adc_scales_node]

    @staticmethod
    def _pad_and_reshape(weights, spec):
        # Pad to even multiple of opu dimension.
        weights_edge = lgf_pb2.EdgeInfo()
        weights_edge.shape.d.extend(weights.shape)
        weights_edge.shape.batch_dim_indx = -1
        weights_edge.dtype.CopyFrom(utils.np_dtype_to_lgf_dtype(weights.dtype))

        num_x, num_y, k, j = opu_op_transform.OPUOpTransform.get_tiled_shape(
            weights_edge, False, spec)
        W_dim = weights.shape

        pad_x = (k * num_x) - W_dim[0]
        pad_y = (j * num_y) - W_dim[1]
        pad = [[0, pad_x], [0, pad_y]]
        W_padded = np.pad(weights, pad, "constant", constant_values=0)

        # Convert [num_x * k, num_y * j] W_padded to full_matrix of size
        # [num_x, num_y, k, j]
        full_matrix = np.split(W_padded, num_x, axis=0)
        full_matrix = np.stack(full_matrix, axis=0)
        full_matrix = np.split(full_matrix, num_y, axis=2)
        full_matrix = np.stack(full_matrix, axis=1)

        # Convert to [num_x, num_y, num_dps, k, k]
        full_matrix = full_matrix.reshape(num_x, num_y, k, j // k, k)
        full_matrix = np.transpose(full_matrix, axes=[0, 1, 3, 2, 4])

        return full_matrix

    @staticmethod
    def matmul_graph(hw_spec,
                     sw_config,
                     sim_params,
                     weights,
                     inp_shape=(1, 4),
                     inp_dtype_t=dtypes_pb2.DT_BFLOAT,
                     inp_dtype_p=16,
                     add_activation=False):
        inp1 = lgf_pb2.EdgeInfo()
        inp1.name = "inp_ten1"
        inp1.port = 0
        inp1.dtype.t = inp_dtype_t
        inp1.dtype.p = inp_dtype_p
        inp1.shape.d.extend(list(inp_shape))

        weights_i = lgf_pb2.EdgeInfo()
        weights_i.name = "weights_ten"
        weights_i.port = 0
        weights_i.dtype.CopyFrom(inp1.dtype)
        weights_i.shape.d.extend(weights.shape)

        outp = lgf_pb2.EdgeInfo()
        outp.CopyFrom(inp1)
        outp.name = "out_ten"
        outp.shape.d[1] = weights.shape[1]
        outp.dtype.t = inp_dtype_t
        outp.dtype.p = inp_dtype_p

        wn = lgf_pb2.LNF()
        wn.name = "weights_ten"
        wn.const.SetInParent()
        wn.outputs.add().CopyFrom(weights_i)
        wn.const.value.CopyFrom(utils.array_to_tensor_pb(weights, weights_i.dtype))
        wn.const.const_type = lgf_pb2.ConstNode.GRAPH_CONST
        wn.supported = True

        mm_tx = matmul_transform.MatMulTransform(hw_spec, sw_config, sim_params)
        mm_nodes = mm_tx.create_supported_nodes("out_ten", inp1, weights_i, outp, [])

        act_nodes = []
        if add_activation:
            act = lgf_pb2.LNF()
            act.name = "act"
            act.sv_max.SetInParent()
            act.supported = True
            act.inputs.add().CopyFrom(mm_nodes[0].outputs[0])
            act.outputs.add().CopyFrom(mm_nodes[0].outputs[0])
            act.outputs[0].name = "act"
            outp = act.outputs[0]
            act_nodes.append(act)

        lg = lgf_graph.LightGraph([wn] + mm_nodes + act_nodes,
                                  input_edges=[inp1],
                                  output_edges=[outp])

        folder = fold_phasify_constants.FoldPhasifyConstants(hw_spec, sw_config,
                                                             sim_params)

        return folder.process_transforms(lg)

    @staticmethod
    def _get_conv2d_output_shape(batch, input_height, input_width, out_chan, ksize,
                                 strides, padding, data_format):
        if data_format == "NHWC":
            b_indx, h_indx, w_indx, c_indx = 0, 1, 2, 3
        elif data_format == "NCHW":
            b_indx, c_indx, h_indx, w_indx = 0, 1, 2, 3
        else:
            raise ValueError("Invalid data format: {}".format(data_format))

        if "SAME" in padding:
            output_height = np.ceil(input_height / strides[h_indx]).astype(int)
            output_width = np.ceil(input_width / strides[w_indx]).astype(int)
        elif padding == "VALID":
            output_height = np.ceil(
                (input_height - ksize[h_indx] + 1) / strides[h_indx]).astype(int)
            output_width = np.ceil(
                (input_width - ksize[w_indx] + 1) / strides[w_indx]).astype(int)
        else:
            raise ValueError("Invlid padding: {}".format(padding))

        output_shape = [0] * 4
        output_shape[b_indx] = batch
        output_shape[h_indx] = output_height
        output_shape[w_indx] = output_width
        output_shape[c_indx] = out_chan

        return output_shape
