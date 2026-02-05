#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
File: convert_onnx.py
Created Date: January 1st 2020
Author: ZL Deng <dawnmsg(at)gmail.com>
---------------------------------------
Last Modified: 6th December 2020 10:58:42 pm
'''

import click
import torch
import warnings
import numpy as np
from os import path
from ribodetector.model import model_cpu as module_arch
from ribodetector.parse_config import ConfigParser


@click.command()
@click.argument('pth', type=click.Path(exists=True))
@click.option('-l', '--length', type=int, default=100, help='The input read length')
@click.option('--opset', type=int, default=18, help='ONNX opset version (default: 18)')
@click.option('--dynamic-seq/--static-seq', default=False,
              help='Export with dynamic sequence length (default: static)')
def convert_to_onnx(pth, length, opset, dynamic_seq):
    cd = path.dirname(path.abspath(__file__))
    config_file = path.join(cd, 'config.json')
    config = ConfigParser.from_json(config_file)
    logger = config.get_logger('convert', 1)
    model = config.init_obj('arch', module_arch)
    state = torch.load(pth, map_location=torch.device('cpu'))
    state_dict = state['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    vocabulary = list('ACGT')
    encodes = np.eye(len(vocabulary))
    x = encodes[np.random.choice(encodes.shape[0], size=length)]
    input_tensor = torch.FloatTensor(np.expand_dims(x, axis=0))

    exported_onnx_file = pth.rsplit('.', 1)[0] + '.onnx'

    logger.info('Converting to ONNX model: {}'.format(exported_onnx_file))

    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    if dynamic_seq:
        dynamic_axes["input"][1] = "sequence"
        dynamic_axes["output"][1] = "sequence"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(model,
                              input_tensor,
                              exported_onnx_file,
                              export_params=True,
                              opset_version=opset,
                              do_constant_folding=True,
                              input_names=["input"],
                              output_names=["output"],
                              dynamic_axes=dynamic_axes,
                              dynamo=False)
        except TypeError:
            # Older torch versions do not support dynamo kwarg
            torch.onnx.export(model,
                              input_tensor,
                              exported_onnx_file,
                              export_params=True,
                              opset_version=opset,
                              do_constant_folding=True,
                              input_names=["input"],
                              output_names=["output"],
                              dynamic_axes=dynamic_axes)
        except Exception as exc:
            if not dynamic_seq:
                raise
            logger.warning('Dynamic sequence export failed: {}. Retrying with static sequence.'.format(exc))
            static_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            try:
                torch.onnx.export(model,
                                  input_tensor,
                                  exported_onnx_file,
                                  export_params=True,
                                  opset_version=opset,
                                  do_constant_folding=True,
                                  input_names=["input"],
                                  output_names=["output"],
                                  dynamic_axes=static_axes,
                                  dynamo=False)
            except TypeError:
                torch.onnx.export(model,
                                  input_tensor,
                                  exported_onnx_file,
                                  export_params=True,
                                  opset_version=opset,
                                  do_constant_folding=True,
                                  input_names=["input"],
                                  output_names=["output"],
                                  dynamic_axes=static_axes)


if __name__ == '__main__':
    convert_to_onnx()
