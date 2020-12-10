#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
File: convert_onnx.py
Created Date: January 1st 2020
Author: ZL Deng <dawnmsg(at)gmail.com>
---------------------------------------
Last Modified: 6th December 2020 10:58:42 pm
'''


# import onnx
# import onnxruntime
import click
import torch
import warnings
import numpy as np
import model.model as module_arch
from parse_config import ConfigParser


@click.command()
@click.argument('pth', type=click.Path(exists=True))
@click.option('-l', '--length', type=int, default=100, help='The input read length')
def convert_to_onnx(pth, length):
    config = ConfigParser.from_json('config.json')
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model,
                          input_tensor,
                          exported_onnx_file,
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"},
                                        "output": {0: "batch_size"}})


if __name__ == '__main__':
    convert_to_onnx()
