# Copyright 2020 The T5 Authors.
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

# Lint as: python3
"""T5 Model Abstract Base class."""

import abc
import functools
import os

from absl import logging
import t5.data
from t5.models import utils
import tensorflow.compat.v1 as tf


def get_eval_dataset(mixture_or_task, sequence_length, split):
  eval_datasets = t5.models.mesh_transformer.mesh_eval_dataset_fn(
      sequence_length=sequence_length,
      dataset_split=split,
      mixture_or_task_name=mixture_or_task.name,
  )

  return eval_datasets[0].dataset_fn()


class T5Model(object):
  """Abstract Base class for T5 Model API."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def train(self, mixture_or_task_name, steps):
    raise NotImplementedError()

  def eval(self,
           mixture_or_task_name,
           predict_fn,
           dataset_fn=None,
           checkpoint_steps=None,
           summary_dir=None,
           split="validation",
           compute_sequence_length=True):
    """Evaluate model on the Mixture or Task.

    Args:
      mixture_or_task_name: str, the name of the Mixture or Task to evaluate
        on. Must be pre-registered in the global `TaskRegistry` or
        `MixtureRegistry.`
      predict_fn: function, returns a list of outputs
      dataset_fn: function, get dataset from task or mixture.
      checkpoint_steps: an iterator with integers for checkpoint steps.
      summary_dir: str, path to write TensorBoard events file summaries for
        eval. If None, use model_dir/eval_{split}.
      split: str, the mixture/task split to evaluate on.
      compute_sequence_length: bool, automatically compute sequence length
        during eval mode.
    """
    vocabulary = utils.get_vocabulary(mixture_or_task_name)

    tasks = t5.data.get_subtasks(
        t5.data.get_mixture_or_task(mixture_or_task_name))
    tasks = utils.get_valid_tasks(tasks, split)

    sequence_length = None if compute_sequence_length else self._sequence_length

    if not tasks:
      logging.info(
          "All provided tasks have metric_fns=[]; eval is not possible.")
      return

    if not dataset_fn:
      dataset_fn = functools.partial(
          get_eval_dataset,
          split=split)

    summary_writer = None

    # Need to create a separate graph for loading in plaintext targets
    # or else TF will complain that we modified the graph
    with tf.Graph().as_default():
      cached_examples, cached_targets, max_sequence_length = \
          utils.get_targets_and_examples(
              tasks=tasks,
              dataset_fn=functools.partial(
                  dataset_fn, sequence_length=sequence_length),
              eval_summary_dir=summary_dir)

    if sequence_length is None:
      logging.info("Setting sequence lengths to %s", max_sequence_length)
      sequence_length = max_sequence_length
    elif (sequence_length["inputs"] < max_sequence_length["inputs"] or
          sequence_length["targets"] < max_sequence_length["targets"]):
      logging.warning(
          "Given sequence lengths are insufficient for some evaluation inputs "
          "or targets. These sequences will be truncated to fit, likely leading"
          " to sub-optimal results. Consider passing `None` for sequence_length"
          " to have them be automatically computed.\n Got: %s,\n Max Lengths: "
          "%s", sequence_length, max_sequence_length)
    elif (sequence_length["inputs"] > max_sequence_length["inputs"] or
          sequence_length["targets"] > max_sequence_length["targets"]):
      logging.warning(
          "Given sequence lengths are longer than necessary for some evaluation"
          " inputs or targets, resulting in wasted computation. Consider "
          "passing `None` for sequence_length to have them be automatically "
          "computed.\n Got: %s,\n Max Lengths: %s",
          sequence_length, max_sequence_length)

    for step in checkpoint_steps:
      logging.info("Evaluating checkpoint step: %d", step)
      outputs = predict_fn(
          step=step,
          vocabulary=vocabulary,
          tasks=tasks,
          eval_dataset_fn=functools.partial(
              dataset_fn, sequence_length=sequence_length),
          cached_examples=cached_examples,
          sequence_length=sequence_length)

      for task in tasks:
        # Extract the portion of decodes corresponding to this dataset
        examples = cached_examples[task.name]
        dataset_size = len(examples)

        predictions = [
            task.postprocess_fn(d, example=ex)
            for d, ex in zip(outputs[:dataset_size], examples)
        ]

        # Remove the used decodes.
        del outputs[:dataset_size]

        if summary_dir:
          predictions_filename = os.path.join(
              summary_dir,
              "{}_{}_predictions".format(task.name, step))
          utils.write_lines_to_file(predictions, predictions_filename)

        with tf.Graph().as_default():
          if summary_dir:
            summary_writer = summary_writer or tf.summary.FileWriter(
                summary_dir)

          for metric_fn in task.metric_fns:
            if summary_dir:
              summary = tf.Summary()
            targets = cached_targets[task.name]
            metric_result = metric_fn(targets, predictions)
            for metric_name, metric_value in metric_result.items():
              tag = "eval/{}/{}".format(task.name, metric_name)
              logging.info("%s at step %d: %.3f", tag, step, metric_value)
              if summary_dir:
                summary.value.add(tag=tag, simple_value=metric_value)
                summary_writer.add_summary(summary, step)
          if summary_dir:
            summary_writer.flush()

      # Only padding should remain.
      expected_pad = -sum(len(t)
                          for t in cached_targets.values()) % self.batch_size
      if outputs and len(outputs) != expected_pad:
        raise ValueError("{} padded outputs, {} expected.".format(
            len(outputs), expected_pad))

  @abc.abstractmethod
  def predict(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def finetune(self, mixture_or_task_name, finetune_steps):
    raise NotImplementedError()
