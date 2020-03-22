"""
The attr_computing.py
provides two methods for computing channel Shapley
and transform them to feature attributions.
"""
import numpy as np
import tensorflow as tf
import lucid.optvis.render as render
from lucid.misc.gradient_override import gradient_override_map
import re


def compute_igsg(img, model, attr_class, layers,
                 flag1, flag_read_attr=True,
                 iter_num=100, SG_path=False,
                 labels=None, save_directory=None):
  with tf.Graph().as_default(), tf.Session() as sess, gradient_override_map({}):
    # img = tf.image.resize_image_with_crop_or_pad(img, model.image_shape[0], model.image_shape[0])
    # imgnp = sess.run(img)
    # imgnp = imgnp.reshape(224, 224, 3)
    # plt.imsave("./doghead224.jpeg", imgnp)
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)
    # grads_cam_T = [T(layer) for layer in layers]
    # logit = T("softmax2_pre_activation")[0]

    logit_layer = "resnet_v1_50/predictions/Reshape"
    # writer = tf.summary.FileWriter('./graph_vis/', sess.graph)
    # writer.add_graph(tf.get_default_graph())
    # writer.close()

    # logit = T("output2")[0]
    # score = T("output2")[0, labels.index(attr_class)]
    logit = T(logit_layer)[0]
    logit4grad = T(logit_layer)[0, labels.index(attr_class)]

    AM_T = list(range(len(layers)))
    AM_T_reverse = list(range(len(layers)))

    channel_attr_list = list(range(len(layers)))

    kept_channel_list = list(range(len(layers)))
    kept_channel_list_reverse = list(range(len(layers)))

    for i_wanted_layer in range(len(layers)):
      layer = layers[i_wanted_layer]
      acts = T(layer).eval()
      attr = np.zeros(acts.shape[1:])
      t_grad = tf.gradients([logit4grad], [T(layer)])[0]

      if not flag_read_attr:
        for n in range(iter_num):
          acts_ = acts * float(n) / iter_num
          if SG_path:
            acts_ *= (np.random.uniform(0, 1, [528]) + np.random.uniform(0, 1,
                                                                         [528])) / 1.5
          grad = t_grad.eval({T(layer): acts_})
          attr += grad[0]
        attr = attr * (1.0 / iter_num) * acts[0]
        attr = np.sum(np.sum(attr, 0), 0)
        layer_name = re.sub(r'/', "", layer)
        np.savetxt(save_directory + "/{}_{}_{}.txt".format(flag1, layer_name, attr_class),
                   attr)
      else:
        layer_name = re.sub(r'/', "", layer)
        attr = np.loadtxt(save_directory + "/{}_{}_{}.txt".
                          format(flag1, layer_name, attr_class)).astype(np.float32)
      # AM_T[i_wanted_layer] = attr * (attr > 0)

      kept_channel_idx = np.squeeze(
        np.argwhere(attr > np.nanmean(np.where(attr > 0, attr, np.nan))))
      acts_squeeze = np.squeeze(acts)
      attr_temp = np.squeeze(attr).astype(np.float32)
      # print(np.count_nonzero(clear_channel_idx))
      channel_attr_list[i_wanted_layer] = attr_temp
      AM_T[i_wanted_layer] = acts_squeeze[..., kept_channel_idx]
      kept_channel_list[i_wanted_layer] = kept_channel_idx

      # # alltests_Shap/test0_4
      # kept_channel_idx = np.squeeze(np.argwhere(attr < 0))
      # # alltests_Shap/test0_6
      kept_channel_idx = np.squeeze(
        np.argwhere(attr < np.nanmean(np.where(attr < 0, attr, np.nan))))
      AM_T_reverse[i_wanted_layer] = acts_squeeze[..., kept_channel_idx]
      kept_channel_list_reverse[i_wanted_layer] = kept_channel_idx

    AM_list = AM_T
    logit_list = sess.run([logit])[0]
    return [AM_list, AM_T_reverse], logit_list, channel_attr_list, \
           [kept_channel_list, kept_channel_list_reverse]


def raw_class_group_attr(img, model, layer, label, labels, group_vecs, override=None):
  """
  The method of Gradient * Activation maps
  """

  # Set up a graph for doing attribution...
  with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)

    # Compute activations
    acts = T(layer).eval()

    if label is None:
      return np.zeros(acts.shape[1:-1])

    # Compute gradient
    score = T("softmax2_pre_activation")[0, labels.index(label)]
    t_grad = tf.gradients([score], [T(layer)])[0]
    grad = t_grad.eval({T(layer): acts})

    # Linear approximation of effect of spatial position
    return [np.sum(group_vec * grad) for group_vec in group_vecs]


def score_f(logit, name, labels):
  if name is None:
    return 0
  elif name == "logsumexp":
    base = tf.reduce_max(logit)
    return base + tf.log(tf.reduce_sum(tf.exp(logit - base)))
  elif name in labels:
    return logit[labels.index(name)]
  else:
    raise RuntimeError("Unsupported")
