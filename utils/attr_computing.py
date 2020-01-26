"""
The attr_computing.py
provides two methods for computing channel Shapley
and transform them to feature attributions.
"""
import numpy as np
import tensorflow as tf
import lucid.optvis.render as render
from lucid.misc.gradient_override import gradient_override_map


def compute_shap(img, model, attr_class, layers,
                  flag1, flag_read_attr=True,
                  iter_num=2 ** 8, labels=None, save_directory=None):
  """
  Using sampling based Shapley method to compute feature attributions
  """
  with tf.Graph().as_default(), tf.Session() as sess, gradient_override_map({}):
    # img = tf.image.resize_image_with_crop_or_pad(img, model.image_shape[0], model.image_shape[0])
    # imgnp = sess.run(img)
    # imgnp = imgnp.reshape(224, 224, 3)
    # plt.imsave("./doghead224.jpeg", imgnp)
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)
    # grads_cam_T = [T(layer) for layer in layers]
    # logit = T("softmax2_pre_activation")[0]

    # score = T("output2")[0, labels.index(attr_class)]
    logit = T("softmax2_pre_activation")[0]
    AM_T = list(range(len(layers)))
    AM_T_reverse = list(range(len(layers)))

    channel_attr_list = list(range(len(layers)))

    kept_channel_list = list(range(len(layers)))
    kept_channel_list_reverse = list(range(len(layers)))
    ori_logit = logit.eval()

    y_label = np.zeros_like(ori_logit)
    y_label[labels.index(attr_class)] = 1

    # index_class_logit = ori_logit[labels.index(attr_class)]
    # detected_label_index = ori_logit.argmax()
    # print("detected label index: {}, real label index: {}, label name: {}"
    #       .format(detected_label_index, labels.index(attr_class), attr_class))
    for i_wanted_layer in range(len(layers)):
      layer = layers[i_wanted_layer]
      acts = T(layer).eval()
      acts_shape = list(acts.shape)
      # part_name = "import/{}:0".format(layer)
      # t_part_input = tf.placeholder(acts.dtype, acts_shape)
      # T_part = import_part_model(model, t_part_input, part_name)
      # part_logit = T_part("softmax2_pre_activation")[0]

      n_features = acts.shape[-1]

      if not flag_read_attr:
        result = np.zeros((1, n_features))
        run_shape = acts_shape.copy()
        # run_shape = np.delete(run_shape, -1).tolist()
        # run_shape.insert(-1, -1)
        reconstruction_shape = [1, acts_shape[-1]]

        for r in range(iter_num):
          p = np.random.permutation(n_features)
          x = acts.copy().reshape(run_shape)

          y = None
          for i in p:
            if y is None:
              y = logit.eval({T(layer): x})
              # y = model.predict(x.reshape(acts_shape))
            x[..., i] = 0
            y0 = logit.eval({T(layer): x})
            # print("Ori logit score: {}, new logit score: {}"
            #       .format(index_class_logit, y0[labels.index(attr_class)]))
            # y0 = model.predict(x.reshape(acts_shape))
            assert y0.shape == y_label.shape, y0.shape
            prediction_delta = np.sum((y - y0) * y_label)
            # if i == 139:
            #   print("AM 139: attr is {}".format(prediction_delta))
            result[:, i] += prediction_delta
            y = y0
        attr = np.squeeze(
          (result.copy() / iter_num).reshape(reconstruction_shape).astype(np.float32))
        np.savetxt(save_directory + "/{}_{}_{}.txt".format(flag1, layer, attr_class), attr)
      else:
        attr = np.loadtxt(save_directory + "/{}_{}_{}.txt".format(flag1, layer, attr_class)).astype(np.float32)

      # arg_attr = attr.argsort()[::-1][:]
      # shap_attr_trans = np.transpose(shap_attr, (2, 0, 1))
      # acts_squeeze_trans = np.transpose(acts_squeeze, (2, 0, 1))

      '''
      # # Use it for debug the attribution maps
      # sort_AMs_idx_b2s = np.argsort(-attr)
      # sort_AMs_idx_s2b = np.argsort(attr)
      # for i_sort_AMs in range(20):
      #   if i_sort_AMs < 10:
      #     AM_backup_idx = sort_AMs_idx_b2s[i_sort_AMs]
      #     print("big shap value: {:+.3f} in channel {} of all {}"
      #           .format(attr[AM_backup_idx], AM_backup_idx, acts.shape[-1]))
      #     print("    for class: {}, image No:{}"
      #           .format(attr_class, i_sort_AMs))
      #   else:
      #     AM_backup_idx = sort_AMs_idx_s2b[i_sort_AMs-10]
      #     print("small shap value: {:+.3f} in channel {} of all {}"
      #           .format(attr[AM_backup_idx], AM_backup_idx, acts.shape[-1]))
      #     print("    for class: {}, image No:{}"
      #           .format(attr_class, i_sort_AMs-10))
      #   AM_backup = acts[..., AM_backup_idx]
      #   AM_backup = AM_backup.reshape([AM_backup.shape[-1], AM_backup.shape[-1]])
      #   AM_backup = resize(AM_backup, (model.image_shape[0], model.image_shape[1]), order=1,
      #                            mode='constant', anti_aliasing=False)
      #   AM_backup = AM_backup / AM_backup.max() * 255
      #   resize_show(AM_backup, xi=img)
      # kept_channel_idx = np.squeeze(np.argwhere(attr > 0))
      '''

      kept_channel_idx = np.squeeze(np.argwhere(attr > np.nanmean(np.where(attr > 0, attr, np.nan))))
      acts_squeeze = np.squeeze(acts)
      attr_temp = np.squeeze(attr).astype(np.float32)
      # print(np.count_nonzero(clear_channel_idx))
      channel_attr_list[i_wanted_layer] = attr_temp
      AM_T[i_wanted_layer] = acts_squeeze[..., kept_channel_idx]
      kept_channel_list[i_wanted_layer] = kept_channel_idx

      kept_channel_idx = np.squeeze(np.argwhere(attr < np.nanmean(np.where(attr < 0, attr, np.nan))))
      AM_T_reverse[i_wanted_layer] = acts_squeeze[..., kept_channel_idx]
      kept_channel_list_reverse[i_wanted_layer] = kept_channel_idx
      # test_AM = acts_squeeze * clear_channel_idx * attr_temp
      # test_AM = np.sum(np.transpose(test_AM, (2, 0, 1)), axis=0)
      # acts_squeeze_trans = np.transpose(AM_T[i_wanted_layer], (2, 0, 1))
      # acts_squeeze_trans_sum = np.sum(acts_squeeze_trans, axis=0)

    AM_list = AM_T
    logit_list = sess.run([logit])[0]
    return [AM_list, AM_T_reverse], logit_list, channel_attr_list, \
           [kept_channel_list, kept_channel_list_reverse]


def compute_igsg(img, model, attr_class, layers,
                    flag1, flag_read_attr=True,
                    iter_num=100, SG_path=False,
                    labels=None, save_directory=None):
  """
  Using IG method to compute feature attributions
  """
  with tf.Graph().as_default(), tf.Session() as sess, gradient_override_map({}):
    # img = tf.image.resize_image_with_crop_or_pad(img, model.image_shape[0], model.image_shape[0])
    # imgnp = sess.run(img)
    # imgnp = imgnp.reshape(224, 224, 3)
    # plt.imsave("./doghead224.jpeg", imgnp)
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)
    # grads_cam_T = [T(layer) for layer in layers]
    # logit = T("softmax2_pre_activation")[0]

    # logit = T("output2")[0]
    # score = T("output2")[0, labels.index(attr_class)]
    logit = T("softmax2_pre_activation")[0]
    logit4grad = T("softmax2_pre_activation")[0, labels.index(attr_class)]

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
            acts_ *= (np.random.uniform(0, 1, [528]) + np.random.uniform(0, 1, [528])) / 1.5
          grad = t_grad.eval({T(layer): acts_})
          attr += grad[0]
        attr = attr * (1.0 / iter_num) * acts[0]
        attr = np.sum(np.sum(attr, 0), 0)
        np.savetxt(save_directory + "/{}_{}_{}.txt".format(flag1, layer, attr_class), attr)
      else:
        attr = np.loadtxt(save_directory + "/{}_{}_{}.txt".
                          format(flag1, layer, attr_class)).astype(np.float32)
      # AM_T[i_wanted_layer] = attr * (attr > 0)

      kept_channel_idx = np.squeeze(np.argwhere(attr > np.nanmean(np.where(attr > 0, attr, np.nan))))
      acts_squeeze = np.squeeze(acts)
      attr_temp = np.squeeze(attr).astype(np.float32)
      # print(np.count_nonzero(clear_channel_idx))
      channel_attr_list[i_wanted_layer] = attr_temp
      AM_T[i_wanted_layer] = acts_squeeze[..., kept_channel_idx]
      kept_channel_list[i_wanted_layer] = kept_channel_idx

      # # alltests_Shap/test0_4
      # kept_channel_idx = np.squeeze(np.argwhere(attr < 0))
      # # alltests_Shap/test0_6
      kept_channel_idx = np.squeeze(np.argwhere(attr < np.nanmean(np.where(attr < 0, attr, np.nan))))
      AM_T_reverse[i_wanted_layer] = acts_squeeze[..., kept_channel_idx]
      kept_channel_list_reverse[i_wanted_layer] = kept_channel_idx

    AM_list = AM_T
    logit_list = sess.run([logit])[0]
    return [AM_list, AM_T_reverse], logit_list, channel_attr_list, \
         [kept_channel_list, kept_channel_list_reverse]


def compute_all_am_shap(img, model, attr_class, layers,
                    flag1, flag_read_attr=True,
                    iter_num=2 ** 8, labels=None, save_directory=None):
  """
  Using sampling based Shapley method to compute feature attributions
  Return all attributions and AM not just positive or negative ones
  """
  with tf.Graph().as_default(), tf.Session() as sess, gradient_override_map({}):
    # img = tf.image.resize_image_with_crop_or_pad(img, model.image_shape[0], model.image_shape[0])
    # imgnp = sess.run(img)
    # imgnp = imgnp.reshape(224, 224, 3)
    # plt.imsave("./doghead224.jpeg", imgnp)
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)
    # grads_cam_T = [T(layer) for layer in layers]
    # logit = T("softmax2_pre_activation")[0]

    # score = T("output2")[0, labels.index(attr_class)]
    logit = T("softmax2_pre_activation")[0]
    AM_T = list(range(len(layers)))
    channel_attr_list = list(range(len(layers)))
    ori_logit = logit.eval()

    y_label = np.zeros_like(ori_logit)
    y_label[labels.index(attr_class)] = 1

    # index_class_logit = ori_logit[labels.index(attr_class)]
    # detected_label_index = ori_logit.argmax()
    # print("detected label index: {}, real label index: {}, label name: {}"
    #       .format(detected_label_index, labels.index(attr_class), attr_class))
    for i_wanted_layer in range(len(layers)):
      layer = layers[i_wanted_layer]
      acts = T(layer).eval()
      acts_shape = list(acts.shape)
      # part_name = "import/{}:0".format(layer)
      # t_part_input = tf.placeholder(acts.dtype, acts_shape)
      # T_part = import_part_model(model, t_part_input, part_name)
      # part_logit = T_part("softmax2_pre_activation")[0]

      n_features = acts.shape[-1]

      if not flag_read_attr:
        result = np.zeros((1, n_features))
        run_shape = acts_shape.copy()
        # run_shape = np.delete(run_shape, -1).tolist()
        # run_shape.insert(-1, -1)
        reconstruction_shape = [1, acts_shape[-1]]

        for r in range(iter_num):
          p = np.random.permutation(n_features)
          x = acts.copy().reshape(run_shape)

          y = None
          for i in p:
            if y is None:
              y = logit.eval({T(layer): x})
              # y = model.predict(x.reshape(acts_shape))
            x[..., i] = 0
            y0 = logit.eval({T(layer): x})
            # print("Ori logit score: {}, new logit score: {}"
            #       .format(index_class_logit, y0[labels.index(attr_class)]))
            # y0 = model.predict(x.reshape(acts_shape))
            assert y0.shape == y_label.shape, y0.shape
            prediction_delta = np.sum((y - y0) * y_label)
            result[:, i] += prediction_delta
            y = y0
        attr = np.squeeze(
          (result.copy() / iter_num).reshape(reconstruction_shape).astype(np.float32))
        np.savetxt(save_directory + "/{}_{}_{}.txt".format(flag1, layer, attr_class), attr)
      else:
        attr = np.loadtxt(save_directory + "/{}_{}_{}.txt".format(flag1, layer, attr_class)).astype(np.float32)

      acts_squeeze = np.squeeze(acts)
      attr_temp = np.squeeze(attr).astype(np.float32)
      channel_attr_list[i_wanted_layer] = attr_temp
      AM_T[i_wanted_layer] = acts_squeeze

    AM_list = AM_T
    logit_list = sess.run([logit])[0]
    return AM_list, logit_list, channel_attr_list


def compute_all_am_igsg(img, model, attr_class, layers,
                        flag1, flag_read_attr=True,
                        iter_num=2 ** 8, SG_path=False,
                        labels=None, save_directory=None):
  """
  Using IG method to compute feature attributions
  Return all attributions and AM not just positive or negative ones
  """
  with tf.Graph().as_default(), tf.Session() as sess, gradient_override_map({}):
    # img = tf.image.resize_image_with_crop_or_pad(img, model.image_shape[0], model.image_shape[0])
    # imgnp = sess.run(img)
    # imgnp = imgnp.reshape(224, 224, 3)
    # plt.imsave("./doghead224.jpeg", imgnp)
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)
    # grads_cam_T = [T(layer) for layer in layers]
    # logit = T("softmax2_pre_activation")[0]

    # score = T("output2")[0, labels.index(attr_class)]
    logit = T("softmax2_pre_activation")[0]
    logit4grad = T("softmax2_pre_activation")[0, labels.index(attr_class)]
    AM_T = list(range(len(layers)))
    channel_attr_list = list(range(len(layers)))

    ori_logit = logit.eval()
    ori_single_logit = logit4grad.eval()

    y_label = np.zeros_like(ori_logit)
    y_label[labels.index(attr_class)] = 1

    # index_class_logit = ori_logit[labels.index(attr_class)]
    # detected_label_index = ori_logit.argmax()
    # print("detected label index: {}, real label index: {}, label name: {}"
    #       .format(detected_label_index, labels.index(attr_class), attr_class))
    for i_wanted_layer in range(len(layers)):
      layer = layers[i_wanted_layer]
      acts = T(layer).eval()

      attr = np.zeros(acts.shape[1:])
      t_grad = tf.gradients([logit4grad], [T(layer)])[0]

      if not flag_read_attr:
        for n in range(iter_num):
          acts_ = acts * float(n) / iter_num
          if SG_path:
            acts_ *= (np.random.uniform(0, 1, [528]) + np.random.uniform(0, 1, [528])) / 1.5
          grad = t_grad.eval({T(layer): acts_})
          attr += grad[0]
        attr = attr * (1.0 / iter_num) * acts[0]
        attr = np.sum(np.sum(attr, 0), 0)

        # normalize IG result for degub differences
        attr = attr * ori_single_logit / np.sum(attr)
        np.savetxt(save_directory + "/{}_{}_{}.txt".format(flag1, layer, attr_class), attr)
      else:
        attr = np.loadtxt(save_directory + "/{}_{}_{}.txt".
                          format(flag1, layer, attr_class)).astype(np.float32)

      attr_temp = np.squeeze(attr).astype(np.float32)
      channel_attr_list[i_wanted_layer] = attr_temp

      acts_squeeze = np.squeeze(acts)
      AM_T[i_wanted_layer] = acts_squeeze

    AM_list = AM_T
    logit_list = sess.run([logit])[0]
    return AM_list, logit_list, channel_attr_list

