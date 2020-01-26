"""
The utils_activation_maps.py
provides all tools for handling the activation maps and
attributions for proper size or format before visualizing
"""
import numpy as np
from skimage.transform import resize
import os
from utils.utils import save_imgs, plot, resize_show, \
  MatrixDecomposer, SklearnCluster
from operator import itemgetter
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_root_dir(img_name, attr_class, flag1):
  name_str = os.path.splitext(os.path.basename(img_name))[-2]
  root_directory = './' + name_str + '/' + attr_class + '_' + flag1
  if not os.path.exists(root_directory):
    os.makedirs(root_directory)
  return root_directory


def create_factorization_dir(root_directory, factorization_method,
                              no_slash_layer_name, reverse_suffix,
                              n_groups):
  save_directory = root_directory + '/' + factorization_method + '/' \
                   + no_slash_layer_name + reverse_suffix + str(n_groups)
  if not os.path.exists(save_directory):
    os.makedirs(save_directory)
  return save_directory


def print_result_from_logit(logit_list, labels):
  sorted_logit = logit_list.argsort()
  pred_index = sorted_logit[-10:][::-1]
  # np.argmax(logit, axis=1)
  print(itemgetter(*pred_index)(labels))
  print(logit_list[sorted_logit[-10:][::-1]])


def debug_show_AM_plus_img(grad_cam_list, img, model):
  # Visualize the activation maps on original image
  for grad_cams_backup in grad_cam_list:
    sort_grad_cams_idx = np.argsort(-(grad_cams_backup.sum(0).sum(0)))
    grad_cams_backup_trans = np.transpose(grad_cams_backup, (2, 0, 1))

    for i_sort_grad_cams in range(sort_grad_cams_idx.shape[0]):
      grad_cam_backup = grad_cams_backup[..., sort_grad_cams_idx[i_sort_grad_cams]]
      grad_cam_backup = grad_cam_backup.reshape([grad_cam_backup.shape[-1], grad_cam_backup.shape[-1]])
      grad_cam_backup = resize(grad_cam_backup, (model.image_shape[0], model.image_shape[1]), order=1,
                               mode='constant', anti_aliasing=False)
      grad_cam_backup = grad_cam_backup / grad_cam_backup.max() * 255
      resize_show(grad_cam_backup, xi=img)
      print("image {} in all {} images".format(i_sort_grad_cams, sort_grad_cams_idx.shape[0]))


def decompose_AM_get_group_num(factorization_method, AM, thres_explained_var):
  for i_n_groups in range(3, 999):
    factor_model = MatrixDecomposer(i_n_groups, factorization_method)
    if factorization_method == 'FactorAnalysis':
      _ = factor_model.fit(AM)
      spatial_factors = factor_model.transform(AM)
    else:
      spatial_factors = factor_model.fit_transform(AM)
    score = factor_model.get_score(AM, spatial_factors)

    if score > thres_explained_var:
      spatial_factors = spatial_factors.transpose(2, 0, 1).astype("float32")
      channel_factors = factor_model._decomposer.components_.astype("float32")
      n_groups = i_n_groups
      return spatial_factors, channel_factors, n_groups


def cluster_AM_get_group_num(cluster_method, AM, **kwargs):
  scaler = MinMaxScaler()
  AM_scaled = scaler.fit_transform(np.reshape(AM, (-1, AM.shape[-1])))
  cluster_max_num = 26
  inertia = np.empty(cluster_max_num)
  secondDerivative = np.empty((cluster_max_num - 5))
  for i_n_groups in range(cluster_max_num):
    cluster_model = SklearnCluster(i_n_groups+1, cluster_method, **kwargs)
    _ = cluster_model.fit_predict(AM_scaled)
    inertia[i_n_groups] = cluster_model._decomposer.inertia_

  for i_n_groups in range(4, cluster_max_num-1):
    secondDerivative[i_n_groups-4] = inertia[i_n_groups+1] + inertia[i_n_groups-1] - 2*inertia[i_n_groups]
  # plt.figure()
  # plt.plot(range(cluster_max_num - 3), inertia)

  n_groups = np.argmax(secondDerivative)+4
  cluster_model = SklearnCluster(n_groups, cluster_method)
  labels = cluster_model.fit_predict(AM)
  return labels, n_groups


def decompose_AM_with_UMAP(AM, n_groups):
  umap_reducer = umap.umap_.UMAP(n_components=n_groups)
  grad_cam_flat = AM.reshape([-1, AM.shape[-1]])

  spatial_factors = umap_reducer.fit_transform(grad_cam_flat)
  spatial_factors = spatial_factors.reshape(AM.shape)
  spatial_factors = spatial_factors.transpose(2, 0, 1).astype("float32")


def map_shap_attr_to_long(channel_factors, channel_attr, kept_channel_index):
  channel_factors_max_index = channel_factors.argmax(axis=0)
  channel_shap = np.zeros((channel_factors.shape[0], channel_attr.shape[0]))
  short_index = []
  long_index = []
  n_groups = channel_factors.shape[0]

  for channel_factors_i in range(n_groups):
    short_index.append(
      np.squeeze(np.argwhere(channel_factors_max_index == channel_factors_i), axis=1))
    map_short_to_long_idx = \
      kept_channel_index[short_index[channel_factors_i]]
    long_index.append(map_short_to_long_idx)
    channel_shap[channel_factors_i, map_short_to_long_idx] = \
      channel_attr[map_short_to_long_idx]
  return channel_factors_max_index, channel_shap, short_index, long_index, n_groups


def map_cluster_label_to_long(labels, channel_attr, kept_channel_index):
  channel_factors_max_index = labels
  n_groups = labels.max() + 1
  channel_shap = np.zeros((n_groups, channel_attr.shape[0]))
  short_index = []
  long_index = []
  for channel_factors_i in range(n_groups):
    short_index.append(
      np.squeeze(np.argwhere(channel_factors_max_index == channel_factors_i), axis=1))
    map_short_to_long_idx = \
      kept_channel_index[short_index[channel_factors_i]]
    long_index.append(map_short_to_long_idx)
    channel_shap[channel_factors_i, map_short_to_long_idx] = \
      channel_attr[map_short_to_long_idx]
  return channel_factors_max_index, channel_shap, short_index, long_index, n_groups


def weight_AM2spatial_factor(AM, spatial_factors, n_groups,
                             short_index, kept_channel_index,
                             channel_attr, i_grad_cam_list_L):
  """
  Weighting Activation maps using feature attributions
  '''
  # Alternatives
  AM = np.squeeze(AM)
  spatial_factors = np.zeros_like(spatial_factors)
  for channel_factors_i in range(n_groups):
    if len(short_index[channel_factors_i]) == 0:
      continue
    temp = np.squeeze(AM[..., short_index[channel_factors_i]])
    if len(temp.shape) == 3:
      spatial_factors[channel_factors_i, ...] = np.sum(temp, axis=-1)
    else:
      spatial_factors[channel_factors_i, ...] = temp
  '''
  """
  spatial_factors = np.zeros_like(spatial_factors)
  for channel_factors_i in range(n_groups):
    if len(short_index[channel_factors_i]) == 0:
      continue
    map_short_to_long_idx = \
      kept_channel_index[short_index[channel_factors_i]]
    temp = np.squeeze(AM[..., short_index[channel_factors_i]])
    temp = temp * channel_attr[map_short_to_long_idx]
    if i_grad_cam_list_L > 0:
      temp *= -1
    if len(temp.shape) == 3:
      spatial_factors[channel_factors_i, ...] = np.sum(temp, axis=-1)
    else:
      spatial_factors[channel_factors_i, ...] = temp
  return spatial_factors


def weight_AM2spatial_factor2(AM, spatial_factors, channel_factors, channel_attr,
                              n_groups, i_grad_cam_list_L):
  """
  Using feature attributions and channel factors to weight activation maps
  """
  spatial_factors = np.zeros_like(spatial_factors)
  for i in range(n_groups):
    temp = channel_factors[i] * channel_attr[i] * AM
    if i_grad_cam_list_L > 0:
      temp *= -1
    if len(temp.shape) == 3:
      spatial_factors[i, ...] = np.sum(temp, axis=-1)
    else:
      spatial_factors[i, ...] = temp
  return spatial_factors


def weight_AM2spatial_factor_FGSM(AM, spatial_factors, n_groups,
                                   short_index, kept_channel_index,
                                   channel_shap, i_grad_cam_list_L):
  """
  The same with weight_AM2spatial_factor,
  but apply to adversarial samples
  """
  spatial_factors = np.zeros_like(spatial_factors)
  for channel_factors_i in range(n_groups):
    if len(short_index[channel_factors_i]) == 0:
      continue
    map_short_to_long_idx = \
      kept_channel_index[short_index[channel_factors_i]]
    temp = AM[..., short_index[channel_factors_i]]
    temp = temp * channel_shap[channel_factors_i, map_short_to_long_idx]
    if i_grad_cam_list_L > 0:
      temp *= -1
    if len(temp.shape) == 3:
      spatial_factors[channel_factors_i, ...] = np.sum(temp, axis=-1)
    else:
      spatial_factors[channel_factors_i, ...] = temp
  return spatial_factors


def get_sort_groups_with_shap_scores(channel_shap):
  every_group_attr = np.sum(channel_shap, axis=1)
  ns_sorted = np.argsort(-every_group_attr)
  return ns_sorted


def sort_groups_features(ns_sorted, spatial_factors, channel_shap, n_groups):
  every_group_attr = np.sum(channel_shap, axis=1)
  every_group_attr_sorted = every_group_attr[ns_sorted]
  spatial_factors = spatial_factors[ns_sorted]
  channel_shap = channel_shap[ns_sorted]

  for i_remove_zero_attr in range(n_groups):
    if every_group_attr_sorted[i_remove_zero_attr] == 0:
      spatial_factors = np.delete(spatial_factors, -1, 0)
      channel_shap = np.delete(channel_shap, -1, 0)
  n_groups = channel_shap.shape[0]
  return every_group_attr_sorted, spatial_factors, channel_shap, n_groups

