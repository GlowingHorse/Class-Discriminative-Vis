import numpy as np
from utils.utils import save_imgs, plot
from skimage.transform import rescale, resize, downscale_local_mean
from re import findall
import tabulate


def gen_spatial_heat_maps(q_spatial, decomposed_channel_num, spatial_factors, save_directory,
                          attr_class, factorization_method, no_slash_layer_name, img, AM, model):
  if q_spatial is not None:
    for i_decomposed_channel_num in range(decomposed_channel_num):
      spatial_factors[i_decomposed_channel_num, ...] = spatial_factors[i_decomposed_channel_num, ...] * (
        spatial_factors[i_decomposed_channel_num, ...] > np.quantile(spatial_factors[i_decomposed_channel_num, ...],
                                                                     q_spatial / 100))

  index_saveimg = 0
  for i_factor in range(spatial_factors.shape[0]):
    factor_resized = resize(spatial_factors[i_factor], (model.image_shape[0], model.image_shape[1]), order=1,
                            mode='constant', anti_aliasing=False)
    imgtype_name1 = 'SpatialHM'
    plot(factor_resized, save_directory, attr_class, factorization_method, no_slash_layer_name,
         imgtype_name1, index_saveimg, xi=img, cmap2='seismic', alpha=0.3)
    index_saveimg = index_saveimg + 1
  print('heat maps have been saved')


def gen_info_txt(channel_shap, decomposed_channel_num, save_directory, factorization_method,
                 attr_class, every_group_attr_sorted):
  channel_shap_max_index = channel_shap.argmax(axis=0)
  # channel_factors_index_temp = np.squeeze(np.argwhere(np.sum(channel_shap, axis=0) == 0))
  channel_shap_max_index[np.squeeze(np.argwhere(np.sum(channel_shap, axis=0) == 0))] = 74
  channel_shap_unique, channel_shap_counts = \
    np.unique(channel_shap_max_index, return_counts=True)
  correspond_channel_shap_index = []
  for channel_factors_i in range(decomposed_channel_num):
    correspond_channel_shap_index_temp = \
      np.argwhere(channel_shap_max_index == channel_factors_i)
    if correspond_channel_shap_index_temp.ndim > 1:
      correspond_channel_shap_index_temp = \
        np.squeeze(correspond_channel_shap_index_temp, axis=1)
    correspond_channel_shap_index.append(list(correspond_channel_shap_index_temp))

  channel_factors_maxindex = dict(zip(channel_shap_unique, channel_shap_counts))
  with open(save_directory + "/" + factorization_method + attr_class + '_SpatialAttrs.txt', 'w') as f:
    f.write("%s\n" % "Original Attrs from map 0->last")
    for item in range(decomposed_channel_num):
      f.write("%s " % str(item))
    f.write("\n\n")
    f.write("%s\n" % "Soft Index for map 0->last")
    for item in every_group_attr_sorted:
      f.write("%.2f " % item)
    f.write("\n\n")
    f.write("%s\n" % "Every component corr to channels' number")
    for k, v in channel_factors_maxindex.items():
      f.write(str(k) + '>>>' + str(v) + '  ')
    f.write("\n\n")
    for k, v in enumerate(correspond_channel_shap_index):
      if len(v) > 5:
        v = v[:5]
      f.write(str(k) + '>>>' + str(v))
      f.write("\n")
