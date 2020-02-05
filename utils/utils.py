"""
utils.py including misc functions, e.g.,
1 the class of matrix decomposition method
2 the class of clusterring method
3 some plot figure operators
4 Image preconditioning method for generating random image
  using different distribution,
  and decorrelate image color space.
"""
from lucid.optvis.param.resize_bilinear_nd import resize_bilinear_nd
import matplotlib.pyplot as plt
# from skimage import data, color
# from skimage.transform import rescale, resize, downscale_local_mean
# import os
# import imageio
# from operator import itemgetter
from re import findall
# import umap
import numpy as np
import sklearn.decomposition
import sklearn.cluster
# from sklearn.utils import check_array
import tensorflow as tf
from decorator import decorator
import lucid.optvis.objectives as objectives


def _make_arg_str(arg):
  arg = str(arg)
  too_big = len(arg) > 15 or "\n" in arg
  return "..." if too_big else arg


@decorator
def wrap_objective(f, *args, **kwds):
  """Decorator for creating Objective factories.

  Changes f from the closure: (args) => () => TF Tensor
  into an Obejective factory: (args) => Objective

  while perserving function name, arg info, docs... for interactive python.
  """
  objective_func = f(*args, **kwds)
  objective_name = f.__name__
  args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
  description = objective_name.title() + args_str
  return objectives.Objective(objective_func, objective_name, description)


def _dot_attr_actmaps(x, y):
  xy_dot = tf.reduce_sum(x * y, -1)
  return tf.reduce_mean(xy_dot)


@wrap_objective
def dot_attr_actmaps(layer, attr, batch=None):
  """Loss func to compute the dot of attribution and activation maps"""
  if batch is None:
    attr = attr[None, None, None]
    return lambda T: _dot_attr_actmaps(T(layer), attr)
  else:
    attr = attr[None, None]
    return lambda T: _dot_attr_actmaps(T(layer)[batch], attr)


class MatrixDecomposer(object):
  """For Matrix Decomposition to the innermost dimension of a tensor.

  This class wraps sklearn.decomposition classes to help them apply to arbitrary
  rank tensors. It saves lots of annoying reshaping.

  See the original sklearn.decomposition documentation:
  http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
  """

  def __init__(self, n_features=3, reduction_alg=None, **kwargs):
    """Constructor for MatrixDecomposer.

    Inputs:
      n_features: Numer of dimensions to reduce inner most dimension to.
      reduction_alg: A string or sklearn.decomposition class.
      kwargs: Additional kwargs to be passed on to the reducer.
    """
    if isinstance(reduction_alg, str):
      reduction_alg = sklearn.decomposition.__getattribute__(reduction_alg)
    self.n_features = n_features
    self._decomposer = reduction_alg(n_features, **kwargs)

  @classmethod
  def _apply_flat(cls, f, acts):
    """Utility for applying f to inner dimension of acts.

    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    orig_shape = acts.shape
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    new_flat = f(acts_flat)
    if not isinstance(new_flat, np.ndarray):
      return new_flat
    shape = list(orig_shape[:-1]) + [-1]
    return new_flat.reshape(shape)

  @classmethod
  def prec_apply_sum(cls, f):
    """Utility for applying f to inner dimension of acts.

    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    new_flat = f()
    new_flat = np.sum(new_flat)
    return new_flat

  def get_precision(self):
    return MatrixDecomposer.prec_apply_sum(self._decomposer.get_precision)

  def get_score(self, AM, W):
    W = np.reshape(W, (-1, W.shape[-1]))
    prediction = np.dot(W, self._decomposer.components_)
    # prediction = self._decomposer.inverse_transform(W)
    prediction = np.reshape(prediction, (-1, prediction.shape[-1]))

    AM = np.reshape(AM, (-1, AM.shape[-1]))
    score = sklearn.metrics.explained_variance_score(AM, prediction)
    return score

  def fit(self, acts):
    return MatrixDecomposer._apply_flat(self._decomposer.fit, acts)

  def fit_transform(self, acts):
    return MatrixDecomposer._apply_flat(self._decomposer.fit_transform, acts)

  def transform(self, acts):
    return MatrixDecomposer._apply_flat(self._decomposer.transform, acts)

  # def transform(self, X):
  #   """
  #   E-step to compute transform X, or factors
  #   for factor analysis
  #   """
  #   orig_shape = X.shape
  #   X_flat = X.reshape([-1, X.shape[-1]])
  #   X_flat = check_array(X_flat)
  #   X_flat = X_flat - self._decomposer.mean_
  #   I = np.eye(len(self._decomposer.components_))
  #   temp = self._decomposer.components_ / self._decomposer.noise_variance_
  #   sigma = np.linalg.inv(I + np.dot(temp, self._decomposer.components_.T))
  #   X_transformed = np.dot(np.dot(X_flat, temp.T), sigma)
  #   shape = list(orig_shape[:-1]) + [-1]
  #   return X_transformed.reshape(shape)


class SklearnCluster(object):
  """Helper for clustering to the innermost dimension of a tensor.

  This class wraps sklearn.cluster classes to help them apply to arbitrary
  rank tensors. It saves lots of annoying reshaping.

  See the original sklearn.decomposition documentation:
  https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
  """
  def __init__(self, n_clusters=6, reduction_alg="KMeans", **kwargs):
    """Constructor for SklearnCluster.

    Inputs:
      n_features: Numer of dimensions to reduce inner most dimension to.
      reduction_alg: A string or sklearn.decomposition class. Defaults to
        "KMeans"
      kwargs: Additional kwargs to be passed on to the reducer.
    """
    if isinstance(reduction_alg, str):
      reduction_alg = sklearn.cluster.__getattribute__(reduction_alg)
    self.n_clusters = n_clusters
    self._decomposer = reduction_alg(n_clusters, **kwargs)

  @classmethod
  def _apply_flat(cls, f, acts):
    """Utility for applying f to inner dimension of acts.

    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    acts_flat = np.transpose(acts_flat, (1, 0))
    labels = f(acts_flat)
    return labels

  def fit_predict(self, acts):
    return SklearnCluster._apply_flat(self._decomposer.fit_predict, acts)

  def __dir__(self):
    dynamic_attrs = dir(self._decomposer)
    return self.__dict__.keys()


def save_imgs(images, save_directory, attr_class, factorization_method
              , no_slash_layer_name, imgtype_name='opt'):
  for i_optimgs in range(len(images)):
    if len(images[i_optimgs]) > 1:
      images_temp = images[i_optimgs]
      w = int(np.sqrt(images_temp.size / 3))
      img = images_temp.reshape(w, w, 3)
      factorization_method = findall('[A-Z]', factorization_method)
      factorization_method = ''.join(factorization_method)
      plt.imsave(save_directory + "/" + attr_class + '_' + factorization_method + '_' +
                 no_slash_layer_name + '_' + imgtype_name + str(i_optimgs) + ".jpg", img)


def save_imgs_seperate_vis(images, save_directory, attr_class, factorization_method
                           , no_slash_layer_name, channel_shap_one, vis_channel_index=None):
  for i_optimgs in range(len(images)):
    if len(images[i_optimgs]) > 1:
      images_temp = images[i_optimgs]
      w = int(np.sqrt(images_temp.size / 3))
      img = images_temp.reshape(w, w, 3)
      factorization_method = findall('[A-Z]', factorization_method)
      factorization_method = ''.join(factorization_method)
      plt.imsave(save_directory + '/' + channel_shap_one[i_optimgs] + attr_class + '_' + factorization_method + '_' +
                 no_slash_layer_name + str(vis_channel_index[i_optimgs][0]) + '.jpg', img)


def plot(data, save_directory, attr_class, factorization_method,
         no_slash_layer_name, imgtype_name, index_saveimg, xi=None, cmap='RdBu_r', cmap2='seismic', alpha=0.8):
  plt.ioff()
  # plt.ion()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap(cmap2)
  cmap_xi.set_bad(alpha=0)
  overlay = xi
  if len(data.shape) == 3:
    data = np.mean(data, 2)
  # axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
  axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  factorization_method = findall('[A-Z]', factorization_method)
  factorization_method = ''.join(factorization_method)
  plt.savefig(save_directory + '/' + attr_class + '_' + factorization_method + '_' +
              no_slash_layer_name + '_' + imgtype_name + str(index_saveimg) + '.jpg')  # 'RdBu_r' 'hot'
  # plt.show()
  # plt.close(1)


def plot_seperate(data, save_directory, attr_class, factorization_method,
         no_slash_layer_name, imgtype_name, score_str, index_num, xi=None, cmap='RdBu_r', alpha=0.8):
  plt.ioff()
  # plt.ion()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap('seismic')
  cmap_xi.set_bad(alpha=0)
  overlay = xi
  if len(data.shape) == 3:
    data = np.mean(data, 2)
  # axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
  axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  factorization_method = findall('[A-Z]', factorization_method)
  factorization_method = ''.join(factorization_method)
  plt.savefig(save_directory + '/' + str(score_str) + attr_class + '_' + factorization_method + '_' +
              no_slash_layer_name + '_' + imgtype_name + index_num + '.jpg')  # 'RdBu_r' 'hot'
  # plt.show()
  # plt.close(1)


def plot_mask_ori_img(data, save_directory, attr_class, factorization_method,
                      no_slash_layer_name, imgtype_name, index_saveimg, xi=None, cmap='RdBu_r', alpha=0.8):
  # mask = data > 0
  mask = data <= np.quantile(data, 85 / 100)
  mask = np.expand_dims(mask, axis=-1)
  masked_img = mask * xi
  plt.imsave(save_directory + "/" + attr_class + '_' + factorization_method + '_' +
             no_slash_layer_name + '_masked_' + imgtype_name + str(index_saveimg) +
             ".jpg", masked_img)

  plt.ioff()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap('seismic')
  cmap_xi.set_bad(alpha=0)
  overlay = xi

  if len(data.shape) == 3:
    data = np.mean(data, 2)
  # axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
  axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  factorization_method = findall('[A-Z]', factorization_method)
  factorization_method = ''.join(factorization_method)
  # Does save the heat maps?
  # plt.savefig(save_directory + '/' + attr_class + '_' + factorization_method + '_' +
  #             no_slash_layer_name + '_' + imgtype_name + str(index_saveimg) + '.jpg')
  # 'RdBu_r' 'hot'
  # plt.close(1)


def resize_show(data, xi=None, cmap='RdBu_r', alpha=0.2):
  plt.ioff()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=300)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap('seismic')
  cmap_xi.set_bad(alpha=0)
  overlay = xi
  if len(data.shape) == 3:
    data = np.mean(data, 2)
  # axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
  axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  plt.show()
  # print("Run show code")


def _rfft2d_freqs(h, w):
  """
  Compute 2d spectrum frequences.
  the precondition methods are same with lucid libriry
  """
  fy = np.fft.fftfreq(h)[:, None]
  # when we have an odd input dimension we need to keep one additional
  # frequency and later cut off 1 pixel
  if w % 2 == 1:
    fx = np.fft.fftfreq(w)[:w//2+2]
  else:
    fx = np.fft.fftfreq(w)[:w//2+1]
  return np.sqrt(fx*fx + fy*fy)


def fft_image(shape, sd=None, decay_power=1, random_seed=0):
  """
  Compute 2d spectrum frequences.
  the precondition methods are same with lucid libriry
  """
  b, h, w, ch = shape
  imgs = []
  for _ in range(b):
    if random_seed > 0:
      np.random.seed(random_seed)
    freqs = _rfft2d_freqs(h, w)
    fh, fw = freqs.shape
    sd = sd or 0.01
    init_val = sd*np.random.randn(2, ch, fh, fw).astype("float32")
    spectrum_var = tf.Variable(init_val)
    spectrum = tf.complex(spectrum_var[0], spectrum_var[1])
    spertum_scale = 1.0 / np.maximum(freqs, 1.0/max(h, w))**decay_power
    # Scale the spectrum by the square-root of the number of pixels
    # to get a unitary transformation. This allows to use similar
    # leanring rates to pixel-wise optimisation.
    spertum_scale *= np.sqrt(w*h)
    scaled_spectrum = spectrum * spertum_scale
    img = tf.spectral.irfft2d(scaled_spectrum)
    # in case of odd input dimension we cut off the additional pixel
    # we get from irfft2d length computation
    img = img[:ch, :h, :w]
    img = tf.transpose(img, [1, 2, 0])
    imgs.append(img)
  return tf.stack(imgs)/4.


def lowres_tensor(shape, underlying_shape, offset=None, sd=None, random_seed=0):
  """Produces a tensor paramaterized by a interpolated lower resolution tensor.

  This is like what is done in a laplacian pyramid, but a bit more general. It
  can be a powerful way to describe images.

  Args:
    shape: desired shape of resulting tensor
    underlying_shape: shape of the tensor being resized into final tensor
    offset: Describes how to offset the interpolated vector (like phase in a
      Fourier transform). If None, apply no offset. If a scalar, apply the same
      offset to each dimension; if a list use each entry for each dimension.
      If a int, offset by that much. If False, do not offset. If True, offset by
      half the ratio between shape and underlying shape (analagous to 90
      degrees).
    sd: Standard deviation of initial tensor variable.
    random_seed: set the random seed.

  Returns:
    A tensor paramaterized by a lower resolution tensorflow variable.
  """
  sd = sd or 0.01

  if random_seed > 0:
    np.random.seed(random_seed)
  init_val = sd*np.random.randn(*underlying_shape).astype("float32")
  underlying_t = tf.Variable(init_val)
  t = resize_bilinear_nd(underlying_t, shape)
  if offset is not None:
    # Deal with non-list offset
    if not isinstance(offset, list):
      offset = len(shape)*[offset]
    # Deal with the non-int offset entries
    for n in range(len(offset)):
      if offset[n] is True:
        offset[n] = shape[n]/underlying_shape[n]/2
      if offset[n] is False:
        offset[n] = 0
      offset[n] = int(offset[n])
    # Actually apply offset by padding and then croping off the excess.
    padding = [(pad, 0) for pad in offset]
    t = tf.pad(t, padding, "SYMMETRIC")
    begin = len(shape)*[0]
    t = tf.slice(t, begin, shape)
  return t


color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))


def _linear_decorelate_color(t):
  """Multiply input by sqrt of emperical (ImageNet) color correlation matrix.

  If you interpret t's innermost dimension as describing colors in a
  decorrelated version of the color space (which is a very natural way to
  describe colors -- see discussion in Feature Visualization article) the way
  to map back to normal colors is multiply the square root of your color
  correlations.
  """
  # check that inner dimension is 3?
  t_flat = tf.reshape(t, [-1, 3])
  color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
  t_flat = tf.matmul(t_flat, color_correlation_normalized.T)
  t = tf.reshape(t_flat, tf.shape(t))
  return t


def to_valid_rgb(t, decorrelate=False):
  """Transform inner dimension of t to valid rgb colors.

  Args:
    t: input tensor, innermost dimension will be interpreted as colors
      and transformed/constrained.
    decorrelate: should the input tensor's colors be interpreted as coming from
      a whitened space or not?

  Returns:
    t with the innermost dimension transformed.
  """
  if decorrelate:
    t = _linear_decorelate_color(t)
  return tf.nn.sigmoid(t)
