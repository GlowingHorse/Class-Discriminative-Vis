import lucid.modelzoo.vision_models as models
import lucid.optvis.render as render
from lucid.misc.io import load
from lucid.misc.io.reading import read
from lucid.optvis.param import color
import lucid.optvis.transform as transform
from utils.attr_computing import raw_class_group_attr, compute_igsg,\
  compute_shap, compute_all_am_shap, compute_shap_test
import utils.utils as utils
from utils.utils import fft_image, lowres_tensor, save_imgs
from utils.utils_activation_maps import create_root_dir, create_factorization_dir, print_result_from_logit
from utils.utils_activation_maps import debug_show_AM_plus_img, decompose_AM_get_group_num
from utils.utils_activation_maps import get_sort_groups_with_shap_scores, sort_groups_features
from utils.utils_activation_maps import map_shap_attr_to_long, weight_AM2spatial_factor
from utils.utils_save import gen_spatial_heat_maps, gen_info_txt


def main():
  # Import a model from the lucid modelzoo
  # Or transform tensorflow slim model from https://github.com/tensorflow/models/tree/master/research/slim
  # the lucid library help you download model automatically.
  model = models.InceptionV1()
  model.load_graphdef()

  # labels_str = read("https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt")
  # labels_str = labels_str.decode("utf-8")
  # labels = [line[line.find(" "):].strip() for line in labels_str.split("\n")]
  # labels = [label[label.find(" "):].strip().replace("_", " ") for label in labels]
  # labels = ["dummy"] + labels
  labels_str = read(model.labels_path)
  labels_str = labels_str.decode("utf-8")
  labels = [line for line in labels_str.split("\n")]

  # layers = ["InceptionV1/Mixed_5c/concat", "InceptionV1/Mixed_5b/concat",
  #                 "InceptionV1/Mixed_4f/concat", "InceptionV1/Mixed_4e/concat",
  #                 "InceptionV1/Mixed_4d/concat", "InceptionV1/Mixed_4c/concat",
  #                 "InceptionV1/Mixed_4b/concat", "InceptionV1/Mixed_3b/concat",
  #                 "InceptionV1/Conv2d_2b_1x1/Relu", "InceptionV1/MaxPool_3a_3x3/MaxPool",
  #                 ]
  # layers = ["mixed5b", "mixed5a", "mixed4e", "mixed4d", "mixed4c",
  #           "mixed4b", "mixed4a", "mixed3b", "mixed3a", "maxpool1"]

  # factorization_methods = ['DictionaryLearning', 'FactorAnalysis', 'FastICA', 'IncrementalPCA',
  #                          'LatentDirichletAllocation', 'MiniBatchDictionaryLearning',
  #                          'MiniBatchSparsePCA', 'NMF', 'PCA', 'SparsePCA',
  #                          'TruncatedSVD']
  # factorization_methods = ['KernelPCA', 'SparseCoder', 'dict_learning', 'dict_learning_online', 'fastica']

  layers = ["mixed4d"]
  factorization_methods = ['FactorAnalysis']

  # attr_classes = ['Egyptian cat', 'golden retriever']
  # attr_classes = ['laptop', 'quilt']  # ['Labrador retriever', 'tennis ball', 'tiger cat']
  # attr_classes = ['tiger cat', 'Labrador retriever']
  # ('Labrador retriever', 'golden retriever')
  # [11.319051   9.532383]
  # ('Labrador retriever', 'golden retriever')
  # [8.349452  8.214619 ]
  attr_classes = ['golden retriever']

  global_random_seed = 5
  image_size = 224  # 224

  # whether load the pre-computed feature attribution
  flag_read_attr = True
  # Shapley value computing method, "Shap" or "IGSG"
  flag1 = "IGSG"
  # iteration times for computing Shapley values
  iter_num = 100
  # pos_flag=1 means only compute positive Shapley
  # = 2 means consider both positive and negative Shapley
  pos_flag = 2
  img_name = "./data/dog_cat224.jpg"

  # ---------------------------------------------------------------------------------------------------
  neuron_groups(img_name, layers, model, attr_classes=attr_classes,
                factorization_methods=factorization_methods, flag1=flag1,
                flag_read_attr=flag_read_attr, iter_num=iter_num,
                SG_path=False, labels=labels, pos_flag=pos_flag,
                thres_explained_var=0.7, vis_random_seed=global_random_seed,
                image_size=image_size)


def neuron_groups(img_name, layers, model, attr_classes, factorization_methods,
                  flag1, flag_read_attr=False, iter_num=100, SG_path=False, labels=None, pos_flag=1,
                  thres_explained_var=0.7, vis_random_seed=0, image_size=0, debug_flag=0):
  img = load(img_name)
  # img = load("./data/doghead224.jpeg")
  # img = load("./data/cathead224.jpeg")
  # img = resize(img, (224, 224, 3), order=1, mode='constant', anti_aliasing=False).astype(np.float32)

  for attr_class in attr_classes:
    root_directory = create_root_dir(img_name, attr_class, flag1)

    if flag1 == "Shap":
      AM_list_L, logit_list, channel_attr_list, kept_channel_list_L \
        = compute_shap(img, model, attr_class, layers,
                          flag1, flag_read_attr=flag_read_attr,
                          iter_num=iter_num, labels=labels, save_directory=root_directory)
    elif flag1 == "IGSG":
      AM_list_L, logit_list, channel_attr_list, kept_channel_list_L \
        = compute_igsg(img, model, attr_class, layers,
                        flag1, flag_read_attr=flag_read_attr,
                        iter_num=iter_num, SG_path=SG_path,
                        labels=labels, save_directory=root_directory)
    else:
      continue

    print_result_from_logit(logit_list, labels)
    for i_pos_neg in range(pos_flag):
      AM_list = AM_list_L[i_pos_neg]
      kept_channel_list = kept_channel_list_L[i_pos_neg]

      if debug_flag == 1:
        debug_show_AM_plus_img(AM_list, img, model)
      if i_pos_neg < 1:
        reverse_suffix = '_pos'
      else:
        reverse_suffix = '_neg'

      for layer_index, AM in enumerate(AM_list):
        layer = layers[layer_index]
        channel_attr = channel_attr_list[layer_index]
        kept_channel_index = kept_channel_list[layer_index]

        for factorization_method in factorization_methods:
          spatial_factors, channel_factors, n_groups = \
            decompose_AM_get_group_num(factorization_method, AM, thres_explained_var)

          # if debug_flag == 1:
          #   decompose_AM_with_UMAP(AM, n_groups)

          channel_factors_max_index, channel_shap, short_index, long_index, \
            n_groups = map_shap_attr_to_long(channel_factors, channel_attr, kept_channel_index)

          # AM = np.squeeze(AM)
          spatial_factors = weight_AM2spatial_factor(AM, spatial_factors, n_groups,
                                                     short_index, kept_channel_index,
                                                     channel_attr, i_pos_neg)

          # If the attribution is negative, channel_shap should be multiply -1
          if i_pos_neg > 0:
            channel_shap = channel_shap * -1

          # Sorting based on sum of Shapley values
          ns_sorted = get_sort_groups_with_shap_scores(channel_shap)
          every_group_attr_sorted, spatial_factors, channel_shap, n_groups =\
            sort_groups_features(ns_sorted, spatial_factors, channel_shap, n_groups)

          no_slash_layer_name = ''.join(layer.split('/'))
          save_directory = create_factorization_dir(root_directory, factorization_method,
                                                    no_slash_layer_name, reverse_suffix,
                                                    n_groups)
          gen_spatial_heat_maps(85, n_groups, spatial_factors, save_directory,
                                attr_class, factorization_method, no_slash_layer_name, img, AM, model)
          gen_info_txt(channel_shap, n_groups, save_directory, factorization_method,
                       attr_class, every_group_attr_sorted)

          # Using feature attributions times activation maps as loss function to update visualization image
          channel_shap = channel_shap.astype("float32")
          obj = sum(utils.dot_attr_actmaps(layer, channel_shap[i], batch=i)
                    for i in range(n_groups))

          '''
          For feature visualization, the library "lucid" will be useful because
          it has implements many loss functions of different literatures, image processing operators,
          and collected several pretrained tensorflow network.
          '''
          transforms = [
            transform.pad(12),
            transform.jitter(8),
            transform.random_scale([n / 100. for n in range(80, 120)]),
            transform.random_rotate(list(range(-10, 10)) + list(range(-5, 5)) + 10 * list(range(-2, 2))),
            transform.jitter(2)
          ]

          # image parameterization with shared params for aligned optimizing images
          def interpolate_f():
            unique = fft_image((n_groups, image_size, image_size, 3), random_seed=vis_random_seed)
            shared = [
              lowres_tensor((n_groups, image_size, image_size, 3),
                            (1, image_size // 2, image_size // 2, 3), random_seed=vis_random_seed),
              lowres_tensor((n_groups, image_size, image_size, 3),
                            (1, image_size // 4, image_size // 4, 3), random_seed=vis_random_seed),
              lowres_tensor((n_groups, image_size, image_size, 3),
                            (1, image_size // 8, image_size // 8, 3), random_seed=vis_random_seed),
              lowres_tensor((n_groups, image_size, image_size, 3),
                            (2, image_size // 8, image_size // 8, 3), random_seed=vis_random_seed),
              lowres_tensor((n_groups, image_size, image_size, 3),
                            (1, image_size // 16, image_size // 16, 3), random_seed=vis_random_seed),
              lowres_tensor((n_groups, image_size, image_size, 3),
                                   (2, image_size // 16, image_size // 16, 3), random_seed=vis_random_seed)
            ]
            return utils.to_valid_rgb(unique + sum(shared), decorrelate=True)

          group_icons = render.render_vis(model, objective_f=obj, param_f=interpolate_f,
                                          transforms=transforms, verbose=False, relu_gradient_override=False)[-1]
          save_imgs(group_icons, save_directory, attr_class, factorization_method, no_slash_layer_name)
          print("Layer {} and class {} -> finished".format(layer, attr_class))


if __name__ == "__main__":
  # execute only if run as a script
  main()
