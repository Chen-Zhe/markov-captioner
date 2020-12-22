import json
import Utility

Constant = Utility.Constant


# Load dataset division definition
def load_image_definition(path = "data/cocotalk.json"):
  with open(path) as json_file:
    cocotalk = json.load(json_file)

  image_defs = dict()
  for img in cocotalk["images"]:
    if img["split"] not in image_defs:
      image_defs[img["split"]] = list()

    image_defs[img["split"]].append((img["id"], img["file_path"]))

  print([(category, len(img_list)) for category, img_list in image_defs.items()])

  return image_defs


def load_and_preprocess_training_image_captions(image_defs,
                                       sentence_len_limit = 16,
                                       min_token_occurrence = 5,
                                       path = "data/dataset_coco.json"):
  with open(path) as json_file:
    dataset_coco = json.load(json_file)

  training_caption_dict = dict()

  training_image_paths = {path: img_id for img_id, path in image_defs["train"] + image_defs["restval"]}

  # preprocessed sentences - collect all tokens
  img_sentences = list()
  for caption in dataset_coco["images"]:
    full_path = caption['filepath'] + "\\" + caption['filename']
    if full_path in training_image_paths:
      img_sentences.append((training_image_paths[full_path],  # convert img id
                            [sentence["tokens"] for sentence in caption["sentences"]]
                            ))

  # replace less frequent tokens with unknown token
  token_freq = dict()

  for img_id, sentences in img_sentences:
    for sentence in sentences:
      for token in sentence:
        token_freq[token] = token_freq.get(token, 0) + 1

  tokens_to_replace = set([token for token, count in token_freq.items() if count < min_token_occurrence])

  for img_id, sentences in img_sentences:
    for sentence in sentences:
      for i in range(len(sentence)):
        if sentence[i] in tokens_to_replace:
          sentence[i] = Constant.unknown_token

  for img_id, sentences in img_sentences:
    training_caption_dict[img_id] = [sentence[:sentence_len_limit] for sentence in sentences]

  return training_caption_dict


# returns weights assigned to each grid for a particular object
# the grids that the object are on receives weight of 1 and the rest are attenuated based on distance
def determine_grid_ids(image_width, image_height, object_bounding_box, grid_size):
  occupying_grids = list()
  # divide the image into a n x n grid
  height_split = image_height / grid_size
  width_split = image_width / grid_size

  # bounding box dimension: x,y,width,height
  w_left, h_upper, obj_width, obj_height = object_bounding_box
  w_right = w_left + obj_width
  h_lower = h_upper + obj_height

  for height_index in range(grid_size):
    for width_index in range(grid_size):
      grid_id = grid_size * height_index + width_index
      # check if this grid intersects with bounding box
      if w_left <= (width_index + 1) * width_split and w_right >= width_index * width_split and h_upper <= (
              height_index + 1) * height_split and h_lower >= height_index * height_split:
        occupying_grids.append(grid_id)

  return set(occupying_grids)


# loads all images, not just training images
def load_and_preprocess_detected_objects(grid_size, path = "annotations2014"):
  with open(f"{path}/instances_val2014.json") as file:
    instances_val = json.load(file)

  with open(f"{path}/instances_train2014.json") as file:
    instances_train = json.load(file)

  category_dict = dict()
  old_cat_to_new_cat_map = dict()

  # normalize the id to 0-starting
  for idx, cat in enumerate(instances_val['categories']):
    category_dict[idx] = cat["name"]
    old_cat_to_new_cat_map[cat["id"]] = idx

  image_dimension = dict()

  for image in instances_val["images"] + instances_train["images"]:
    image_dimension[image["id"]] = (image["width"], image["height"])

  objects_dict = dict()

  for annot in instances_val["annotations"] + instances_train["annotations"]:
    img_id = annot["image_id"]

    if img_id not in objects_dict:
      objects_dict[img_id] = dict()

    width, height = image_dimension[img_id]

    object_cat = old_cat_to_new_cat_map[annot["category_id"]]

    if object_cat not in objects_dict[img_id]:
      objects_dict[img_id][object_cat] = set()

    objects_dict[img_id][object_cat].update(determine_grid_ids(width, height, annot["bbox"], grid_size))

  return (category_dict, objects_dict)
