import DataLoader
import sys

ngram_n = int(sys.argv[1])
grid_size = int(sys.argv[2])

image_defs = DataLoader.load_image_definition("data/cocotalk.json")

category_dict, objects_dict = DataLoader.load_and_preprocess_detected_objects(grid_size = grid_size)

training_caption_dict = DataLoader.load_and_preprocess_training_image_captions(image_defs, path = "data/dataset_coco.json")

import MarkovCaptioner
markov_captioner = MarkovCaptioner.MarkovCaptioner(ngram_n = ngram_n, grid_size = grid_size)
markov_captioner.fit(training_caption_dict, objects_dict, len(category_dict))

import Utility
Utility.pickle_save(markov_captioner, f"location-encoded-markov-captioner-{ngram_n}-gram-{grid_size}-grid")