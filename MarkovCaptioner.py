import numpy as np
from sklearn.preprocessing import LabelEncoder
import sparse
import Utility
import Heatmap

Constant = Utility.Constant


class BeamSearchCandidate:
  def __init__(self, init_states, state_count=dict(), log_prob=0.0):
    self.state_count = state_count
    self.states = init_states
    self.log_prob = log_prob

  def append(self, state, log_prob):
    # make a copy of the states
    new_state_count = self.state_count.copy()
    new_state_count[state] = new_state_count.get(state, 0) + 1
    return BeamSearchCandidate(self.states + [state], new_state_count, self.log_prob + log_prob)

  def __lt__(self, other):
    return self.log_prob < other.log_prob


class MarkovCaptioner:
  def __init__(self, ngram_n, grid_size):
    self.word_encoder = None
    self.ngram_n = ngram_n
    self.grid_size = grid_size

    self.word_count = None
    self.word_log_prob = None
    self.object_word_prob = None
    self.state_transition_prob_matrix = None
    self.denominator_conditional_prob_matrix = None

    self.start_token_index = None
    self.end_token_index = None

    self.num_words = None
    self.num_obj_cats = None

  def fit(self, training_caption_dict, image_object_dict, num_categories,
          train_markov = True, train_object_word = True):
    def create_ngram(tokens, n):
      """enumerate all ngrams from the list of tokens with automatic start and end paddings"""
      tokens_with_end = tokens + [self.end_token_index]
      return [tuple([self.start_token_index] * max(0, n - i - 1)
                    + tokens_with_end[max(0, i + 1 - n): i + 1]) for i in range(len(tokens_with_end))]

    # captions are yet to have start/end tokens added
    # unknown token depends on the data. Do not add artificially
    unique_words = {Constant.start_token, Constant.end_token}

    unmatch_count = 0
    matched_count = 0

    for img_id, ngram_lists in training_caption_dict.items():
      # make sure that the training data exists from both datasets
      if img_id not in image_object_dict:
        unmatch_count += 1
        continue

      matched_count += 1

      for ngrams in ngram_lists:
        unique_words.update(ngrams)

    print(f"{matched_count} images will be used for training")
    print(f"{unmatch_count} images unmatched")
    print(len(unique_words), "unique words")

    word_encoder = LabelEncoder()
    word_encoder.fit(list(unique_words))

    self.word_encoder = word_encoder

    self.start_token_index, self.end_token_index = word_encoder.transform([Constant.start_token,
                                                                           Constant.end_token])

    self.num_words = len(unique_words)
    self.num_obj_cats = num_categories

    if train_markov:
      # Count(w_t)
      word_count = np.zeros(self.num_words)
      # Count(w_t-2, w_t-1, w_t)
      state_transition_occurrence_matrix = sparse.DOK([self.num_words] * self.ngram_n)

    if train_object_word:
      # P(obj_cat | w_t)
      # flatten grid index dimension
      object_word_occurrence = np.zeros((self.num_obj_cats * self.grid_size ** 2, self.num_words))

    # MLE
    for img_id, sentence_lists in training_caption_dict.items():

      # make sure that the training data exists from both datasets
      if img_id not in image_object_dict:
        continue

      object_list = image_object_dict[img_id]

      for sentence in sentence_lists:
        encoded_sentence = word_encoder.transform(sentence).tolist()

        # add 1 start and end token per sentence for counting purpose
        for word in [self.start_token_index, self.end_token_index] + encoded_sentence:
          if train_markov:
            # add to word prob
            word_count[word] += 1

          if train_object_word:
            # add to object-word prob
            for object_id, grid_ids in object_list.items():
              for grid_id in grid_ids:
                object_word_occurrence[self.num_obj_cats * grid_id + object_id][word] += 1

        if train_markov:
          # add to markov chain prob
          # create_ngram automatically pads the start and end of the encoded sentence
          for ngram in create_ngram(encoded_sentence, self.ngram_n):
            state_transition_occurrence_matrix[ngram] += 1

    if train_markov:
      # P(w_t-2, w_t-1 | w_t)
      self.state_transition_prob_matrix = state_transition_occurrence_matrix / word_count # automatically converts from DOK to COO
      # P(w_t-2 | w_t-1)
      self.denominator_conditional_prob_matrix = state_transition_occurrence_matrix.to_coo().sum(-1) / word_count

      # impute the count of <start> and <end> token as the average count of all other regular words
      # this alleviates the problem of <end> token being generated too soon
      word_count_copy = word_count.copy()
      mask = np.ones(len(word_count_copy), dtype=bool)
      mask[[self.start_token_index, self.end_token_index]] = False
      word_count_copy[[self.start_token_index, self.end_token_index]] = word_count_copy[mask].mean()

      # P(w_t)
      self.word_log_prob = np.log(word_count_copy / word_count_copy.sum())
      # for debugging purpose
      self.word_count = word_count

    if train_object_word:
      self.object_word_prob = object_word_occurrence / word_count

  def generate_caption(self, given_objects, sentence_length_limit = 16, beam_width = 5, decay_factor = 1e-2):
    """
    generate a caption sentence given the object id and locations using beam search
    """
    sentence_candidates = [BeamSearchCandidate([self.start_token_index] * (self.ngram_n - 1))]

    # prepare seen/unseen object vector beforehand
    objects_unseen = np.ones(self.num_obj_cats * self.grid_size ** 2)

    for obj_id, grid_ids in given_objects:
      for grid_id in grid_ids:
        # flatten object location list and write to array
        objects_unseen[self.num_obj_cats * grid_id + obj_id] = 0

    # word index -> sum of object log prob
    object_log_prob_cache = dict()

    def calculate_obj_log_prob(word_index):
      if word_index in object_log_prob_cache:
        return object_log_prob_cache[word_index]
      else:
        probs = np.abs(objects_unseen - self.object_word_prob[:, word_index])
        probs[probs < 1e-20] = 1e-20 # suppress div-by-0 warning by substituting 0 with a small number
        total_log_prob = np.log(probs).sum()
        object_log_prob_cache[word_index] = total_log_prob
        return total_log_prob

    for i in range(sentence_length_limit):
      # a list of beam search candidates
      index_candidates = list()

      for candidate in sentence_candidates:
        if candidate.states[-1] == self.end_token_index:
          # pass through if candidate already generated <end>
          index_candidates.append(candidate)
          continue

        generated_words = candidate.states
        next_candidates = list()

        # get the preceding (n-gram_n - 1) words
        preceding_words_tuple = tuple(generated_words[-(self.ngram_n - 1):])

        # calculate the denominator probability
        denom_log_prob = self.denominator_conditional_prob_matrix[preceding_words_tuple] + \
                         self.word_log_prob[preceding_words_tuple[-1]] + calculate_obj_log_prob(preceding_words_tuple[-1])

        # we assume that the training sentences are complete
        # such that if a transitional probability is 0, we'll not use the word at all
        inverse_transitional_probs = self.state_transition_prob_matrix[preceding_words_tuple]

        next_word_indices = np.where(inverse_transitional_probs > 0.0)[0]
        joint_markov_chain_probs = np.log(inverse_transitional_probs[next_word_indices].todense()) + \
                                   self.word_log_prob[next_word_indices]

        for word_idx, joint_markov_chain_prob in zip(next_word_indices, joint_markov_chain_probs):
          # vectorized object seen probability accumulation
          object_prob = calculate_obj_log_prob(word_idx)

          word_decay_log_prob = np.log(decay_factor ** candidate.state_count.get(word_idx, 0))

          total_pob = joint_markov_chain_prob + object_prob + word_decay_log_prob - denom_log_prob

          next_candidates.append(candidate.append(word_idx, total_pob))

        # optimization: find the most probable next candidates for the current candidate
        next_candidates.sort(reverse=True)
        index_candidates.extend(next_candidates[:beam_width])

      # find the most probable next candidates of the index
      index_candidates.sort(reverse=True)
      sentence_candidates = index_candidates[:beam_width]

    human_readable_sentences = list()

    for candidate in sentence_candidates:
      encoded_sentence = candidate.states[self.ngram_n - 1:]
      if encoded_sentence[-1] == self.end_token_index:
        encoded_sentence = encoded_sentence[:-1]
      human_readable_sentences.append(" ".join(self.word_encoder.inverse_transform(encoded_sentence)))

    return human_readable_sentences

  def show_object_word_heatmap(self, word, category_id, **kwargs):
    """
    display the pseudo global attention of a given word on a given category on each grid
    """
    reshaped_object_word_array = self.object_word_prob.reshape(self.grid_size ** 2, self.num_obj_cats, self.num_words)
    return Heatmap.heatmap(reshaped_object_word_array[:, category_id,
                           self.word_encoder.transform([word])[0]].reshape(self.grid_size, self.grid_size),
                           row_labels=range(self.grid_size), col_labels=range(self.grid_size), vmin=0, **kwargs)
