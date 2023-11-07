from typing import List, Tuple, Dict

# N-gram language models often include special "beginning-of-string" (BOS) and "end-of-string" (EOS) control tokens for knowing whether an n-gram is at the beginning, middle, or end of the sequence.
BOS = "<BOS>"
EOS = "<EOS>"


# Function for generating n-grams out of the given sequence of tokens.
# N-grams are represented as tuples. Note that this function does not have control tokens
def build_ngrams(tokens: List[str], n: int) -> List[Tuple[str]]:
    tuple_list = []
    for i in range(len(tokens) - n + 1):
        token = tokens[i]
        next_tokens = tokens[i + 1 : i + n]
        if len(next_tokens) < n - 1:
            next_tokens.extend([""] * (n - 1 - len(next_tokens)))
        tuple_list.append((token, *next_tokens))
    return tuple_list


# Function for generating n-grams out of the given sequence of tokens.
# Control tokens included.
def build_ngrams_ctrl(tokens: List[str], n: int) -> List[Tuple[str]]:
    tokens = [BOS] * (n - 1) + tokens + [EOS] * (n - 1)
    tuple_list = []

    for i in range(len(tokens) - n + 1):
        token = tokens[i]
        next_tokens = tokens[i + 1 : i + n]
        tuple_list.append((token, *next_tokens))

    return tuple_list


# Function to compute Maximum Likelihood Estimations. In MLE we first count the number of times each word follows an n-gram of size `n-1`. In other words we count the occurences of the next word given that you have n-1 word in the sequence. You can build this structure as a Python Dictionary. Key would be tuples containing the sequence and value would be the dictionary of the occurences of the next words in the sequence. Example below:
# {
#     ('<BOS>', '<BOS>'): {'It': 1, 'How': 1},
#     ('<BOS>', 'It'): {'shows': 1},
#     ('<BOS>', 'How'): {'would': 1},
#     ...
#     ('my', 'dear'): {'Watson': 2},
#     ('dear', 'Watson'): {',': 1, '?': 1},
#     ...
# }
def count_ngrams(
    texts: List[List[str]], n: int
) -> Dict[Tuple[str, ...], Dict[str, int]]:
    frequency_dict: Dict[Tuple, Dict[str, int]] = {}

    def update_frequency_dict(tuples: Tuple[str], next_token: str):
        key: Tuple[str] = tuples[:-1]
        try:
            if next_token in frequency_dict[key]:
                frequency_dict[key][next_token] += 1
            else:
                frequency_dict[key][next_token] = 1
        except KeyError:
            frequency_dict[key] = {next_token: 1}

    for item in texts:
        tuple_emitter = build_ngrams_ctrl(item, n)
        for item in tuple_emitter:
            next_token = item[-1]
            update_frequency_dict(item, next_token)

    return frequency_dict


# Function to make the N-gram model. Here we convert the count from the n-grams and make it into probability
def build_ngram_model(
    texts: List[List[str]], n: int
) -> Dict[Tuple[str, ...], Dict[str, float]]:
    frequency_dict = count_ngrams(texts, n)
    for dicts in frequency_dict.values():
        for key, value in dicts.items():
            dicts[key] = value / len(texts)

    return frequency_dict
