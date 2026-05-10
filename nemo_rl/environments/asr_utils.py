# import os
import re
# import numpy as np
# import unicodedata
import math
try:
    import editdistance
except ImportError:
    editdistance = None

from nemo_rl.environments.normalizer.asr_normalizer import (
    EnglishNumberNormalizer,
    EnglishSpellingNormalizer,
    remove_symbols_and_diacritics,
)
from nemo_rl.environments.normalizer.english_abbreviations import english_spelling_normalizer


def _edit_distance(ref_words: list[str], pred_words: list[str]) -> int:
    if editdistance is not None:
        return int(editdistance.eval(ref_words, pred_words))

    prev = list(range(len(pred_words) + 1))
    for i, ref in enumerate(ref_words, start=1):
        cur = [i]
        for j, pred in enumerate(pred_words, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ref == pred else 1),
                )
            )
        prev = cur
    return prev[-1]


# def uniform_subsample(start, end, output_samples: int):
#     return np.linspace(start, end, output_samples)


# sampling = {
#     "uniform": uniform_subsample,
# }


# def get_actual_options(num_options, start_letter="A"):
#     """
#     Given number of options and the start letter, the function returns sth like this:
#     e.g. num_options=3, start_letter="B", returns: "B, C, or D"
#     Can be non-English letters.
#     """
#     candidate_answers = [chr(ord(start_letter)+idx) for idx in range(num_options)]
#     candidate_answers_all_but_last = ",".join(candidate_answers[:-1])
#     actual_options = f"{candidate_answers_all_but_last}, or {candidate_answers[-1]}"
#     return actual_options


class EnglishTextNormalizer:
    def __init__(self, english_spelling_mapping=english_spelling_normalizer):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # prefect tenses, ideally it should be any past participles, but it's harder..
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)

    def _has_non_filler_words(self, s: str) -> bool:
        """Check whether *s* contains at least one word that is NOT a filler."""
        filler_set = {"hmm", "mm", "mhm", "mmm", "uh", "um"}
        return any(
            w not in filler_set
            for w in re.sub(r"[^a-z' ]", " ", s).split()
            if w
        )

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        # Only strip filler words when the text contains substantive
        # (non-filler) words.  For filler-only utterances like "uh-huh",
        # "um", or "mm-hmm", the filler IS the content and removing it
        # destroys the reference for WER computation.
        if self._has_non_filler_words(s):
            s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep some symbols for numerics

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s


# class EvaluationTokenizer:
#     """Generic evaluation-time tokenizer for ASR evaluation."""

#     SPACE = ' '
#     SPACE_ESCAPE = '\u2581'

#     def __init__(self, tokenizer=None, lowercase=True, punctuation_removal=False, character_tokenization=False):
#         self.tokenizer = tokenizer
#         self.lowercase = lowercase
#         self.punctuation_removal = punctuation_removal
#         self.character_tokenization = character_tokenization

#     @classmethod
#     def remove_punctuation(cls, sent: str):
#         """Remove punctuation based on Unicode category."""
#         return cls.SPACE.join(t for t in sent.split(cls.SPACE) if not all(unicodedata.category(c)[0] == "P" for c in t))

#     def tokenize(self, sent: str):
#         """Tokenize text for evaluation."""
#         if self.tokenizer:
#             tokenized = self.tokenizer()(sent)
#         else:
#             tokenized = sent

#         if self.punctuation_removal:
#             tokenized = self.remove_punctuation(tokenized)

#         if self.character_tokenization:
#             tokenized = self.SPACE.join(list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE)))

#         if self.lowercase:
#             tokenized = tokenized.lower()

#         return tokenized


# def remove_special_tokens(text, language='en'):
#     """Remove special tokens and normalize text."""
#     PUNCS = "!,.?;:"

#     # Remove special tokens like <|...|>
#     text = re.sub(r"<\|.*?\|>", " ", text)

#     # Replace consecutive spaces with single space
#     text = re.sub(r"\s+", " ", text)

#     # Normalize punctuation spacing
#     text = re.sub(f" ?([{PUNCS}])", r"\1", text)

#     # Remove leading space
#     text = text.lstrip(" ")

#     # For Chinese, remove all spaces
#     if language == "zh":
#         text = re.sub(r"\s+", "", text)

#     return text


# def compute_wer(refs, hyps, language='en'):
#     """Compute Word Error Rate."""
#     normalizer = EnglishTextNormalizer()
#     tokenizer = EvaluationTokenizer(lowercase=True, punctuation_removal=True)

#     total_edits = 0
#     total_words = 0

#     for ref, hyp in zip(refs, hyps):
#         # Normalize reference and hypothesis
#         ref_norm = normalizer(ref)
#         hyp_norm = normalizer(hyp)

#         # Tokenize
#         ref_tokens = tokenizer.tokenize(ref_norm).split()
#         hyp_tokens = tokenizer.tokenize(hyp_norm).split()

#         # Compute edit distance
#         edits = editdistance.eval(ref_tokens, hyp_tokens)
#         total_edits += edits
#         total_words += len(ref_tokens)

#     return total_edits / total_words if total_words > 0 else 0.0


def asr_wer(results):
    """
    Compute Word Error Rate for ASR evaluation.

    Args:
        results (list): List of dictionaries with 'gt' and 'pred' keys

    Returns:
        float: Word Error Rate as a percentage
    """
    normalizer = EnglishTextNormalizer()

    total_edits = 0
    total_words = 0

    for result in results:
        if result['gt'] is None or (isinstance(result['gt'], float) and math.isnan(result['gt'])):
            continue
        gt_text = normalizer(result['gt'])

        if result['pred'] is None or (isinstance(result['pred'], float) and math.isnan(result['pred'])):
            pred_text = ""
        else:
            pred_text = normalizer(str(result['pred']))

        # Tokenize into words
        gt_words = gt_text.split()
        pred_words = pred_text.split()

        # Compute edit distance, capped at the number of reference words so
        # that WER never exceeds 100%.  Without the cap, short references
        # (e.g. 1-word GTs) can produce WER >> 100% when the prediction is
        # longer, leading to disproportionately harsh negative rewards that
        # destabilise RL training on short ASR utterances.
        edits = min(_edit_distance(gt_words, pred_words), len(gt_words))
        total_edits += edits
        total_words += len(gt_words)

    # Compute WER
    if total_words == 0:
        return 0.0

    wer = (total_edits / total_words) * 100
    return wer


# def compute_wer_from_file(eval_file):
#     """
#     Compute WER from an evaluation file.

#     Args:
#         eval_file (str): Path to evaluation file (xlsx, csv, or json)

#     Returns:
#         float: Word Error Rate as a percentage
#     """
#     data = load(eval_file)

#     # Convert to list of dictionaries for WER computation
#     results = []
#     for _, row in data.iterrows():
#         results.append({
#             'gt': row['answer'],
#             'pred': row['prediction']
#         })

#     return asr_wer(results)


# def retrieve_subtitles(
#     subtitles,
#     timestamps_seconds = [],
#     fps = None, frame_indices = [],
#     subtitles_are_sorted = False,
#     timestamps_are_sorted = False,
# ):
#     """
#     Retrieve a list of subtitles from a srt file or a list of tuple of
#         [subtitle_start_second, subtitle_end_second, subtitle_text]
#     Args:
#         subtitles: A srt file path or a list of SSAFile objects.
#         fps: video fps. WARNING: only applicable for fixed fps videos.
#             For variable fps videos, please use timestamps_seconds.
#         frame_indices: if fps is not None, frame_indices can be used to
#             calculate the time stamps of each frame in the list.
#         timestamps_seconds: a list of tuple of start and end seconds
#         subtitles_are_sorted: if subtitlesare sorted
#             according to the start and end seconds.
#         timestamps_are_sorted: if timestamps_are_sorted are sorted
#             according to the start and end seconds.
#     Return:
#         A list of strings
#     """
#     retrieved_subtitles = []

#     if isinstance(subtitles, str):
#         subtitles_are_sorted = False
#         loaded_subtitles = pysubs2.load(subtitles, encoding='utf-8')
#         loaded_subtitles = [(sub.start/1000.0, sub.end/1000.0, sub.text) for sub in loaded_subtitles]
#     loaded_subtitles = [(s,e,t.replace('\\N', ' ')) for s,e,t in loaded_subtitles]
#     loaded_subtitles = [(s,e,t) for s,e,t in loaded_subtitles if t.strip()]

#     if not subtitles_are_sorted:
#         loaded_subtitles = sorted([sub for sub in loaded_subtitles], key=lambda x: (x[0], x[1]))

#     if timestamps_seconds == []:
#         timestamps_are_sorted = False
#         assert fps is not None and frame_indices != [], \
#             "Both fps and frame_indices must be known for subtitles retrieval if " \
#             "timestamps_seconds is not provided and video is of fixed fps."
#         frame_indices = sorted(frame_indices)
#         timestamps_seconds_start = [
#             pysubs2.make_time(fps=fps, frames=frame_id)/1000.0 for frame_id in frame_indices
#         ]
#         timestamps_seconds = [(s, s+1/fps) for s in timestamps_seconds_start]

#     if not timestamps_are_sorted:
#         timestamps_seconds = sorted(timestamps_seconds, key=lambda x: (x[0], x[1]))

#     loaded_subtitles_idx_offset = 0
#     for frame_start, frame_end in timestamps_seconds:
#         retrieved_subtitle_text = ''
#         loaded_subtitles_idx = loaded_subtitles_idx_offset
#         for sub_start, sub_end, sub_text in loaded_subtitles[loaded_subtitles_idx_offset:]:
#             if frame_start >= sub_start and frame_end <= sub_end:
#                 retrieved_subtitle_text = sub_text
#                 loaded_subtitles_idx_offset = loaded_subtitles_idx
#                 break
#             loaded_subtitles_idx += 1
#         retrieved_subtitles.append(retrieved_subtitle_text)

#     return retrieved_subtitles


# def extract_characters_regex(s):
#     s = s.strip()
#     answer_prefixes = [
#         'The best answer is',
#         'The correct answer is',
#         'The answer is',
#         'The answer',
#         'The best option is'
#         'The correct option is',
#         'Best answer:'
#         'Best option:',
#         'Answer:',
#         'Option:',
#     ]
#     for answer_prefix in answer_prefixes:
#         s = s.replace(answer_prefix, '')

#     if len(s.split()) > 10 and not re.search('[ABCD]', s):
#         return ''
#     matches = re.search(r'[ABCD]', s)
#     if matches is None:
#         return ''
#     return matches[0]
