MCQA_INSTRUCTION = 'Your answer MUST be in the format of "(NUMBER) STATEMENT". For example, if the answer was (4) A pen, you would ONLY output "(4) A pen". Do NOT include any other text. Using information from the given audio. {}'

ACCEPTABLE_MCQA_ANSWERS = {"1", "2", "3", "4"}

MCQA_ANSWER_PARSING_REGEX_PATTERN = r"\((\d)\)"
