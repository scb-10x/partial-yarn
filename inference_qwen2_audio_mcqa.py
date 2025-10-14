import argparse
import json
import re
from tqdm import tqdm
from pathlib import Path
import logging
import math
import sys

import librosa
import numpy as np
import torch
from transformers import set_seed


sys.path.append(str(Path(__file__).parent))
from constants import MCQA_ANSWER_PARSING_REGEX_PATTERN, ACCEPTABLE_MCQA_ANSWERS, MCQA_INSTRUCTION
from models.processing_qwen2_audio import Qwen2AudioProcessor
from models.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration


logging.basicConfig(level=logging.WARNING)

MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"

ATTENTION_IMPLEMENTATION = "sdpa"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dataset-path", type=str, required=True, help="Path to test dataset.")
    parser.add_argument("--enable-partial-yarn", action="store_true", help="Whether to use Partial YaRN.")
    parser.add_argument("--interpolation-start-dim", type=int, default=0, help="Cutoff dimension index.")
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)

    args = parser.parse_args()
    return args


def parse_response_for_mcqa_answer(response):
    answer = re.search(MCQA_ANSWER_PARSING_REGEX_PATTERN, response)
    if answer:
        return answer.group(1)

    # Catch one sided parenthesis
    answer = re.search(r"(\d)\)", response)
    if answer:
        return answer.group(1)

    # Catch a single dot
    answer = re.search(r"(\d)\.", response)
    if answer:
        return answer.group(1)

    return None


def evaluate_model_single_sample_mcqa(model, sample, processor) -> bool:
    target_answer = sample["text"]
    if target_answer not in ACCEPTABLE_MCQA_ANSWERS:
        logging.warning(
            f"Evaluation sample target answer ({target_answer}) is not in acceptable form {ACCEPTABLE_MCQA_ANSWERS}."
        )

    question = sample["Q"]
    question = MCQA_INSTRUCTION.replace("the given audio", "all the given audios").format(question)

    audio, sr = librosa.load(sample["path"], sr=processor.feature_extractor.sampling_rate)
    assert sr == processor.feature_extractor.sampling_rate

    chunk_size = int(sr * 30)
    num_audio_chunks = math.ceil(audio.shape[0] / sr / 30)
    split_indices = np.arange(chunk_size, audio.shape[0], chunk_size)
    audio_chunks = np.split(audio, split_indices, axis=0)
    if audio_chunks[-1].size == 0:
        audio_chunks = audio_chunks[:-1]
    assert len(audio_chunks) == num_audio_chunks

    input_content = [{"type": "audio", "audio_array": audio_chunk} for audio_chunk in audio_chunks]
    input_content += [{"type": "text", "text": question}]

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_content},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    inputs = processor(
        text=text,
        audio=audio_chunks,
        return_tensors="pt",
        padding=True,
        truncation=False,
        sampling_rate=processor.feature_extractor.sampling_rate,
    ).to("cuda")

    with torch.inference_mode():
        generate_ids = model.generate(**inputs, max_new_tokens=512)
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer = parse_response_for_mcqa_answer(response)

    if answer is None or answer not in ACCEPTABLE_MCQA_ANSWERS:
        logging.info(f"Error parsing model response: {response}")
        logging.info(f"The inputted prompt is: {conversation}")
        logging.info(f"Target answer is: {target_answer}")

        return None

    return answer == target_answer


if __name__ == "__main__":
    set_seed(25)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse_args()

    partial_yarn_args = dict(
        enable_partial_yarn=args.enable_partial_yarn,
        interpolation_start_dim=args.interpolation_start_dim,
        partial_interpolation_attention_temperature=args.attention_temperature,
    )
    print("Partial Yarn config:", partial_yarn_args)

    print("Model name:", args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=ATTENTION_IMPLEMENTATION,
        **partial_yarn_args,
    ).eval()
    processor = Qwen2AudioProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    with open(args.test_dataset_path, "r") as f:
        test_dataset = json.load(f)["annotation"]

    total_errored = 0
    total_samples = 0
    correct_samples = 0
    for test_sample in tqdm(test_dataset):
        assert test_sample["task"] == "QA"
        result = evaluate_model_single_sample_mcqa(model=model, sample=test_sample, processor=processor)

        if result is None:
            total_errored += 1

        correct_samples += bool(result)
        total_samples += 1

    qa_accuracy = correct_samples / total_samples
    print("MCQA Evaluation accuracy =", qa_accuracy)
    print("Errored responses =", total_errored)
