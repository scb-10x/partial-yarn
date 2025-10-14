
# Partial YaRN: Audio-Only Context Extension for Large Audio-Language Models

This repository contains the PyTorch code of **Partial YaRN** and **Virtual Longform Audio Training (VLAT)**. Our work focuses on extending the audio context window of Large Audio-Language Models (LALMs) without degrading the sophisticated language capabilities of the base Large Language Model (LLM).

The core idea is to apply context extension methods, such as YaRN, in a targeted manner. Instead of altering the positional encodings of the entire input sequence, Partial YaRN modifies *only* the audio tokens. This preserves the original positional information for text tokens, thereby protecting the LLM's pretrained language understanding.

![Partial YaRN Concept](assets/partial_yarn.png)

## 📌 Table of Contents
* [Repository Structure](#-repository-structure)
* [Installation](#-installation)
* [Usage](#-usage)
* [How It Works](#-how-it-works)
* [Citation](#-citation)

## 📂 Repository Structure

The repository is organized as follows:

```
.
├── assets
│   ├── audio.wav                   # Sample audio file
│   └── sample_data.json            # Sample data
├── models
│   ├── modeling_qwen2_audio.py     # Main LALM model definition
│   ├── modeling_qwen2.py           # Modified Qwen2 text backbone with Partial YaRN RoPE implementation
│   └── processing_qwen2_audio.py   # Model's processor
└── inference_qwen2_audio_mcqa.py   # Main script for running evaluation
```

-   **`models/modeling_qwen2.py`**: This is the most critical file, containing our custom implementations of Rotary Position Embeddings (RoPE). The `Qwen2PartialYarnRotaryEmbedding` class implements the logic for applying positional interpolation exclusively to audio tokens.
-   **`inference_qwen2_audio_mcqa.py`**: This script serves as the main entry point for running inference and evaluating the model on a multiple-choice question-answering task. It handles data loading, processing, and model evaluation, and includes command-line arguments to enable Partial YaRN.
-   **`assets/`**: This directory contains sample data and images needed to run the code and understand the project.

## ⚙️ Installation

To set up the environment and install the required dependencies, please follow these steps. We recommend using a Python virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

The primary script for running experiments is `inference_qwen2_audio_mcqa.py`. It evaluates the model's performance on a dataset specified in JSON format.

### Running Inference

To run the evaluation, execute the script with the path to your test dataset. You can use the provided `sample_data.json` for a quick test.

**Command Structure:**
```bash
python inference_qwen2_audio_mcqa.py --test-dataset-path <path_to_your_data.json> [OPTIONS]
```

### Command-Line Arguments

-   `--test-dataset-path` (required): Path to the JSON file containing the test dataset.
-   `--model-path`: The Hugging Face path or local directory of the model to be evaluated. Defaults to `Qwen/Qwen2-Audio-7B-Instruct`.
-   `--enable-partial-yarn`: **This is the key flag.** Add it to activate the Partial YaRN context extension method.
-   `--interpolation-start-dim`: Specifies the dimension index from which to start applying interpolation in Partial YaRN. A value greater than 0 enables Partial YaRN logic.
-   `--attention-temperature`: A scaling factor applied to the attention scores of audio tokens when using Partial YaRN.

### Examples

1.  **Running the baseline Qwen2-Audio model (without Partial YaRN):**
    ```bash
    python inference_qwen2_audio_mcqa.py \
        --test-dataset-path assets/sample_data.json \
        --model-path Qwen/Qwen2-Audio-7B-Instruct
    ```

2.  **Running the model with Partial YaRN enabled:**
    This command activates Partial YaRN, starting interpolation from the 32nd dimension of the rotary embeddings and applying an attention temperature of 1.2.
    ```bash
    python inference_qwen2_audio_mcqa.py \
        --test-dataset-path assets/sample_data.json \
        --model-path Qwen/Qwen2-Audio-7B-Instruct \
        --enable-partial-yarn \
        --interpolation-start-dim 32 \
        --attention-temperature 1.2
    ```

## 💡 How It Works

The core of our method is implemented in **`models/modeling_qwen2.py`**. We have modified the standard Rotary Position Embedding (RoPE) module of the Qwen2 architecture.

1.  **RoPE Module Swap**: The default `Qwen2RotaryEmbedding` is replaced with our custom `Qwen2PartialYarnRotaryEmbedding` class within the `Qwen2Model` definition.
2.  **Dynamic Position ID Calculation**: The `forward` method of our custom RoPE class receives an `audio_mask`. This mask identifies which tokens in the input sequence correspond to audio.
3.  **Targeted Interpolation**:
    -   If a token is identified as **text**, its original positional ID is used.
    -   If a token is identified as **audio**, we calculate a new, interpolated positional ID. This is done by scaling the positions of the audio segment from its actual token length down to the model's original trained audio length (e.g., mapping a 60s audio segment's tokens into the positional space of a 30s segment).
4.  **Inference Script Integration**: The `inference_qwen2_audio_mcqa.py` script passes the necessary flags (`enable_partial_yarn`, `interpolation_start_dim`, etc.) during model initialization, ensuring that our custom RoPE module is used and configured correctly.

This targeted approach allows the model to handle long audio inputs by "compressing" their positional information into the range it was trained on, all while ensuring the text processing capabilities remain unaffected.