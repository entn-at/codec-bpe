# codec-bpe
![codec_bpe.png](img/codec_bpe.png)

Codec BPE is an implementation of [Acoustic BPE](https://arxiv.org/abs/2310.14580) (Shen et al., 2024), extended for RVQ-based Neural Audio Codecs such as [EnCodec](https://github.com/facebookresearch/encodec) (Défossez et al., 2022), [DAC](https://github.com/descriptinc/descript-audio-codec) (Kumar et al., 2023), and [Mimi](https://huggingface.co/kyutai/mimi) (Défossez et al., 2024), and [FunCodec](https://funcodec.github.io/) (Du et al., 2024). Built on top of the [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) library.

Codec BPE flattens multi-level codes from Residual Vector Quantizers (RVQ) and converts them into unicode strings for tokenization into compressed token sequences. For example, a single Codec BPE token might represent a 4-gram of codes from 4 codebooks representing a single acoustic unit, a 6-gram comprising a whole acoustic unit and half of the next one, or even an 8-gram represnting two whole acoustic units. Depending on the codec, vocab size and type of audio, this can yield savings of 2-5x in sequence length compared to directly modeling the flattened codebooks.

Codec BPE can also be used with single-level codecs such as that used in the Acoustic BPE paper. In this case, a single Codec BPE token could represent one or more codes where each code represents a whole acoustic unit.

**Using Codec BPE allows efficient audio language modeling with multi-level codecs to be done with vanilla LLM architectures, meaning no custom architecture is needed to deal with modeling the RVQ. Your model will already be compatible with the full ecosystem of training and inference tools available for [HuggingFace Transformers](https://github.com/huggingface/transformers), such as [vLLM](https://github.com/vllm-project/vllm) and [Ollama](https://ollama.com/)!**

## 🚀 Updates
**2025-03-09**
- Added support for [FunCodec](https://funcodec.github.io/) from Alibaba DAMO Speech Lab! Use `--codec_model alibaba-damo/...` when encoding audio with `codec_bpe.audio_to_codes` to encode using the FunCodec model. Model paths on the HuggingFace hub are listed [here](https://github.com/modelscope/FunCodec?tab=readme-ov-file#available-models). See [here](#train-a-tokenizer-from-audio-files) for a usage example.

**2024-09-20**

- Added support for Kyutai Lab's [Mimi codec](https://huggingface.co/kyutai/mimi), an amazing new codec with a 12.5 Hz framerate! Use `--codec_model kyutai/mimi` when encoding audio with `codec_bpe.audio_to_codes` to encode using the Mimi model. See [here](#train-a-tokenizer-from-audio-files) for a usage example.

**2024-09-19**

- Initial Release!

## Setup
```bash
pip install codec-bpe
```
If you want to use the `--codec_type funcodec` option with `codec_bpe.audio_to_codes`, run:
```bash
pip install codec-bpe[funcodec]
```

## Supported Codecs
| Model                                                               | Sample Rate (kHz)* | Framerate (Hz)* | Max Codebooks | Codebook Size | Max Bandwidth (kbps)* | Training Domain |
|:--------------------------------------------------------------------|:------------------:|:--------------:|:--------------:|:-------------:|:---------------------:|:---------------:|
| [🤗 EnCodec 24khz](https://huggingface.co/facebook/encodec_24khz)  | 24                 | 75             | 32             | 1024          | 24                    | General         |
| [🤗 DAC 44khz](https://huggingface.co/descript/dac_44khz)          | 44.1               | 86.1328125     | 9              | 1024          | 7.8                   | General         |
| [🤗 DAC 24khz](https://huggingface.co/descript/dac_24khz)          | 24                 | 75             | 32             | 1024          | 24                    | General         |
| [🤗 DAC 16khz](https://huggingface.co/descript/dac_16khz)          | 16                 | 50             | 12             | 1024          | 6                     | General         |
| [🤗 Mimi](https://huggingface.co/kyutai/mimi)                      | 24                 | 12.5           | 32             | 2048          | 4.4                   | Speech          |
| [🤗 FunCodec zh_en-general-16k-nq32ds640](https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch) | 16 | 25 | 32 | 1024 | 8  | General         |
| [🤗 FunCodec zh_en-general-16k-nq32ds320](https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch) | 16 | 50 | 32 | 1024 | 16 | General         |
| [🤗 FunCodec en-libritts-16k-nq32ds640](https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch)     | 16 | 25 | 32 | 1024 | 8  | Audiobooks      |
| [🤗 FunCodec en-libritts-16k-nq32ds320](https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch)     | 16 | 50 | 32 | 1024 | 16 | Audiobooks      |

\* Sample Rate (kHz) is the sampling rate of the audio input to the codec.

\* Framerate (Hz) is the number of timesteps (acoustic units of size `num_codebooks`) per second output by the codec.

\* Bandwidth (kbps) = `framerate (Hz) x num_codebooks x log2(codebook_size)`.

## Usage

### Convert audio codes to and from unicode strings
Use your codec of choice (e.g., EnCodec, DAC, Mimi, FunCodec) to encode your audio into a torch tensor or numpy array of codes of shape (num_codebooks, length), then use the provided converter methods to convert to and from unicode strings.

**Note:** In the Acoustic BPE paper, a single-level codec was used (HuBERT + k-means), where each encoded timestep consisted of a single code which was converted to a single unicode character. Here, we support multi-level codecs based on Residual Vector Quantizers. If num_codebooks > 1, a flattening pattern is used to interleave all codebooks into a single level before mapping to unicode. For example, if 4 codebooks are used then each encoded timestep would consist of 4 codes (one from each codebook) and would be converted to a unicode 4-gram.

Example: audio language modeling using EnCodec 24 kHz at 3 kbps (4 codebooks):
```python
import torch
import librosa
import soundfile as sf
from transformers import (
    EncodecModel, 
    AutoModelForCausalLM,
    AutoProcessor, 
    AutoTokenizer,
)
from codec_bpe import codes_to_chars, chars_to_codes

# load a Codec BPE tokenizer and compatible language model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("output/my_tokenizer")
model = AutoModelForCausalLM.from_pretrained("output/my_model").to(device)

# load the EnCodec model
encodec_modelname = "facebook/encodec_24khz"
encodec_model = EncodecModel.from_pretrained(encodec_modelname).to(device)
encodec_processor = AutoProcessor.from_pretrained(encodec_modelname)

# (1) encode audio using EnCodec
audio, sr = librosa.load("some_audio.mp3", sr=encodec_model.config.sampling_rate, mono=True)
inputs = encodec_processor(raw_audio=audio, sampling_rate=sr, return_tensors="pt").to(device)
with torch.no_grad():
    encoded_audio = encodec_model.encode(**inputs, bandwidth=3.0).audio_codes[0, 0]

# (2) convert the audio codes to a unicode string and tokenize it
unicode_str = codes_to_chars(encoded_audio, codebook_size=encodec_model.config.codebook_size)
inputs = tokenizer(unicode_str, return_tensors="pt").to(device)

# (3) generate tokens from the model
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=300)

# (4) detokenize the output back into a unicode string and convert it back to audio codes
unicode_str_2 = tokenizer.decode(outputs[0], skip_special_tokens=False)
encoded_audio_2 = chars_to_codes(
    unicode_str_2, 
    num_codebooks=encoded_audio.shape[0], 
    codebook_size=encodec_model.config.codebook_size, 
    return_tensors="pt",
).to(device)

# (5) decode the generated audio using EnCodec
with torch.no_grad():
    audio_2 = encodec_model.decode(encoded_audio_2.unsqueeze(0).unsqueeze(0), [None]).audio_values[0, 0]
sf.write("some_audio_output.wav", audio_2.cpu().numpy(), sr)
```

### Train a tokenizer from audio files
To train a tokenizer from audio files:

1. Use your codec of choice (e.g., EnCodec, DAC, Mimi, FunCodec) to encode each audio file into a directory of numpy arrays (.npy files):
    ```bash
    # encode audio files using EnCodec 24 kHz at 3 kbps (4 codebooks)
    python -m codec_bpe.audio_to_codes \
        --audio_path path/to/audio \
        --codec_model facebook/encodec_24khz \
        --bandwidth 3.0 \
        --batch_size 8

    # encode audio files using first 4 codebooks of DAC 44kHz
    python -m codec_bpe.audio_to_codes \
        --audio_path path/to/audio \
        --codec_model descript/dac_44khz \
        --n_quantizers 4 \
        --batch_size 8

    # encode audio files using first 6 codebooks of Mimi (24kHz)
    python -m codec_bpe.audio_to_codes \
        --audio_path path/to/audio \
        --codec_model kyutai/mimi \
        --n_quantizers 6 \
        --batch_size 8

    # encode audio files using FunCodec (16kHz) at 1.5 kbps (6 codebooks)
    python -m codec_bpe.audio_to_codes \
        --audio_path path/to/audio \
        --codec_model alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch \
        --bandwidth 1500 \
        --batch_size 8
    ```

2. Suppose you want to use the first 4 codebooks of [EnCodec 24 kHz](https://huggingface.co/facebook/encodec_24khz), run:
    ```bash
    python -m codec_bpe.train_tokenizer \
        --codes_path output/codes/encodec_24khz/mono \
        --chunk_size_secs 30 \
        --vocab_size 30000 \
        --pad_token "<pad>"
    ```
    Here: 
    - `chunk_size_secs` specifies the number of timesteps (in seconds) that get converted to unicode and returned to the underlying Tokenizers trainer at a time.
    - `vocab_size` specifies the number of tokens (including the base vocabulary of individual unicode characters) that you want your tokenizer to have. The base vocabulary size is `num_codebooks` x `codebook_size`. For example, the command above would yield a tokenizer with a base vocabulary of 4096 individual unicode character tokens, each representing a single code from a single codebook, and 25,904 merged "ngram" tokens.

    By default, the following additional arguments are automatically initialized from the `codec_info.json` file output by `codec_bpe.audio_to_codes`:
    - `num_codebooks` specifies how many codebooks should be used (in a flattened pattern) when converting each timestep to unicode. For example, EnCodec 24kHz uses 2 codebooks at 1.5 kbps, 4 codebooks at 3 kbps, 8 codebooks at 6 kbps, etc. Note: when encoding the audio files, you should use at least as many codebooks as you plan to specify here.
    - `codebook_size` specifies the size of the codebook. EnCodec 24 kHz uses a codebook size of 1024.
    - `codec_framerate` specifies the framerate (number of timesteps per second) of the codec. EnCodec 24 kHz generates 75 timesteps per second.
    
    You may also pass these arguments explicitly. For example:
    ```bash
    python -m codec_bpe.train_tokenizer \
        --codes_path output/codes/encodec_24khz/mono \
        --num_codebooks 4 \
        --codebook_size 1024 \
        --codec_framerate 75 \
        --chunk_size_secs 30 \
        --vocab_size 30000 \
        --pad_token "<pad>"
    ```
    This is useful if you are using audio codes that you generated with a tool other than the `codec_bpe.audio_to_codes` script, or if you wish to use a lower number of codebooks
    for training the tokenizer than you used for encoding the audio files.

    See [train_tokenizer.py](codec_bpe/train_tokenizer.py) for a complete list of supported arguments.

#### Controlling the granularity of Codec BPE tokens
The `max_token_codebook_ngrams` argument can be used to control how many codes can be merged into a single Codec BPE token. This is useful to avoid repetitive patterns in the audio manifesting as redundant tokens in the vocabulary. For example, if long segments of silence exist in the training audio then you may end up with hundreds of tokens that just represent different lengths of silence.

To avoid this, you can set `max_token_codebook_ngrams` to the maximum number of codebook ngrams (whole acoustic units) you want to allow a single token to represent. For example, if you set `max_token_codebook_ngrams = 2` while `num_codebooks` is set to 4, then a single Codec BPE token may only hold up to 8 codes:
```bash
python -m codec_bpe.train_tokenizer \
    --codes_path output/codes/encodec_24khz/mono \
    --chunk_size_secs 30 \
    --vocab_size 30000 \
    --pad_token "<pad>" \
    --max_token_codebook_ngrams 2
```

**It is highly recommended to set this argument to a value <= 2 to ensure that your `vocab_size` budget gets distributed across diverse acoustic patterns in your training data.**

Setting `max_token_codebook_ngrams = 0` will skip tokenizer training and simply output a base vocabulary of `num_codebooks x codebook_size` tokens, each representing a single code from a single codebook. This is useful if you want to directly model individual codes from the flattened codebooks instead of combining them into n-grams.

### Extend an existing Transformers PreTrainedTokenizer
You may want to train a new Codec BPE tokenizer and then export its trained vocabulary to an existing Transformers tokenizer. For example, extending the Llama, Mistral, Qwen, etc. tokenizers for multimodal text-audio language modeling.

Suppose you have trained your Codec BPE tokenizer and saved it to `output/encodec_bpe_4cb_30k` and you want to extend the Mistral-7B-v0.1 tokenizer with its vocabulary, run:
```bash
python -m codec_bpe.extend_tokenizer \
    --existing_tokenizer mistralai/Mistral-7B-v0.1 \
    --codec_bpe_tokenizer output/encodec_bpe_4cb_30k \
    --audio_start_token "<audio>" \ # optional
    --audio_end_token "</audio>"    # optional
```
This will simply add every token in `output/encodec_bpe_4cb_30k/tokenizer.json` to the `mistralai/Mistral-7B-v0.1` tokenizer as a special token and save a copy of the latter. 

#### Avoiding vocabulary conflicts
If the added Codec BPE unicode tokens would conflict with existing tokens in the vocabulary, you can override the default unicode offset using the `unicode_offset` argument for both `codec_bpe.train_tokenizer` and `codec_bpe.extend_tokenizer`. By default, unicode characters from the [CJK Unified Ideographs](https://symbl.cc/en/unicode-table/#cjk-unified-ideographs) block are used, following the Acoustic BPE paper. You can set `unicode_offset` to a different value to use a different unicode block that doesn't conflict with your existing vocabulary.
