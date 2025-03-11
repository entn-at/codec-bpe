import librosa
import os
import shutil
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import math
import json
from transformers import SequenceFeatureExtractor
from typing import Optional, List, Union, Tuple
from enum import Enum
from tqdm import tqdm

class CodecTypes(Enum):
    ENCODEC = "encodec"
    DAC = "dac"
    MIMI = "mimi"
    FUNCODEC = "funcodec"

    @classmethod
    def try_get_codec_type(cls, codec_model):
        codec_model = codec_model.lower()
        if "audio_codec" in codec_model:
            return cls.FUNCODEC
        if "encodec" in codec_model:
            return cls.ENCODEC
        if "dac" in codec_model:
            return cls.DAC
        if "mimi" in codec_model:
            return cls.MIMI
        raise ValueError(f"Could not infer codec type from codec model: {codec_model}. Please specify --codec_type.")

    def __str__(self):
        return self.value
    def __eq__(self, value):
        return str(self) == value

SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".flac", ".opus"]

def process_batch(
    batch: List[np.ndarray], 
    batch_info: List[Tuple[str, str, int, int]],
    args: argparse.Namespace, 
    model: torch.nn.Module, 
    device: Union[str, torch.device],
    sr: int, 
    processor: Optional[SequenceFeatureExtractor] = None,
):
    num_codebooks = None
    framerate = None
    errored_files = []
    if not batch:
        return errored_files
    
    try:        
        if args.codec_type == CodecTypes.FUNCODEC:
            # Process audio to get padded input tensor
            max_chunk_len = max([chunk.shape[-1] for chunk in batch])
            batch_tensors = [F.pad(torch.from_numpy(chunk), (0, max_chunk_len-chunk.shape[-1])) for chunk in batch]
            input_values = torch.stack(batch_tensors).unsqueeze(1).to(device)

            # Encode the batch
            with torch.no_grad():
                encoded_batch, _, _, _ = model(
                    input_values,
                    bit_width=int(args.bandwidth) if args.bandwidth is not None else None,
                    run_mod="encode",
                )
            # Permute dimensions to match expected format
            audio_codes = torch.permute(encoded_batch[0], (1, 0, 2))
        else:
            # Process audio to get padded input tensor
            inputs = processor(raw_audio=batch, sampling_rate=sr, return_tensors="pt").to(device)
            input_values = inputs.input_values
            
            encode_kwargs = {}
            if args.codec_type == CodecTypes.DAC:
                encode_kwargs["n_quantizers"] = args.n_quantizers
            elif args.codec_type == CodecTypes.MIMI:
                encode_kwargs["num_quantizers"] = args.n_quantizers
            elif args.codec_type == CodecTypes.ENCODEC:
                encode_kwargs["bandwidth"] = args.bandwidth
            
            # Encode the batch
            with torch.no_grad():
                outputs = model.encode(**inputs, **encode_kwargs)
            audio_codes = outputs.audio_codes
            
        # Save the non-padded part of the encoded audio
        num_codebooks = audio_codes.shape[-2]
        samples_per_frame = math.ceil(input_values.shape[-1] / audio_codes.shape[-1])
        framerate = sr / samples_per_frame
        batch_dim = 1 if args.codec_type == CodecTypes.ENCODEC else 0
        for i, (file_path, numpy_root, channel, start_secs) in enumerate(batch_info):
            encoded_chunk = audio_codes.select(batch_dim, i).unsqueeze(batch_dim)
            non_padded_len = math.ceil(batch[i].shape[-1] / samples_per_frame)
            encoded_chunk = encoded_chunk[..., :non_padded_len]

            # Save encoded chunk to numpy file
            file_name_noext = os.path.basename(os.path.splitext(file_path)[0])
            numpy_filepath = os.path.join(numpy_root, f"{file_name_noext}_c{channel}_t{start_secs:06d}.npy")
            os.makedirs(os.path.dirname(numpy_filepath), exist_ok=True)
            np.save(numpy_filepath, encoded_chunk.cpu().numpy(), allow_pickle=False)

    except Exception as e:
        print(f"Error encoding batch: {e}")
        errored_files.extend(set([info[0] for info in batch_info]))
    
    return num_codebooks, framerate, errored_files

def write_codec_info(
    codec_type: CodecTypes,
    codec_model: str, 
    codes_path: str, 
    sr: int, 
    num_codebooks: int, 
    codebook_size: int, 
    framerate: float,
):
    codec_info = {
        "codec_type": str(codec_type),
        "codec_model": codec_model,
        "sampling_rate": sr,
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        "framerate": framerate,
    }
    codec_info_path = os.path.join(codes_path, "codec_info.json")
    with open(codec_info_path, "w") as f:
        json.dump(codec_info, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert audio files to numpy files containing audio codes using a Codec"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="audio",
        help="Directory containing the audio files",
    )
    parser.add_argument(
        "--codes_path",
        type=str,
        default="output/codes",
        help="Directory to save the numpy codes files",
    )
    parser.add_argument(
        "--chunk_size_secs", type=int, default=60, help="Chunk size in seconds"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of audio chunks to process in a single batch",
    )
    parser.add_argument(
        "--codec_type", 
        type=str, 
        choices=list(CodecTypes),
        default=None,
        help="Type of codec to use for encoding. None to infer the type from --codec_model.",
    )
    parser.add_argument(
        "--codec_model",
        type=str,
        default="facebook/encodec_24khz",
        help="Codec model path on the HuggingFace Model Hub.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help=(
            "Bandwidth for encoding. Only applies if --codec_type is 'encodec' or 'funcodec'. "
            "Values may be provided in kbps (e.g. 1.5) or in bps (e.g. 1500)."
            "For FunCodec, valid ranges for this parameter are listed in the 'Bitrate' column at "
            "https://github.com/modelscope/FunCodec?tab=readme-ov-file#available-models. "
            "For EnCodec, valid values are 1.5, 3.0, 6.0, 12.0, and 24.0 (kpbs). "
            "None uses the max bandwidth with FunCodec and the min bandwidth with EnCodec."
        ),
    )
    parser.add_argument(
        "--n_quantizers",
        type=int,
        default=None,
        help=(
            "Number of quantizers (codebooks) to use for encoding. None to use all quantizers. "
            "Only applies if --codec_type is 'dac' or 'mimi'."
        ),
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="Encode stereo audio channels separately instead of converting to mono",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=SUPPORTED_EXTENSIONS,
        help="Audio file extensions to convert. Formats must be supported by a librosa backend.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing numpy codes directories. If not set, audio corresponding to existing "
            "numpy codes directories will be skipped."
        ),
    )
    args = parser.parse_args()

    if args.codec_type is None:
        args.codec_type = CodecTypes.try_get_codec_type(args.codec_model)
    
    # support bandwidth in kbps or bps
    if args.bandwidth is not None:
        if args.codec_type == CodecTypes.FUNCODEC and args.bandwidth <= 16.0:
            args.bandwidth *= 1000
        if args.codec_type == CodecTypes.ENCODEC and args.bandwidth > 24.0:
            args.bandwidth /= 1000

    # load the codec model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.codec_type == CodecTypes.FUNCODEC:
        from funcodec.bin.codec_inference import Speech2Token
        from huggingface_hub import snapshot_download
        cache_path = snapshot_download(args.codec_model)
        config_file = os.path.join(cache_path, "config.yaml")
        model_pth = os.path.join(cache_path, "model.pth")
        model = Speech2Token(config_file, model_pth, device=device)
        model.eval()
    else:
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(args.codec_model).to(device)
        processor = AutoProcessor.from_pretrained(args.codec_model)

    # traverse the audio directory recursively and convert in each subdirectory containing
    # audio fileswith the specified extensions
    codec_name_for_path = args.codec_model.split("/")[-1]
    codes_path = os.path.join(
        args.codes_path, codec_name_for_path, "stereo" if args.stereo else "mono"
    )
    num_audio_files = 0
    num_numpy_files = 0
    num_skipped_dirs = 0
    errored_audio_files = []
    batch = []
    batch_info = []
    batch_processor = processor if args.codec_type != CodecTypes.FUNCODEC else None
    sr = (
        model.model_args.sampling_rate if args.codec_type == CodecTypes.FUNCODEC 
        else model.config.sampling_rate
    )
    for root, dirs, files in os.walk(args.audio_path):
        files = sorted([f for f in files if os.path.splitext(f)[1] in args.extensions])
        if len(files) == 0:
            continue
        numpy_root = root.replace(args.audio_path, codes_path)
        if os.path.exists(numpy_root):
            if args.overwrite:
                shutil.rmtree(numpy_root)
            else:
                print(f"Skipping {root} because {numpy_root} already exists.")
                num_skipped_dirs += 1
                continue
        print(f"Converting in {root}...")
        for file in tqdm(files, desc="Files"):
            file_path = os.path.join(root, file)
            num_audio_files += 1
            try:
                # Load the audio file
                audio, _ = librosa.load(file_path, sr=sr, mono=not args.stereo)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                errored_audio_files.append(file_path)
                continue

            # Encode it in chunks of size chunk_size_secs on each channel independently
            start = 0
            while True:
                end = start + args.chunk_size_secs * sr
                audio_chunk = audio[..., start:end]
                if len(audio_chunk.shape) == 1:
                    audio_chunk = np.expand_dims(audio_chunk, axis=0)
                for channel in range(audio_chunk.shape[0]):
                    batch.append(audio_chunk[channel])
                    batch_info.append((file_path, numpy_root, channel, start // sr))
                    
                    # Process batch if it reaches the specified size
                    if len(batch) == args.batch_size:
                        num_codebooks, framerate, errored_files = process_batch(
                            batch, batch_info, args, model, device, sr, batch_processor
                        )
                        num_numpy_files += len(batch) if not errored_files else 0
                        errored_audio_files.extend(errored_files)
                        batch.clear()
                        batch_info.clear()

                if end >= audio.shape[-1]:
                    break
                start = end

    # Process any remaining chunks in the batch
    if batch:
        num_codebooks, framerate, errored_files = process_batch(
            batch, batch_info, args, model, device, sr, batch_processor
        )
        num_numpy_files += len(batch) if not errored_files else 0
        errored_audio_files.extend(errored_files)

    # Write codec info to the base codes directory
    if num_audio_files > 0:
        codebook_size = (
            model.model_args.quantizer_conf.codebook_size if args.codec_type == CodecTypes.FUNCODEC 
            else model.config.codebook_size
        )
        write_codec_info(args.codec_type, args.codec_model, codes_path, sr, num_codebooks, codebook_size, framerate)

    # Print summary
    errored_audio_files = sorted(set(errored_audio_files))
    print(f"Attempted to convert {num_audio_files} audio files:")
    print(f"{num_audio_files-len(errored_audio_files)} Succeeded.")
    print(f"{len(errored_audio_files)} Errored.")
    print(f"{num_numpy_files} numpy files created.")
    print(f"{num_skipped_dirs} directories skipped.")
    if errored_audio_files:
        print("\nErrored files:")
        for file in errored_audio_files:
            print(file)
