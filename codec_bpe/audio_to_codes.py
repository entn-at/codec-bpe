import librosa
import os
import shutil
import argparse
import numpy as np
import torch
from enum import Enum
from tqdm import tqdm

class CodecTypes(Enum):
    ENCODEC = "encodec"
    DAC = "dac"
    MIMI = "mimi"
    FUNCODEC = "funcodec"

    def __str__(self):
        return self.value
    def __eq__(self, value):
        return str(self) == value

SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".flac", ".opus"]

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
        "--codec_type", 
        type=str, 
        choices=list(CodecTypes),
        default="encodec",
        help="Type of codec to use for encoding. Default is 'encodec'.",
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
    num_converted_audio_files = 0
    num_numpy_files = 0
    num_skipped_dirs = 0
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
            file_name_noext, _ = os.path.splitext(file)
            try:
                # Load the audio file
                num_audio_files += 1
                sr = (
                    model.model_args.sampling_rate if args.codec_type == CodecTypes.FUNCODEC 
                    else model.config.sampling_rate
                )
                audio, sr = librosa.load(file_path, sr=sr, mono=not args.stereo)

                # Encode it in chunks of size chunk_size_secs on each channel independently
                start = 0
                while True:
                    end = start + args.chunk_size_secs * sr
                    audio_chunk = audio[..., start:end]
                    if len(audio_chunk.shape) == 1:
                        audio_chunk = np.expand_dims(audio_chunk, axis=0)
                    for channel in range(audio_chunk.shape[0]):
                        channel_chunk = audio_chunk[channel]
                        if args.codec_type == CodecTypes.FUNCODEC:
                            with torch.no_grad():
                                encoded_chunk, _, _, _ = model(
                                    torch.from_numpy(channel_chunk).to(device).unsqueeze(0),
                                    bit_width=int(args.bandwidth),
                                    run_mod="encode",
                                )
                                encoded_chunk = torch.permute(encoded_chunk[0], (1, 0, 2))
                        else:
                            # prepare for model
                            inputs = processor(raw_audio=channel_chunk, sampling_rate=sr, return_tensors="pt").to(device)
                            encode_kwargs = {}
                            if args.codec_type == CodecTypes.DAC:
                                encode_kwargs["n_quantizers"] = args.n_quantizers
                            elif args.codec_type == CodecTypes.MIMI:
                                encode_kwargs["num_quantizers"] = args.n_quantizers
                            elif args.codec_type == CodecTypes.ENCODEC:
                                encode_kwargs["bandwidth"] = args.bandwidth
                            # encode
                            with torch.no_grad():
                                encoded_chunk = model.encode(**inputs, **encode_kwargs).audio_codes

                        # Save the numpy file
                        start_secs = start // sr
                        numpy_filepath = os.path.join(numpy_root, f"{file_name_noext}_c{channel}_t{start_secs:06d}.npy")
                        os.makedirs(os.path.dirname(numpy_filepath), exist_ok=True)
                        np.save(numpy_filepath, encoded_chunk.cpu().numpy(), allow_pickle=False)
                        num_numpy_files += 1

                    if end >= audio.shape[-1]:
                        break
                    start = end
                num_converted_audio_files += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Attempted to convert {num_audio_files} audio files:")
    print(f"{num_converted_audio_files} Succeeded.")
    print(f"{num_audio_files-num_converted_audio_files} Failed.")
    print(f"{num_numpy_files} numpy files created.")
    print(f"{num_skipped_dirs} directories skipped.")
