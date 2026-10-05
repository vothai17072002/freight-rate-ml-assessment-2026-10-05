"""Replace narration while copying every original video packet and timestamp."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import wave

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, creationflags=FLAGS)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def wav_seconds(path):
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


def video_hash(ffmpeg, path, timed=False):
    command = [ffmpeg, "-v", "error"]
    if timed:
        command.append("-copyts")
    command += ["-i", str(path), "-map", "0:v:0", "-c", "copy",
                "-f", "framehash" if timed else "hash", "-hash", "sha256", "-"]
    return run(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/walkthrough.mp4")
    parser.add_argument("--metadata", type=Path, default=ROOT / "reports/walkthrough_metadata.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/walkthrough_male.mp4")
    parser.add_argument("--voice", default="Microsoft David Desktop")
    parser.add_argument("--speech-rate", type=int, default=2)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate output file to preserve the original video.")
    original = json.loads(args.metadata.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if source_hash != original["mp4_sha256"]:
        raise ValueError("Input video does not match its recorded metadata.")
    assets = ROOT / "reports/video_assets/male_voice"
    assets.mkdir(parents=True, exist_ok=True)
    specs = []
    for slide in original["slides"]:
        specs.append({"number": slide["number"], "narration": slide["narration"],
                      "wave_path": str(assets / f"raw_{slide['number']:02d}.wav")})
    spec_path = assets / "speech_inputs.json"
    spec_path.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    script = assets / "synthesize.ps1"
    script.write_text("""param([string]$SpecPath, [string]$VoiceName, [int]$SpeechRate)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$slideSpecs = Get-Content -LiteralPath $SpecPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($slideSpec in $slideSpecs) {
    $speechSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        $speechSynth.SelectVoice($VoiceName)
        $speechSynth.Rate = $SpeechRate
        $speechSynth.Volume = 100
        $speechSynth.SetOutputToWaveFile($slideSpec.wave_path)
        $speechSynth.Speak($slideSpec.narration)
    } finally { $speechSynth.Dispose() }
}
""", encoding="utf-8")
    run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(script), "-SpecPath", str(spec_path), "-VoiceName", args.voice,
         "-SpeechRate", str(args.speech_rate)])
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    segments = []
    fit_records = []
    for slide, spec in zip(original["slides"], specs):
        raw_seconds = wav_seconds(spec["wave_path"])
        desired_speech = slide["audio_seconds"]
        tempo = raw_seconds / desired_speech
        if not 0.5 <= tempo <= 2:
            raise ValueError(f"Unsafe timing adjustment for slide {slide['number']}: {tempo}")
        fitted = assets / f"fitted_{slide['number']:02d}.wav"
        # atempo changes duration without changing pitch; original pause is retained.
        audio_filter = (f"atempo={tempo:.10f},apad,atrim=duration={desired_speech:.10f},"
                        f"apad,atrim=duration={slide['duration_seconds']:.10f}")
        run([ffmpeg, "-v", "error", "-y", "-i", spec["wave_path"], "-af", audio_filter,
             "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(fitted)])
        segments.append(fitted)
        fit_records.append({"number": slide["number"], "original_speech_seconds": desired_speech,
                            "raw_male_speech_seconds": round(raw_seconds, 6), "tempo_factor": tempo,
                            "original_slide_seconds": slide["duration_seconds"],
                            "fitted_segment_seconds": wav_seconds(fitted), "wave_path": str(fitted)})
    combined = assets / "male_narration.wav"
    with wave.open(str(combined), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(44100)
        for segment in segments:
            with wave.open(str(segment), "rb") as source:
                assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 44100)
                output.writeframes(source.readframes(source.getnframes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run([ffmpeg, "-v", "error", "-y", "-copyts", "-i", str(args.input), "-i", str(combined),
         "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "1",
         "-movflags", "+faststart", str(args.output)])
    before_stream = video_hash(ffmpeg, args.input)
    after_stream = video_hash(ffmpeg, args.output)
    before_timed = video_hash(ffmpeg, args.input, timed=True)
    after_timed = video_hash(ffmpeg, args.output, timed=True)
    if before_stream != after_stream or before_timed != after_timed:
        raise ValueError("Video payload or timestamps changed; output rejected.")
    run([ffmpeg, "-v", "error", "-i", str(args.output), "-f", "null", "-"])
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(args.output)],
                           capture_output=True, text=True, creationflags=FLAGS)
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe.stderr)
    if not match:
        raise ValueError("Output duration could not be read.")
    duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
    if duration != original["duration_seconds"]:
        raise ValueError(f"Duration changed: {original['duration_seconds']} -> {duration}")
    metadata = copy.deepcopy(original)
    metadata.update({"video_file": args.output.relative_to(ROOT).as_posix(), "voice": args.voice,
                     "speech_rate": args.speech_rate, "duration_seconds": duration,
                     "disclosure": "Prepared with AI assistance; Microsoft David male synthetic narration; no claim to be the candidate's own voice.",
                     "requirements": "Windows System.Speech + Microsoft David Desktop; imageio-ffmpeg",
                     "build_command": ".venv\\Scripts\\python.exe -m src.change_walkthrough_voice",
                     "mp4_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})
    for slide, record in zip(metadata["slides"], fit_records):
        slide["wave_path"] = record["wave_path"]
    metadata["voice_replacement"] = {"source_video": args.input.relative_to(ROOT).as_posix(),
        "source_mp4_sha256": source_hash, "source_voice": original["voice"],
        "video_stream_sha256": after_stream.strip().split("=", 1)[1],
        "timed_video_packet_table_sha256": hashlib.sha256(after_timed.encode()).hexdigest(),
        "video_packet_payloads_and_timestamps_identical": True, "narration_text_unchanged": True,
        "audio_fit": fit_records, "timing_policy": "Fit male speech to each original speech window without pitch shift, preserving original slide intervals."}
    metadata_path = args.output.with_name(args.output.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "duration_seconds": duration,
                      "voice": args.voice, "original_video_and_timing_preserved": True}))


if __name__ == "__main__":
    main()
