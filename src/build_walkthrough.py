"""Build a disclosed synthetic-narration MP4 draft, fully headless on Windows.

Reproduce: .venv\\Scripts\\python.exe -m src.build_walkthrough
Requires Pillow and imageio-ffmpeg plus Windows System.Speech/Zira Desktop.
The artifact is a local upload draft, not a Loom recording or uploaded link.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import wave

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports/video_assets"
VOICE = "Microsoft Zira Desktop"
WIDTH, HEIGHT, FPS = 1280, 720, 24
BG, PANEL, WHITE, MUTED, TEAL, GOLD = "#102033", "#1B3047", "#F3F7FA", "#BCD0DE", "#38D8BC", "#FFC76E"
FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def font(size, mono=False, bold=False):
    name = "consola.ttf" if mono else ("segoeuib.ttf" if bold else "segoeui.ttf")
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def wrapped(draw, xy, text, size=28, width=1120, color=WHITE, spacing=10, mono=False):
    face = font(size, mono)
    x, y = xy
    for paragraph in text.split("\n"):
        lines, line = [], ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=face) > width and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        for line in lines:
            draw.text((x, y), line, font=face, fill=color)
            y += size + spacing
    return y


def base_slide(number, title, eyebrow):
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((48, 38, 60, 106), radius=5, fill=TEAL)
    d.text((80, 34), eyebrow.upper(), font=font(19, bold=True), fill=TEAL)
    d.text((80, 67), title, font=font(40, bold=True), fill=WHITE)
    d.line((48, 657, 1232, 657), fill="#39516A", width=1)
    d.text((48, 675), "Prepared with AI assistance | synthetic narration | local upload draft", font=font(19), fill=MUTED)
    d.text((1162, 675), f"{number:02d} / 06", font=font(19), fill=MUTED)
    return im, d


def card(d, xy, title, main, foot, width=360, color=TEAL):
    x, y = xy
    d.rounded_rectangle((x, y, x + width, y + 170), radius=16, fill=PANEL)
    d.text((x + 24, y + 21), title, font=font(23, bold=True), fill=MUTED)
    d.text((x + 24, y + 60), main, font=font(34, bold=True), fill=color)
    wrapped(d, (x + 24, y + 116), foot, size=21, width=width - 48, color=MUTED, spacing=3)


def render_slides(metrics, specs):
    hold = metrics["holdout_results"]["full"]
    cold = metrics["cold_city_stress"]["metrics"]
    selection = metrics["selection_results"]["primary"]
    slides = []
    im, d = base_slide(1, "Freight rate prediction", "Problem and evaluation design")
    wrapped(d, (64, 150), "48,000 labeled loads: Jan-Oct 2025\n12,000 future loads: Nov-Dec 2025; labels unavailable", size=29, spacing=16)
    card(d, (64, 294), "FIT", "Jan-Jun", "28,806 loads", width=354)
    card(d, (462, 294), "SELECT", "Jul-Aug", "9,671 loads", width=354)
    card(d, (860, 294), "LOCKED HOLDOUT", "Sep-Oct", "9,523 loads", width=354, color=GOLD)
    wrapped(d, (64, 510), "Holdout fit: Jan-Aug. After scoring: refit all 48,000 labeled loads.\nChronological splits reflect prediction of future shipments.", size=27, spacing=15)
    slides.append(im)

    im, d = base_slide(2, "Clean inputs; exclude the unstable proxy", "Data findings")
    card(d, (64, 145), "MISSING WEIGHT", "300", "training rows", width=354)
    card(d, (462, 145), "NEGATIVE WEIGHT", "292", "training rows", width=354, color=GOLD)
    card(d, (860, 145), "MISSING MARKET", "374", "training rows", width=354)
    wrapped(d, (64, 354), "Invalid weight -> missing + quality indicator\nNumerical medians fitted on training rows only\nPositive price outliers retained in every official score", size=27, spacing=17)
    d.rounded_rectangle((64, 514, 1214, 620), radius=14, fill=PANEL)
    wrapped(d, (86, 533), "quote_signal excluded: its monthly relationship changes and reverses.\nProxy instability is documented; leakage is not claimed as proven.", size=25, width=1098, spacing=9)
    slides.append(im)

    im, d = base_slide(3, "CatBoost models log dollars per mile", "Locked model choice")
    wrapped(d, (64, 152), "Equipment, weight, location, route, calendar and available market data\n700 iterations | depth 6 | learning rate 0.055 | seed 42", size=27, spacing=16)
    d.rounded_rectangle((64, 271, 1214, 396), radius=15, fill=PANEL)
    d.text((87, 292), "training target = log(posted_rate / distance)", font=font(27, mono=True), fill=TEAL)
    d.text((87, 345), "predicted rate = distance * exp(predicted log rate/mile)", font=font(26, mono=True), fill=WHITE)
    card(d, (64, 441), "JUL-AUG SELECTION RMSE", f"${selection['rmse']:,.2f}", "full model, quote excluded", width=554)
    card(d, (660, 441), "JUL-AUG SELECTION MAE", f"${selection['mae']:,.2f}", "config frozen before holdout", width=554)
    slides.append(im)

    im, d = base_slide(4, "Measured holdout and cold-city stress", "Evaluation evidence")
    d.rounded_rectangle((64, 154, 622, 497), radius=18, fill=PANEL)
    d.rounded_rectangle((656, 154, 1214, 497), radius=18, fill=PANEL)
    for x, title, count, result in [(88, "SEP-OCT HOLDOUT", "9,523 untouched loads", hold), (680, "CONTROLLED COLD-CITY STRESS", "8 excluded cities; 2,004 loads", cold)]:
        d.text((x, 177), title, font=font(22, bold=True), fill=TEAL)
        d.text((x, 219), count, font=font(25), fill=MUTED)
        for y, key, label in [(278, "rmse", "RMSE"), (340, "mae", "MAE"), (402, "wape", "WAPE")]:
            value = f"{100 * result[key]:.2f}%" if key == "wape" else f"${result[key]:,.2f}"
            d.text((x, y), label, font=font(27), fill=MUTED)
            d.text((x + 210, y - 3), value, font=font(32, bold=True), fill=WHITE)
    wrapped(d, (64, 529), "Cold-city stress uses Jul-Aug labels and is post-selection.\nNeither test measures unknown Nov-Dec outcomes.\nLarge positive-rate tails contribute to the gap between RMSE and MAE.", size=26, spacing=13)
    slides.append(im)

    im, d = base_slide(5, "Key code: training-only preprocessing", "Implementation walkthrough")
    d.text((64, 147), "src/features.py", font=font(24, bold=True), fill=TEAL)
    d.rounded_rectangle((64, 189, 1214, 273), radius=14, fill=PANEL)
    d.text((84, 208), 'x["weight_missing"] = (x.weight.isna() | (x.weight <= 0)).astype(int)', font=font(22, mono=True), fill=WHITE)
    d.text((84, 240), 'x["weight"] = x.weight.where(x.weight > 0)', font=font(22, mono=True), fill=WHITE)
    d.text((64, 302), "src/model.py - exact implementation excerpts", font=font(24, bold=True), fill=TEAL)
    d.rounded_rectangle((64, 345, 1214, 539), radius=14, fill=PANEL)
    code = ['self.medians_ = x[nums].median().fillna(0.0)', 'x[nums] = x[nums].fillna(self.medians_)', 'y = np.log(y / frame.distance.to_numpy())', 'pred = np.exp(pred) * frame.distance.to_numpy()', 'return np.maximum(1.0, pred)']
    for line, y in zip(code, range(360, 525, 32)):
        d.text((84, y), line, font=font(24, mono=True), fill=WHITE if not line.startswith("pred") else TEAL)
    d.text((64, 578), "Run: python -m src.run_all", font=font(29, mono=True), fill=GOLD)
    slides.append(im)

    im, d = base_slide(6, "December scenario and submission", "Artifacts and practical limits")
    chart = Image.open(ROOT / "reports/scorer_results/candidate_december.png").convert("RGB")
    chart.thumbnail((788, 432), Image.Resampling.LANCZOS)
    im.paste(chart, (64 + (788 - chart.width) // 2, 153 + (432 - chart.height) // 2))
    d.rounded_rectangle((884, 154, 1214, 576), radius=16, fill=PANEL)
    wrapped(d, (907, 178), "Fixed scenario", size=27, width=290, color=TEAL)
    wrapped(d, (907, 229), "Lexington ->\nFort Wayne\n360 miles\nDry Van\n32,000 lb", size=24, width=286, spacing=10)
    wrapped(d, (907, 432), f"31 December dates\n${metrics['december']['minimum']:.2f}-${metrics['december']['maximum']:.2f}", size=25, width=285, color=GOLD, spacing=12)
    wrapped(d, (64, 601), "12,000 predictions checked by supplied scorer; final labels are held by the assessor.", size=24, width=1152, color=MUTED)
    slides.append(im)
    for i, (im, spec) in enumerate(zip(slides, specs), 1):
        path = ASSETS / f"slide_{i:02d}.png"
        im.save(path)
        spec["image_path"] = str(path)


def narration(metrics):
    hold = metrics["holdout_results"]["full"]
    stress = metrics["cold_city_stress"]["metrics"]
    return [
        "This walkthrough is a local upload draft, prepared with AI assistance and synthetic narration. The task predicts a freight load's posted rate. Forty eight thousand labeled loads cover January through October, and twelve thousand unlabeled loads cover November and December. The solution fits January through June, selects on July and August, and freezes configuration before September and October evaluation. The final refit uses all labeled history.",
        "Data checks found three hundred missing weights, two hundred ninety two negative weights, and three hundred seventy four missing market values. Invalid weights become missing, with a quality indicator. Numerical medians come only from training rows. Positive price outliers stay in the data and scores. Quote signal shows unstable monthly relationships, including reversals, so the production models exclude it. This is evidence of proxy instability, not proof of leakage.",
        "Among quote excluded candidates, CatBoost predicts the logarithm of dollars per mile. This separates distance scaling from equipment, weight, route, calendar, and market effects. Prediction restores dollars by exponentiating and multiplying by distance. The configuration uses seven hundred iterations, depth six, and seed forty two. Selection uses dollar R M S E; configurations remain fixed for the later holdout.",
        f"On nine thousand five hundred twenty three untouched September and October loads, R M S E is {hold['rmse']:.2f} dollars, M A E is {hold['mae']:.2f} dollars, and weighted absolute percentage error is {hold['wape']*100:.2f} percent. A separate, controlled test excludes eight cities from earlier training and scores two thousand four later loads, with M A E {stress['mae']:.2f} dollars. This stress test is post selection and does not measure actual November or December accuracy.",
        "Features create the invalid weight flag without modifying source files. The model fits numerical medians on its training partition, reuses those medians at prediction, and transforms the target into log dollars per mile. The highlighted inference line restores dollars. A one dollar floor ensures feasibility, without removing training labels. The pipeline command generates models, predictions, the supplied scorer's chart, and the PDF report.",
        "The December chart uses a dedicated model trained on pickup, delivery, distance, equipment, weight, and date. The Lexington to Fort Wayne scenario changes only its date, without inventing missing market or quote values. December labels are unavailable, so this is a scenario, not measured future accuracy. The submission contains twelve thousand matched load identifiers and predictions. The scorer validates formatting; the assessor computes accuracy after submission.",
    ]


def run_hidden(command):
    subprocess.run(command, check=True, creationflags=FLAGS, capture_output=True, text=True)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    metrics_path = ROOT / "artifacts/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    specs = [{"number": i, "narration": text, "wave_path": str(ASSETS / f"narration_{i:02d}.wav")} for i, text in enumerate(narration(metrics), 1)]
    render_slides(metrics, specs)
    spec_path = ASSETS / "speech_inputs.json"
    spec_path.write_text(json.dumps(specs, ensure_ascii=True, indent=2), encoding="utf-8")
    ps_path = ASSETS / "synthesize.ps1"
    ps_path.write_text("""param([string]$SpecPath)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$slideSpecs = Get-Content -LiteralPath $SpecPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($slideSpec in $slideSpecs) {
    $speechSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        $speechSynth.SelectVoice('Microsoft Zira Desktop')
        $speechSynth.Rate = 2
        $speechSynth.Volume = 100
        $speechSynth.SetOutputToWaveFile($slideSpec.wave_path)
        $speechSynth.Speak($slideSpec.narration)
    } finally { $speechSynth.Dispose() }
}
""", encoding="utf-8")
    run_hidden(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(ps_path), "-SpecPath", str(spec_path)])
    audio_seconds = 0.0
    for spec in specs:
        with wave.open(spec["wave_path"], "rb") as wav:
            seconds = wav.getnframes() / wav.getframerate()
        spec["audio_seconds"] = round(seconds, 4)
        audio_seconds += seconds
    pad_each = max(0.7, (122 - audio_seconds) / 6)
    if audio_seconds + 6 * pad_each > 179:
        raise ValueError(f"Narration too long: {audio_seconds:.2f}s; reduce speech text or increase synthesis rate.")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clips, offset = [], 0.0
    for spec in specs:
        duration = round((spec["audio_seconds"] + pad_each) * FPS) / FPS
        spec["start_seconds"] = round(offset, 4)
        spec["duration_seconds"] = duration
        offset += duration
        clip = ASSETS / f"clip_{spec['number']:02d}.mp4"
        run_hidden([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-i", spec["image_path"], "-i", spec["wave_path"], "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-crf", "21", "-pix_fmt", "yuv420p", "-r", str(FPS), "-vf", "scale=1280:720", "-af", "apad", "-t", f"{duration:.4f}", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "1", str(clip)])
        clips.append(clip)
    concat = ASSETS / "concat.txt"
    concat.write_text("\n".join("file '" + p.as_posix() + "'" for p in clips), encoding="utf-8")
    output = ROOT / "reports/walkthrough.mp4"
    run_hidden([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(output)])
    # Decode verification checks the actual completed MP4, not just its inputs.
    run_hidden([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(output), "-f", "null", "-"])
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(output)], capture_output=True, text=True, creationflags=FLAGS)
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe.stderr)
    if not match:
        raise ValueError("Cannot verify MP4 duration.")
    duration = int(match[1])*3600 + int(match[2])*60 + float(match[3])
    if not 120 <= duration <= 180:
        raise ValueError(f"Actual MP4 duration outside 2-3 minutes: {duration}")
    for i in [1, 5, 6]:
        spec = specs[i - 1]
        run_hidden([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(spec["start_seconds"] + 2), "-i", str(output), "-frames:v", "1", str(ASSETS / f"verified_frame_{i:02d}.png")])
    metadata = {
        "status": "local upload draft; no Loom upload or link has been created",
        "disclosure": "Prepared with AI assistance; neutral Microsoft Zira synthetic narration; no claim to be the candidate's own voice.",
        "video_file": "reports/walkthrough.mp4", "duration_seconds": duration,
        "audio_seconds_from_wav_frames": round(audio_seconds, 4),
        "resolution": [WIDTH, HEIGHT], "fps": FPS, "video_codec": "H.264", "audio_codec": "AAC",
        "voice": VOICE, "speech_rate": 2,
        "narration_word_count": sum(len(spec["narration"].split()) for spec in specs),
        "build_command": ".venv\\Scripts\\python.exe -m src.build_walkthrough",
        "requirements": "Windows System.Speech + Microsoft Zira Desktop; Pillow; imageio-ffmpeg",
        "headless": True, "decode_check": "Full MP4 decoded by ffmpeg with no errors",
        "verified_frame_files": [f"reports/video_assets/verified_frame_{i:02d}.png" for i in [1, 5, 6]],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [metrics_path, ROOT / "src/features.py", ROOT / "src/model.py", ROOT / "src/train.py", ROOT / "src/run_all.py", ROOT / "reports/scorer_results/candidate_december.png"]},
        "mp4_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "slides": specs,
    }
    (ROOT / "reports/walkthrough_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ["video_file", "duration_seconds", "narration_word_count", "voice", "decode_check"]}), flush=True)


if __name__ == "__main__":
    main()
