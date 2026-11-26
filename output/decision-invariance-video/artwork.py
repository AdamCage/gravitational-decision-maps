"""Lay out generated analogy illustrations and insert them into two film chapters."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps

from render import OUT, PROJECT, encoder, read_timeline


DEFINITIONS = {
    "02": ("delivery-city.png", "Представьте город", "Два пункта доставки", 0.48, 7.70),
    "05": ("carousel.png", "Представьте карусель", "По пути можно пересечь линию", 2.30, 9.80),
}


def font(size, bold=False):
    candidates = (["C:/Windows/Fonts/segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold
                  else ["C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise OSError("Install Segoe UI or DejaVu Sans for Cyrillic titles")


def insert(scene, timeline, options):
    name, headline, subtitle, begin, end = DEFINITIONS[scene]
    dimensions = timeline["width"], timeline["height"]
    with Image.open(PROJECT / "media" / "visuals" / name) as raw:
        card = ImageOps.fit(raw.convert("RGB"), dimensions, Image.Resampling.LANCZOS)
    layer = Image.new("RGBA", dimensions)
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle((92, 74, 536, 139), radius=18, fill=(8, 19, 31, 234))
    draw.text((119, 105), "АНАЛОГИЯ  /  ИЛЛЮСТРАЦИЯ ИИ",
              font=font(25, True), fill="#B9F7EF", anchor="lm")
    draw.rounded_rectangle((92, 778, 852, 989), radius=27, fill=(8, 19, 31, 235))
    draw.rectangle((124, 820, 130, 939), fill="#57E6D0")
    draw.text((155, 847), headline, font=font(52, True), fill="#F2F8F5", anchor="lm")
    draw.text((156, 914), subtitle, font=font(34), fill="#B7C9D8", anchor="lm")
    card = Image.alpha_composite(card.convert("RGBA"), layer).convert("RGB")
    overlay = OUT / "clips" / f"{scene}-illustration.png"
    card.save(overlay)
    clip = OUT / "clips" / f"{scene}.mp4"
    target = clip.with_name(f"{scene}.artwork.mp4")
    fade = .35
    filters = (
        f"[1:v]format=yuva420p,fade=t=in:st={begin}:d={fade}:alpha=1,"
        f"fade=t=out:st={end-fade}:d={fade}:alpha=1[art];"
        "[0:v][art]overlay=shortest=1:format=yuv420[out]"
    )
    fps = timeline["fps"]
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-filter_complex_threads", "2", "-i", str(clip),
                    "-loop", "1", "-framerate", str(fps), "-i", str(overlay),
                    "-filter_complex", filters, "-map", "[out]", "-an", *options,
                    "-profile:v", "high", "-g", str(2 * fps), "-bf", "2",
                    "-color_range", "tv", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", str(target)], check=True)
    count = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                     "-show_entries", "stream=nb_frames", "-of", "json", str(target)])
    expected = next(item["frames"] for item in timeline["scenes"] if item["id"] == scene)
    if int(json.loads(count)["streams"][0]["nb_frames"]) != expected:
        raise ValueError(f"Artwork changed the number of frames in scene {scene}")
    target.replace(clip)
    print(f"{scene}: {name} integrated at {begin:.2f}–{end:.2f} s", flush=True)


def main():
    timeline = read_timeline()
    options = encoder("auto")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda scene: insert(scene, timeline, options), DEFINITIONS))


if __name__ == "__main__":
    main()
