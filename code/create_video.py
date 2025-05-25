import os
import glob
import numpy as np
from PIL import Image
import cairosvg
from moviepy.editor import ImageSequenceClip, AudioFileClip
import VocalTractLab as vtl
import shutil
from svg_to_new_form import modify_svg

NPY_DIR = "predictions"
AUDIO_DIR = "audios_16hz"
OUTPUT_DIR = "."

def write_tract_file(array, path):
    with open(path, "w") as f:
        f.write("# The first two lines (below the comment lines) indicate the name of the vocal fold model and the number of states.\n")
        f.write("# The following lines contain the control parameters of the vocal folds and the vocal tract (states)\n")
        f.write("# in steps of 110 audio samples (corresponding to about 2.5 ms for the sampling rate of 44100 Hz).\n")
        f.write("# For every step, there is one line with the vocal fold parameters followed by\n")
        f.write("# one line with the vocal tract parameters.\n\n")
        f.write("Geometric glottis\n")
        f.write(f"{array.shape[0]}\n")
        for row in array:
            f.write("101.594 0 0.0998 0.0998 0.1 1.22204 0 0.054 0 25 -10\n")
            f.write(" ".join(str(x) for x in row) + " 0 0 0\n")

def convert_svgs_to_pngs(svg_dir, png_dir, width, height):
    os.makedirs(png_dir, exist_ok=True)
    svg_files = sorted(glob.glob(os.path.join(svg_dir, "*.svg")), key=lambda f: int(''.join(filter(str.isdigit, f))))
    for idx, svg_file in enumerate(svg_files):
        modify_svg(svg_file)
        png_path = os.path.join(png_dir, f"{idx}.png")
        cairosvg.svg2png(url=svg_file, write_to=png_path, output_width=width, output_height=height, background_color="white")

def make_gif_from_pngs(png_dir, gif_path, frame_duration=100):
    png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")), key=lambda f: int(''.join(filter(str.isdigit, f))))
    images = [Image.open(png) for png in png_files]
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=frame_duration, loop=0)

def make_video_from_pngs_and_audio(png_dir, video_path, audio_path):
    png_files = sorted([os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith(".png")], key=lambda f: int(''.join(filter(str.isdigit, f))))
    if not png_files:
        return
    audio_clip = AudioFileClip(audio_path)
    fps = len(png_files) / audio_clip.duration
    video_clip = ImageSequenceClip(png_files, fps=fps).set_audio(audio_clip)
    video_clip.write_videofile(video_path, codec="libx264")

def move_related_files(base_name, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for file in glob.glob(f"{base_name}*"):
        if file != base_name:
            shutil.move(file, os.path.join(dest_dir, os.path.basename(file)))

def main():
    npy_files = glob.glob(os.path.join(NPY_DIR, "*.npy"))
    for npy_path in npy_files:
        name = os.path.splitext(os.path.basename(npy_path))[0]
        wav_path = os.path.join(AUDIO_DIR, f"{name}.wav")
        tract_path = f"{name}.tract"
        arr = np.load(npy_path)
        write_tract_file(arr, tract_path)
        vtl.tract_sequence_to_svg(tract_path, fps=360)

        svg_dir = os.path.join(OUTPUT_DIR, f"{name}_svg")
        png_dir = os.path.join(OUTPUT_DIR, f"{name}_png")
        gif_path = os.path.join(OUTPUT_DIR, f"{name}.gif")
        video_path = os.path.join(OUTPUT_DIR, f"{name}.mp4")

        convert_svgs_to_pngs(svg_dir, png_dir, width=300, height=300)
        make_gif_from_pngs(png_dir, gif_path)
        make_video_from_pngs_and_audio(png_dir, video_path, wav_path)
        move_related_files(name, os.path.join(OUTPUT_DIR, name))

        print(f"Finished: {name}")

if __name__ == "__main__":
    main()
