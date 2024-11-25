import argparse
import collections
import os
import subprocess


def download_and_uncompress(url, output_dir):
    scene_file = os.path.basename(url)
    print(f'Downloading {url} ...')
    if not os.path.exists(scene_file):
        subprocess.run(['wget', url])
    subprocess.run(['tar', '-xzf', scene_file])
    subprocess.run(['rm', scene_file])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='replica', choices=['mp3d', 'replica'])
    parser.add_argument('--rir-type', default='binaural_rirs', choices=['binaural_rirs', 'ambisonic_rirs'])
    output_dir="D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/media/aluo/big2/soundspaces_full/binaural_rirs"
    aws_root_dir = 'http://dl.fbaipublicfiles.com/SoundSpaces/'
   # scenes = os.listdir(os.path.join('data/metadata/', args.dataset))
    scene='room_2'
    scene_file = os.path.join(aws_root_dir, 'binaural_rirs','replica',scene+'.tar.gz')
    print("scene_file:"+scene_file)
    download_and_uncompress(scene_file, output_dir)


if __name__ == '__main__':
    main()