import os


def download_gg(id, dst):
    os.system(f"wget --no-check-certificate 'https://docs.google.com/uc?export=download&id={id}' -O {dst}")


def jacksons_five_full(dst_dir='.'):
    download_gg('1ACTLuOY3cBHmqmUXdVoypJsMTtIYXHrH', os.path.join(dst, 'jacksons_five_full.mkv'))
    return os.path.join(dst_dir, 'jacksons_five_full.mkv')
