from flerken.video.utils import apply_tree, apply_single


def reencode_25_yuv420p(video_path: str, dst_path: str, *args, **kwargs):
    kwargs['input_options'] = ['-y']
    kwargs['output_options'] = ['-r', '25', '-pix_fmt', 'yuv420p']
    return apply_single(video_path, dst_path, *args, **kwargs)


def av_dataset_25fps(src, dst, ext=None):
    from multiprocessing import cpu_count
    apply_tree(src, dst, multiprocessing=0, fn=reencode_25_yuv420p, ext=ext)


if __name__ == '__main__':
    av_dataset_25fps('/media/jfm/SlaveEVO970/voxceleb/', '/media/jfm/Slave/voxceleb')
