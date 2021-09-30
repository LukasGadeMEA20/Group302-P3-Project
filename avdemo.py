import av

container = av.open(path_to_video)

for frame in container.decode(video=0):
    frame.to_image().save('frame-%04d.jpg' % frame.index)