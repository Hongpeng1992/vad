import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

n_frames = 100 #features.shape[0] #* 128
frames = []
for i in range(n_frames):
    frame = (np.random.randn(640*480) * 256).astype(np.uint8)
    frame = frame.reshape(640, 480).T
#     im = plt.imshow(frame, cmap='gray', animated=True)
    frames.append(frame)

