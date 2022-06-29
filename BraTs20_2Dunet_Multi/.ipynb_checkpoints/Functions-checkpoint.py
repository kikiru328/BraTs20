def read_image(path):
    import nibabel as nib
    import numpy as np
    image = nib.load(path)
    image = (image.dataobj)
    return np.asarray(image)
