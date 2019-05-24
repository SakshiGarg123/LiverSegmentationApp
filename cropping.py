from PIL import Image
import numpy as np

def crop_file(maskfile,inputfile,outputfile):
    liver = Image.open(maskfile)
    liver = liver.resize((256, 256))
    mask = np.array(liver)
    mask = mask / 255.
    # Mask of non-black pixels (assuming image has a single channel).
    mask = mask > 0
    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    if (coords.size == 0):
        print('no liver found, skipping....')
        return
    print('cropping...')

    # Bounding box of non-black pixels.
    print(coords.min(axis=0))
    x0, y0,z0 = coords.min(axis=0) - 5
    x1, y1,z1 = coords.max(axis=0) + 5  # slices are exclusive at the top

    # Get the contents of the bounding box.

    vol = Image.open(inputfile)
    vol = vol.resize((256, 256))
    vol = np.array(vol) / 255.
    vol = np.stack((vol,) * 3, axis=-1)
    vol = vol * mask
    img = Image.fromarray(np.uint8(vol * 255))

    img.crop((y0, x0, y1, x1)).resize((256, 256)).save(outputfile)