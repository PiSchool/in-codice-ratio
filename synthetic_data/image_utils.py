import cv2
import random
import numpy as np

def open_grayscale_img(filename, crop=False):
    if crop:
        return crop_white_space(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def open_sample(filename):
    return add_padding_center(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

def open_many_samples(filenames):
    samples_images = []

    for filename in filenames:
        samples_images.append(open_sample(filename))

    return np.array(samples_images)

def open_many_samples_center(filenames):
    samples_images = []

    for filename in filenames:
        samples_images.append(add_padding_center(open_grayscale_img(filename, crop=True), width=56))

    return np.array(samples_images)

def write_image(image, filename):
    cv2.imwrite(filename, image)

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise( ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise( ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(int(window_len/2)-1):-(int(window_len/2))]

def augment(images, to_n,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.2,
            seed=1337):
    random.seed(seed)
    img_n = len(images)
    needed = to_n - img_n
    augmented_images = []
    i = 0
    while len(augmented_images) < needed:
        new_image = cv2.bitwise_not(images[i%img_n].copy())
        h, w = new_image.shape[:2]
        #cv2.imshow('Before',new_image)

        # random zooming
        zoom_factor = random.uniform(-zoom_range, zoom_range)
        new_image = cv2.resize(new_image, (0,0), fx=1.0+zoom_factor, fy=1.0+zoom_factor)

        # random shearing
        shear_factor = random.uniform(-shear_range, shear_range)
        pt1 = np.float32([[0,0],[0,h],[w,0]])
        pt2 = np.float32([[w*shear_factor, 0], [0,h], [w*(1+shear_factor),0]])
        shear_mx = cv2.getAffineTransform(pt1,pt2)
        new_image = cv2.warpAffine(new_image, shear_mx, (w,h))

        # random rotation
        rotn_factor = random.randint(-rotation_range, rotation_range)
        rotn_mx = cv2.getRotationMatrix2D((w/2, h/2), angle=rotn_factor, scale=1.0)
        new_image = cv2.warpAffine(new_image, rotn_mx, (w,h))

        # random vertical/horizontal shift
        hshift_factor = random.uniform(-width_shift_range, width_shift_range)
        vshift_factor = random.uniform(-height_shift_range, height_shift_range)
        shift_mx = np.float32([[1,0,w*hshift_factor],[0,1,h*vshift_factor]])
        new_image = cv2.warpAffine(new_image, shift_mx, (w,h))
        new_image = cv2.threshold(new_image, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # print "rotn:", rotn_factor
        # print "shift: v",vshift_factor*h, "h", hshift_factor*w
        # print "shear:", shear_factor*w
        # print "zoom:", zoom_factor
        # print new_image.shape
        # cv2.imshow('After',new_image)
        # cv2.waitKey(0)

        augmented_images.append(cv2.bitwise_not(new_image))

        i += 1

    samples = np.concatenate((images, np.array(augmented_images, dtype='uint8')))
    return samples


def extract_sample_from_coordinates(page, coordinates, clean=True):
    if isinstance(page, str):
        page = open_grayscale_img(page)

    x, y, w, h = coordinates

    crop = page[y:y+h, x:x+w]
    if clean:
        crop = clean_unconnected_noise(crop)
        black_pixels = count_black_pixels_per_column(crop)
        white_space_ixs = space_start_ixs(black_pixels)
        # rimozione dello spazio in eccesso
        for space_start in white_space_ixs:
            if(space_start > crop.shape[1]/3):
                crop = crop[:,:space_start]
        if(np.any(crop == 0)):
            crop = crop_white_space(crop)

    return crop

def count_black_pixels(image):
    return (image.shape[0]*image.shape[1]) - cv2.countNonZero(image)

def clean_unconnected_noise(image, strength=0.3):
    # copia dell'immagine, invertita per trovare i contorni
    image = cv2.copyMakeBorder(image, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=255)
    image_copy = cv2.bitwise_not(image.copy())

    # cv2.imshow('segment', cv2.resize(image, (0,0), fx=4.0,fy=4.0))
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_area = max([cv2.contourArea(c) for c in contours])
        threshold = max_area*strength

        for cnt in contours:
            if (cv2.contourArea(cnt) < threshold):
                cv2.fillPoly(image, [cnt], (255,255,255))
    # cv2.imshow('segment', cv2.resize(image, (0,0), fx=4.0,fy=4.0))
    # cv2.waitKey(0)
    return image

def space_start_ixs(black_pixels, space_width=3, tolerance=0):
    """
        ritorna una lista di indici in corrispondenza dei quali
        c'e uno spazio <= a tolerance di dimensione space_width
    """
    ixs = []
    for i in range(0, len(black_pixels)-space_width):
        sub_array = np.array(black_pixels[i:i+space_width])
        if np.all(sub_array <= tolerance):
            ixs.append(i)
    return ixs


def character_height(segment, top_baseline, bottom_baseline):
    upper_half = segment[:top_baseline]
    lower_half = segment[bottom_baseline:]

    upper_elements = upper_half.shape[0]*upper_half.shape[1]
    lower_elements = lower_half.shape[0]*lower_half.shape[1]

    if (cv2.countNonZero(upper_half) < upper_elements) and (cv2.countNonZero(lower_half) == lower_elements):
        return 'high'
    if (cv2.countNonZero(upper_half) == upper_elements) and (cv2.countNonZero(lower_half) < lower_elements):
        return 'low'
    if (cv2.countNonZero(upper_half) == upper_elements) and (cv2.countNonZero(lower_half) == lower_elements):
        return 'mid'

def find_baseline(image, margin=5, min_line_height=10):
    """
        trova i limiti superiori e inferiori dei caratteri mediani.
        margin determina la tolleranza rispetto al valore trovato,
        min_line_height la distanza minima fra un picco e l'altro.
    """
    black_count = count_black_pixels_per_row(image)
    max_1st_ix = np.argmax(black_count) # prima baseline
    # ora occorre stabilire se la successiva baseline si trova al di sopra o al di sotto di essa:
    # considero le righe al di sopra e al di sotto della prima baseline con un margine
    # pari a min_line_height (controllando di non eccedere la dimensione dell'immagine)
    start = (max_1st_ix + min_line_height)
    end = (max_1st_ix - min_line_height)
    if end < 1:
        end = 1
    if start > image.shape[0]-1:
        start = image.shape[0]-1

    upper_half = black_count[:end]
    lower_half = black_count[start:]
    # considero i valori massimi per questi sottoinsiemi
    upper_max_ix = np.argmax(upper_half)
    lower_max_ix = np.argmax(lower_half) + (len(black_count)-len(lower_half))

    max_2nd_ix = lower_max_ix # seconda baseline
    # scelgo l'indice della colonna con piu' pixel neri
    if black_count[upper_max_ix] > black_count[lower_max_ix]:
        max_2nd_ix = upper_max_ix

    # l'indice con valore piu' basso sara' la baseline piu' alta, e viceversa
    upper_bound, lower_bound = sorted((max_1st_ix, max_2nd_ix))

    result = [upper_bound-5, lower_bound+5]
    if result[0] < 0:
        result[0] = 0
    if result[1] > image.shape[0]-1:
        result[1] = image.shape[0]-1
    return result

def crop_white_space(image, threshold=255):
    """
        rimuove lo spazio bianco eccedente, in alto, in basso, a dx e a sx.
        NB: funziona solo se l'immagine e' in grayscale
    """
    # una maschera di valori booleani. Ha la stessa struttura dell'immagine.
    # True se il pixel non e' bianco.
    img_mask = image < threshold
    # mask.any(1), mask.any(0) producono rispettivamente le maschere per righe e colonne:
    # True se la riga (o la colonna) contiene almeno un pixel nero.
    # sono monodimensionali.
    row_mask = img_mask.any(1)
    col_mask = img_mask.any(0)
    # np.ix_ costruisce gli indici che genereranno il prodotto fra le due maschere
    return image[np.ix_(row_mask, col_mask)]

def add_padding(img,color=255,width=34,height=56):
    """
        Aggiunge un bordo bianco in alto e a destra fino a raggiungere
        le width e height desiderate
    """
    top = max(0, height - img.shape[0])
    right = max(0, width - img.shape[1])
    return cv2.copyMakeBorder(img, top=top, bottom=0, left=0, right=right, borderType=cv2.BORDER_CONSTANT, value=color)

def add_padding_center(img, color=255, width=34, height=56):
    """
        Aggiunge una cornice bianca attorno all'immagine
        fino a raggiungere le width e height desiderate
    """
    vertical_border = max(0, height - img.shape[0])
    horizontal_border = max(0, width - img.shape[1])
    top = vertical_border/2
    bottom = vertical_border - top
    left = horizontal_border/2
    right = horizontal_border - left
    return cv2.copyMakeBorder(img, top=int(top), bottom=int(bottom), left=int(left), right=int(right), borderType=cv2.BORDER_CONSTANT, value=color)

def count_black_pixels_per_column(image):
    """
        conta i pixel neri di ciascuna colonna dell'immagine.
        ritorna un array 1D contenente i valori di nero in ordine:
        a[0] corrisponde al conteggio di pixel neri della prima colonna da sx.
    """
    black_counts = []
    for i in np.arange(image.shape[1]):
        # pixel totali - pixel diversi da 0 (= neri)
        black_pixels = len(image[:,i]) - cv2.countNonZero(image[:,i])
        black_counts.append(black_pixels)

    return np.array(black_counts)

def count_black_pixels_per_row(image):
    """
        conta i pixel neri di ciascuna riga dell'immagine.
        ritorna un array 1D contenente i valori di nero in ordine:
        a[0] corrisponde al conteggio di pixel neri della prima riga dall'alto.
    """
    black_counts = []
    for row in image:
        black_pixels = len(row) - cv2.countNonZero(row)
        black_counts.append(black_pixels)

    return np.array(black_counts)
