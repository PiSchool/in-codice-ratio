import matplotlib.pyplot as plt
from random import sample
import os
import re
import json
import string
import numpy as np
import cv2
import networkx as nx


PATTERN2SYMBOL = json.load(open('abbr_matchings.json'))
MATCHINGS = [re.compile(k) if k!="\\." else re.compile("\.") for k in PATTERN2SYMBOL.keys()]

HIGH_SYMBOLS = ['A','B','b','C','D','d_tall','E','F','f','G','h','k','L','l',\
                'N','O','Q','R','S','T','U','l_stroke',"s_tall",'w','W']
MID_SYMBOLS = ["a","c","d_mid","e",'I',"i",'M',"m","n","o","r",'r_2',"t","u","s_mid",'P','tyronian_note','H','prae','b_semicolon']
LOW_SYMBOLS = ['j',"s_low","g","p",'per','pro',"q",'que','qui','rum','x','y','z','9','dot']
UPPER_SYMBOLS = ['apostrophe','curl']

OLD_SYMBOLS = ['a','b','c','d_mid','d_tall','e','f','g','h','i','l','m','n','o','p','q','r','s_tall','s_mid','s_low','t','u']


def count_black_pixels_per_row(image):
    black_counts = []
    for row in image:
        black_pixels = len(row) - cv2.countNonZero(row)
        black_counts.append(black_pixels)

    return np.array(black_counts)

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

def get_connected_components_bbxs(image):
    """
        return x,y,w,h bounding boxes for each connected component in the image,
        excluding the background.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(image))
    stats = sorted(stats, key=lambda s: s[4])
    conn_comp_stats = stats[:-1]
    return conn_comp_stats

def largest_boundingbox(bbxs):
    """
        compute largest bounding box given a list of bounding boxes
    """
    x1 = min([bbx[0] for bbx in bbxs])
    y1 = min([bbx[1] for bbx in bbxs])
    x2 = max([bbx[0]+bbx[2] for bbx in bbxs])
    y2 = max([bbx[1]+bbx[3] for bbx in bbxs])
    return x1, y1, x2, y2


def generate_word(word, dataset_filenames):
    alt_decomp = [(a.span(),a.re.pattern) for m in MATCHINGS for a in m.finditer(word)]
    alt_graph = nx.DiGraph()

    for (start, end), pattern in alt_decomp:
        alt_graph.add_edge(start, end, label=pattern)
    try:
        node_path = sample(list(nx.all_simple_paths(alt_graph, 0, len(word))), 1)[0]
    except:
        print(word)
        raise
    pattern_seq = [alt_graph.get_edge_data(u,v)['label'] for u, v in zip(node_path, node_path[1:])]

    symbol_seq = []
    for p in pattern_seq:
        patt = sample(PATTERN2SYMBOL[p],1)[0]
        if isinstance(patt, list):
            symbol_seq += patt
        else:
            symbol_seq.append(patt)

    images = []
    for i,c in enumerate(symbol_seq):
        s = sample(dataset_filenames[c], 1)[0]
        img = cv2.imread(s, cv2.IMREAD_GRAYSCALE)

        bbxs = get_connected_components_bbxs(img)

        if c in OLD_SYMBOLS:
            char_x, char_y, char_w, char_h, char_a = bbxs[-1]
            char_x1, char_y1, char_x2, char_y2 = char_x, char_y, char_x+char_w, char_y+char_h
        else:
            char_x1, char_y1, char_x2, char_y2 = largest_boundingbox(bbxs)

        images.append((c, img[char_y1:char_y2,char_x1:char_x2]))

    width = np.sum([img.shape[1] for c, img in images])*2
    height = np.max([img.shape[0] for c, img in images])*2

    midline = int(height/2)

    blank = np.ones((height, width), dtype='uint8')*255

    for i,(c, image) in enumerate(images):
        # compute bounding box for cropped image to find rightmost x
        crop_start = midline-int(image.shape[0]/2)
        crop_end = crop_start + image.shape[0]
        img_crop = blank.copy()[crop_start:crop_end,:]

        bbxs = get_connected_components_bbxs(img_crop)

        if len(bbxs) > 0:
            start_x = max([bbx[0]+bbx[2] for bbx in bbxs])
        else:
            start_x = int(np.sum([im.shape[1] for c, im in images[:i]]))

        if c in MID_SYMBOLS:
            start_y = midline - int(image.shape[0]/2)
        if c in HIGH_SYMBOLS:
            start_y = midline - int((image.shape[0]/2)*1.3)
        if c in LOW_SYMBOLS:
            start_y = midline - int((image.shape[0]/2)*0.7)
        if c in UPPER_SYMBOLS:
            start_y = midline - int((image.shape[0]/2)*2)

        end_x = start_x + image.shape[1]
        end_y = start_y + image.shape[0]
        try:
            blank[start_y:end_y,start_x:end_x] = cv2.bitwise_and(blank[start_y:end_y,start_x:end_x], image, mask=image)
        except:
            print(blank[start_y:end_y,start_x:end_x].shape, image.shape)
            raise

    # bounding box on the generated word to get rid of extra white space
    bbxs = get_connected_components_bbxs(blank)
    blank_x1, blank_y1, blank_x2, blank_y2 = largest_boundingbox(bbxs)

    return blank[blank_y1:blank_y2,blank_x1:blank_x2]

def generate_lines(text, maxlen, dataset_filenames, dst_folder='synthetic_lines/', offset=0):
    """
        text: array of words (str)
        maxlen: maximum width of the line image

        return: # of lines generated
    """
    line_count = 0
    linelen = 0
    text = list(text)
    words_in_line = []

    for i,wordstr in enumerate(text):
        w = generate_word(wordstr, dataset_filenames)
        linelen += w.shape[1]
        words_in_line.append((wordstr, w))
        if linelen > maxlen or i == (len(text)-1): # we have completed a row, generate image
            minspace, maxspace = (4,11) #px
            height = np.max([wil.shape[0] for _,wil in words_in_line])*2
            width = np.sum([wil.shape[1] for _,wil in words_in_line]) + len(words_in_line)*11
            blankline = np.ones((height, width), dtype='uint8')*255
            start_x = np.random.randint(minspace,maxspace)
            linestr = ''

            for s, wil in words_in_line:
                end_x = start_x + wil.shape[1]

                blackpx_w = count_black_pixels_per_row(wil)
                blackpx_w_smooth = smooth(blackpx_w, window_len=int(wil.shape[0]/3))

                loc_maxima = np.r_[True, blackpx_w_smooth[1:] >= blackpx_w_smooth[:-1]] &\
                                  np.r_[blackpx_w_smooth[:-1] >= blackpx_w_smooth[1:], True]
                loc_minima = np.r_[True, blackpx_w_smooth[1:] <= blackpx_w_smooth[:-1]] &\
                                  np.r_[blackpx_w_smooth[:-1] <= blackpx_w_smooth[1:], True]
                max_ixs = np.where(loc_maxima)[0]
                min_ixs = np.where(loc_minima)[0]
                top2_ixs = max_ixs[np.argsort(blackpx_w_smooth[max_ixs])[-2:]] # top 2 maximum values

                if len(top2_ixs) == 2:
                    top1, top2 = sorted(top2_ixs)
                    midval = min_ixs[min_ixs>top1]
                    midval = midval[midval<top2]
                    midval = midval[height/4<midval]
                    midval = midval[midval<height*3/4]
                    if len(midval) > 0:
                        start_y = int(height/2 - midval[-1])
                    else:
                        start_y = int(height/2 - wil.shape[0]/2)
                else:
                    start_y = int(height/2 - wil.shape[0]/2)

                end_y = start_y + wil.shape[0]

                blankline[start_y:end_y,start_x:end_x] = cv2.bitwise_and(blankline[start_y:end_y,start_x:end_x], wil)
                start_x += wil.shape[1] + np.random.randint(minspace,maxspace)
                linestr += s+' '

            # bounding box on the generated line to get rid of extra white space
            bbxs = get_connected_components_bbxs(blankline)
            blank_x1, blank_y1, blank_x2, blank_y2 = largest_boundingbox(bbxs)

            cv2.imwrite(dst_folder+str(offset+line_count)+'.png',blankline[blank_y1:blank_y2,blank_x1:blank_x2])
            with open(dst_folder+str(offset+line_count)+'.txt', mode='w') as f:
                f.write(linestr+'\n')
            #reset
            line_count += 1
            linelen = 0
            words_in_line = []

    return line_count


if __name__ == '__main__':
    dataset_folder = 'character_samples/'
    corpus_folder = 'corpus/'
    dest_folder = 'ocr/'

    dataset_filenames = {
        char: [dataset_folder+char+'/'+f for f in os.listdir(dataset_folder+char)]
        for char in os.listdir(dataset_folder)
    }

    corpus_files = sorted([corpus_folder+f for f in os.listdir(corpus_folder)])
    tot_imgs = 0

    for corpus_file in corpus_files:
        print(corpus_file)
        corpus = open(corpus_file)
        text = re.sub(r'[0-9]|\'|"|,|:|;|\(|\)|\[|\]|<|>|!|\?|—|-|†', '', corpus.read().replace('\n',' ').replace('.', ' . '))

        line_imgs = generate_lines(filter(lambda x: len(x)>0,text.split(' ')[1:]),
                                   1200, dataset_filenames, dst_folder=dest_folder, offset=tot_imgs)
        tot_imgs += line_imgs
        print('Lines generated:', line_imgs)
    print("total lines:",tot_imgs)
