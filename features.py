import tqdm
import sys
import cv2
import collections
import numpy as np

ImageFeatures = collections.namedtuple(
    'ImageFeatures', 
    [
        'path',  # path to the image from which the descriptors were extracted
        'label',  # Image label as a string
        'data',  # Numpy array containing image features
    ]
)


# opencv documentation:
# https://docs.opencv.org/3.4.3/d6/d00/tutorial_py_root.html

def checkForSIFT():
    """Checks if the SIFT descriptor is available in the used OpenCV install"""
    try:
        cv2.xfeatures2d.SIFT_create(nfeatures=500)
    except:
        print('SIFT is unavailable in your current OpenCV install.')
        return False
    return True


def checkForSURF():
    """Checks if the SURF descriptor is available in the used OpenCV install"""
    try:
        cv2.xfeatures2d.SURF_create()
    except:
        print('SURF is unavailable in your current OpenCV install.')
        return False
    return True


def extractShiTomasiCorners(input_image, num_features=100, min_distance=20, visualize=False):
    """Extracts interest points using the Shi-Tomashi point detector and visualizes them on demand"""
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, num_features, qualityLevel=0.01, minDistance=min_distance)
    corners = np.int0(corners)
    
    if visualize:
        corner_image = input_image.copy()
        for i in corners:
            x, y = i.ravel()
            cv2.circle(corner_image, (x, y), 5, (0, 0, 255), -1)
        return corner_image, corners
    
    else:
        return corners


def extractDAISY(input_image, corner_coords, radius=16, rings=3):
    """Extracts interest points (Shi-Tomashi) and uses them to identify interesting image patches
    from which DAISY feature descriptors are extracted."""
    daisy = cv2.xfeatures2d.DAISY_create(radius=radius, q_radius=rings, q_hist=8)
    # A quick look into the c++ code at https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src
    # /daisy.cpp suggests that the size parameter is (as expected) not used, so we set it to -1 to indicate it's an
    # unused dummy value
    keypoints = [cv2.KeyPoint(float(x), float(y), _size=-1) for coords in corner_coords for (x, y) in coords]
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = daisy.compute(gray, keypoints)
    return descriptors


def extractDAISYCallback(input_image):
    """Callback which can be handed to extractFeatures.
    Extracts interest points (Shi-Tomashi) and uses them to identify interesting image patches
    from which DAISY feature descriptors are extracted."""
    corners = extractShiTomasiCorners(input_image, num_features=500)
    return extractDAISY(input_image, corners, radius=16, rings=3)


def extractFREAKCallback(input_image):
    """Callback which can be handed to extractFeatures.
    Extracts interest points (Shi-Tomashi) and uses them to identify interesting image patches
    from which FREAK feature descriptors are extracted."""
    # corner detector parameters
    num_features = 500
    min_distance = 20
    # use shi tomasi again to extract corners
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, num_features, qualityLevel=0.01, minDistance=min_distance)
    corners = np.int0(corners)
    
    freak = cv2.xfeatures2d.FREAK_create()
    keypoints = [cv2.KeyPoint(float(x), float(y), _size=-1) for coords in corners for (x, y) in coords]
    keypoints, descriptors = freak.compute(gray, keypoints)
    descriptors = np.float32(descriptors)
    descriptors /= np.max(descriptors)
    return descriptors


def extractLUCIDCallback(input_image):
    """Callback which can be handed to extractFeatures.
    Extracts interest points (Shi-Tomashi) and uses them to identify interesting image patches
    from which LUCID feature descriptors are extracted.  """
    # corner detector parameters
    num_features = 500
    min_distance = 20
    # use shi tomasi again to extract corners
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, num_features, qualityLevel=0.01, minDistance=min_distance)
    corners = np.int0(corners)
    lucid = cv2.xfeatures2d.LUCID_create()
    keypoints = [cv2.KeyPoint(float(x), float(y), _size=-1) for coords in corners for (x, y) in coords]
    keypoints, descriptors = lucid.compute(input_image, keypoints)
    descriptors = np.float32(descriptors)
    descriptors /= np.max(descriptors)
    return descriptors


def extractVGGCallback(input_image):
    """Callback which can be handed to extractFeatures.
    Extracts interest points (Shi-Tomashi) and uses them to identify interesting image patches
    from which VGG feature descriptors are extracted.  """
    # corner detector parameters
    num_features = 500
    min_distance = 20
    # use shi tomasi again to extract corners
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, num_features, qualityLevel=0.01, minDistance=min_distance)
    corners = np.int0(corners)
    vgg = cv2.xfeatures2d.VGG_create()
    keypoints = [cv2.KeyPoint(float(x), float(y), _size=-1) for coords in corners for (x, y) in coords]
    keypoints, descriptors = vgg.compute(input_image, keypoints)
    descriptors = np.float32(descriptors)
    descriptors /= np.max(descriptors)
    return descriptors


def extractORBCallback(input_image):
    """Callback which can be handed to extractFeatures.
    Extracts interest points (FAST) and uses them to identify interesting image patches
    from which ORiented Brief feature descriptors are extracted.  """
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    descriptors = np.float32(descriptors)
    descriptors /= np.max(descriptors)
    return descriptors


def extractSIFTCallback(input_image):
    """Callback which can be handed to extractFeatures.
    Extracts interest points (SIFT) and uses them to identify interesting image patches
    from which SIFT feature descriptors are extracted.  """
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def extractSURFCallback(input_image):
    """Callback which can be handed to extractFeatures.
    Extracts interest points (SURF) and uses them to identify interesting image patches
    from which SURF feature descriptors are extracted.  """
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    return descriptors


def extractBoostDescCallback(input_image):
    """Callback which can be handed to extractFeatures.
    Extracts interest points (Shi-Tomashi) and uses them to identify interesting image patches
    from which boosting feature descriptors are extracted.  """
    # corner detector parameters
    num_features = 500
    min_distance = 20
    # use shi tomasi again to extract corners
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, num_features, qualityLevel=0.01, minDistance=min_distance)
    corners = np.int0(corners)
    boost_desc = cv2.xfeatures2d.BoostDesc_create()
    keypoints = [cv2.KeyPoint(float(x), float(y), _size=-1) for coords in corners for (x, y) in coords]
    keypoints, descriptors = boost_desc.compute(input_image, keypoints)
    descriptors = np.float32(descriptors)
    descriptors /= np.max(descriptors)
    return descriptors


def extractFeatures(image_paths, descriptor_dict, label):
    """ Extracts image features for a given set of images based on a given dict of feature extractor callbacks.
    Returns a dict of extracted features using the ImageFeatures namedtuple data structure"""
    extracted_features = dict((key, []) for (key, value) in descriptor_dict.items())
    # force python to empty stdout buffers for tqdm progress counter
    sys.stdout.flush()
    for img_path in tqdm.tqdm(image_paths, total=len(image_paths)):
        # load image
        current_image = cv2.imread(img_path)
        for (descriptor_name, descriptor_function) in descriptor_dict.items():
            descriptors = descriptor_function(current_image)
            image_features = ImageFeatures(img_path, label, descriptors)
            extracted_features[descriptor_name].append(image_features)
    return extracted_features
