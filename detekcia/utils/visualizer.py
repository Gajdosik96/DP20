import abc
import argparse
import colorsys
import logging
from time import sleep, time
from typing import Tuple, Union, Optional, List, Any

import cv2
import numpy as np
from screeninfo import Monitor, screeninfo

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


class Drawer:
    """
        Drawer class contains all basic drawing functions. There are most commonly used functions.
    """

    @classmethod
    def create_unique_color_float(cls, tag: int, hue_step: float = 0.05):
        """Create a unique RGB color code for a given track id (tag).

        The color code is generated in HSV color space by moving along the
        hue angle and gradually changing the saturation.

        Parameters
        ----------
        tag : int
            The unique target identifying tag.
        hue_step : float
            Difference between two neighboring color codes in HSV space (more
            specifically, the distance in hue channel).

        Returns
        -------
        (float, float, float)
            RGB color code in range [0, 1]

        """
        h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
        r, g, b = colorsys.hsv_to_rgb(h, 1., v)
        return r, g, b

    @classmethod
    def create_unique_color_uchar(cls, tag: int, hue_step: float = 0.41):
        """Create a unique RGB color code for a given track id (tag).

        The color code is generated in HSV color space by moving along the
        hue angle and gradually changing the saturation.

        Parameters
        ----------
        tag : int
            The unique target identifying tag.
        hue_step : float
            Difference between two neighboring color codes in HSV space (more
            specifically, the distance in hue channel).

        Returns
        -------
        (int, int, int)
            RGB color code in range [0, 255]

        """
        r, g, b = cls.create_unique_color_float(tag, hue_step)
        return int(255 * r), int(255 * g), int(255 * b)

    @classmethod
    def is_in_bounds(cls, mat: np.ndarray, roi: (int, int, int, int)):
        """Check if ROI is fully contained in the image.

        Parameters
        ----------
        mat : ndarray
            An ndarray of ndim>=2.
        roi : (int, int, int, int)
            Region of interest (x, y, width, height) where (x, y) is the top-left
            corner.

        Returns
        -------
        bool
            Returns true if the ROI is contain in mat.

        """
        if roi[0] < 0 or roi[0] + roi[2] >= mat.shape[1]:
            return False
        if roi[1] < 0 or roi[1] + roi[3] >= mat.shape[0]:
            return False
        return True

    @classmethod
    def view_roi(cls, mat: np.ndarray, roi: (int, int, int, int)):
        """Get sub-array.

        The ROI must be valid, i.e., fully contained in the image.

        Parameters
        ----------
        mat : ndarray
            An ndarray of ndim=2 or ndim=3.
        roi : (int, int, int, int)
            Region of interest (x, y, width, height) where (x, y) is the top-left
            corner.

        Returns
        -------
        ndarray
            A view of the roi.

        """
        sx, ex = roi[0], roi[0] + roi[2]
        sy, ey = roi[1], roi[1] + roi[3]
        if mat.ndim == 2:
            return mat[sy:ey, sx:ex]
        else:
            return mat[sy:ey, sx:ex, :]

    @classmethod
    def rectangle(cls,
                  frame: np.ndarray,
                  x: Union[float, int],
                  y: Union[float, int],
                  w: Union[float, int],
                  h: Union[float, int],
                  label: Optional[str] = None,
                  color: Tuple[int, int, int] = (0, 0, 255),
                  thickness: int = 2):
        """Draw a rectangle.

        Parameters
        ----------
        frame : np.ndarray
            Input frame where the rectangle is drawn.
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.
        color: Tuple [int, int, int]
            Rectangle color.
        thickness: float | int
            Thickness of the rectangle outline, if positive. Negative thickness means that a filled circle is to be
            drawn.
        """
        font_scale = min(round(frame.shape[0] * frame.shape[1]) / 307200, 2.5)

        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        cv2.rectangle(frame, pt1, pt2, color, thickness)

        if label is not None:
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)

            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
            cv2.rectangle(frame, pt1, pt2, color, -1)
            cv2.putText(frame, label, center, cv2.FONT_HERSHEY_PLAIN,
                        font_scale, (255, 255, 255), thickness)

    @classmethod
    def rectangle_filled_with_opacity(cls,
                  frame: np.ndarray,
                  x: Union[float, int],
                  y: Union[float, int],
                  w: Union[float, int],
                  h: Union[float, int],
                  label: Optional[str] = None,
                  color: Tuple[int, int, int] = (0, 0, 255),
                  thickness: int = -1,
                  alpha: float = 0.5):
        """Draw a rectangle.

        Parameters
        ----------
        frame : np.ndarray
            Input frame where the rectangle is drawn.
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.
        color: Tuple [int, int, int]
            Rectangle color.
        thickness: float | int
            Thickness of the rectangle outline, if positive. Negative thickness means that a filled circle is to be
            drawn.
        """
        overlay = frame.copy()

        font_scale = min(round(frame.shape[0] * frame.shape[1]) / 307200, 2.5)

        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        if label is not None:
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, 1)

            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
            cv2.rectangle(frame, pt1, pt2, color, -1)
            cv2.putText(frame, label, center, cv2.FONT_HERSHEY_PLAIN,
                        font_scale, (255, 255, 255), 2)
        # apply the overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    @classmethod
    def circle(cls,
               frame: np.ndarray,
               x: Union[float, int],
               y: Union[float, int],
               radius: Union[float, int],
               label: Optional[str] = None,
               color: Tuple[int, int, int] = (0, 0, 255),
               thickness: int = 2):
        """Draw a circle.

        Parameters
        ----------
        frame : np.ndarray
            Input frame where the rectangle is drawn.
        x : float | int
            Center of the circle (x-axis).
        y : float | int
            Center of the circle (y-axis).
        radius : float | int
            Radius of the circle in pixels.
        label : Optional[str]
            A text label that is placed at the center of the circle.
        color: Tuple [int, int, int]
            Circle color.
        thickness: float | int
            Thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn.

        """

        circle_add = min(round(frame.shape[0] * frame.shape[1]) / (307200 * 2), 4)

        image_size = int(radius + 2 + circle_add)  # actually half size
        roi = int(x - image_size), int(y - image_size), int(2 * image_size), int(2 * image_size)
        if not cls.is_in_bounds(frame, roi):
            return

        image = cls.view_roi(frame, roi)
        center = image.shape[1] // 2, image.shape[0] // 2
        cv2.circle(
            image, center, int(radius + circle_add), color, thickness)
        if label is not None:
            cv2.putText(
                frame, label, (int(center[0] + x), int(center[1] + y)), cv2.FONT_HERSHEY_PLAIN,
                2, color, 2)

    @classmethod
    def gaussian(cls,
                 frame: np.ndarray,
                 mean,
                 covariance,
                 label: Optional[str] = None,
                 color: Tuple[int, int, int] = (0, 0, 0)):
        """Draw 95% confidence ellipse of a 2-D Gaussian distribution.

        Parameters
        ----------
        frame : np.ndarray
            Input frame where the rectangle is drawn.
        mean : array_like
            The mean vector of the Gaussian distribution (ndim=1).
        covariance : array_like
            The 2x2 covariance matrix of the Gaussian distribution.
        label : Optional[str]
            A text label that is placed at the center of the ellipse.
        color: Tuple [int, int, int]
            Ellipse color.

        """
        # chi2inv(0.95, 2) = 5.9915
        vals, vecs = np.linalg.eigh(5.9915 * covariance)
        indices = vals.argsort()[::-1]
        vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

        center = int(mean[0] + .5), int(mean[1] + .5)
        axes = int(vals[0] + .5), int(vals[1] + .5)
        angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
        cv2.ellipse(
            frame, center, axes, angle, 0, 360, color, 2)
        if label is not None:
            cv2.putText(frame, label, center, cv2.FONT_HERSHEY_PLAIN,
                        2, color, 2)

    @classmethod
    def annotate(cls,
                 frame: np.ndarray,
                 x: Union[float, int],
                 y: Union[float, int],
                 text: str,
                 color: Tuple[int, int, int] = (0, 0, 0),
                 font_size: float = 0.5,
                 font_thickness: int = 1,
                 font_type=cv2.FONT_HERSHEY_SIMPLEX):
        """Draws a text string at a given location.

        Parameters
        ----------
        frame : np.ndarray
            Input frame where the rectangle is drawn.
        x : int | float
            Bottom-left corner of the text in the image (x-axis).
        y : int | float
            Bottom-left corner of the text in the image (y-axis).
        text : str
            The text to be drawn.
        color: Tuple [int, int, int]
            Ellipse color.
        font_size: float
            Font size of text.
        font_thickness: int
            Thickness of text.
        font_type: cv2 Font

        """
        cv2.putText(frame, text, (int(x), int(y)), font_type,
                    fontScale=font_size, color=color, thickness=font_thickness)

    @classmethod
    def line(cls, frame: np.ndarray,
             pt1: Tuple[int, int],
             pt2: Tuple[int, int],
             label: str = None,
             color: Tuple[int, int, int] = (255, 0, 0),
             thickness: int = 1
             ):
        """Draw a line.

        Parameters
        ----------
        frame : np.ndarray
            Input frame where the line is drawn.
        pt1: Tuple [int, int]
            First point of the line segment.
        pt2: Tuple [int, int]
            Second point of the line segment.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.
        color: Tuple [int, int, int]
            Line color.
        thickness: float | int
            Thickness of the line.

        """

        cv2.line(frame, tuple(pt1), tuple(pt2), color, thickness)

        if label is not None:
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
            cv2.rectangle(frame, tuple(pt1), tuple(pt2), color, -1)

            if (color[0] + color[1] + color[2]) > 375:
                text_color = (10, 10, 10)
            else:
                text_color = (245, 245, 245)

            cv2.putText(frame, label, center, cv2.FONT_HERSHEY_PLAIN,
                        1, text_color, thickness)

    @classmethod
    def colored_points(cls, frame, points, colors=None, skip_index_check=False, color=(0, 0, 255)):
        """Draw a collection of points.

        The point size is fixed to 1.

        Parameters
        ----------
        frame : np.ndarray
            Input frame where the points is drawn.
        points : ndarray
            The Nx2 array of image locations, where the first dimension is
            the x-coordinate and the second dimension is the y-coordinate.
        colors : Optional[ndarray]
            The Nx3 array of colors (dtype=np.uint8). If None, the current
            color attribute is used.
        skip_index_check : Optional[bool]
            If True, index range checks are skipped. This is faster, but
            requires all points to lie within the image dimensions.
        color: Tuple [int, int, int]
            Points color.

        """
        if not skip_index_check:
            cond1, cond2 = points[:, 0] >= 0, points[:, 0] < 480
            cond3, cond4 = points[:, 1] >= 0, points[:, 1] < 640
            indices = np.logical_and.reduce((cond1, cond2, cond3, cond4))
            points = points[indices, :]
        if colors is None:
            colors = np.repeat(
                color, len(points)).reshape(3, len(points)).T
        indices = (points + .5).astype(np.int)
        frame[indices[:, 1], indices[:, 0], :] = colors

    @classmethod
    def draw_bounded_text(cls, frame: np.ndarray,
                          string: Optional[str] = "",
                          position: Tuple[int, int] = (20, 20),
                          bkg_color: Tuple[int, int, int] = (0, 0, 0),
                          text_color: Tuple[int, int, int] = (255, 255, 255),
                          font_size: float = 0.5,
                          font_thickness: int = 1,
                          font_type=cv2.FONT_HERSHEY_SIMPLEX):
        """Draw text bounded with rectangle.

            Parameters
            ----------
            frame : np.ndarray
                Input frame where the line is drawn.
            string : Optional[str]
                String inside bounded box.
            position: Tuple[int, int]
                Position of the bouded box.
            bkg_color : Tuple [int, int, int]
                Text bacground color. It should be contrasting to text color.
            text_color: Tuple [int, int, int]
                Text color.
            font_size: float
                Font size of text.
            font_thickness: int
                Thickness of text.
            font_type: cv font type
                OpenCV font type.
        """

        text_size = cv2.getTextSize(string, font_type, font_size, font_thickness)
        pt1 = position

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        cv2.rectangle(frame, pt1, pt2, bkg_color, -1)
        cv2.putText(frame, string, center, font_type,
                    font_size, text_color, font_thickness)

    @classmethod
    def draw_marks(cls,
                   image: np.ndarray,
                   marks: List[Tuple[Union[float, int], Union[float, int]]],
                   color: Tuple[int, int, int] = (255, 255, 255)):
        """Draw mark points on image with different size.

            Parameters
            ----------
            image : np.ndarray
                Input frame where the line is drawn.
            marks : List[Tuple[Union[float,int], Union[float,int]]]
                Second point of the line segment.
            color: Tuple [int, int, int]
                Marks color.
        """

        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)

    @classmethod
    def apply_mask(cls, image: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float] = (0, 1, 0), alpha: float = 0.5, variant: str = 'color') -> np.ndarray:
        """Function that applying specific mask on the input_output image.

            Parameters
            ----------
            :param image: np.ndarray
                Input source frame.
            :param mask: np.ndarray
                Mask of specific classes that belongs to input_output frame.
            :param color: Tuple[float, float, float]
                Color representation in range of <0, 1> for each channel. Number of each channel will be multiply by 255.
            :param alpha: float
                Opacity for used color in range <0, 1>.
            :param variant: str
                Specific type applying mask on image. (i.e. 'color', 'grey', 'black')

            :rtype: np.ndarray
            :return:
                Return output image after mask application
        """

        if variant == 'color':
            null_visualization = image
        elif variant == 'grey':
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = np.stack((gray_image,) * 3, axis=-1)
            null_visualization = gray_image
            alpha = 0
        else:
            black_image = np.zeros(shape=(image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)
            null_visualization = black_image
            alpha = 0

        for i in range(3):
            image[:, :, i] = np.where(
                mask == 0,
                null_visualization[:, :, i],
                image[:, :, i] * (1 - alpha) + alpha * color[i] * 255,
            )
        return image


class Visualizer(object):

    @abc.abstractmethod
    def visualize(self, frame: np.ndarray):
        """Required to be implemented."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)


class DummyVisualizer(Visualizer):

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def visualize(self, frame: np.ndarray, title: str = None, fps: int = 0):
        cv2.imshow('Dummy visualizer', frame)
        cv2.waitKey(1)


class WindowVisualizer(Visualizer):

    def __init__(self, width=640, height=480, title='window_visualizer'):
        super().__init__()
        self.width = width
        self.height = height
        self.title = title

        self.logger = logging.getLogger(self.__class__.__name__)

    def visualize(self, frame: np.ndarray, title: str = None, fps: int = 0):
        show_frame = cv2.resize(frame, (self.width, self.height))
        cv2.imshow(self.title, show_frame)
        self.dumb_stopper(fps=fps)


class FullScreenVisualizer(Visualizer):
    def __init__(self, screen_id=0, title='fullscreen'):
        super().__init__()
        self.screen_id = screen_id
        self.screen = screeninfo.get_monitors()[screen_id]
        self.title = title

        self.logger = logging.getLogger(self.__class__.__name__)


    def fullscreen(self, image: np.array, screen: Monitor, title: str = None, fps: int = 0):
        window_name = title if title else self.title
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, image)

        self.dumb_stopper(fps)

    def visualize(self, frame: np.ndarray, title: str = None, fps: int = 0):
        self.fullscreen(image=frame, screen=self.screen, title=title, fps=fps)


class FileVisualizer(Visualizer):

    def __init__(self, width=640, height=480, filename=None, fps=20):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.width = width
        self.height = height
        self.filename = filename
        self.fps = fps

        self.last_stored_time = 0

        if filename is not None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(filename=filename,
                                                apiPreference=cv2.CAP_FFMPEG,
                                                fourcc=fourcc,
                                                fps=fps,
                                                frameSize=(int(self.width),
                                                           int(self.height)))

    def visualize(self, frame: np.ndarray, title: str = None, fps: int = 0):
        if frame is None:
            self.logger.info(f"Frame is None, sleeping for 1 sec.")
            sleep(1)

        show_frame = cv2.resize(frame, (self.width, self.height))

        if self.video_writer is not None:
            self.video_writer.write(show_frame)
        else:
            self.logger.warning("Video writer is not initialized.")


class CompositeVisualizer(Visualizer):

    def __init__(self):
        super().__init__()
        self._children = set()

    def visualize(self, frame: np.ndarray, title: str = None, fps: int = 0):
        for child in self._children:
            child.visualize(frame, title, fps)

    def add(self, visualizer):
        self._children.add(visualizer)

    def remove(self, visualizer):
        self._children.discard(visualizer)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--type',
                        dest='type',
                        choices=('dummy', 'window', 'file', 'fullscreen'),
                        default='dummy',
                        help='Choose type of video reader. As default you can read images from video file or stream. '
                             'In case of Basler input_output or input_output from video/image directory you need to change this '
                             'option. '
                        )

    parser.add_argument('--width',
                        dest='width',
                        type=int,
                        default=640,
                        help='Width of input_output video. It is optional but recommended. Sometimes it is not possible to set'
                             'reading input_output or you need to set correct values.')

    parser.add_argument('--height',
                        dest='height',
                        type=int,
                        default=480,
                        help='Height of input_output video. It is optional but recommended. Sometimes it is not possible to '
                             'set reading input_output or you need to set correct values.')

    parser.add_argument('--filename',
                        dest='filename',
                        type=str,
                        default='video.avi',
                        help='Filename where you will save your input_output.')

    parser.add_argument('-V', '--version', action='version', version='%(prog)s 1.0')

    return parser.parse_args()


if __name__ == '__main__':

    params = parse_args()

    vis = DummyVisualizer()

    if params.type == 'dummy':
        vis = DummyVisualizer()
    elif params.type == 'window':
        vis = WindowVisualizer(params.height, params.width)
    elif params.type == 'file':
        vis = FileVisualizer(params.height, params.width, params.filename)
    elif params.type == 'fullscreen':
        vis = FullScreenVisualizer()
    else:
        logging.error('Wrong visualizer type.')

    for i in range(10):
        blank = np.zeros((720, 1024, 3), np.uint8)
        message = f'Visualizer test. frame #{i}'
        Drawer.draw_bounded_text(blank, message)

        vis.visualize(blank)
        sleep(1)
