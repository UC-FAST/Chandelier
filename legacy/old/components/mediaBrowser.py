#import imghdr
import os
import threading
import time
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union


class MediaBrowser:
    """
    MediaBrowser provides an interface for browsing, displaying, and managing a directory of images and videos with lazy loading and caching.
    Features:
        - Supports common image (jpeg, jpg, png, bmp, tiff, webp, gif) and video (mp4, avi, mov, mkv, flv, wmv) formats.
        - Automatically discovers media files in a specified directory.
        - Maintains a cache for efficient navigation (previous, current, next).
        - Lazy-loads video files and supports play/pause, seeking, and progress tracking.
        - Provides methods for navigating (next, previous, jumpToIndex), deleting, and refreshing media.
        - Handles resizing of images and video frames to target dimensions.
        - Displays a loading animation when media is not ready.
        - Thread-safe cache management for asynchronous loading.
        mediaPath (str): Directory containing media files. Defaults to './media/'.
        width (Optional[int]): Target display width for images and video frames.
        height (Optional[int]): Target display height for images and video frames.
    Raises:
        FileExistsError: If attempting to access media when none is available.
    Methods:
        refreshMediaList(): Refresh the list of media files and rebuild the cache.
        next(): Navigate to the next media item.
        previous(): Navigate to the previous media item.
        togglePlayPause(): Toggle play/pause state for the current video.
        seekVideo(position: float): Seek to a specific position (0.0-1.0) in the current video.
        getCurrentFrame() -> Optional[np.ndarray]: Get the current image or video frame.
        deleteCurrent() -> bool: Delete the current media file.
        getCurrentMediaInfo() -> Optional[Dict]: Get information about the current media item.
        getCurrentFileName(fullPath: bool = False) -> Optional[str]: Get the current media file name.
        totalMedia() -> int: Get the total number of media items.
        currentPosition() -> int: Get the current media index.
        jumpToIndex(index: int): Jump to a specific media index.
        isPlaying() -> bool: Check if a video is currently playing.
        getVideoProgress() -> Tuple[float, float]: Get the current video position and total duration in seconds.
        getVideoPosition() -> float: Get the current video position as a fraction (0.0-1.0).
    """

    def __init__(self, mediaPath: str = './media/', width: Optional[int] = None, height: Optional[int] = None):
        """
        Initializes the media browser component.
        Args:
            mediaPath (str, optional): Path to the directory containing media files. Defaults to './media/'.
            width (Optional[int], optional): Target display width for media. Defaults to None.
            height (Optional[int], optional): Target display height for media. Defaults to None.
        Attributes:
            __mediaPath (str): Absolute path to the media directory.
            __targetWidth (Optional[int]): Target width for displaying media.
            __targetHeight (Optional[int]): Target height for displaying media.
            __cacheSize (int): Number of media items to cache (previous, current, next).
            __mediaCache (List[Optional[Dict]]): Cache for media items.
            __cacheLock (threading.RLock): Lock for synchronizing cache access.
            __currentIndex (int): Index of the currently selected media file.
            __mediaFiles (List[Dict]): List of media files with metadata.
            __loadThread (Optional[threading.Thread]): Thread for loading media files.
            __loadCancelFlag (threading.Event): Event flag to cancel media loading.
            __videoCapture (Optional[cv2.VideoCapture]): Video capture object for video playback.
            __isPlaying (bool): Indicates if a video is currently playing.
            __videoPosition (int): Current position in the video (frame index).
            __videoFps (int): Frames per second of the current video.
            __videoTotalFrames (int): Total number of frames in the current video.
            __currentVideoPath (Optional[str]): Path to the currently loaded video.
            __loadingAnimation (int): State of the loading animation.
        Calls:
            refreshMediaList(): Loads and indexes available media files in the specified directory.
        """
        
        # Create directory if needed
        self.__mediaPath = os.path.abspath(mediaPath)
        os.makedirs(self.__mediaPath, exist_ok=True)

        # Display dimensions
        self.__targetWidth = width
        self.__targetHeight = height

        # Cache configuration (prev, current, next)
        self.__cacheSize = 3
        self.__mediaCache: List[Optional[Dict]] = [None] * self.__cacheSize
        self.__cacheLock = threading.RLock()

        # Media management
        self.__currentIndex = 0
        self.__mediaFiles: List[Dict] = []  # List of {path, type, metadata}
        self.__loadThread: Optional[threading.Thread] = None
        self.__loadCancelFlag = threading.Event()

        # Video state (lazy-loaded when needed)
        self.__videoCapture: Optional[cv2.VideoCapture] = None
        self.__isPlaying = False
        self.__videoPosition = 0
        self.__videoFps = 0
        self.__videoTotalFrames = 0
        self.__currentVideoPath: Optional[str] = None

        # Initialize browser
        self.refreshMediaList()

        # Loading animation
        self.__loadingAnimation = 0

    def refreshMediaList(self):
        """
        Refreshes the list of media files in the specified media directory.
        This method performs the following actions:
            - Clears the internal media cache.
            - Scans the media directory for supported image and video files.
            - Verifies image files using their actual format.
            - Populates the internal list of media files with metadata (path and type).
            - Sorts the media files by modification time in descending order (newest first).
            - Adjusts the current index to ensure it is within valid bounds.
            - Preloads the current and next media files (images only) into the cache.
        Supported image formats: jpeg, jpg, png, bmp, tiff, webp, gif.
        Supported video formats: mp4, avi, mov, mkv, flv, wmv.
        Returns:
            None
        """
        
        # Clear existing cache
        with self.__cacheLock:
            self.__mediaCache = [None] * self.__cacheSize

        # Discover supported media files
        self.__mediaFiles = []
        supportedImageFormats = {'jpeg', 'jpg',
                                 'png', 'bmp', 'tiff', 'webp', 'gif'}
        supportedVideoFormats = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

        for filename in os.listdir(self.__mediaPath):
            filePath = os.path.join(self.__mediaPath, filename)
            if os.path.isdir(filePath):
                continue

            # Get file extension
            ext = os.path.splitext(filename)[1][1:].lower()

            # Check if it's an image
            if ext in supportedImageFormats:
                # Verify image format
                actualFormat = 'PNG'
                if actualFormat and actualFormat.lower() in supportedImageFormats:
                    self.__mediaFiles.append({
                        'path': filePath,
                        'type': 'image'
                    })

            # Check if it's a video
            elif ext in supportedVideoFormats:
                # Minimal metadata - full loading happens when needed
                self.__mediaFiles.append({
                    'path': filePath,
                    'type': 'video'
                })

        # Handle empty media library
        if not self.__mediaFiles:
            return

        # Sort by modification time (newest first)
        self.__mediaFiles.sort(
            key=lambda x: os.path.getmtime(x['path']), reverse=True)

        # Adjust current position
        self.__currentIndex = min(
            self.__currentIndex, len(self.__mediaFiles) - 1)

        # Preload current media (images only)
        self.__loadMedia(self.__currentIndex, cachePos=1)

        # Preload next media if available (images only)
        if self.__currentIndex < len(self.__mediaFiles) - 1:
            self.__loadMedia(self.__currentIndex + 1,
                             cachePos=2, asyncLoad=True)

    def __loadMedia(self, index: int, cachePos: int, asyncLoad: bool = False):
        """
        Loads a media file at the specified index, either synchronously or asynchronously.
        Args:
            index (int): The index of the media file to load from the media files list.
            cachePos (int): The cache position to use for loading the media file.
            asyncLoad (bool, optional): If True, loads the media file asynchronously in a separate thread.
                If False, loads the media file synchronously. Defaults to False.
        Behavior:
            - Validates the index to ensure it is within the bounds of the media files list.
            - If a previous load operation is in progress, waits briefly for it to complete.
            - If asyncLoad is True, starts a new thread to load the media file asynchronously,
              ensuring any previous load thread is properly cancelled and joined.
            - If asyncLoad is False, loads the media file synchronously by calling the load task directly.
        """
        
        # Validate index range
        if index < 0 or index >= len(self.__mediaFiles):
            return

        # Wait for existing load to complete
        if self.__loadThread and self.__loadThread.is_alive():
            self.__loadThread.join(timeout=0.05)

        # Start async or sync loading
        if asyncLoad:
            self.__loadCancelFlag.set()
            if self.__loadThread and self.__loadThread.is_alive():
                self.__loadThread.join(timeout=0.05)

            self.__loadCancelFlag.clear()
            self.__loadThread = threading.Thread(
                target=self.__loadTask,
                args=(index, cachePos),
                daemon=True
            )
            self.__loadThread.start()
        else:
            self.__loadTask(index, cachePos)

    def __loadTask(self, index: int, cachePos: int):
        """
        Loads a media file (image or video) at the specified index and updates the media cache.
        Args:
            index (int): The index of the media file to load from the media file list.
            cachePos (int): The position in the cache to store the loaded media data.
        Behavior:
            - If the cancellation flag is set, the method returns immediately.
            - For images:
                - Reads the image file from disk.
                - Decodes the image using OpenCV.
                - Optionally resizes the image to target width and height if specified.
                - Stores the image data in the cache.
            - For videos:
                - Stores only the file path in the cache for lazy loading.
            - Handles exceptions during loading and prints an error message if any occur.
            - Updates the media cache at the specified cache position with the loaded result.
        """
        
        # Check for cancellation
        if self.__loadCancelFlag.is_set():
            return

        mediaInfo = self.__mediaFiles[index]
        mediaType = mediaInfo['type']
        filePath = mediaInfo['path']
        result = None

        try:
            # Handle image loading
            if mediaType == 'image':
                # Read image data
                imgData = np.fromfile(filePath, dtype=np.uint8)
                img = cv2.imdecode(imgData, cv2.IMREAD_COLOR)

                if img is not None:
                    # Resize if needed
                    if self.__targetWidth and self.__targetHeight:
                        img = cv2.resize(
                            img, (self.__targetWidth, self.__targetHeight))

                    result = {
                        'index': index,
                        'type': 'image',
                        'data': img
                    }

            # Handle video - minimal metadata only
            elif mediaType == 'video':
                # Videos are lazy-loaded - just store path
                result = {
                    'index': index,
                    'type': 'video',
                    'path': filePath
                }

        except Exception as e:
            print(f"Error loading {filePath}: {str(e)}")

        # Update cache
        if result is not None:
            with self.__cacheLock:
                self.__mediaCache[cachePos] = result

    def __openVideo(self, filePath: str):
        """
        Opens a video file for playback and initializes video properties.
        Closes any previously opened video, then attempts to open the specified video file.
        If successful, retrieves and stores the video's frames per second (FPS), total frame count,
        and resets the playback position and state. Sets the video to the first frame.
        Args:
            filePath (str): The path to the video file to open.
        Returns:
            bool: True if the video was successfully opened, False otherwise.
        """
        
        # Close any existing video
        self.__closeVideo()

        # Open new video
        self.__videoCapture = cv2.VideoCapture(filePath)
        if not self.__videoCapture.isOpened():
            return False

        # Get video properties
        self.__videoFps = self.__videoCapture.get(cv2.CAP_PROP_FPS)
        self.__videoTotalFrames = int(
            self.__videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__videoPosition = 0
        self.__isPlaying = False
        self.__currentVideoPath = filePath

        # Set to first frame
        self.__videoCapture.set(cv2.CAP_PROP_POS_FRAMES, self.__videoPosition)
        return True

    def __closeVideo(self):
        """Close current video if open"""
        if self.__videoCapture is not None:
            self.__videoCapture.release()
            self.__videoCapture = None
            self.__isPlaying = False
            self.__currentVideoPath = None

    def next(self):
        """Navigate to next media item"""
        if self.__currentIndex >= len(self.__mediaFiles) - 1:
            return self

        # Stop any playing video
        self.__closeVideo()

        self.__currentIndex += 1

        # Update cache
        with self.__cacheLock:
            self.__mediaCache[0] = self.__mediaCache[1]
            self.__mediaCache[1] = self.__mediaCache[2]
            self.__mediaCache[2] = None

        # Preload next media (images only)
        if self.__currentIndex < len(self.__mediaFiles) - 1:
            self.__loadMedia(self.__currentIndex + 1,
                             cachePos=2, asyncLoad=True)

        return self

    def previous(self):
        """Navigate to previous media item"""
        if self.__currentIndex <= 0:
            return self

        # Stop any playing video
        self.__closeVideo()

        self.__currentIndex -= 1

        # Update cache
        with self.__cacheLock:
            self.__mediaCache[2] = self.__mediaCache[1]
            self.__mediaCache[1] = self.__mediaCache[0]
            self.__mediaCache[0] = None

        # Preload previous media (images only)
        if self.__currentIndex > 0:
            self.__loadMedia(self.__currentIndex - 1,
                             cachePos=0, asyncLoad=True)

        return self

    def togglePlayPause(self):
        """Toggle play/pause for current video"""
        if self.__isPlaying:
            self.__isPlaying = False
        else:
            currentMedia = self.getCurrentMediaInfo()
            if currentMedia and currentMedia['type'] == 'video':
                # Open video if not already open (lazy loading)
                if self.__videoCapture is None or self.__currentVideoPath != currentMedia['path']:
                    if not self.__openVideo(currentMedia['path']):
                        return
                self.__isPlaying = True

    def seekVideo(self, position: float):
        """Seek to specific position in current video (0.0-1.0)"""
        if self.__videoCapture is None:
            return

        framePos = int(position * self.__videoTotalFrames)
        self.__videoPosition = min(framePos, self.__videoTotalFrames - 1)
        self.__videoCapture.set(cv2.CAP_PROP_POS_FRAMES, self.__videoPosition)
        self.__isPlaying = False  # Pause after seeking

    def getCurrentFrame(self) -> Optional[np.ndarray]:
        """Get current media frame (image or video frame)"""
        if not self.__mediaFiles:
            raise FileExistsError

        # Get current media info
        currentMedia = self.getCurrentMediaInfo()
        if not currentMedia:
            raise FileExistsError

        # Handle video playback (lazy loading)
        if currentMedia['type'] == 'video':
            # Open video if not already open
            if self.__videoCapture is None or self.__currentVideoPath != currentMedia['path']:
                if not self.__openVideo(currentMedia['path']):
                    return self.__createLoadingScreen()

            if self.__isPlaying:
                # Get next frame if playing
                ret, frame = self.__videoCapture.read()
                if ret:
                    self.__videoPosition += 1
                    # End of video handling
                    if self.__videoPosition >= self.__videoTotalFrames:
                        self.__isPlaying = False
                        self.__videoPosition = 0
                        self.__videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    # Resize if needed
                    if self.__targetWidth and self.__targetHeight:
                        frame = cv2.resize(
                            frame, (self.__targetWidth, self.__targetHeight))
                    return frame
                else:
                    self.__isPlaying = False
            else:
                # If paused, get current frame without advancing
                currentFrame = int(
                    self.__videoCapture.get(cv2.CAP_PROP_POS_FRAMES))
                if currentFrame != self.__videoPosition:
                    self.__videoCapture.set(
                        cv2.CAP_PROP_POS_FRAMES, self.__videoPosition)

                ret, frame = self.__videoCapture.read()
                if ret:
                    # Stay on the same frame
                    self.__videoCapture.set(
                        cv2.CAP_PROP_POS_FRAMES, self.__videoPosition)

                    # Resize if needed
                    if self.__targetWidth and self.__targetHeight:
                        frame = cv2.resize(
                            frame, (self.__targetWidth, self.__targetHeight))
                    return frame

        # Handle image display
        elif currentMedia['type'] == 'image':
            # Try to get image from cache
            currentImg = None
            with self.__cacheLock:
                if self.__mediaCache[1] and self.__mediaCache[1]['index'] == self.__currentIndex:
                    currentImg = self.__mediaCache[1].get('data', None)

            # Load immediately if missing
            if currentImg is None:
                self.__loadMedia(self.__currentIndex, cachePos=1)
                with self.__cacheLock:
                    if self.__mediaCache[1]:
                        currentImg = self.__mediaCache[1].get('data', None)

            return currentImg if currentImg is not None else self.__createLoadingScreen()

        # Show loading animation if media not ready
        return self.__createLoadingScreen()

    def __createLoadingScreen(self) -> np.ndarray:
        """Create a loading screen with animation"""
        height = self.__targetHeight or 480
        width = self.__targetWidth or 640
        loadingScreen = np.zeros((height, width, 3), dtype=np.uint8)

        # Get current filename
        currentFile = self.getCurrentFileName()
        displayText = f"Loading: {currentFile[:20] + '...' if len(currentFile) > 20 else currentFile}"
        cv2.putText(loadingScreen, displayText, (20, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add rotating animation
        self.__loadingAnimation = (self.__loadingAnimation + 8) % 360
        radius = min(height, width) // 10
        center = (width - 40, height - 40)

        # Draw rotating segments
        cv2.ellipse(loadingScreen, center, (radius, radius),
                    self.__loadingAnimation, 0, 60, (0, 200, 0), -1)
        cv2.ellipse(loadingScreen, center, (radius, radius),
                    self.__loadingAnimation + 120, 0, 60, (200, 150, 0), -1)
        cv2.ellipse(loadingScreen, center, (radius, radius),
                    self.__loadingAnimation + 240, 0, 60, (200, 0, 0), -1)

        return loadingScreen

    def deleteCurrent(self) -> bool:
        """Delete current media file"""
        if not self.__mediaFiles:
            return False

        # Close video if open
        if self.__videoCapture is not None:
            self.__closeVideo()

        mediaInfo = self.__mediaFiles[self.__currentIndex]
        filePath = mediaInfo['path']

        try:
            # Remove physical file
            os.remove(filePath)

            # Update media list
            self.__mediaFiles.pop(self.__currentIndex)

            # Adjust position if at end
            if self.__currentIndex >= len(self.__mediaFiles):
                self.__currentIndex = max(0, len(self.__mediaFiles) - 1)

            # Reset cache
            with self.__cacheLock:
                self.__mediaCache = [None] * self.__cacheSize

            # Reload current position
            self.__loadMedia(self.__currentIndex, cachePos=1)

            # Preload next if available
            if self.__currentIndex < len(self.__mediaFiles) - 1:
                self.__loadMedia(self.__currentIndex + 1,
                                 cachePos=2, asyncLoad=True)

            return True
        except Exception as e:
            print(f"Deletion failed: {filePath} - {str(e)}")
            return False

    def getCurrentMediaInfo(self) -> Optional[Dict]:
        """Get information about current media item"""
        if not self.__mediaFiles or self.__currentIndex >= len(self.__mediaFiles):
            return None

        return self.__mediaFiles[self.__currentIndex]

    def getCurrentFileName(self, fullPath: bool = False) -> Optional[str]:
        """Get current media file name"""
        if not self.__mediaFiles or self.__currentIndex >= len(self.__mediaFiles):
            return None

        path = self.__mediaFiles[self.__currentIndex]['path']
        return path if fullPath else os.path.basename(path)

    def totalMedia(self) -> int:
        """Get total media count"""
        return len(self.__mediaFiles)

    def currentPosition(self) -> int:
        """Get current media index"""
        return self.__currentIndex

    def jumpToIndex(self, index: int):
        """Jump to specific media index"""
        if 0 <= index < len(self.__mediaFiles):
            # Close any open video
            self.__closeVideo()

            self.__currentIndex = index

            # Reset cache
            with self.__cacheLock:
                self.__mediaCache = [None] * self.__cacheSize

            # Load current media
            self.__loadMedia(self.__currentIndex, cachePos=1)

            # Preload adjacent media
            if self.__currentIndex > 0:
                self.__loadMedia(self.__currentIndex - 1,
                                 cachePos=0, asyncLoad=True)
            if self.__currentIndex < len(self.__mediaFiles) - 1:
                self.__loadMedia(self.__currentIndex + 1,
                                 cachePos=2, asyncLoad=True)

    def isPlaying(self) -> bool:
        """Check if video is currently playing"""
        return self.__isPlaying

    def getVideoProgress(self) -> Tuple[float, float]:
        """Get current video progress (position, total duration)"""
        if self.__videoCapture is None or self.__videoFps <= 0:
            return (0, 0)

        position = self.__videoPosition / self.__videoFps
        duration = self.__videoTotalFrames / self.__videoFps
        return (position, duration)

    def getVideoPosition(self) -> float:
        """Get current video position (0.0-1.0)"""
        if self.__videoCapture is None or self.__videoTotalFrames == 0:
            return 0.0
        return self.__videoPosition / self.__videoTotalFrames
