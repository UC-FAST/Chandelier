from collections import deque


class SlidingWindowFilter:
    """
    A sliding window filter for calculating the moving average of a data stream.

    Attributes:
        __maxSize (int): The maximum number of elements in the sliding window.
        __deque (collections.deque): The deque holding the current window of data.
        __sum (float): The sum of the elements in the current window.

    Methods:
        __init__(maxSize):
            Initializes the filter with a specified window size.
            Args:
                maxSize (int): The maximum number of elements in the window. Must be at least 1.
            Raises:
                ValueError: If maxSize is less than 1.

        addData(data):
            Adds a new data point to the window. If the window is full, removes the oldest data point.
            Args:
                data (float): The new data point to add.
            Returns:
                SlidingWindowFilter: The instance itself for method chaining.

        calc():
            Calculates the average of the current window.
            Returns:
                float: The average of the data in the window.
            Raises:
                ValueError: If there is no data in the window.
    """

    def __init__(self, maxSize):
        if maxSize < 1:
            raise ValueError("maxSize must be at least 1")
        self.__maxSize = maxSize
        self.__deque = deque(maxlen=maxSize)
        self.__sum = 0.0

    def addData(self, data):
        """
        Adds a new data point to the sliding window filter.

        If the window has reached its maximum size, removes the oldest data point before adding the new one.
        Updates the running sum accordingly.

        Args:
            data (numeric): The new data point to add to the sliding window.

        Returns:
            self: Returns the instance to allow method chaining.
        """
        if len(self.__deque) == self.__maxSize:
            self.__sum -= self.__deque[0]
        self.__deque.append(data)
        self.__sum += data
        return self

    def calc(self):
        size = len(self.__deque)
        if size == 0:
            raise ValueError("No data in the window")
        return self.__sum / size
