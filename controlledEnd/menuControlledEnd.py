from enum import StrEnum
import inspect
import json
import math
import os
import queue
import time
import typing

import cv2
import numpy as np


from utils import configLoader, Logger, LogMsg
from frameDecorator.colors import Colors
from .controlledEnd import ControlledEnd


class MenuItemType(StrEnum):
    BOOL = 'bool'
    MENU = 'menu'
    IRQ = 'irq'
    MSG = 'msg'
    OPTION = 'option'
    NUMERAL = 'numeral'


class MenuControlledEnd(ControlledEnd):
    """
    MenuControlledEnd is a menu controller class for a graphical user interface, inheriting from ControlledEnd.
    It manages a menu system with multiple options, supporting navigation, selection, and configuration of menu items.
    The class is designed for use with OpenCV for rendering and supports various input actions (button presses, rotary encoder, etc.).
    Attributes:
        __options (list): List of current menu options.
        __option_list (dict): Dictionary of all menu options loaded from a JSON file.
        __path (str): Path to the menu configuration JSON file.
        __width (int): Width of the menu display.
        __height (int): Height of the menu display.
        __row_count (int): Number of rows (menu items) per page.
        __font_height (int): Height of the font used for menu text.
        __font_scale (float): Font scale for normal text.
        __highlight_font_scale (float): Font scale for highlighted text.
        __padding (tuple): Padding for the menu display (left, top, right, bottom).
        __show_index (bool): Whether to show index numbers for menu items.
        __show_preview (bool): Whether to show a preview of the selected item.
        __thickness (int): Thickness of the text.
        __space_height (int): Vertical space between menu items.
        __current_options (list): List of currently visible menu options.
        __page_count (int): Total number of pages in the menu.
        __current_page (int): Index of the current page.
        __current_index (int): Index of the currently selected item on the current page.
        __current_menu_id (str): ID of the current menu.
        __select_index (int): Index of the currently selected option for editing.
        __title (str): Title of the current menu.
        __from (str): Sender ID for message passing.
        __value_temp: Temporary value for editing options.
        __route_list (list): Stack for tracking menu navigation history.
        __theme (dict): Color theme for the menu display.
        __frame_list (queue): Queue for storing rendered frames.
        __config (ConfigLoader): Configuration loader instance.
    Methods:
        __init__(...): Initialize the menu controller with display and menu parameters.
        __page_count_calc(): Calculate the number of pages based on enabled options.
        options: Property to get the current option list.
        set_option(key): Set the current menu options by key.
        dump_config(): Save the current menu configuration to file.
        __space_calc(): Calculate vertical spacing for menu items.
        __draw_slide_bar(frame): Draw the slide bar for page navigation.
        __gen_item_start_coordinate(...): Generate coordinates for menu item rendering.
        __jump_to_previous(): Navigate to the previous menu in the route stack.
        __set_enable_state(option): Enable or disable options based on dependencies.
        select(): Handle selection of the current menu item.
        unselect(): Finalize editing of an option and send updates.
        up_action(), up_one_step(), __page_up(): Navigate up in the menu.
        down_action(), down_one_step(), __page_down(): Navigate down in the menu.
        __jump_by_index(index), __jump_by_id(target, record): Jump to a submenu by index or ID.
        __draw_content(frame): Draw menu item text.
        __draw_underline_preview(frame): Draw preview of the selected item.
        __draw_data(frame): Draw data values for menu items.
        __draw_cursor(frame): Draw the selection cursor.
        __draw_title(background): Draw the menu title.
        __numerical_slide_bar(frame): Draw a slider for numerical options.
        __option_menu(frame): Draw the option selection menu.
        decorate(): Render the current menu state to a frame.
        __next_step(), __previous_step(): Change the step size for numerical options.
        __value_plus(), __value_minus(): Increment or decrement a numerical value.
        __option_up(), __option_down(): Navigate through option values.
        center_press_action(), up_press_action(), down_press_action(), left_press_action(), right_press_action(): Handle button press actions.
        circle_press_action(), cross_press_action(), cross_long_press_action(): Handle special button actions.
        rotary_encoder_counter_clockwise(), rotary_encoder_clockwise(), rotary_encoder_select(): Handle rotary encoder actions.
        shutter_press_action(), square_press_action(): Placeholder for additional actions.
        msg_receiver(sender, msg): Receive and process external messages.
        on_enter(last_id): Initialize the menu when entering.
        on_exit(): Clean up when exiting the menu.
        main_loop(): Generator yielding rendered frames for display.
        """

    def __init__(
            self,
            _id='MenuControlledEnd',
            path: None | str = None,
            width: int = 320,
            height: int = 240,
            padding: tuple = (10, 10, 10, 10),
            row_count: int = 4,
            show_index: bool = False,
            show_preview: bool = True,
            font_height: int = 24,
            thickness: int = 1
    ):
        """
            Initializes a MenuControlledEnd instance with customizable menu display options.
            Parameters:
                _id (str): Identifier for the menu instance. Defaults to 'MenuControlledEnd'.
                path (str, optional): Path to a JSON file containing menu options. If provided, menu options are loaded from this file.
                width (int): Width of the menu display in pixels. Defaults to 320.
                height (int): Height of the menu display in pixels. Defaults to 240.
                padding (tuple): Padding around the menu content, specified as (left, top, right, bottom). Defaults to (10, 10, 10, 10).
                row_count (int): Number of rows to display in the menu. Defaults to 4.
                show_index (bool): Whether to display the index of each menu option. Defaults to False.
                show_preview (bool): Whether to show a preview for menu options. Defaults to True.
                font_height (int): Height of the font used for menu text. Defaults to 24.
                thickness (int): Thickness of the font and menu borders. Defaults to 1.
            Attributes initialized:
                - Loads menu options from file if path is provided.
                - Sets up font scaling for normal and highlighted text.
                - Initializes menu state variables (current options, page, index, etc.).
                - Sets up color themes for menu display.
                - Loads configuration from './config.json'.
            """

        ControlledEnd.__init__(self, _id)
        self.__options: None | dict = None
        self.__option_list = None
        self.__path, self.__width, self.__height, self.__row_count = path, width, height, row_count
        if self.__path:
            with open(self.__path) as f:
                self.__menu_options = json.load(f)

        self.__font_height = font_height
        self.__font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC, self.__font_height
        )
        self.__padding = padding
        self.__show_index, self.__show_preview, self.__thickness = show_index, show_preview, thickness
        self.__space_height = 0
        self.__current_options: typing.List[dict] = list()
        self.__page_count = 0
        self.__current_page = 0
        self.__current_index = 0
        self.__current_menu_id = None
        self.__select_index = None  # Index selected
        self.__title = None
        self.__from = None
        self.__value_temp = None
        self.__route_list: typing.List[tuple] = list()
        self.__theme = {
            'background': Colors.black.value,
            'cursor': Colors.steelblue.value,
            'text': Colors.white.value,
            'boolTrue': Colors.darkolivegreen.value,
            'boolFalse': Colors.darkred.value,
            MenuItemType.NUMERAL: Colors.darkgoldenrod.value,
            MenuItemType.MSG: Colors.gray.value,
            MenuItemType.OPTION: Colors.palevioletred.value,
            MenuItemType.IRQ: Colors.darkorchid.value,
            'cursorDisable': Colors.darkgray.value,
            'textDisable': Colors.gray.value
        }
        __theme_test = {
            'background': Colors.industrialBlue.value,
            'cursor': Colors.industrialGreen.value,
            'text': Colors.white.value,
            'boolTrue': Colors.industrialYellow.value,
            'boolFalse': Colors.darkred.value,
            MenuItemType.NUMERAL: Colors.darkgoldenrod.value,
            MenuItemType.MSG: Colors.gray.value,
            MenuItemType.OPTION: Colors.palevioletred.value,
            MenuItemType.IRQ: Colors.darkorchid.value,
            'cursorDisable': Colors.darkgray.value,
            'textDisable': Colors.gray.value
        }

        self.__frame_list = None
        self.__config = configLoader.ConfigLoader('./config.json')

        self.__logger = Logger()

        self.__logger.info(
            LogMsg(
                content=f'MenuControlledEnd init finished path={self.__path} resolution ={self.__width}x{self.__height}',
                module=self.__module__,
                filename=os.path.basename(os.path.abspath(__file__)),
                currentframe=inspect.currentframe(),
                pid=os.getpid()
            )
        )

    def __page_count_calc(self):
        """
        Calculates the total number of pages required to display enabled options.
        Iterates through the list of options, counting only those that are enabled.
        Determines the highest index of enabled options and calculates the number of pages
        needed based on the row count per page. Ensures that there is at least one page.
        Updates:
            self.__page_count (int): The calculated number of pages.
        """

        enable_option_range = 0
        for index, option in enumerate(self.__options, start=1):
            enable = option.get('enable', True)
            if enable:
                enable_option_range = index
        self.__page_count = math.ceil(enable_option_range / self.__row_count)
        if self.__page_count == 0:
            self.__page_count = 1

    @property
    def options(self):
        """
        Returns the list of available options.
        Returns:
            list: The list of options stored in the __option_list attribute.
        """

        return self.__option_list

    def set_option(self, key):
        """
        Sets the current menu option based on the provided key.
        This method updates the internal state of the menu, including the list of options,
        current menu ID, navigation route, title, available options, and pagination details.
        It resets the selection index, current page, and current option index, and prepares
        the options to be displayed on the first page.
        Args:
            key (str): The key identifying the menu options to display.
        Side Effects:
            Updates several internal attributes related to menu state and navigation.
        """

        self.__option_list = self.__menu_options[key]
        self.__current_menu_id = "0"
        self.__route_list = list()
        self.__title = self.__title = self.__option_list[self.__current_menu_id].get(
            'title', None)
        self.__options = self.__option_list[self.__current_menu_id]['options']
        self.__page_count_calc()
        self.__space_calc()
        self.__select_index = None
        self.__current_page = 0
        self.__current_index = 0
        self.__current_options = self.__options[0:self.__row_count]

    def dump_config(self):
        """
        Saves the current menu options to a JSON file.
        Writes the contents of self.__menu_options to the file specified by self.__path
        in JSON format with indentation for readability.
        Raises:
            IOError: If the file cannot be opened or written to.
            TypeError: If self.__menu_options contains non-serializable objects.
        """

        with open(self.__path, 'w') as f:
            json.dump(self.__menu_options, f, indent=4)

    def __space_calc(self):
        line_count = self.__row_count
        if self.__show_preview:
            line_count += 1
        if self.__title is not None:
            line_count += 1
        total_height = self.__height - self.__padding[1] - self.__padding[3]
        total_space = total_height - line_count * self.__font_height
        space_count = line_count
        self.__space_height = total_space // space_count

    def __draw_slide_bar(self, frame):
        """
        Draws a vertical slide bar (pagination indicator) on the given frame.

        The slide bar visually represents the current page among multiple pages, with up and down arrows indicating navigation availability. The bar and arrows are styled according to the current theme and padding settings.

        Args:
            frame (numpy.ndarray): The image/frame on which to draw the slide bar.

        Side Effects:
            Modifies the input frame in place by drawing the slide bar, navigation arrows, and the current page indicator.

        Visual Elements:
            - Background rectangle for the slide bar.
            - Top and bottom navigation arrows (solid or outlined depending on page position).
            - Highlighted rectangle indicating the current page.

        Depends on:
            - self.__width (int): Width of the frame.
            - self.__height (int): Height of the frame.
            - self.__padding (tuple): Padding values for the frame (left, top, right, bottom).
            - self.__current_page (int): Index of the current page.
            - self.__page_count (int): Total number of pages.
            - self.__theme (dict): Color theme for drawing.
        """
        ref_length = min(self.__width, self.__height)

        top_solid, bottom_solid = True, True
        if self.__current_page == self.__page_count - 1:
            bottom_solid = False
        if self.__current_page == 0:
            top_solid = False

        background_coordinate = (
            (self.__width - self.__padding[2] -
             ref_length // 21, self.__padding[1]),
            (self.__width, self.__height - self.__padding[3])
        )
        cv2.rectangle(
            frame, background_coordinate[0], background_coordinate[1], self.__theme['background'], -1)
        top_coordinate = (
            (self.__width - self.__padding[2],
             self.__padding[1] + ref_length // 21),
            (self.__width - self.__padding[2] - ref_length //
             21, self.__padding[1] + ref_length // 21),
            (self.__width - self.__padding[2] -
             ref_length // 42, self.__padding[1]),
            (self.__width - self.__padding[2],
             self.__padding[1] + ref_length // 21),
        )

        if top_solid:
            polygon = np.array(top_coordinate)
            cv2.fillConvexPoly(frame, polygon, self.__theme['cursor'])
        else:
            last = top_coordinate[0]
            for i in top_coordinate[1:]:
                cv2.line(frame, last, i, self.__theme['cursor'])
                last = i

        bottom_coordinate = (
            (self.__width - self.__padding[2], self.__height -
             self.__padding[1] - ref_length // 21),
            (self.__width - self.__padding[2] - ref_length // 21,
             self.__height - self.__padding[3] - ref_length // 21),
            (self.__width - self.__padding[2] - ref_length //
             42, self.__height - self.__padding[3]),
            (self.__width - self.__padding[2], self.__height -
             self.__padding[1] - ref_length // 21),
        )

        if bottom_solid:
            polygon = np.array(bottom_coordinate)
            cv2.fillConvexPoly(frame, polygon, self.__theme['cursor'])
        else:
            last = bottom_coordinate[0]
            for i in bottom_coordinate[1:]:
                cv2.line(frame, last, i, self.__theme['cursor'])
                last = i

        slide_bar_height_total = self.__height - \
            self.__padding[1] - self.__padding[3] - \
            ref_length // 21 * 2 - ref_length // 42 * 2
        slide_bar_height = slide_bar_height_total // self.__page_count
        slide_bar_width = ref_length // 21
        slide_bar_offset = (
            self.__width - self.__padding[2] - slide_bar_width,
            self.__padding[1] + ref_length // 21 + ref_length // 42
        )

        begin_coordinate = (
            slide_bar_offset[0],
            self.__current_page * slide_bar_height + slide_bar_offset[1]
        )

        end_coordinate = (
            slide_bar_offset[0] + slide_bar_width,
            (self.__current_page + 1) * slide_bar_height + slide_bar_offset[1]
        )

        cv2.rectangle(frame, begin_coordinate, end_coordinate,
                      self.__theme['cursor'], -1)

    def __gen_item_start_coordinate(self, item_count=None, ignore_title=False):
        """
        Generates the starting (x, y) coordinates for menu items, accounting for padding, spacing, font height, and optional title.

        Args:
            item_count (int, optional): The number of items to generate coordinates for. If None, uses self.__row_count.
            ignore_title (bool, optional): If True, ignores the title when calculating the starting coordinate. Defaults to False.

        Yields:
            list: The [x, y] coordinate for each menu item.
        """
        if not item_count:
            item_count = self.__row_count
        if self.__title is not None and not ignore_title:
            first_start_coordinate = (
                self.__padding[0],
                self.__padding[1] +
                int(self.__space_height * 1.5) + self.__font_height,
            )
        else:
            first_start_coordinate = (
                self.__padding[0],
                self.__padding[1] + self.__space_height
            )

        temp = list(first_start_coordinate)

        for _ in range(item_count):
            yield temp
            temp[1] += self.__font_height + self.__space_height

    def __jump_to_previous(self):
        last = self.__route_list.pop()
        self.__jump_by_id(last[0], record=False)
        self.__current_index = last[1]

    def __set_enable_state(self, option):
        """
        Updates the 'enable' state of options based on the provided configuration.

        Args:
            option (dict): A dictionary containing the following optional keys:
                - 'setDisable' (list): List of option IDs to be disabled or enabled based on 'value'.
                - 'setEnable' (list): List of option IDs to be enabled or disabled based on 'value'.
                - 'enableWith' (list): List of option IDs to be enabled or disabled together with the current option.
                - 'value' (bool): Determines whether to enable or disable options in 'setDisable' and 'setEnable'.
                - 'enable' (bool, optional): If True (default), enables options in 'enableWith'; if False, disables them.

        Side Effects:
            Modifies the 'enable' state of options in self.__options in place.
        """
        set_disable: list = option.get('setDisable', [])
        set_enable: list = option.get('setEnable', [])
        enable_with: list = option.get('enableWith', [])
        if not set_disable and not set_enable and not enable_with:
            return
        ready_to_enable, ready_to_disable = [], []

        for i in self.__options:
            if i['id'] in set_disable:
                if option['value']:
                    ready_to_disable.append(i)
                else:
                    ready_to_enable.append(i)
            if i['id'] in set_enable:
                if option['value']:
                    ready_to_enable.append(i)
                else:
                    ready_to_disable.append(i)

        if option.get('enable', True):
            for i in self.__options:
                if i['id'] in enable_with:
                    if i in ready_to_disable:
                        continue
                    ready_to_enable.append(i)
        else:
            for i in self.__options:
                if i['id'] in enable_with:
                    if i in ready_to_enable:
                        ready_to_enable.remove(i)
                    ready_to_disable.append(i)

        for i in ready_to_enable:
            i['enable'] = True
        for j in ready_to_disable:
            j['enable'] = False

    def select(self):
        t = self.__current_options[self.__current_index]['type'].lower()
        if t == MenuItemType.BOOL:
            self.__current_options[self.__current_index]['value'] = not self.__current_options[self.__current_index][
                'value']
            self.__set_enable_state(
                self.__current_options[self.__current_index])
            self._msg_sender(
                self._id,
                self.__from,
                (
                    self.__current_menu_id,
                    self.__menu_options[self.__from]
                )
            )
            self.__page_count_calc()
            receiver = self.__current_options[self.__current_index].get(
                'receiver', None)
            if receiver:
                self._msg_sender(
                    receiver,
                    receiver,
                    self.__current_options[self.__current_index]['value']
                )
            self.dump_config()
        elif t == MenuItemType.MENU:
            self.__jump_by_index(self.__current_index)
            self.decorate()
            return
        elif t == MenuItemType.IRQ:
            self._irq(self.__current_options[self.__current_index]['value'])
            return
        elif t == MenuItemType.MSG:
            self._msg_sender(
                self._id,
                self.__current_options[self.__current_index]['receiver'],
                (
                    self.__current_options[self.__current_index]['value'],
                    self.__menu_options[self.__from]
                )
            )
            self._irq(self.__from)
        elif t == MenuItemType.OPTION or t == MenuItemType.NUMERAL:
            self.__value_temp = self.__current_options[self.__current_index]['value']
            self.__select_index = self.__current_index
        else:
            raise RuntimeError()

    def unselect(self):
        """
        Handles the unselection of the current menu option.

        This method performs the following actions:
        - Updates the current option's value with the temporary value.
        - Resets the temporary value.
        - Sends a message with the current menu state to the originating sender.
        - If the current option has a receiver and a value, sends the updated option to the receiver.
        - Resets the selection index.
        - Dumps (saves) the current configuration.
        """
        self.__current_options[self.__select_index]['value'] = self.__value_temp
        self.__value_temp = None
        self._msg_sender(
            self._id,
            self.__from,
            (
                self.__current_menu_id,
                self.__menu_options[self.__from]
            )
        )
        # If have reveiver in option, send value to it
        receiver = self.__current_options[self.__select_index].get(
            'receiver', None)
        if receiver and 'value' in self.__current_options[self.__select_index].keys():
            self._msg_sender(
                None,
                receiver,
                self.__current_options[self.__select_index]
            )
        self.__select_index = None
        self.dump_config()

    def up_action(self):
        times = 1
        for i in self.__options[self.__current_page * self.__row_count + self.__current_index - 1::-1]:
            enable = i.get('enable', True)
            if not enable:
                times += 1
            else:
                break
        else:
            times = 0

        for i in range(times):
            self.up_one_step()

    def up_one_step(self):
        if self.__current_page == 0 and self.__current_index == 0:
            return
        elif self.__current_page != 0 and self.__current_index == 0:
            self.__page_up()
            return
        self.__current_index -= 1

    def __page_up(self):
        if self.__current_page == 0:
            return
        self.__current_page -= 1
        self.__current_index = self.__row_count - 1
        self.__current_options = self.__options[
            self.__current_page * self.__row_count:(self.__current_page + 1) * self.__row_count
        ]

    def down_action(self):
        times = 1
        for i in self.__options[self.__current_page * self.__row_count + self.__current_index + 1:]:
            enable = i.get('enable', True)
            if not enable:
                times += 1
            else:
                break
        else:
            times = 0

        for i in range(times):
            self.down_one_step()

    def down_one_step(self):
        if self.__current_page * self.__row_count + self.__current_index + 1 == len(self.__options):
            return
        elif self.__current_index == self.__row_count - 1:
            self.__page_down()
            return
        self.__current_index += 1

    def __page_down(self):
        if self.__current_page == self.__page_count:
            return
        self.__current_page += 1
        self.__current_index = 0
        self.__current_options = self.__options[
            self.__current_page * self.__row_count:(self.__current_page + 1) * self.__row_count
        ]

    def __jump_by_index(self, index):
        self.__jump_by_id(self.__current_options[index]['value'])

    def __jump_by_id(self, target, record=True):
        if record:
            self.__route_list.append(
                (self.__current_menu_id, self.__current_index))
        self.__current_menu_id = target
        self.__title = self.__option_list[self.__current_menu_id].get(
            'title', None)
        self.__options = self.__option_list[self.__current_menu_id]['options']

        self.__page_count_calc()
        self.__space_calc()
        self.__select_index = None
        self.__current_page = 0
        self.__current_index = 0
        self.__current_options = self.__options[0:self.__row_count]

    def __draw_content(self, frame):
        """
        Draws the menu options onto the provided frame.
        This method iterates through the current menu options and renders each option's text and a background rectangle
        on the given frame using OpenCV. The text color depends on whether the option is enabled or disabled. If
        `self.__show_index` is True, the index of each option is displayed before its content.
        Args:
            frame (numpy.ndarray): The image/frame on which the menu content will be drawn.
        Returns:
            None
        """

        if not self.__show_index:
            return
        for index, i in zip(
                range(len(self.__current_options)),
                self.__gen_item_start_coordinate()
        ):
            enable = self.__current_options[index].get('enable', True)
            color = self.__theme['text'] if enable else self.__theme['textDisable']

            if self.__show_index:
                cv2.putText(
                    frame,
                    "{} {}".format(
                        index + 1, self.__current_options[index]['content']),
                    (i[0], i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    color
                )
            else:
                cv2.putText(
                    frame,
                    "{}".format(self.__current_options[index]['content']),
                    (i[0], i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    color
                )
            cv2.rectangle(
                frame,
                (
                    self.__width -
                    self.__padding[2] - self.__width // 21 -
                    self.__width // 21,
                    i[1] - self.__space_height,
                ),
                (
                    self.__width - self.__padding[2] - self.__width // 21,
                    i[1] + self.__space_height + self.__font_height,
                ),
                self.__theme['background'],
                -1
            )

    def __draw_underline_preview(self, frame):
        if not self.__show_preview:
            return
        line = self.__row_count
        if self.__title is not None:
            line += 1

        t = self.__current_options[self.__current_index]['type'].lower()

        if t == MenuItemType.BOOL:
            if self.__current_options[self.__current_index]['value']:
                background_color = self.__theme['boolTrue']
                text = 'Y'
            else:
                background_color = self.__theme['boolFalse']
                text = 'N'
        elif t == MenuItemType.IRQ:
            background_color = self.__theme[t]
            text = str(self.__current_options[self.__current_index]['value'])
        elif t == MenuItemType.OPTION:
            background_color = self.__theme[t]
            value = self.__current_options[self.__current_index]['value']['content']
            text = str(value)
        elif t == MenuItemType.NUMERAL:
            value = self.__current_options[self.__current_index]['value']
            if isinstance(value, float):
                value = round(value, 2)
            text = str(value)
            background_color = self.__theme[t]
        elif t == MenuItemType.MSG:
            background_color = self.__theme[t]
            text = str(
                self.__current_options[self.__current_index]['receiver'])
        elif t == MenuItemType.MENU:
            return
        else:
            raise TypeError(t)

        coordinate = (
            (
                self.__padding[0],
                line * (self.__font_height + self.__space_height) +
                self.__space_height + self.__padding[1]
            ),
            (
                self.__width - self.__padding[2] -
                self.__width // 21 - self.__width // 21,
                (line + 1) * (self.__font_height +
                              self.__space_height) + self.__padding[1]
            )
        )

        cv2.rectangle(frame, coordinate[0],
                      coordinate[1], background_color, -1)

        cv2.putText(
            frame,
            str(text),
            (
                coordinate[0][0], coordinate[1][1]
            ),
            cv2.FONT_ITALIC,
            self.__font_scale,
            self.__theme['text'],
        )

        cv2.rectangle(
            frame,
            (coordinate[1][0], coordinate[0][1]),
            (self.__width - self.__padding[2] -
             self.__width // 21, coordinate[1][1]),
            self.__theme['background'],
            -1
        )

    def __draw_data(self, frame):
        """
        Draws the data options onto the provided frame if preview is enabled.

        This method iterates through the current menu options and renders their visual representation
        on the given frame using OpenCV drawing functions. It highlights the currently selected option,
        displays option values with appropriate background colors based on their type, and handles
        different option types such as boolean, message, menu, IRQ, option, and numeral.

        Args:
            frame: The image/frame (as a NumPy array) on which the menu options will be drawn.

        Raises:
            TypeError: If an unknown option type is encountered in the current options.
        """
        if not self.__show_preview:
            return
        for index, i in zip(
                range(len(self.__current_options)),
                self.__gen_item_start_coordinate()
        ):
            if not self.__current_options[index].get('enable', True):
                continue
            if self.__current_index == index:
                cv2.rectangle(
                    frame,
                    (
                        self.__width -
                        self.__padding[2] -
                        self.__width // 21 - self.__width // 21,
                        i[1] - self.__space_height
                    ),
                    (
                        self.__width -
                        self.__padding[2] - self.__width // 42 - 1,
                        i[1] + self.__font_height + self.__space_height
                    ),
                    self.__theme['background'],
                    - 1
                )
                self.__draw_underline_preview(frame)
                # continue
            font_color = self.__theme['text']
            t = self.__current_options[index]['type'].lower()
            # Show little hint value
            if t == MenuItemType.BOOL:
                if self.__current_options[index]['value']:
                    value = 'Y'
                    background_color = self.__theme['boolTrue']
                else:
                    value = 'N'
                    background_color = self.__theme['boolFalse']
            elif t == MenuItemType.MSG:
                value = self.__current_options[index]['receiver']
                background_color = self.__theme[t]
            elif t == MenuItemType.MENU:
                continue
            elif t == MenuItemType.IRQ:
                value = self.__current_options[index]['value']
                background_color = self.__theme[t]

            elif t == MenuItemType.OPTION:
                # When option have extra value, show it's name
                value = self.__current_options[index]['value']['content']
                background_color = self.__theme[t]
            elif t == MenuItemType.NUMERAL:
                value = self.__current_options[index]['value']
                if isinstance(value, float):
                    value = round(value, 2)
                background_color = self.__theme[t]
            else:
                raise TypeError(t)

            cv2.rectangle(
                frame,
                (
                    self.__width -
                    self.__padding[2] - 3 *
                    (self.__width // 42) - 8 * self.__font_height,
                    i[1] - self.__space_height // 3
                ),
                (
                    self.__width -
                    self.__padding[2] - self.__width // 21 -
                    self.__width // 21,
                    i[1] + self.__font_height + self.__space_height // 3
                ),
                background_color,
                -1
            )
            cv2.putText(
                frame,
                "{}".format(value),
                (
                    self.__width - self.__padding[
                        2] - self.__width // 42 - self.__width // 21 - 8 * self.__font_height,
                    i[1] + self.__font_height
                ),
                cv2.FONT_ITALIC,
                self.__font_scale,
                font_color
            )
            cv2.rectangle(
                frame,
                (
                    self.__width -
                    self.__padding[2] - self.__width // 21 -
                    self.__width // 21,
                    i[1] - self.__space_height
                ),
                (
                    self.__width - self.__padding[2] - self.__width // 42 - 1,
                    i[1] + self.__font_height + self.__space_height
                ),
                self.__theme['background'],
                -1
            )

    def __draw_cursor(self, frame):
        for index, i in enumerate(self.__gen_item_start_coordinate()):
            if self.__current_index == index:
                enable = self.__current_options[index].get('enable', True)
                color = self.__theme['cursor'] if enable else self.__theme['cursorDisable']
                cv2.rectangle(
                    frame,
                    (
                        i[0],
                        i[1] - self.__space_height // 3
                    ),
                    (
                        self.__width -
                        self.__padding[2] -
                        self.__width // 21 - self.__width // 21,
                        i[1] + self.__font_height + self.__space_height // 3
                    ),
                    color,
                    -1
                )
                break

    def __draw_title(self, background):
        if not self.__title:
            return
        coordinate = (
            self.__padding[0],
            self.__padding[1] + self.__font_height
        )

        cv2.putText(
            background,
            self.__title,
            coordinate,
            cv2.FONT_ITALIC,
            self.__font_scale,
            self.__theme['text'],
            self.__thickness
        )
        cv2.rectangle(
            background,
            (
                self.__width -
                self.__padding[2] - self.__width // 21 -
                self.__width // 21 + 1,
                self.__padding[1] - self.__space_height
            ),
            (
                self.__width - self.__padding[2] - self.__width // 21 - 1,
                self.__padding[1] + self.__font_height + self.__space_height
            ),
            self.__theme['background'],
            -1
        )

    def __numerical_slide_bar(self, frame):
        """
        Draws a numerical slide bar UI component on the given frame.

        This method visualizes a slider for adjusting a numerical value within a specified range.
        It displays the minimum and maximum labels, arrow indicators for increment/decrement,
        the current value, the range, and the step size. The slider bar is filled according to
        the current value, and the UI is rendered using OpenCV drawing functions.

        Args:
            frame (numpy.ndarray): The image/frame on which the slider UI will be drawn.

        Visual Elements:
            - "min" and "max" labels at the ends of the slider.
            - Left and right arrow indicators, filled or outlined depending on value limits.
            - A filled rectangle representing the current value's position on the slider.
            - The current value, centered below the slider.
            - The range (min <--> max) and step size displayed below the slider.

        Uses:
            - self.__current_options: List of option dictionaries containing 'min', 'max', 'value', and 'step'.
            - self.__current_index: Index of the currently selected option.
            - self.__value_temp: Temporary value for the slider, if set.
            - self.__theme: Dictionary containing color settings for text and cursor.
            - self.__font_height, self.__font_scale: Font size settings.
            - self.__width, self.__padding: UI layout settings.
            - self.__gen_item_start_coordinate: Helper method to generate coordinates for UI elements.
        """
        value = self.__value_temp if self.__value_temp is not None else self.__current_options[self.__current_index][
            'value']
        mi = self.__current_options[self.__current_index]['min']
        ma = self.__current_options[self.__current_index]['max']
        for index, i in enumerate(self.__gen_item_start_coordinate(item_count=5, ignore_title=True)):
            if index == 0:
                cv2.putText(
                    frame,
                    "min",
                    (i[0], i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    self.__theme['text'],
                )
                cv2.putText(
                    frame,
                    "max",
                    (int(
                        self.__width - self.__padding[2] - self.__font_height * 3), i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    self.__theme['text'],
                )

            elif index == 1:
                right_coordinate = (
                    (i[0] + self.__font_height, i[1]),
                    (i[0] + self.__font_height, i[1] + self.__font_height),
                    (i[0], i[1] + self.__font_height // 2),
                    (i[0] + self.__font_height, i[1]),
                )

                if value <= mi:  # No fill
                    last = right_coordinate[0]
                    for _ in right_coordinate[1:]:
                        cv2.line(frame, last, _, self.__theme['cursor'])
                        last = _
                else:
                    polygon = np.array(right_coordinate)
                    cv2.fillConvexPoly(frame, polygon, self.__theme['cursor'])

                left_coordinate = (
                    (self.__width -
                     self.__padding[2] - self.__font_height, i[1]),
                    (self.__width -
                     self.__padding[2] - self.__font_height, i[1] + self.__font_height),
                    (self.__width - self.__padding[2],
                     i[1] + self.__font_height // 2),
                    (self.__width -
                     self.__padding[2] - self.__font_height, i[1]),
                )

                if value >= ma:
                    last = left_coordinate[0]
                    for _ in left_coordinate[1:]:
                        cv2.line(frame, last, _, self.__theme['cursor'])
                        last = _
                else:
                    polygon = np.array(left_coordinate)
                    cv2.fillConvexPoly(frame, polygon, self.__theme['cursor'])

                slide_bar_total_width = self.__width - self.__padding[0] - self.__padding[2] - (
                    self.__font_height + self.__width // 42) * 2
                slide_bar_width = int(
                    slide_bar_total_width * (value-1) / (ma - mi))
                slide_bar_coordinate = (
                    (i[0] + self.__font_height + self.__width // 42, i[1]),
                    (i[0] + self.__font_height + self.__width //
                     42 + slide_bar_width, i[1] + self.__font_height),
                )
                cv2.rectangle(
                    frame,
                    slide_bar_coordinate[0],
                    slide_bar_coordinate[1],
                    self.__theme['cursor'],
                    -1
                )

            elif index == 2:
                if isinstance(value, float):
                    value = round(value, 2)
                cv2.putText(
                    frame,
                    str(value),
                    (int((self.__width - self.__font_height *
                     len(str(value))) // 2), i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    self.__theme['text'],
                )

            elif index == 3:
                cv2.putText(
                    frame,
                    "{} <--> {}".format(mi, ma),
                    (i[0], i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    self.__theme['text'],
                )
            elif index == 4:
                cv2.putText(
                    frame,
                    "Step {}".format(
                        self.__current_options[self.__current_index]['step']),
                    (i[0], i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    self.__theme['text'],
                )
                return

    def __option_menu(self, frame):
        # Gt Current Value
        value = self.__value_temp if self.__value_temp is not None else self.__current_options[
            self.__current_index]['value']
        # Get Current Options
        options: list = self.__current_options[self.__current_index]['options']
        # Get Current Index of Selected Option
        select_index = options.index(value)
        value = value['content']
        for _, (index, i) in zip(
                range(
                    len(self.__current_options[self.__current_index]['options']) + 1),
                enumerate(self.__gen_item_start_coordinate(
                    item_count=self.__row_count + 1, ignore_title=True))
        ):
            if index == 0:
                cv2.putText(
                    frame,
                    self.__current_options[self.__current_index]['content'],
                    (i[0], i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    self.__theme['text']
                )

            elif index == 2:
                right_coordinate = (
                    (i[0], i[1]),
                    (i[0] + self.__font_height, i[1] + self.__font_height // 2),
                    (i[0], i[1] + self.__font_height),
                    (i[0], i[1]),
                )
                polygon = np.array(right_coordinate)
                cv2.fillConvexPoly(frame, polygon, self.__theme['cursor'])
                cv2.putText(
                    frame,
                    str(value),
                    (i[0] + self.__font_height + self.__width //
                     42, i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    self.__theme['text']
                )
            else:  # Render the rest of the options
                try:
                    option = options[(select_index + index - 2) % len(options)]
                    text = option['content']
                except IndexError:
                    break
                cv2.putText(
                    frame,
                    text,
                    (i[0] + self.__font_height + self.__width //
                     42, i[1] + self.__font_height),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    self.__theme['text']
                )

    def decorate(self):
        sketch = np.full((self.__height, self.__width, 3),
                         self.__theme['background'], np.uint8)
        if self.__select_index is None:
            if self.__title is not None:
                self.__draw_title(sketch)
            self.__draw_cursor(sketch)
            self.__draw_content(sketch)
            self.__draw_data(sketch)
            self.__draw_slide_bar(sketch)
        else:
            t = self.__current_options[self.__current_index]['type'].lower()
            if t == MenuItemType.NUMERAL:
                self.__numerical_slide_bar(sketch)
            elif t == MenuItemType.OPTION:
                self.__option_menu(sketch)

        self.__frame_list.put(sketch)

    def __next_step(self):
        item: dict = self.__current_options[self.__current_index]
        if 'stepOptions' not in item.keys():
            return
        current_step_index = item['stepOptions'].index(item['step'])
        if current_step_index == len(item['stepOptions']) - 1:
            item['step'] = item['stepOptions'][0]
        else:
            item['step'] = item['stepOptions'][current_step_index + 1]

    def __previous_step(self):
        item: dict = self.__current_options[self.__current_index]
        if 'stepOptions' not in item.keys():
            return
        current_step_index = item['stepOptions'].index(item['step'])
        if current_step_index == 0:
            item['step'] = item['stepOptions'][-1]
        else:
            item['step'] = item['stepOptions'][current_step_index - 1]

    def __value_plus(self):
        item = self.__current_options[self.__select_index]
        if item['type'] != MenuItemType.NUMERAL:
            return
        value, step, ma = self.__value_temp, item['step'], item['max']
        value += step
        if value >= ma:
            value = ma
        self.__value_temp = value

    def __value_minus(self):
        item = self.__current_options[self.__select_index]
        if item['type'] != MenuItemType.NUMERAL:
            return
        value, step, mi = self.__value_temp, item['step'], item['min']
        value -= step
        if value <= mi:
            value = mi
        self.__value_temp = value

    def __option_up(self):
        item = self.__current_options[self.__select_index]
        if item['type'] != MenuItemType.OPTION:
            return
        value, option_list = self.__value_temp, item['options']
        option_index = option_list.index(value)
        self.__value_temp = option_list[option_index - 1]

    def __option_down(self):
        item = self.__current_options[self.__select_index]
        if item['type'] != MenuItemType.OPTION:
            return
        value, option_list = self.__value_temp, item['options']
        option_index = option_list.index(value)
        try:
            self.__value_temp = option_list[option_index + 1]
        except IndexError:
            self.__value_temp = option_list[0]

    def center_press_action(self):
        if self.__select_index is not None:
            self.unselect()
        else:
            self.select()
        self.decorate()

    def up_press_action(self):
        if self.__select_index is None:
            self.up_action()
            self.decorate()
            time.sleep(0.2)
        else:
            if self.__current_options[self.__select_index]['type'] == MenuItemType.NUMERAL:
                self.__previous_step()
                self.decorate()
                time.sleep(0.2)
            else:
                self.__option_up()
                self.decorate()
                time.sleep(0.2)

    def down_press_action(self):
        if self.__select_index is None:
            self.down_action()
            self.decorate()
            time.sleep(0.2)
        else:
            if self.__current_options[self.__select_index]['type'] == MenuItemType.NUMERAL:
                self.__next_step()
                self.decorate()
                time.sleep(0.2)
            else:
                self.__option_down()
                self.decorate()
                time.sleep(0.2)

    def left_press_action(self):
        if self.__select_index is None:
            self.up_action()
            self.decorate()
            time.sleep(0.2)
        else:
            if self.__current_options[self.__select_index]['type'] == MenuItemType.NUMERAL:
                self.__value_minus()
                self.decorate()
                time.sleep(0.1)
            else:
                self.__option_up()
                self.decorate()
                time.sleep(0.2)

    def right_press_action(self):
        if self.__select_index is None:
            self.down_action()
            self.decorate()
            time.sleep(0.2)
        else:
            if self.__current_options[self.__select_index]['type'] == MenuItemType.NUMERAL:
                self.__value_plus()
                self.decorate()
                time.sleep(0.1)
            else:
                self.__option_down()
                self.decorate()
                time.sleep(0.2)

    def circle_press_action(self):
        if self.__select_index is not None:
            self.unselect()
        else:
            self.select()
        self.decorate()

    def cross_press_action(self):
        if self.__select_index is not None:
            self.__select_index = None
            self.decorate()
        else:
            try:
                self.__jump_to_previous()
                self.decorate()
            except IndexError:
                self._irq(self.__from)

    def cross_long_press_action(self):
        self._irq(self.__from)

    def rotary_encoder_counter_clockwise(self):
        if self.__select_index is None:
            self.down_action()
            self.decorate()
            time.sleep(0.05)
        else:
            if self.__current_options[self.__select_index]['type'] == MenuItemType.NUMERAL:
                self.__value_plus()
                self.decorate()
                time.sleep(0.1)
            else:
                self.__option_down()
                self.decorate()
                time.sleep(0.2)

    def rotary_encoder_clockwise(self):
        if self.__select_index is None:
            self.up_action()
            self.decorate()
            time.sleep(0.05)
        else:
            if self.__current_options[self.__select_index]['type'] == MenuItemType.NUMERAL:
                self.__value_minus()
                self.decorate()
                time.sleep(0.1)
            else:
                self.__option_up()
                self.decorate()
                time.sleep(0.2)

    def rotary_encoder_select(self):
        if self.__select_index is not None:
            self.unselect()
        else:
            self.select()
        self.decorate()

    def shutter_press_action(self):
        if self.__select_index is not None:
            self.unselect()
        else:
            self.select()
        self.decorate()

    def square_press_action(self):
        pass

    def msg_receiver(self, sender, msg):
        self.set_option(msg)
        self.__from = sender
        self._msg_sender(
            self._id,
            self.__from,
            (
                self.__current_menu_id,
                self.__menu_options[self.__from]
            )
        )

    def on_enter(self, last_id):
        self.__frame_list = queue.SimpleQueue()
        self.decorate()

    def on_exit(self):
        self.__frame_list.put(
            np.full(
                (self.__height, self.__width, 3),
                self.__theme['background'], np.uint8),
            block=True
        )

    def main_loop(self):
        while True:
            yield self.__frame_list.get(True)
