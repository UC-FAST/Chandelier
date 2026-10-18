from __future__ import annotations

import math
import queue
import time
from pathlib import Path

import cv2
import numpy as np

from chandelier.config import load_menu_config, save_menu_config
from chandelier.constants import EndId, MenuItemType
from chandelier.i18n import language_from_options, tr, translator
from chandelier.log import get_logger
from chandelier.menu_defaults import ensure_menu_defaults
from chandelier.ui.controlled_end import ControlledEnd
from chandelier.ui.decorators.icons import draw_icon
from chandelier.ui.text_render import get_renderer
from chandelier.ui.themes import (
    get_theme,
    set_active_theme,
    theme_id_from_options,
)


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

    OPTION_VISIBLE_COUNT = 5

    def __init__(
            self,
            _id: str = EndId.MENU.value,
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
        if path is None:
            raise ValueError("menu configuration path is required")
        self._options: list[dict] | None = None
        self._option_list: dict | None = None
        self._path = Path(path)
        self._width, self._height, self._row_count = width, height, row_count
        self._logger = get_logger(__name__)
        self._menu_options = load_menu_config(self._path)
        if ensure_menu_defaults(self._menu_options):
            try:
                save_menu_config(self._menu_options, self._path)
            except OSError:
                self._logger.warning(
                    "could not persist migrated menu defaults", exc_info=True
                )
        self._apply_theme(theme_id_from_options(self._menu_options))
        language = language_from_options(self._menu_options)
        translator.set_language(language or "en")

        self._font_height = font_height
        self._font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC, self._font_height
        )
        self._small_font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC, max(8, self._font_height - 4)
        )
        self._big_font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC, self._font_height + 10
        )
        self._renderer = get_renderer()
        self._text_queue: list[tuple] = []
        self._confirm_power = False
        self._confirm_option: dict | None = None
        self._retro_style: str | None = None
        self._theme_preview_active = False
        self._theme_preview_original: str | None = None
        self._padding = padding
        self._show_index, self._show_preview, self._thickness = show_index, show_preview, thickness
        self._header_height = max(20, self._font_height + 6)
        self._footer_height = max(16, self._font_height)
        self._preview_height = 36 if self._show_preview else 0
        self._content_top = self._header_height + 2
        self._content_bottom = (
            self._height - self._footer_height - self._preview_height - 2
        )
        available = max(
            self._row_count * (self._font_height + 2),
            self._content_bottom - self._content_top,
        )
        self._row_height = max(
            self._font_height + 4, available // max(1, self._row_count)
        )
        self._icon_size = min(16, max(10, self._font_height))
        self._space_height = max(0, self._row_height - self._font_height)
        self._current_options: list[dict] = []
        self._page_count = 0
        self._current_page = 0
        self._current_index = 0
        self._current_menu_id = None
        self._select_index = None  # Index selected
        self._title = None
        self._from = None
        self._value_temp = None
        self._route_list: list[tuple] = []
        self._frame_list: queue.SimpleQueue | None = None
        self._logger.info(
            "MenuControlledEnd initialised path=%s resolution=%sx%s theme=%s",
            self._path,
            self._width,
            self._height,
            self._menu_theme.id,
        )

    def _read_theme_id(self) -> str:
        return theme_id_from_options(self._menu_options)

    def _apply_theme(self, theme_id: str) -> None:
        self._menu_theme = get_theme(theme_id)
        set_active_theme(self._menu_theme.id)
        # Kept for older drawing helpers that still index a dict.
        self._theme = self._menu_theme.legacy_dict()

    def _preview_theme(self) -> None:
        """Apply the theme currently highlighted in the editor."""

        if not self._theme_preview_active or self._select_index is None:
            return
        option = self._current_options[self._select_index]
        if option.get("id") != "theme":
            return
        value = self._value_temp
        if isinstance(value, dict):
            value = value.get("value")
        if value is not None:
            self._apply_theme(str(value))

    def _cancel_theme_preview(self) -> None:
        if (
            self._theme_preview_active
            and self._theme_preview_original is not None
        ):
            self._apply_theme(self._theme_preview_original)
        self._theme_preview_active = False
        self._theme_preview_original = None

    def _finish_theme_preview(self) -> None:
        self._theme_preview_active = False
        self._theme_preview_original = None

    def _page_count_calc(self):
        """
        Calculates the total number of pages required to display enabled options.
        Iterates through the list of options, counting only those that are enabled.
        Determines the highest index of enabled options and calculates the number of pages
        needed based on the row count per page. Ensures that there is at least one page.
        Updates:
            self._page_count (int): The calculated number of pages.
        """

        enable_option_range = 0
        for index, option in enumerate(self._options, start=1):
            enable = option.get('enable', True)
            if enable:
                enable_option_range = index
        self._page_count = math.ceil(enable_option_range / self._row_count)
        if self._page_count == 0:
            self._page_count = 1

    @property
    def options(self):
        """
        Returns the list of available options.
        Returns:
            list: The list of options stored in the __option_list attribute.
        """

        return self._option_list

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

        self._option_list = self._menu_options[key]
        self._current_menu_id = "0"
        self._route_list = list()
        self._title = self._title = self._option_list[self._current_menu_id].get(
            'title', None)
        self._options = self._option_list[self._current_menu_id]['options']
        self._apply_enable_conditions()
        self._page_count_calc()
        self._space_calc()
        self._select_index = None
        self._current_page = 0
        self._current_index = 0
        self._current_options = self._options[0:self._row_count]

    def dump_config(self):
        """
        Saves the current menu options to a JSON file.
        Writes the contents of self._menu_options to the file specified by self._path
        in JSON format with indentation for readability.
        Raises:
            IOError: If the file cannot be opened or written to.
            TypeError: If self._menu_options contains non-serializable objects.
        """

        save_menu_config(self._menu_options, self._path)

    def _space_calc(self):
        line_count = self._row_count
        if self._show_preview:
            line_count += 1
        if self._title is not None:
            line_count += 1
        total_height = self._height - self._padding[1] - self._padding[3]
        total_space = total_height - line_count * self._font_height
        space_count = line_count
        self._space_height = total_space // space_count

    def _draw_slide_bar(self, frame):
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
            - self._width (int): Width of the frame.
            - self._height (int): Height of the frame.
            - self._padding (tuple): Padding values for the frame (left, top, right, bottom).
            - self._current_page (int): Index of the current page.
            - self._page_count (int): Total number of pages.
            - self._theme (dict): Color theme for drawing.
        """
        ref_length = min(self._width, self._height)

        top_solid, bottom_solid = True, True
        if self._current_page == self._page_count - 1:
            bottom_solid = False
        if self._current_page == 0:
            top_solid = False

        background_coordinate = (
            (self._width - self._padding[2] -
             ref_length // 21, self._padding[1]),
            (self._width, self._height - self._padding[3])
        )
        cv2.rectangle(
            frame, background_coordinate[0], background_coordinate[1], self._theme['background'], -1)
        top_coordinate = (
            (self._width - self._padding[2],
             self._padding[1] + ref_length // 21),
            (self._width - self._padding[2] - ref_length //
             21, self._padding[1] + ref_length // 21),
            (self._width - self._padding[2] -
             ref_length // 42, self._padding[1]),
            (self._width - self._padding[2],
             self._padding[1] + ref_length // 21),
        )

        if top_solid:
            polygon = np.array(top_coordinate)
            cv2.fillConvexPoly(frame, polygon, self._theme['cursor'])
        else:
            last = top_coordinate[0]
            for i in top_coordinate[1:]:
                cv2.line(frame, last, i, self._theme['cursor'])
                last = i

        bottom_coordinate = (
            (self._width - self._padding[2], self._height -
             self._padding[1] - ref_length // 21),
            (self._width - self._padding[2] - ref_length // 21,
             self._height - self._padding[3] - ref_length // 21),
            (self._width - self._padding[2] - ref_length //
             42, self._height - self._padding[3]),
            (self._width - self._padding[2], self._height -
             self._padding[1] - ref_length // 21),
        )

        if bottom_solid:
            polygon = np.array(bottom_coordinate)
            cv2.fillConvexPoly(frame, polygon, self._theme['cursor'])
        else:
            last = bottom_coordinate[0]
            for i in bottom_coordinate[1:]:
                cv2.line(frame, last, i, self._theme['cursor'])
                last = i

        slide_bar_height_total = self._height - \
            self._padding[1] - self._padding[3] - \
            ref_length // 21 * 2 - ref_length // 42 * 2
        slide_bar_height = slide_bar_height_total // self._page_count
        slide_bar_width = ref_length // 21
        slide_bar_offset = (
            self._width - self._padding[2] - slide_bar_width,
            self._padding[1] + ref_length // 21 + ref_length // 42
        )

        begin_coordinate = (
            slide_bar_offset[0],
            self._current_page * slide_bar_height + slide_bar_offset[1]
        )

        end_coordinate = (
            slide_bar_offset[0] + slide_bar_width,
            (self._current_page + 1) * slide_bar_height + slide_bar_offset[1]
        )

        cv2.rectangle(frame, begin_coordinate, end_coordinate,
                      self._theme['cursor'], -1)

    def _gen_item_start_coordinate(self, item_count=None, ignore_title=False):
        """
        Generates the starting (x, y) coordinates for menu items, accounting for padding, spacing, font height, and optional title.

        Args:
            item_count (int, optional): The number of items to generate coordinates for. If None, uses self._row_count.
            ignore_title (bool, optional): If True, ignores the title when calculating the starting coordinate. Defaults to False.

        Yields:
            list: The [x, y] coordinate for each menu item.
        """
        if not item_count:
            item_count = self._row_count
        if self._title is not None and not ignore_title:
            first_start_coordinate = (
                self._padding[0],
                self._padding[1] +
                int(self._space_height * 1.5) + self._font_height,
            )
        else:
            first_start_coordinate = (
                self._padding[0],
                self._padding[1] + self._space_height
            )

        temp = list(first_start_coordinate)

        for _ in range(item_count):
            yield temp
            temp[1] += self._font_height + self._space_height

    def _jump_to_previous(self):
        last = self._route_list.pop()
        self._jump_by_id(last[0], record=False)
        self._current_index = last[1]

    def _apply_enable_conditions(self) -> None:
        """Apply declarative ``enable_if`` dependencies for option values."""

        if not self._options:
            return
        values = {}
        for option in self._options:
            if not isinstance(option, dict):
                continue
            value = option.get("value")
            if isinstance(value, dict):
                value = value.get("value")
            values[option.get("id")] = value

        for option in self._options:
            condition = option.get("enable_if")
            if not isinstance(condition, dict):
                continue
            option["enable"] = (
                values.get(condition.get("id")) == condition.get("value")
            )

    def _set_enable_state(self, option):
        """
        Updates the 'enable' state of options based on the provided configuration.

        Args:
            option (dict): A dictionary containing the following optional keys:
                - 'set_disable' (list): List of option IDs to be disabled or enabled based on 'value'.
                - 'set_enable' (list): List of option IDs to be enabled or disabled based on 'value'.
                - 'enable_with' (list): List of option IDs to be enabled or disabled together with the current option.
                - 'value' (bool): Determines whether to enable or disable options in 'set_disable' and 'set_enable'.
                - 'enable' (bool, optional): If True (default), enables options in 'enable_with'; if False, disables them.

        Side Effects:
            Modifies the 'enable' state of options in self._options in place.
        """
        set_disable: list = option.get('set_disable', [])
        set_enable: list = option.get('set_enable', [])
        enable_with: list = option.get('enable_with', [])
        if not set_disable and not set_enable and not enable_with:
            return
        ready_to_enable, ready_to_disable = [], []

        for i in self._options:
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
            for i in self._options:
                if i['id'] in enable_with:
                    if i in ready_to_disable:
                        continue
                    ready_to_enable.append(i)
        else:
            for i in self._options:
                if i['id'] in enable_with:
                    if i in ready_to_enable:
                        ready_to_enable.remove(i)
                    ready_to_disable.append(i)

        for i in ready_to_enable:
            i['enable'] = True
        for j in ready_to_disable:
            j['enable'] = False

    def select(self):
        t = self._current_options[self._current_index]['type'].lower()
        if t == MenuItemType.BOOL:
            self._current_options[self._current_index]['value'] = not self._current_options[self._current_index][
                'value']
            self._set_enable_state(
                self._current_options[self._current_index])
            self._apply_enable_conditions()
            self._msg_sender(
                self._id,
                self._from,
                (
                    self._current_menu_id,
                    self._menu_options[self._from]
                )
            )
            self._page_count_calc()
            receiver = self._current_options[self._current_index].get(
                'receiver', None)
            if receiver:
                self._msg_sender(
                    receiver,
                    receiver,
                    self._current_options[self._current_index]['value']
                )
            self.dump_config()
        elif t == MenuItemType.MENU:
            self._jump_by_index(self._current_index)
            self.decorate()
            return
        elif t == MenuItemType.IRQ:
            self._irq(self._current_options[self._current_index]['value'])
            return
        elif t == MenuItemType.MSG:
            option = self._current_options[self._current_index]
            if option.get("id") == "reboot" and not self._confirm_power:
                # Destructive action: require a second confirmation press.
                self._confirm_power = True
                self._confirm_option = option
                return
            self._confirm_power = False
            self._confirm_option = None
            self._send_current_message()
        elif t == MenuItemType.OPTION or t == MenuItemType.NUMERAL:
            option = self._current_options[self._current_index]
            if option.get("id") == "theme":
                self._theme_preview_active = True
                self._theme_preview_original = self._menu_theme.id
            self._value_temp = option['value']
            self._select_index = self._current_index
        else:
            raise RuntimeError()

    def _send_current_message(self, option: dict | None = None) -> None:
        option = option or self._current_options[self._current_index]
        self._msg_sender(
            self._id,
            option["receiver"],
            (option["value"], self._menu_options[self._from]),
        )
        self._irq(self._from)

    def _send_power_action(self, option: dict) -> None:
        self._msg_sender(
            None,
            option.get("receiver", EndId.CONTROLLER.value),
            option,
        )
        self._irq(self._from or EndId.CAMERA.value)

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
        self._current_options[self._select_index]['value'] = self._value_temp
        self._value_temp = None
        self._apply_enable_conditions()
        self._page_count_calc()
        self._msg_sender(
            self._id,
            self._from,
            (
                self._current_menu_id,
                self._menu_options[self._from]
            )
        )
        edited = self._current_options[self._select_index]
        receiver = edited.get("receiver")

        if edited.get("id") == "power":
            # Destructive action: ask for confirmation before sending the
            # selected action to the controller.
            self._confirm_power = True
            self._confirm_option = edited
        elif receiver and "value" in edited:
            self._msg_sender(None, receiver, edited)

        self._select_index = None
        self.dump_config()
        if edited.get("id") == "theme":
            self._apply_theme(self._read_theme_id())
            self._finish_theme_preview()
        elif edited.get("id") == "language":
            value = edited.get("value")
            if isinstance(value, dict):
                value = value.get("value")
            translator.set_language(value)

    def up_action(self):
        times = 1
        for i in self._options[self._current_page * self._row_count + self._current_index - 1::-1]:
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
        if self._current_page == 0 and self._current_index == 0:
            return
        elif self._current_page != 0 and self._current_index == 0:
            self._page_up()
            return
        self._current_index -= 1

    def _page_up(self):
        if self._current_page == 0:
            return
        self._current_page -= 1
        self._current_index = self._row_count - 1
        self._current_options = self._options[
            self._current_page * self._row_count:(self._current_page + 1) * self._row_count
        ]

    def down_action(self):
        times = 1
        for i in self._options[self._current_page * self._row_count + self._current_index + 1:]:
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
        if self._current_page * self._row_count + self._current_index + 1 == len(self._options):
            return
        elif self._current_index == self._row_count - 1:
            self._page_down()
            return
        self._current_index += 1

    def _page_down(self):
        if self._current_page >= self._page_count - 1:
            return
        self._current_page += 1
        self._current_index = 0
        self._current_options = self._options[
            self._current_page * self._row_count:(self._current_page + 1) * self._row_count
        ]

    def _jump_by_index(self, index):
        self._jump_by_id(self._current_options[index]['value'])

    def _jump_by_id(self, target, record=True):
        if record:
            self._route_list.append(
                (self._current_menu_id, self._current_index))
        self._current_menu_id = target
        self._title = self._option_list[self._current_menu_id].get(
            'title', None)
        self._options = self._option_list[self._current_menu_id]['options']
        self._apply_enable_conditions()

        self._page_count_calc()
        self._space_calc()
        self._select_index = None
        self._current_page = 0
        self._current_index = 0
        self._current_options = self._options[0:self._row_count]

    def _draw_content(self, frame):
        """
        Draws the menu options onto the provided frame.
        This method iterates through the current menu options and renders each option's text and a background rectangle
        on the given frame using OpenCV. The text color depends on whether the option is enabled or disabled. If
        `self._show_index` is True, the index of each option is displayed before its content.
        Args:
            frame (numpy.ndarray): The image/frame on which the menu content will be drawn.
        Returns:
            None
        """

        for index, i in zip(
                range(len(self._current_options)),
                self._gen_item_start_coordinate()
        ):
            enable = self._current_options[index].get('enable', True)
            color = self._theme['text'] if enable else self._theme['textDisable']

            content = tr(self._current_options[index]['content'])
            text = f"{index + 1} {content}" if self._show_index else content
            self._text(
                frame,
                text,
                (i[0], i[1] + self._font_height),
                cv2.FONT_ITALIC,
                self._font_scale,
                color
            )
            cv2.rectangle(
                frame,
                (
                    self._width -
                    self._padding[2] - self._width // 21 -
                    self._width // 21,
                    i[1] - self._space_height,
                ),
                (
                    self._width - self._padding[2] - self._width // 21,
                    i[1] + self._space_height + self._font_height,
                ),
                self._theme['background'],
                -1
            )

    def _draw_underline_preview(self, frame):
        if not self._show_preview:
            return
        line = self._row_count
        if self._title is not None:
            line += 1

        t = self._current_options[self._current_index]['type'].lower()

        if t == MenuItemType.BOOL:
            if self._current_options[self._current_index]['value']:
                background_color = self._theme['boolTrue']
                text = 'Y'
            else:
                background_color = self._theme['boolFalse']
                text = 'N'
        elif t == MenuItemType.IRQ:
            background_color = self._theme[t]
            text = str(self._current_options[self._current_index]['value'])
        elif t == MenuItemType.OPTION:
            background_color = self._theme[t]
            value = self._current_options[self._current_index]['value']['content']
            text = str(value)
        elif t == MenuItemType.NUMERAL:
            value = self._current_options[self._current_index]['value']
            if isinstance(value, float):
                value = round(value, 2)
            text = str(value)
            background_color = self._theme[t]
        elif t == MenuItemType.MSG:
            background_color = self._theme[t]
            option = self._current_options[self._current_index]
            text = str(option.get('badge', option['receiver']))
        elif t == MenuItemType.MENU:
            return
        else:
            raise TypeError(t)

        coordinate = (
            (
                self._padding[0],
                line * (self._font_height + self._space_height) +
                self._space_height + self._padding[1]
            ),
            (
                self._width - self._padding[2] -
                self._width // 21 - self._width // 21,
                (line + 1) * (self._font_height +
                              self._space_height) + self._padding[1]
            )
        )

        cv2.rectangle(frame, coordinate[0],
                      coordinate[1], background_color, -1)

        self._text(
            frame,
            str(text),
            (
                coordinate[0][0], coordinate[1][1]
            ),
            cv2.FONT_ITALIC,
            self._font_scale,
            self._theme['text'],
        )

        cv2.rectangle(
            frame,
            (coordinate[1][0], coordinate[0][1]),
            (self._width - self._padding[2] -
             self._width // 21, coordinate[1][1]),
            self._theme['background'],
            -1
        )

    def _draw_data(self, frame):
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
        if not self._show_preview:
            return
        for index, i in zip(
                range(len(self._current_options)),
                self._gen_item_start_coordinate()
        ):
            if not self._current_options[index].get('enable', True):
                continue
            if self._current_index == index:
                cv2.rectangle(
                    frame,
                    (
                        self._width -
                        self._padding[2] -
                        self._width // 21 - self._width // 21,
                        i[1] - self._space_height
                    ),
                    (
                        self._width -
                        self._padding[2] - self._width // 42 - 1,
                        i[1] + self._font_height + self._space_height
                    ),
                    self._theme['background'],
                    - 1
                )
                self._draw_underline_preview(frame)
                # continue
            font_color = self._theme['text']
            t = self._current_options[index]['type'].lower()
            # Show little hint value
            if t == MenuItemType.BOOL:
                if self._current_options[index]['value']:
                    value = 'Y'
                    background_color = self._theme['boolTrue']
                else:
                    value = 'N'
                    background_color = self._theme['boolFalse']
            elif t == MenuItemType.MSG:
                value = self._current_options[index].get(
                    'badge', self._current_options[index]['receiver']
                )
                background_color = self._theme[t]
            elif t == MenuItemType.MENU:
                continue
            elif t == MenuItemType.IRQ:
                value = self._current_options[index]['value']
                background_color = self._theme[t]

            elif t == MenuItemType.OPTION:
                # When option have extra value, show it's name
                value = self._current_options[index]['value']['content']
                background_color = self._theme[t]
            elif t == MenuItemType.NUMERAL:
                value = self._current_options[index]['value']
                if isinstance(value, float):
                    value = round(value, 2)
                background_color = self._theme[t]
            else:
                raise TypeError(t)

            cv2.rectangle(
                frame,
                (
                    self._width -
                    self._padding[2] - 3 *
                    (self._width // 42) - 8 * self._font_height,
                    i[1] - self._space_height // 3
                ),
                (
                    self._width -
                    self._padding[2] - self._width // 21 -
                    self._width // 21,
                    i[1] + self._font_height + self._space_height // 3
                ),
                background_color,
                -1
            )
            self._text(
                frame,
                "{}".format(value),
                (
                    self._width - self._padding[
                        2] - self._width // 42 - self._width // 21 - 8 * self._font_height,
                    i[1] + self._font_height
                ),
                cv2.FONT_ITALIC,
                self._font_scale,
                font_color
            )
            cv2.rectangle(
                frame,
                (
                    self._width -
                    self._padding[2] - self._width // 21 -
                    self._width // 21,
                    i[1] - self._space_height
                ),
                (
                    self._width - self._padding[2] - self._width // 42 - 1,
                    i[1] + self._font_height + self._space_height
                ),
                self._theme['background'],
                -1
            )

    def _draw_cursor(self, frame):
        for index, i in enumerate(self._gen_item_start_coordinate()):
            if self._current_index == index:
                enable = self._current_options[index].get('enable', True)
                color = self._theme['cursor'] if enable else self._theme['cursorDisable']
                cv2.rectangle(
                    frame,
                    (
                        i[0],
                        i[1] - self._space_height // 3
                    ),
                    (
                        self._width -
                        self._padding[2] -
                        self._width // 21 - self._width // 21,
                        i[1] + self._font_height + self._space_height // 3
                    ),
                    color,
                    -1
                )
                break

    def _draw_title(self, background):
        if not self._title:
            return
        coordinate = (
            self._padding[0],
            self._padding[1] + self._font_height
        )

        self._text(
            background,
            self._title,
            coordinate,
            cv2.FONT_ITALIC,
            self._font_scale,
            self._theme['text'],
            self._thickness
        )
        cv2.rectangle(
            background,
            (
                self._width -
                self._padding[2] - self._width // 21 -
                self._width // 21 + 1,
                self._padding[1] - self._space_height
            ),
            (
                self._width - self._padding[2] - self._width // 21 - 1,
                self._padding[1] + self._font_height + self._space_height
            ),
            self._theme['background'],
            -1
        )

    def _numerical_slide_bar(self, frame):
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
            - self._current_options: List of option dictionaries containing 'min', 'max', 'value', and 'step'.
            - self._current_index: Index of the currently selected option.
            - self._value_temp: Temporary value for the slider, if set.
            - self._theme: Dictionary containing color settings for text and cursor.
            - self._font_height, self._font_scale: Font size settings.
            - self._width, self._padding: UI layout settings.
            - self._gen_item_start_coordinate: Helper method to generate coordinates for UI elements.
        """
        value = self._value_temp if self._value_temp is not None else self._current_options[self._current_index][
            'value']
        mi = self._current_options[self._current_index]['min']
        ma = self._current_options[self._current_index]['max']
        for index, i in enumerate(self._gen_item_start_coordinate(item_count=5, ignore_title=True)):
            if index == 0:
                self._text(
                    frame,
                    "min",
                    (i[0], i[1] + self._font_height),
                    cv2.FONT_ITALIC,
                    self._font_scale,
                    self._theme['text'],
                )
                self._text(
                    frame,
                    "max",
                    (int(
                        self._width - self._padding[2] - self._font_height * 3), i[1] + self._font_height),
                    cv2.FONT_ITALIC,
                    self._font_scale,
                    self._theme['text'],
                )

            elif index == 1:
                right_coordinate = (
                    (i[0] + self._font_height, i[1]),
                    (i[0] + self._font_height, i[1] + self._font_height),
                    (i[0], i[1] + self._font_height // 2),
                    (i[0] + self._font_height, i[1]),
                )

                if value <= mi:  # No fill
                    last = right_coordinate[0]
                    for _ in right_coordinate[1:]:
                        cv2.line(frame, last, _, self._theme['cursor'])
                        last = _
                else:
                    polygon = np.array(right_coordinate)
                    cv2.fillConvexPoly(frame, polygon, self._theme['cursor'])

                left_coordinate = (
                    (self._width -
                     self._padding[2] - self._font_height, i[1]),
                    (self._width -
                     self._padding[2] - self._font_height, i[1] + self._font_height),
                    (self._width - self._padding[2],
                     i[1] + self._font_height // 2),
                    (self._width -
                     self._padding[2] - self._font_height, i[1]),
                )

                if value >= ma:
                    last = left_coordinate[0]
                    for _ in left_coordinate[1:]:
                        cv2.line(frame, last, _, self._theme['cursor'])
                        last = _
                else:
                    polygon = np.array(left_coordinate)
                    cv2.fillConvexPoly(frame, polygon, self._theme['cursor'])

                slide_bar_total_width = self._width - self._padding[0] - self._padding[2] - (
                    self._font_height + self._width // 42) * 2
                slide_bar_width = int(
                    slide_bar_total_width * (value-1) / (ma - mi))
                slide_bar_coordinate = (
                    (i[0] + self._font_height + self._width // 42, i[1]),
                    (i[0] + self._font_height + self._width //
                     42 + slide_bar_width, i[1] + self._font_height),
                )
                cv2.rectangle(
                    frame,
                    slide_bar_coordinate[0],
                    slide_bar_coordinate[1],
                    self._theme['cursor'],
                    -1
                )

            elif index == 2:
                if isinstance(value, float):
                    value = round(value, 2)
                self._text(
                    frame,
                    str(value),
                    (int((self._width - self._font_height *
                     len(str(value))) // 2), i[1] + self._font_height),
                    cv2.FONT_ITALIC,
                    self._font_scale,
                    self._theme['text'],
                )

            elif index == 3:
                self._text(
                    frame,
                    "{} <--> {}".format(mi, ma),
                    (i[0], i[1] + self._font_height),
                    cv2.FONT_ITALIC,
                    self._font_scale,
                    self._theme['text'],
                )
            elif index == 4:
                self._text(
                    frame,
                    "Step {}".format(
                        self._current_options[self._current_index]['step']),
                    (i[0], i[1] + self._font_height),
                    cv2.FONT_ITALIC,
                    self._font_scale,
                    self._theme['text'],
                )
                return

    def _option_menu(self, frame):
        # Gt Current Value
        value = self._value_temp if self._value_temp is not None else self._current_options[
            self._current_index]['value']
        # Get Current Options
        options: list = self._current_options[self._current_index]['options']
        # Get Current Index of Selected Option
        select_index = options.index(value)
        value = value['content']
        for _, (index, i) in zip(
                range(
                    len(self._current_options[self._current_index]['options']) + 1),
                enumerate(self._gen_item_start_coordinate(
                    item_count=self._row_count + 1, ignore_title=True))
        ):
            if index == 0:
                self._text(
                    frame,
                    self._current_options[self._current_index]['content'],
                    (i[0], i[1] + self._font_height),
                    cv2.FONT_ITALIC,
                    self._font_scale,
                    self._theme['text']
                )

            elif index == 2:
                right_coordinate = (
                    (i[0], i[1]),
                    (i[0] + self._font_height, i[1] + self._font_height // 2),
                    (i[0], i[1] + self._font_height),
                    (i[0], i[1]),
                )
                polygon = np.array(right_coordinate)
                cv2.fillConvexPoly(frame, polygon, self._theme['cursor'])
                self._text(
                    frame,
                    str(value),
                    (i[0] + self._font_height + self._width //
                     42, i[1] + self._font_height),
                    cv2.FONT_ITALIC,
                    self._font_scale,
                    self._theme['text']
                )
            else:  # Render the rest of the options
                try:
                    option = options[(select_index + index - 2) % len(options)]
                    text = option['content']
                except IndexError:
                    break
                self._text(
                    frame,
                    text,
                    (i[0] + self._font_height + self._width //
                     42, i[1] + self._font_height),
                    cv2.FONT_ITALIC,
                    self._font_scale,
                    self._theme['text']
                )

    # ------------------------------------------------------------------
    # Theme-aware renderer
    # ------------------------------------------------------------------
    def _rounded_rect(
        self,
        frame,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color,
        radius: int,
        thickness: int = -1,
    ) -> None:
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        if radius == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            return
        if thickness < 0:
            cv2.rectangle(
                frame, (x1 + radius, y1), (x2 - radius, y2), color, -1
            )
            cv2.rectangle(
                frame, (x1, y1 + radius), (x2, y2 - radius), color, -1
            )
            for cx, cy in (
                (x1 + radius, y1 + radius),
                (x2 - radius, y1 + radius),
                (x1 + radius, y2 - radius),
                (x2 - radius, y2 - radius),
            ):
                cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
            return

        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        for center, start, end in (
            ((x1 + radius, y1 + radius), 180, 270),
            ((x2 - radius, y1 + radius), 270, 360),
            ((x1 + radius, y2 - radius), 90, 180),
            ((x2 - radius, y2 - radius), 0, 90),
        ):
            cv2.ellipse(
                frame,
                center,
                (radius, radius),
                0,
                start,
                end,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _pixel_height(self, font_scale) -> int:
        if font_scale == self._big_font_scale:
            return self._font_height + 10
        if font_scale == self._small_font_scale:
            return max(8, self._font_height - 4)
        return self._font_height

    def _text(
        self,
        frame,
        text,
        org,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=None,
        color=(255, 255, 255),
        thickness=1,
        line_type=cv2.LINE_AA,
    ):
        """Queue Unicode text for drawing after all shapes are composed."""

        text = tr(text)
        if not text:
            return
        self._text_queue.append(
            (
                str(text),
                (int(org[0]), int(org[1])),
                self._pixel_height(font_scale),
                tuple(color),
                max(1, int(thickness)),
            )
        )

    def _measure(self, text, font_face=None, font_scale=None, thickness=1):
        text = tr(text)
        if not text:
            return (0, 0), 0
        width, height = self._renderer.measure(
            str(text), self._pixel_height(font_scale)
        )
        return (width, height), height

    def _flush_text(self, frame) -> None:
        queued = self._text_queue
        self._text_queue = []
        for text, org, height, color, thickness in queued:
            self._renderer.draw(
                frame, text, org, height, color, thickness
            )

    def _fit_text(
        self, text: str, max_width: int, scale: float | None = None
    ) -> str:
        text = tr(text)
        if max_width <= 0 or not text:
            return ""
        scale = self._font_scale if scale is None else scale
        if self._measure(text, self._menu_theme.font_face, scale, 1)[0][0] <= max_width:
            return text
        ellipsis = ".."
        while text and self._measure(
            text + ellipsis, self._menu_theme.font_face, scale, 1
        )[0][0] > max_width:
            text = text[:-1]
        return (text + ellipsis) if text else ""

    def _format_value(self, option: dict, value) -> str:
        fmt = option.get("format")
        if fmt == "shutter":
            try:
                micros = float(value)
            except (TypeError, ValueError):
                return str(value)
            if micros >= 1_000_000:
                return f"{micros / 1_000_000:.1f}s"
            if micros >= 1000:
                return f"1/{max(1, round(1_000_000 / micros))}"
            return f"{micros:.0f}us"
        if fmt == "ev":
            try:
                return f"{float(value):+.1f}EV"
            except (TypeError, ValueError):
                return str(value)
        if fmt == "seconds":
            try:
                return f"{float(value):g}s"
            except (TypeError, ValueError):
                return str(value)
        if fmt == "gain":
            try:
                return f"x{float(value):.2f}"
            except (TypeError, ValueError):
                return str(value)
        if fmt == "percent":
            return f"{value}%"
        if fmt == "us":
            return f"{value}us"
        if fmt == "degrees":
            try:
                return f"{float(value):.0f}deg"
            except (TypeError, ValueError):
                return str(value)
        if fmt == "lens":
            try:
                return f"{float(value) / 100.0:.2f} dpt"
            except (TypeError, ValueError):
                return str(value)
        if isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)

    def _display_value(self, option: dict) -> str:
        item_type = option["type"].lower()
        value = option.get("value")
        if item_type == MenuItemType.BOOL:
            return "ON" if value else "OFF"
        if item_type == MenuItemType.OPTION:
            return str(value.get("content", "")) if isinstance(value, dict) else str(value)
        if item_type == MenuItemType.NUMERAL:
            return self._format_value(option, value)
        if item_type == MenuItemType.MSG:
            return str(option.get("badge", option.get("receiver", "SEND")))
        if item_type == MenuItemType.IRQ:
            return "GO"
        if item_type == MenuItemType.MENU:
            return ">"
        return str(value)

    def _icon_name(self, option: dict) -> str:
        if option.get("icon"):
            return option["icon"]
        return {
            MenuItemType.BOOL.value: "sliders",
            MenuItemType.MENU.value: "display",
            MenuItemType.IRQ.value: "monitor",
            MenuItemType.MSG.value: "image",
            MenuItemType.OPTION.value: "sliders",
            MenuItemType.NUMERAL.value: "sliders",
        }.get(option["type"].lower(), "sliders")

    def _row_geometry(self, index: int) -> tuple[int, int]:
        y1 = self._content_top + index * self._row_height
        return y1, y1 + self._row_height

    def _draw_header(self, frame) -> None:
        theme = self._menu_theme
        cv2.rectangle(
            frame, (0, 0), (self._width, self._header_height), theme.header_bg, -1
        )
        cv2.rectangle(frame, (0, 0), (3, self._header_height), theme.accent, -1)
        title = self._title or ""
        self._text(
            frame,
            title,
            (self._padding[0] + 4, self._header_height - 5),
            theme.font_face,
            self._font_scale,
            theme.text,
            1,
            cv2.LINE_AA,
        )

        page_text = f"{self._current_page + 1}/{max(1, self._page_count)}"
        (text_width, text_height), _ = self._measure(
            page_text, theme.font_face, self._small_font_scale, 1
        )
        pill_width = text_width + 12
        x2 = self._width - self._padding[2]
        x1 = x2 - pill_width
        y1 = (self._header_height - 14) // 2
        self._rounded_rect(frame, x1, y1, x2, y1 + 14, theme.surface_alt, 7)
        self._text(
            frame,
            page_text,
            (x1 + 6, y1 + 11),
            theme.font_face,
            self._small_font_scale,
            theme.accent,
            1,
            cv2.LINE_AA,
        )

    def _draw_footer(self, frame) -> None:
        theme = self._menu_theme
        y0 = self._height - self._footer_height
        cv2.rectangle(
            frame, (0, y0), (self._width, self._height), theme.footer_bg, -1
        )
        if self._select_index is not None:
            hints = "OK SAVE   < > VALUE   X CANCEL"
        elif self._current_options:
            item_type = self._current_options[self._current_index]["type"].lower()
            ok = {
                MenuItemType.MENU.value: "OPEN",
                MenuItemType.IRQ.value: "GO",
                MenuItemType.MSG.value: "SEND",
                MenuItemType.BOOL.value: "TOGGLE",
                MenuItemType.OPTION.value: "EDIT",
                MenuItemType.NUMERAL.value: "EDIT",
            }.get(item_type, "SELECT")
            hints = f"OK {ok}   < > MOVE   X BACK"
        else:
            hints = "X BACK"
        self._text(
            frame,
            hints,
            (self._padding[0], self._height - 4),
            theme.font_face,
            self._small_font_scale,
            theme.text_dim,
            1,
            cv2.LINE_AA,
        )

    def _draw_value_badge(
        self, frame, option: dict, x1: int, y1: int, x2: int, y2: int, enabled: bool
    ) -> None:
        theme = self._menu_theme
        item_type = option["type"].lower()
        if item_type == MenuItemType.MENU:
            cx = x2 - 5
            cy = (y1 + y2) // 2
            color = theme.accent if enabled else theme.text_dim
            cv2.line(frame, (cx - 4, cy - 4), (cx, cy), color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx - 4, cy + 4), (cx, cy), color, 1, cv2.LINE_AA)
            return

        text = self._display_value(option)
        if item_type == MenuItemType.BOOL:
            background = theme.success if option.get("value") else theme.danger
            foreground = theme.accent_text
        else:
            background = theme.surface_alt
            foreground = (
                theme.color_for_type(item_type) if enabled else theme.text_dim
            )

        (text_width, text_height), _ = self._measure(
            text, theme.font_face, self._small_font_scale, 1
        )
        badge_width = min(x2 - x1, max(34, text_width + 12))
        badge_height = min(18, max(14, y2 - y1 - 6))
        text = self._fit_text(
            text,
            max(0, badge_width - 12),
            self._small_font_scale,
        )
        bx1 = x2 - badge_width
        by1 = (y1 + y2 - badge_height) // 2
        by2 = by1 + badge_height
        self._rounded_rect(
            frame, bx1, by1, x2, by2, background, badge_height // 2
        )
        if item_type == MenuItemType.BOOL:
            self._rounded_rect(
                frame, bx1, by1, x2, by2, background, badge_height // 2
            )
        else:
            self._rounded_rect(
                frame, bx1, by1, x2, by2, theme.border, badge_height // 2, 1
            )
        self._text(
            frame,
            text,
            (bx1 + 6, by2 - 5),
            theme.font_face,
            self._small_font_scale,
            foreground,
            1,
            cv2.LINE_AA,
        )

    def _draw_row(self, frame, index: int, option: dict) -> None:
        theme = self._menu_theme
        y1, y2 = self._row_geometry(index)
        x1 = self._padding[0]
        x2 = self._width - self._padding[2]
        selected = index == self._current_index and self._select_index is None
        enabled = bool(option.get("enable", True))

        if selected:
            if theme.cursor_style == "fill":
                self._rounded_rect(
                    frame, x1, y1 + 1, x2, y2 - 2, theme.surface_alt, 5
                )
            elif theme.cursor_style == "bar":
                self._rounded_rect(
                    frame, x1, y1 + 1, x2, y2 - 2, theme.surface, 5
                )
            else:
                self._rounded_rect(
                    frame, x1, y1 + 1, x2, y2 - 2, theme.accent, 5, 1
                )
            if enabled:
                cv2.rectangle(
                    frame,
                    (x1 + 1, y1 + 5),
                    (x1 + 4, y2 - 6),
                    theme.accent,
                    -1,
                )
        elif index % 2 == 0:
            cv2.rectangle(
                frame, (x1, y1), (x2, y2 - 1), theme.surface, -1
            )

        cv2.line(
            frame,
            (x1, y2 - 1),
            (x2, y2 - 1),
            theme.border,
            1,
            cv2.LINE_AA,
        )

        icon_color = (
            theme.accent
            if selected and enabled
            else (theme.text_dim if not enabled else theme.text_dim)
        )
        icon_x = x1 + 5
        icon_y = y1 + (self._row_height - self._icon_size) // 2
        draw_icon(
            frame,
            self._icon_name(option),
            icon_x,
            icon_y,
            self._icon_size,
            icon_color,
        )

        label_x = icon_x + self._icon_size + 5
        if self._show_index:
            self._text(
                frame,
                str(index + 1),
                (label_x, y1 + (self._row_height + self._font_height) // 2),
                theme.font_face,
                self._small_font_scale,
                theme.text_dim,
                1,
                cv2.LINE_AA,
            )
            label_x += 12

        value_text = self._display_value(option)
        value_badge_width = max(
            34,
            self._measure(
                value_text,
                theme.font_face,
                self._small_font_scale,
                1,
            )[0][0]
            + 12,
        )
        value_width = min(
            int((x2 - x1) * 0.45), value_badge_width
        )
        value_left = x2 - value_width - 4
        label = self._fit_text(
            str(option.get("content", "")), max(0, value_left - label_x - 4)
        )
        baseline = y1 + (self._row_height + self._font_height) // 2
        self._text(
            frame,
            label,
            (label_x, baseline),
            theme.font_face,
            self._font_scale,
            theme.text if enabled else theme.text_dim,
            1,
            cv2.LINE_AA,
        )
        self._draw_value_badge(
            frame, option, value_left, y1, x2, y2, enabled
        )

    def _draw_preview(self, frame) -> None:
        if not self._show_preview or not self._current_options:
            return
        theme = self._menu_theme
        option = self._current_options[self._current_index]
        y2 = self._height - self._footer_height - 2
        y1 = y2 - self._preview_height
        x1 = self._padding[0]
        x2 = self._width - self._padding[2]
        self._rounded_rect(frame, x1, y1, x2, y2, theme.surface, 5)
        cv2.rectangle(
            frame, (x1 + 1, y1 + 5), (x1 + 4, y2 - 5), theme.accent, -1
        )

        label = str(option.get("content", ""))
        self._text(
            frame,
            self._fit_text(label, (x2 - x1) // 2),
            (x1 + 10, y1 + 13),
            theme.font_face,
            self._small_font_scale,
            theme.text_dim,
            1,
            cv2.LINE_AA,
        )
        value = self._display_value(option)
        self._text(
            frame,
            self._fit_text(value, (x2 - x1) // 2),
            (x1 + 10, y2 - 6),
            theme.font_face,
            self._font_scale,
            theme.accent if option.get("enable", True) else theme.text_dim,
            1,
            cv2.LINE_AA,
        )

        item_type = option["type"].lower()
        hint = option.get("hint")
        if not hint:
            if item_type == MenuItemType.NUMERAL:
                hint = (
                    f"{option.get('min')} - {option.get('max')}  "
                    f"{tr('step')} {option.get('step')}"
                )
            elif item_type == MenuItemType.MENU:
                hint = "open >"
            elif item_type == MenuItemType.IRQ:
                hint = "switch screen"
            elif item_type == MenuItemType.MSG:
                hint = "send"
            elif item_type == MenuItemType.BOOL:
                hint = "toggle"
        if hint:
            self._text(
                frame,
                self._fit_text(str(hint), (x2 - x1) // 2 - 10),
                (x1 + (x2 - x1) // 2 + 6, y1 + 13),
                theme.font_face,
                self._small_font_scale,
                theme.text_dim,
                1,
                cv2.LINE_AA,
            )

    def _draw_numeric_edit(self, frame) -> None:
        theme = self._menu_theme
        option = self._current_options[self._current_index]
        value = self._value_temp if self._value_temp is not None else option["value"]
        try:
            minimum = float(option["min"])
            maximum = float(option["max"])
            numeric = float(value)
        except (KeyError, TypeError, ValueError):
            minimum, maximum, numeric = 0.0, 1.0, 0.0
        span = maximum - minimum

        text = self._format_value(option, value)
        (text_width, _), _ = self._measure(
            text, theme.font_face, self._big_font_scale, 1
        )
        self._text(
            frame,
            text,
            ((self._width - text_width) // 2, self._content_top + 42),
            theme.font_face,
            self._big_font_scale,
            theme.accent,
            1,
            cv2.LINE_AA,
        )
        (title_width, _), _ = self._measure(
            option.get("content", ""), theme.font_face, self._small_font_scale, 1
        )
        self._text(
            frame,
            option.get("content", ""),
            ((self._width - title_width) // 2, self._content_top + 16),
            theme.font_face,
            self._small_font_scale,
            theme.text_dim,
            1,
            cv2.LINE_AA,
        )

        x1 = self._padding[0] + 8
        x2 = self._width - self._padding[2] - 8
        y = self._content_top + 72
        self._rounded_rect(frame, x1, y - 3, x2, y + 3, theme.surface_alt, 3)
        fraction = 0.0 if span <= 0 else (numeric - minimum) / span
        fraction = min(1.0, max(0.0, fraction))
        knob_x = int(x1 + (x2 - x1) * fraction)
        self._rounded_rect(frame, x1, y - 2, knob_x, y + 2, theme.accent, 2)
        cv2.circle(frame, (knob_x, y), 5, theme.accent, -1, cv2.LINE_AA)

        self._text(
            frame,
            str(option.get("min", "")),
            (x1, y + 18),
            theme.font_face,
            self._small_font_scale,
            theme.text_dim,
            1,
            cv2.LINE_AA,
        )
        max_text = str(option.get("max", ""))
        (max_width, _), _ = self._measure(
            max_text, theme.font_face, self._small_font_scale, 1
        )
        self._text(
            frame,
            max_text,
            (x2 - max_width, y + 18),
            theme.font_face,
            self._small_font_scale,
            theme.text_dim,
            1,
            cv2.LINE_AA,
        )
        self._text(
            frame,
            f"{tr('step')} {option.get('step', '')}",
            (x1, y + 34),
            theme.font_face,
            self._small_font_scale,
            theme.text_dim,
            1,
            cv2.LINE_AA,
        )

    @classmethod
    def _option_window(cls, selected: int, count: int) -> tuple[int, int]:
        visible = min(cls.OPTION_VISIBLE_COUNT, count)
        start = min(
            max(0, selected - cls.OPTION_VISIBLE_COUNT // 2),
            max(0, count - visible),
        )
        return start, visible

    def _draw_option_edit(self, frame) -> None:
        theme = self._menu_theme
        option = self._current_options[self._current_index]
        value = self._value_temp if self._value_temp is not None else option["value"]
        choices = option.get("options", [])
        try:
            selected = choices.index(value)
        except ValueError:
            selected = 0

        if not choices:
            return

        # Second screen line: fixed arrows plus the option name.  Only the
        # option window below it scrolls.
        arrow_y = self._content_top + 14
        label = self._fit_text(
            str(option.get("content", "")),
            self._width - 2 * self._padding[0] - 28,
            self._small_font_scale,
        )
        (label_width, _), _ = self._measure(
            label, theme.font_face, self._small_font_scale, 1
        )
        self._text(
            frame,
            label,
            ((self._width - label_width) // 2, arrow_y),
            theme.font_face,
            self._small_font_scale,
            theme.text_dim,
            1,
            cv2.LINE_AA,
        )
        self._text(
            frame,
            "<",
            (self._padding[0] + 6, arrow_y),
            theme.font_face,
            self._font_scale,
            theme.accent,
            1,
            cv2.LINE_AA,
        )
        self._text(
            frame,
            ">",
            (self._width - self._padding[2] - 14, arrow_y),
            theme.font_face,
            self._font_scale,
            theme.accent,
            1,
            cv2.LINE_AA,
        )

        # Show a window of choices around the selection instead of only the
        # current/previous/next values.
        start, visible = self._option_window(selected, len(choices))
        x1 = self._padding[0]
        x2 = self._width - self._padding[2]
        list_top = self._content_top + 34
        available = max(1, self._content_bottom - list_top)
        row_height = min(
            28,
            max(18, available // max(1, visible)),
        )
        block_height = row_height * visible
        block_top = list_top + max(
            0, (available - block_height) // 2
        )
        retro = self._retro_style
        if retro in {"luna", "win7", "win10"}:
            radius = 4 if retro == "win10" else 6
            self._rounded_rect(
                frame,
                x1,
                block_top - 4,
                x2,
                block_top + block_height + 4,
                theme.surface,
                radius,
            )
            if retro == "win10":
                cv2.rectangle(
                    frame,
                    (x1, block_top - 4),
                    (x2, block_top + block_height + 4),
                    theme.border,
                    1,
                    cv2.LINE_AA,
                )
        elif retro == "win2000":
            self._bevel_rect(
                frame,
                x1,
                block_top - 4,
                x2,
                block_top + block_height + 4,
                theme.surface_alt,
                raised=False,
            )

        for row in range(visible):
            index = start + row
            choice = choices[index]
            chosen = index == selected
            scale = (
                self._big_font_scale
                if chosen
                else self._small_font_scale
            )
            text = self._fit_text(
                str(choice.get("content", "")),
                self._width - 2 * self._padding[0] - 24,
                scale,
            )
            height = self._pixel_height(scale)
            row_top = block_top + row * row_height
            if chosen and retro == "win7":
                selection_bg = self._blend(
                    theme.surface_alt, theme.accent, 0.25
                )
                self._rounded_rect(
                    frame,
                    x1 + 2,
                    row_top + 1,
                    x2 - 2,
                    row_top + row_height - 2,
                    selection_bg,
                    3,
                )
            elif chosen and retro == "win10":
                self._rounded_rect(
                    frame,
                    x1 + 2,
                    row_top + 1,
                    x2 - 2,
                    row_top + row_height - 2,
                    theme.accent,
                    2,
                )
            elif chosen and retro:
                if retro == "luna":
                    self._rounded_rect(
                        frame,
                        x1 + 2,
                        row_top + 1,
                        x2 - 2,
                        row_top + row_height - 2,
                        theme.accent,
                        4,
                    )
                else:
                    cv2.rectangle(
                        frame,
                        (x1 + 2, row_top + 1),
                        (x2 - 2, row_top + row_height - 2),
                        theme.accent,
                        -1,
                    )
            elif chosen:
                self._rounded_rect(
                    frame,
                    x1,
                    row_top + 1,
                    x2,
                    row_top + row_height - 2,
                    theme.surface_alt,
                    5,
                )
                cv2.rectangle(
                    frame,
                    (x1 + 2, row_top + 6),
                    (x1 + 5, row_top + row_height - 7),
                    theme.accent,
                    -1,
                )
            (text_width, _), _ = self._measure(
                text, theme.font_face, scale, 1
            )
            if chosen and retro:
                text_color = theme.selection_text or theme.accent_text
            elif chosen:
                text_color = theme.accent
            else:
                text_color = theme.text_dim
            self._text(
                frame,
                text,
                (
                    (self._width - text_width) // 2,
                    row_top + (row_height + height) // 2 - 1,
                ),
                theme.font_face,
                scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

    def _draw_power_confirm(self, frame) -> None:
        theme = self._menu_theme
        width = min(250, self._width - 20)
        height = 86
        x1 = (self._width - width) // 2
        y1 = (self._height - height) // 2
        x2 = x1 + width
        y2 = y1 + height
        retro = self._retro_style
        if retro == "luna":
            self._rounded_rect(frame, x1, y1, x2, y2, theme.surface, 8)
            cv2.rectangle(
                frame, (x1 + 1, y1 + 1), (x2 - 1, y1 + 22),
                theme.header_bg, -1,
            )
            self._rounded_rect(frame, x1, y1, x2, y2, theme.accent, 8, 1)
        elif retro == "win2000":
            self._bevel_rect(
                frame, x1, y1, x2, y2, theme.surface_alt, raised=True
            )
            cv2.rectangle(
                frame, (x1 + 2, y1 + 2), (x2 - 2, y1 + 20),
                theme.header_bg, -1,
            )
        elif retro == "win7":
            self._rounded_rect(frame, x1, y1, x2, y2, theme.surface, 8)
            cv2.rectangle(
                frame, (x1 + 1, y1 + 1), (x2 - 1, y1 + 22),
                theme.header_bg, -1,
            )
            self._rounded_rect(frame, x1, y1, x2, y2, theme.accent, 8, 1)
        elif retro == "win10":
            self._rounded_rect(frame, x1, y1, x2, y2, theme.surface, 4)
            cv2.rectangle(
                frame, (x1 + 1, y1 + 1), (x2 - 1, y1 + 22),
                theme.accent, -1,
            )
            self._rounded_rect(frame, x1, y1, x2, y2, theme.border, 4, 1)
        else:
            self._rounded_rect(frame, x1, y1, x2, y2, theme.surface, 8)
            self._rounded_rect(frame, x1, y1, x2, y2, theme.danger, 8, 2)

        action = ""
        if isinstance(self._confirm_option, dict):
            action = self._confirm_option.get("value")
            if isinstance(action, dict):
                action = action.get("value")
        title = tr(
            "Shut down system?"
            if action == "shutdown"
            else "Reboot system?"
        )
        (title_width, _), _ = self._measure(
            title, theme.font_face, self._font_scale, 1
        )
        title_color = (
            theme.accent_text if retro else theme.text
        )
        title_y = y1 + 16 if retro else y1 + 32
        self._text(
            frame,
            title,
            ((self._width - title_width) // 2, title_y),
            theme.font_face,
            self._font_scale,
            title_color,
            1,
            cv2.LINE_AA,
        )
        hint = tr("OK = YES    X = NO")
        (hint_width, _), _ = self._measure(
            hint, theme.font_face, self._small_font_scale, 1
        )
        self._text(
            frame,
            hint,
            ((self._width - hint_width) // 2, y2 - 16),
            theme.font_face,
            self._small_font_scale,
            theme.text_dim,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _blend(
        first: tuple[int, int, int],
        second: tuple[int, int, int],
        fraction: float,
    ) -> tuple[int, int, int]:
        return tuple(
            int(first[index] + (second[index] - first[index]) * fraction)
            for index in range(3)
        )

    @staticmethod
    def _bevel_rect(
        frame,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        fill,
        raised: bool = True,
    ) -> None:
        """Win2000-style 3D bevel."""

        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        light = (255, 255, 255)
        dark = (128, 128, 128)
        top_left, bottom_right = (
            (light, dark) if raised else (dark, light)
        )
        cv2.line(frame, (x1, y1), (x2, y1), top_left, 1)
        cv2.line(frame, (x1, y1), (x1, y2), top_left, 1)
        cv2.line(frame, (x1, y2), (x2, y2), bottom_right, 1)
        cv2.line(frame, (x2, y1), (x2, y2), bottom_right, 1)

    def _draw_retro_header(self, frame, luna: bool) -> None:
        theme = self._menu_theme
        height = self._header_height
        if luna:
            top = (255, 150, 60)
            bottom = theme.header_bg
            for y in range(height):
                color = self._blend(
                    top, bottom, y / max(1, height - 1)
                )
                cv2.line(frame, (0, y), (self._width, y), color, 1)
            cv2.line(
                frame,
                (0, height - 1),
                (self._width, height - 1),
                (255, 255, 255),
                1,
            )
        else:
            cv2.rectangle(
                frame, (0, 0), (self._width, height), theme.header_bg, -1
            )
            cv2.line(
                frame,
                (0, height - 1),
                (self._width, height - 1),
                (255, 255, 255),
                1,
            )

        title = self._title or ""
        self._text(
            frame,
            title,
            (self._padding[0] + 4, height - 5),
            theme.font_face,
            self._font_scale,
            theme.accent_text,
            1,
            cv2.LINE_AA,
        )

        page_text = f"{self._current_page + 1}/{max(1, self._page_count)}"
        (page_width, _), _ = self._measure(
            page_text, theme.font_face, self._small_font_scale, 1
        )
        if luna:
            pill_width = page_width + 12
            x2 = self._width - self._padding[2]
            x1 = x2 - pill_width
            y1 = (height - 14) // 2
            self._rounded_rect(
                frame, x1, y1, x2, y1 + 14, theme.surface_alt, 7
            )
            self._text(
                frame,
                page_text,
                (x1 + 6, y1 + 11),
                theme.font_face,
                self._small_font_scale,
                theme.accent,
                1,
                cv2.LINE_AA,
            )
        else:
            controls = 3
            box = 11
            gap = 2
            x = self._width - self._padding[2] - controls * box - gap
            page_x = x - page_width - 6
            self._text(
                frame,
                page_text,
                (page_x, height - 5),
                theme.font_face,
                self._small_font_scale,
                theme.accent_text,
                1,
                cv2.LINE_AA,
            )
            for index in range(controls):
                bx = x + index * (box + gap)
                self._bevel_rect(
                    frame,
                    bx,
                    4,
                    bx + box,
                    4 + box,
                    theme.surface_alt,
                    raised=True,
                )
                if index == 0:  # minimise
                    cv2.line(
                        frame,
                        (bx + 3, 4 + box - 4),
                        (bx + box - 3, 4 + box - 4),
                        (0, 0, 0),
                        1,
                    )
                elif index == 1:  # maximise
                    cv2.rectangle(
                        frame,
                        (bx + 3, 7),
                        (bx + box - 3, 4 + box - 3),
                        (0, 0, 0),
                        1,
                    )
                else:  # close
                    cv2.line(
                        frame,
                        (bx + 3, 7),
                        (bx + box - 3, 4 + box - 3),
                        (0, 0, 0),
                        1,
                    )
                    cv2.line(
                        frame,
                        (bx + box - 3, 7),
                        (bx + 3, 4 + box - 3),
                        (0, 0, 0),
                        1,
                    )

    def _draw_win7_header(self, frame) -> None:
        theme = self._menu_theme
        height = self._header_height
        top = (210, 160, 100)
        bottom = theme.header_bg
        for y in range(height):
            color = self._blend(top, bottom, y / max(1, height - 1))
            cv2.line(frame, (0, y), (self._width, y), color, 1)
        cv2.line(
            frame,
            (0, height - 1),
            (self._width, height - 1),
            theme.accent,
            1,
        )
        title = self._title or ""
        self._text(
            frame,
            title,
            (self._padding[0] + 4, height - 5),
            theme.font_face,
            self._font_scale,
            theme.header_text or theme.text,
            1,
            cv2.LINE_AA,
        )
        page_text = f"{self._current_page + 1}/{max(1, self._page_count)}"
        (page_width, _), _ = self._measure(
            page_text, theme.font_face, self._small_font_scale, 1
        )
        pill_width = page_width + 12
        x2 = self._width - self._padding[2]
        x1 = x2 - pill_width
        y1 = (height - 14) // 2
        self._rounded_rect(
            frame, x1, y1, x2, y1 + 14, theme.surface_alt, 7
        )
        self._text(
            frame,
            page_text,
            (x1 + 6, y1 + 11),
            theme.font_face,
            self._small_font_scale,
            theme.accent,
            1,
            cv2.LINE_AA,
        )

    def _draw_win10_header(self, frame) -> None:
        theme = self._menu_theme
        height = self._header_height
        cv2.rectangle(
            frame, (0, 0), (self._width, height), theme.header_bg, -1
        )
        cv2.rectangle(
            frame, (0, 0), (3, height), theme.accent, -1
        )
        title = self._title or ""
        self._text(
            frame,
            title,
            (self._padding[0] + 4, height - 5),
            theme.font_face,
            self._font_scale,
            theme.header_text or theme.text,
            1,
            cv2.LINE_AA,
        )
        page_text = f"{self._current_page + 1}/{max(1, self._page_count)}"
        (page_width, _), _ = self._measure(
            page_text, theme.font_face, self._small_font_scale, 1
        )
        self._text(
            frame,
            page_text,
            (self._width - self._padding[2] - page_width, height - 5),
            theme.font_face,
            self._small_font_scale,
            theme.header_text_dim or theme.text_dim,
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (0, height - 1),
            (self._width, height - 1),
            theme.accent,
            1,
        )

    def _draw_retro_row(self, frame, index: int, option: dict) -> None:
        theme = self._menu_theme
        style = self._retro_style
        y1, y2 = self._row_geometry(index)
        x1 = self._padding[0]
        x2 = self._width - self._padding[2]
        selected = index == self._current_index and self._select_index is None
        enabled = bool(option.get("enable", True))

        if style == "luna":
            if selected:
                self._rounded_rect(
                    frame, x1 + 3, y1 + 1, x2 - 3, y2 - 2,
                    theme.accent, 4,
                )
                cv2.line(
                    frame,
                    (x1 + 8, y1 + 3),
                    (x2 - 8, y1 + 3),
                    (255, 220, 180),
                    1,
                )
            else:
                row_bg = (
                    theme.surface if index % 2 == 0
                    else theme.surface_alt
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2 - 1), row_bg, -1)
            cv2.line(
                frame, (x1, y2 - 1), (x2, y2 - 1),
                theme.border, 1,
            )
            foreground = (
                theme.accent_text if selected
                else (theme.text if enabled else theme.text_dim)
            )
            icon_color = (
                theme.accent_text if selected
                else (theme.accent if enabled else theme.text_dim)
            )
        elif style == "win2000":
            if selected:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2 - 1), theme.accent, -1
                )
            else:
                self._bevel_rect(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2 - 1,
                    theme.surface,
                    raised=False,
                )
            foreground = (
                theme.accent_text if selected
                else (theme.text if enabled else theme.text_dim)
            )
            icon_color = (
                theme.accent_text if selected
                else (theme.text if enabled else theme.text_dim)
            )
        elif style == "win7":
            if selected:
                selection_bg = self._blend(
                    theme.surface_alt, theme.accent, 0.25
                )
                self._rounded_rect(
                    frame, x1 + 2, y1 + 1, x2 - 2, y2 - 2,
                    selection_bg, 3,
                )
                cv2.rectangle(
                    frame,
                    (x1 + 3, y1 + 2),
                    (x1 + 6, y2 - 3),
                    theme.accent,
                    -1,
                )
            else:
                row_bg = (
                    theme.surface if index % 2 == 0
                    else theme.surface_alt
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2 - 1), row_bg, -1)
            cv2.line(
                frame, (x1, y2 - 1), (x2, y2 - 1),
                theme.border, 1,
            )
            foreground = (
                (theme.selection_text or theme.text)
                if selected
                else (theme.text if enabled else theme.text_dim)
            )
            icon_color = (
                theme.accent if selected
                else (theme.accent if enabled else theme.text_dim)
            )
        else:  # win10
            if selected:
                self._rounded_rect(
                    frame, x1 + 2, y1 + 1, x2 - 2, y2 - 2,
                    theme.accent, 2,
                )
            else:
                row_bg = (
                    theme.surface if index % 2 == 0
                    else theme.surface_alt
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2 - 1), row_bg, -1)
            cv2.line(
                frame, (x1, y2 - 1), (x2, y2 - 1),
                theme.border, 1,
            )
            foreground = (
                (theme.selection_text or theme.accent_text)
                if selected
                else (theme.text if enabled else theme.text_dim)
            )
            icon_color = (
                (theme.selection_text or theme.accent_text)
                if selected
                else (theme.text_dim if enabled else theme.text_dim)
            )

        icon_x = x1 + 5
        icon_y = y1 + (self._row_height - self._icon_size) // 2
        draw_icon(
            frame,
            self._icon_name(option),
            icon_x,
            icon_y,
            self._icon_size,
            icon_color,
        )

        label_x = icon_x + self._icon_size + 5
        baseline = y1 + (self._row_height + self._font_height) // 2
        if self._show_index:
            self._text(
                frame,
                str(index + 1),
                (label_x, baseline),
                theme.font_face,
                self._small_font_scale,
                theme.text_dim if not selected else foreground,
                1,
                cv2.LINE_AA,
            )
            label_x += 12

        item_type = option["type"].lower()
        value_text = self._display_value(option)
        value_width = 34
        if item_type != MenuItemType.MENU.value:
            value_width = min(
                int((x2 - x1) * 0.45),
                max(
                    34,
                    self._measure(
                        value_text,
                        theme.font_face,
                        self._small_font_scale,
                        1,
                    )[0][0]
                    + 12,
                ),
            )
        value_left = x2 - value_width - 4
        label = self._fit_text(
            str(option.get("content", "")),
            max(0, value_left - label_x - 4),
        )
        self._text(
            frame,
            label,
            (label_x, baseline),
            theme.font_face,
            self._font_scale,
            foreground,
            1,
            cv2.LINE_AA,
        )

        if item_type == MenuItemType.MENU.value:
            self._text(
                frame,
                ">",
                (x2 - 14, baseline),
                theme.font_face,
                self._font_scale,
                foreground if selected else theme.accent,
                1,
                cv2.LINE_AA,
            )
            return

        badge_height = min(18, max(14, y2 - y1 - 6))
        value_fit = self._fit_text(
            value_text,
            max(0, value_width - 12),
            self._small_font_scale,
        )
        (text_width, _), _ = self._measure(
            value_fit, theme.font_face, self._small_font_scale, 1
        )
        badge_width = min(value_width, max(34, text_width + 12))
        bx1 = x2 - badge_width - 4
        by1 = (y1 + y2 - badge_height) // 2
        by2 = by1 + badge_height
        if style == "luna":
            background = (
                theme.surface_alt if not selected else theme.surface
            )
            foreground_badge = theme.accent
            self._rounded_rect(
                frame, bx1, by1, bx1 + badge_width, by2,
                background, badge_height // 2,
            )
            cv2.rectangle(
                frame, (bx1, by1), (bx1 + badge_width, by2),
                theme.accent, 1, cv2.LINE_AA,
            )
        elif style == "win2000":
            background = theme.surface_alt
            foreground_badge = theme.text
            self._bevel_rect(
                frame, bx1, by1, bx1 + badge_width, by2,
                background, raised=True,
            )
        elif style == "win7":
            background = theme.surface
            foreground_badge = theme.accent
            self._rounded_rect(
                frame, bx1, by1, bx1 + badge_width, by2,
                background, 4,
            )
            cv2.rectangle(
                frame, (bx1, by1), (bx1 + badge_width, by2),
                theme.accent, 1, cv2.LINE_AA,
            )
        else:  # win10
            background = theme.surface_alt
            foreground_badge = (
                theme.selection_text or theme.text
            )
            self._rounded_rect(
                frame, bx1, by1, bx1 + badge_width, by2,
                background, 2,
            )
            cv2.rectangle(
                frame, (bx1, by1), (bx1 + badge_width, by2),
                theme.border, 1, cv2.LINE_AA,
            )
        self._text(
            frame,
            value_fit,
            (
                bx1 + (badge_width - text_width) // 2,
                by2 - 5,
            ),
            theme.font_face,
            self._small_font_scale,
            foreground_badge,
            1,
            cv2.LINE_AA,
        )

    def _draw_retro_preview(self, frame) -> None:
        if not self._show_preview or not self._current_options:
            return
        theme = self._menu_theme
        option = self._current_options[self._current_index]
        y2 = self._height - self._footer_height - 2
        y1 = y2 - self._preview_height
        x1 = self._padding[0]
        x2 = self._width - self._padding[2]
        if self._retro_style == "win2000":
            self._bevel_rect(
                frame, x1, y1, x2, y2, theme.surface_alt, raised=True
            )
        else:
            radius = 4 if self._retro_style == "win10" else 5
            self._rounded_rect(
                frame, x1, y1, x2, y2, theme.surface, radius
            )
            cv2.rectangle(
                frame, (x1 + 2, y1 + 5), (x1 + 5, y2 - 5),
                theme.accent, -1,
            )
            if self._retro_style == "win10":
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    theme.border, 1, cv2.LINE_AA,
                )

        label = self._fit_text(
            str(option.get("content", "")),
            (x2 - x1) // 2,
            self._small_font_scale,
        )
        self._text(
            frame,
            label,
            (x1 + 10, y1 + 13),
            theme.font_face,
            self._small_font_scale,
            theme.text_dim,
            1,
            cv2.LINE_AA,
        )
        value = self._fit_text(
            self._display_value(option),
            (x2 - x1) // 2,
        )
        self._text(
            frame,
            value,
            (x1 + 10, y2 - 6),
            theme.font_face,
            self._font_scale,
            theme.accent if option.get("enable", True) else theme.text_dim,
            1,
            cv2.LINE_AA,
        )

    def _decorate_retro(self, style: str) -> None:
        self._retro_style = style
        theme = self._menu_theme
        sketch = np.full(
            (self._height, self._width, 3), theme.background, np.uint8
        )
        if style == "luna":
            self._draw_retro_header(sketch, True)
        elif style == "win2000":
            self._draw_retro_header(sketch, False)
        elif style == "win7":
            self._draw_win7_header(sketch)
        else:
            self._draw_win10_header(sketch)
        if self._select_index is None:
            for index, option in enumerate(self._current_options):
                self._draw_retro_row(sketch, index, option)
            self._draw_retro_preview(sketch)
        else:
            item_type = self._current_options[
                self._current_index
            ]["type"].lower()
            if item_type == MenuItemType.NUMERAL:
                self._draw_numeric_edit(sketch)
            elif item_type == MenuItemType.OPTION:
                self._draw_option_edit(sketch)
            else:
                for index, option in enumerate(self._current_options):
                    self._draw_retro_row(sketch, index, option)
        self._draw_footer(sketch)
        self._flush_text(sketch)
        if self._confirm_power:
            self._draw_power_confirm(sketch)
            self._flush_text(sketch)
        self._frame_list.put(sketch)

    def _decorate_classic(self) -> None:
        """Draw the original pre-0.3 menu layout.

        The classic theme keeps the italic Hershey text, slide bar, packed
        value blocks and the original edit screens instead of the modern
        header/card/footer renderer.
        """

        self._retro_style = None
        sketch = np.full(
            (self._height, self._width, 3),
            self._theme["background"],
            np.uint8,
        )
        if self._select_index is None:
            if self._title is not None:
                self._draw_title(sketch)
            if self._current_options:
                self._draw_cursor(sketch)
                self._draw_content(sketch)
                self._draw_data(sketch)
            self._draw_slide_bar(sketch)
        else:
            item_type = self._current_options[
                self._current_index
            ]["type"].lower()
            if item_type == MenuItemType.NUMERAL:
                self._numerical_slide_bar(sketch)
            elif item_type == MenuItemType.OPTION:
                self._option_menu(sketch)
        self._flush_text(sketch)
        if self._confirm_power:
            self._draw_power_confirm(sketch)
            self._flush_text(sketch)
        self._frame_list.put(sketch)

    def decorate(self):
        style = self._menu_theme.render_style
        if style == "classic":
            self._decorate_classic()
            return
        if style in {"luna", "win2000", "win7", "win10"}:
            self._decorate_retro(style)
            return

        self._retro_style = None
        theme = self._menu_theme
        sketch = np.full(
            (self._height, self._width, 3), theme.background, np.uint8
        )
        self._draw_header(sketch)
        if self._select_index is None:
            for index, option in enumerate(self._current_options):
                self._draw_row(sketch, index, option)
            self._draw_preview(sketch)
        else:
            item_type = self._current_options[
                self._current_index
            ]["type"].lower()
            if item_type == MenuItemType.NUMERAL:
                self._draw_numeric_edit(sketch)
            elif item_type == MenuItemType.OPTION:
                self._draw_option_edit(sketch)
            else:
                for index, option in enumerate(self._current_options):
                    self._draw_row(sketch, index, option)
        self._draw_footer(sketch)
        self._flush_text(sketch)
        if self._confirm_power:
            self._draw_power_confirm(sketch)
            self._flush_text(sketch)
        self._frame_list.put(sketch)

    def _next_step(self):
        item: dict = self._current_options[self._current_index]
        if 'step_options' not in item.keys():
            return
        current_step_index = item['step_options'].index(item['step'])
        if current_step_index == len(item['step_options']) - 1:
            item['step'] = item['step_options'][0]
        else:
            item['step'] = item['step_options'][current_step_index + 1]

    def _previous_step(self):
        item: dict = self._current_options[self._current_index]
        if 'step_options' not in item.keys():
            return
        current_step_index = item['step_options'].index(item['step'])
        if current_step_index == 0:
            item['step'] = item['step_options'][-1]
        else:
            item['step'] = item['step_options'][current_step_index - 1]

    def _value_plus(self):
        item = self._current_options[self._select_index]
        if item['type'] != MenuItemType.NUMERAL:
            return
        value, step, ma = self._value_temp, item['step'], item['max']
        value += step
        if value >= ma:
            value = ma
        self._value_temp = value

    def _value_minus(self):
        item = self._current_options[self._select_index]
        if item['type'] != MenuItemType.NUMERAL:
            return
        value, step, mi = self._value_temp, item['step'], item['min']
        value -= step
        if value <= mi:
            value = mi
        self._value_temp = value

    def _option_up(self):
        item = self._current_options[self._select_index]
        if item['type'] != MenuItemType.OPTION:
            return
        value, option_list = self._value_temp, item['options']
        option_index = option_list.index(value)
        self._value_temp = option_list[option_index - 1]
        self._preview_theme()

    def _option_down(self):
        item = self._current_options[self._select_index]
        if item['type'] != MenuItemType.OPTION:
            return
        value, option_list = self._value_temp, item['options']
        option_index = option_list.index(value)
        try:
            self._value_temp = option_list[option_index + 1]
        except IndexError:
            self._value_temp = option_list[0]
        self._preview_theme()

    def center_press_action(self):
        if self._confirm_power:
            option = self._confirm_option
            self._confirm_power = False
            self._confirm_option = None
            if option is not None and option.get("id") == "power":
                self._send_power_action(option)
            else:
                self._send_current_message(option)
            self.decorate()
            return
        if self._select_index is not None:
            self.unselect()
        else:
            self.select()
        self.decorate()

    def up_press_action(self):
        if self._select_index is None:
            self.up_action()
            self.decorate()
            time.sleep(0.2)
        else:
            if self._current_options[self._select_index]['type'] == MenuItemType.NUMERAL:
                self._previous_step()
                self.decorate()
                time.sleep(0.2)
            else:
                self._option_up()
                self.decorate()
                time.sleep(0.2)

    def down_press_action(self):
        if self._select_index is None:
            self.down_action()
            self.decorate()
            time.sleep(0.2)
        else:
            if self._current_options[self._select_index]['type'] == MenuItemType.NUMERAL:
                self._next_step()
                self.decorate()
                time.sleep(0.2)
            else:
                self._option_down()
                self.decorate()
                time.sleep(0.2)

    def left_press_action(self):
        if self._select_index is None:
            self.up_action()
            self.decorate()
            time.sleep(0.2)
        else:
            if self._current_options[self._select_index]['type'] == MenuItemType.NUMERAL:
                self._value_minus()
                self.decorate()
                time.sleep(0.1)
            else:
                self._option_up()
                self.decorate()
                time.sleep(0.2)

    def right_press_action(self):
        if self._select_index is None:
            self.down_action()
            self.decorate()
            time.sleep(0.2)
        else:
            if self._current_options[self._select_index]['type'] == MenuItemType.NUMERAL:
                self._value_plus()
                self.decorate()
                time.sleep(0.1)
            else:
                self._option_down()
                self.decorate()
                time.sleep(0.2)

    def circle_press_action(self):
        self.center_press_action()

    def cross_press_action(self):
        if self._confirm_power:
            self._confirm_power = False
            self._confirm_option = None
            self.decorate()
            return
        if self._select_index is not None:
            self._cancel_theme_preview()
            self._select_index = None
            self.decorate()
        else:
            try:
                self._jump_to_previous()
                self.decorate()
            except IndexError:
                self._irq(self._from)

    def cross_long_press_action(self):
        self._cancel_theme_preview()
        self._irq(self._from)

    def rotary_encoder_counter_clockwise(self):
        if self._select_index is None:
            self.down_action()
            self.decorate()
            time.sleep(0.05)
        else:
            if self._current_options[self._select_index]['type'] == MenuItemType.NUMERAL:
                self._value_plus()
                self.decorate()
                time.sleep(0.1)
            else:
                self._option_down()
                self.decorate()
                time.sleep(0.2)

    def rotary_encoder_clockwise(self):
        if self._select_index is None:
            self.up_action()
            self.decorate()
            time.sleep(0.05)
        else:
            if self._current_options[self._select_index]['type'] == MenuItemType.NUMERAL:
                self._value_minus()
                self.decorate()
                time.sleep(0.1)
            else:
                self._option_up()
                self.decorate()
                time.sleep(0.2)

    def rotary_encoder_select(self):
        self.center_press_action()

    def shutter_press_action(self):
        if self._confirm_power:
            self.center_press_action()
            return
        if self._select_index is not None:
            self.unselect()
        else:
            self.select()
        self.decorate()

    def square_press_action(self):
        pass

    def msg_receiver(self, sender, msg):
        self.set_option(msg)
        self._from = sender
        self._msg_sender(
            self._id,
            self._from,
            (
                self._current_menu_id,
                self._menu_options[self._from]
            )
        )

    def on_enter(self, last_id):
        self._frame_list = queue.SimpleQueue()
        self._confirm_power = False
        self._confirm_option = None
        self._finish_theme_preview()
        self.decorate()

    def on_exit(self):
        self._cancel_theme_preview()
        self._frame_list.put(
            np.full(
                (self._height, self._width, 3),
                self._theme['background'], np.uint8),
            block=True
        )

    def main_loop(self):
        while True:
            yield self._frame_list.get(True)
