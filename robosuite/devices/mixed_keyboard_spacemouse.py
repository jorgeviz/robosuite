"""
Driver class for Mixed Keyboard/Space Mouse controller.
"""
import glfw
import threading
import numpy as np
from .keyboard import Keyboard
from .spacemouse import SpaceMouse, to_int16, scale_to_control, convert
from .device import Device
from robosuite.utils.transform_utils import rotation_matrix


class MixedSpaceMouseKeyboard(Device):
    """ SpaceMouse / Keyboard Handler
    """

    def __init__(self, sm_vendor_id, sm_product_id, mode="pure_translation"):
        """ initialize SpaceMouse and Keyboard drivers
        """
        # Mode
        self.control_mode = mode
        # Devices
        self.space_mouse = SpaceMouse(
            vendor_id=sm_vendor_id,
            product_id=sm_product_id,
            listener=False
        )
        self.keyboard = Keyboard()
        self.axis_var = 0  # `fixed_rot_axis` aux var
        self.smouse_read = "pos"  # `swap_pos` aux var

        self._display_controls()
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = False
        # Speed of horizontal Keyboard steps
        self._pos_step = 0.05  
        # Space Mouse handling thread
        # launch a new listener thread to listen 
        # to SpaceMouse
        self.thread = threading.Thread(target=self.handle_smouse)
        self.thread.daemon = True
        self.thread.start()
    
    def close(self):
        """ Close event listeners 
        """
        self.space_mouse.close()
        self.keyboard.close()
    
    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys/Control", "Command")
        print_command("Right Mouse Button", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("UP-DOWN-RIGHT-DOWN", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command(
            "Twist mouse about an axis", "rotate arm about a corresponding axis"
        )
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        if self.control_mode == "swap_pos":
            self.pos = np.zeros(3)  # (x, y, z)
            self.last_pos = np.zeros(3)
        else:
            self.pos = np.zeros(2)  # (x, y)
            self.last_pos = np.zeros(2)
        self.grasp = False
    
    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
    
    def handle_smouse(self):
        """Listener method that keeps pulling new messages."""

        t_last_click = -1
        while True:
            d = self.space_mouse.device.read(8)
            # print("reading len", len(d))
            if d is not None and self._enabled:
                if d[0] == 1:  ## readings from 6-DoF sensor
                    self.x = convert(d[3], d[4], 100.)
                    self.y = convert(d[1], d[2], 100.)
                    self.z = convert(d[5], d[6], 100.) * -1.0

                    d_2 = self.space_mouse.device.read(8) # Read orientation pkg
                    self.roll = convert(d_2[1], d_2[2], 350.)
                    self.pitch = convert(d_2[3], d_2[4], 350.)
                    self.yaw = convert(d_2[5], d_2[6], 350.)

                    self.space_mouse._control = [
                        self.x,
                        self.y,
                        self.z,
                        self.roll,
                        self.pitch,
                        self.yaw,
                    ]

                elif d[0] == 3:  ## readings from the side buttons
                    print("Reading from buttons")
                    # press left button
                    # if d[1] == 1:
                    #    pass
                    # release left button
                    if d[1] == 0:
                        print("Release left button")
                        # self.single_click_and_hold = False
                        if self.control_mode != "swap_pos":
                            self.grasp = not self.grasp # toggle gripper

                    if d[1] == 2:
                        print("Release right button")
                        if self.control_mode == "swap_pos":
                            # SMouse read Translations
                            if self.smouse_read == "rot":
                                self.smouse_read = "pos"
                                print("---- Swapped to read Position")
                            else:
                                self.smouse_read = "rot"
                                print("---- Swapped to read Rotation")
                        else:
                            # right button is for reset
                            self._reset_state = 1
                            self._enabled = False
                            self._reset_internal_state()
        
    def on_press(self, window, key, scancode, action, mods):
        """
        Key handler for key presses.
        """
        # controls for moving position
        if key == glfw.KEY_UP:
            if self.control_mode == "swap_pos":
                # Swap Pos uses keys for Z
                self.pos[2] += self._pos_step
            else:
                self.pos[0] -= self._pos_step  # dec x
        elif key == glfw.KEY_DOWN:
            if self.control_mode == "swap_pos":
                # Swap Pos uses keys for Z
                self.pos[2] -= self._pos_step
            else:
                self.pos[0] += self._pos_step  # inc x
        elif key == glfw.KEY_LEFT:
            self.pos[1] -= self._pos_step  # dec y
        elif key == glfw.KEY_RIGHT:
            self.pos[1] += self._pos_step  # inc y
        elif key == glfw.KEY_0:
            self.axis_var = 0 # Rotation over axis X
        elif key == glfw.KEY_1:
            self.axis_var = 1 # Rotation over axis Y
        elif key == glfw.KEY_2:
            self.axis_var = 2 # Rotation over axis Z
        elif key == glfw.KEY_Q:
            # right button is for reset
            self._reset_state = 1
            self._enabled = False
            self._reset_internal_state()
        elif key == glfw.KEY_SPACE:
            if self.control_mode == "swap_pos":
                # Use Space bar for toggling grasp
                self.grasp = not self.grasp

    def on_release(self, window, key, scancode, action, mods):
        """
        Key handler for key releases.
        """
        pass

    def get_controller_state(self):
        """Returns the current state of
            the keyboard, a dictionary of pos, 
            orn, grasp, and reset.
        """
        if self.control_mode == "mixed" or self.control_mode == "fixed_rot_axis":
            # X-Y Position from keyboard
            dpos = np.zeros(3)
            _dpos = self.pos - self.last_pos
            dpos[0], dpos[1] = _dpos[0], _dpos[1]
            # Z position from SMouse
            dpos[2] = self.space_mouse.control[2] * 0.005
            self.last_pos = np.array(self.pos)
        if self.control_mode == "pure_translation":
            # X,Y,Z from Smouse
            dpos = self.space_mouse.control[:3] * 0.005
        if self.control_mode == "mixed":
            # Rotation from SMouse
            roll, pitch, yaw = self.space_mouse.control[3:] * 0.005
        if self.control_mode == "pure_translation":
            # Set rotation to zero
            roll, pitch, yaw = 0., 0., 0.
        if self.control_mode == "fixed_rot_axis":
            # Set rotation for the axis selected
            roll, pitch, yaw = tuple([
                self.space_mouse.control[self.axis_var + 3] * 0.005 \
                    if i == self.axis_var else 0. \
                    for i in range(3)
            ])
        if self.control_mode == "swap_pos":
            if self.smouse_read == "pos":
                ## Just use translation
                # X,Y from Smouse
                dpos = np.zeros(3)
                _dpos = self.space_mouse.control[:2] * 0.005
                dpos[0], dpos[1] = _dpos[0], _dpos[1]
                # Z from keyboard
                #dpos[2] = self.pos[2] - self.last_pos[2]
                dpos[2] = self.space_mouse.control[-1] * 0.005
                # self.last_pos = np.array(self.pos)
                roll, pitch, yaw = 0., 0., 0.
            else:
                ## Just use rotation
                dpos = np.zeros(3)
                roll, pitch, yaw = self.space_mouse.control[3:] * 0.005
        
            # print("Pos: ", dpos)
            # print("Orient: ", roll, pitch, yaw)
        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1., 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1., 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.], point=None)[:3, :3]
        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
        )