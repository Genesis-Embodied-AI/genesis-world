from typing import TYPE_CHECKING

import torch
import numpy as np

from genesis.utils.geom import quat_to_xyz

if TYPE_CHECKING:
    from genesis.engine.entities import DroneEntity


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


class DronePIDController:
    def __init__(self, drone: "DroneEntity", dt, base_rpm, pid_params):
        self.__pid_pos_x = PIDController(kp=pid_params[0][0], ki=pid_params[0][1], kd=pid_params[0][2])
        self.__pid_pos_y = PIDController(kp=pid_params[1][0], ki=pid_params[1][1], kd=pid_params[1][2])
        self.__pid_pos_z = PIDController(kp=pid_params[2][0], ki=pid_params[2][1], kd=pid_params[2][2])

        self.__pid_vel_x = PIDController(kp=pid_params[3][0], ki=pid_params[3][1], kd=pid_params[3][2])
        self.__pid_vel_y = PIDController(kp=pid_params[4][0], ki=pid_params[4][1], kd=pid_params[4][2])
        self.__pid_vel_z = PIDController(kp=pid_params[5][0], ki=pid_params[5][1], kd=pid_params[5][2])

        self.__pid_att_roll = PIDController(kp=pid_params[6][0], ki=pid_params[6][1], kd=pid_params[6][2])
        self.__pid_att_pitch = PIDController(kp=pid_params[7][0], ki=pid_params[7][1], kd=pid_params[7][2])
        self.__pid_att_yaw = PIDController(kp=pid_params[8][0], ki=pid_params[8][1], kd=pid_params[8][2])

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()

    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        return quat_to_xyz(quat, rpy=True, degrees=True)

    def __mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel) -> torch.Tensor:
        M1 = self.__base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        M2 = self.__base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        M3 = self.__base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        M4 = self.__base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        return torch.Tensor([M1, M2, M3, M4])

    def update(self, target) -> np.ndarray:
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_att = self.__get_drone_att()

        err_pos_x = target[0] - curr_pos[0]
        err_pos_y = target[1] - curr_pos[1]
        err_pos_z = target[2] - curr_pos[2]

        vel_des_x = self.__pid_pos_x.update(err_pos_x, self.__dt)
        vel_des_y = self.__pid_pos_y.update(err_pos_y, self.__dt)
        vel_des_z = self.__pid_pos_z.update(err_pos_z, self.__dt)

        error_vel_x = vel_des_x - curr_vel[0]
        error_vel_y = vel_des_y - curr_vel[1]
        error_vel_z = vel_des_z - curr_vel[2]

        x_vel_del = self.__pid_vel_x.update(error_vel_x, self.__dt)
        y_vel_del = self.__pid_vel_y.update(error_vel_y, self.__dt)
        thrust_des = self.__pid_vel_z.update(error_vel_z, self.__dt)

        err_roll = 0.0 - curr_att[0]
        err_pitch = 0.0 - curr_att[1]
        err_yaw = 0.0 - curr_att[2]

        roll_del = self.__pid_att_roll.update(err_roll, self.__dt)
        pitch_del = self.__pid_att_pitch.update(err_pitch, self.__dt)
        yaw_del = self.__pid_att_yaw.update(err_yaw, self.__dt)

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)

        return prop_rpms.cpu().numpy()
