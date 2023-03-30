import copy
import numpy as np
import matplotlib.pyplot as plt

# 제어 시간 상수
tau = 0.1

# PID 게인
Kp = np.array([10., 10., 10.]) * 1.
Ki = np.array([0.1, 0.1, 0.1]) * 1.
Kd = np.array([1., 1., 1.]) * 1

# 초기 상태
x = np.array([0., 0., 0.])
x_d = np.array([0., 0., 0.])
xdot = np.array([0., 0., 0.])
x_err = np.array([0., 0., 0.])
x_int_err = np.array([0., 0., 0.])
xdot_old = np.array([0.0, 0.0, 0.0])


# 목표 경로
path = np.array([[1., 0., 0.], [1., 0.1, 0.], [1., 0.2, 0.], [1., 0.3, 0.],
                 [1., 0.4, 0.], [1., 0.5, 0.], [1., 0.6, 0.], [
                     1., 0.7, 0.], [1., 0.8, 0.], [1., 0.9, 0.], [1., 1., 0.],
                 [0.9, 1., 0.], [0.8, 1., 0.], [0.7, 1., 0.], [0.6, 1., 0.],
                 [0.5, 1., 0.], [0.4, 1., 0.], [0.3, 1., 0.], [0.2, 1., 0.],
                 [0.1, 1., 0.], [0., 1., 0.], [0., 0.9, 0.], [0., 0.8, 0.],
                 [0., 0.7, 0.], [0., 0.6, 0.], [0., 0.5, 0.], [0., 0.4, 0.],
                 [0., 0.3, 0.], [0., 0.2, 0.], [
                     0., 0.1, 0.], [0., 0., 0.], [0., 0., 0.1],
                 [0., 0., 0.2], [0., 0., 0.3], [0., 0., 0.4], [0., 0., 0.5],
                 [0., 0., 0.6], [0., 0., 0.7], [0., 0., 0.8], [0., 0., 0.9]])

tol = 0.05
next_wp_idx = 0

# 제어 주기
dt = 0.01

# 결과 저장
t = []
x_hist = []
x_d_hist = []

# 제어 주기마다 실행
for i in range(1000):

    # 현재 위치와 다음 웨이포인트 간의 거리 계산
    dist = np.linalg.norm(x - path[next_wp_idx])

    # 일정 거리 이내로 다가왔을 때 다음 웨이포인트를 목표 위치로 설정
    if dist < tol:
        next_wp_idx += 1
        if next_wp_idx >= len(path):
            next_wp_idx = len(path) - 1
        x_d = path[next_wp_idx]
    else:
        x_d = path[next_wp_idx]

    print(f"curr waypoint idx: {next_wp_idx}/{len(path)}")

    # 오차 계산
    x_err = x_d - x
    x_int_err += x_err * dt

    err_norm = np.linalg.norm(x_err)
    print(f"curr err to the next waypoint: {err_norm:.3f}")

    if err_norm < tol and next_wp_idx == len(path) - 1:
        print(f"the robot reached the final waypoint")
        print(
            f"Terminate the navigation. {i} steps were used to the destination.")
        break

    # PID 제어 신호 계산
    u = Kp * x_err + Ki * x_int_err + Kd * (xdot - xdot_old) / tau

    # 로봇 다이나믹스 모델
    xdot = u
    x += xdot * dt
    print(f"curr position {x}")

    # 결과 저장
    t.append(i * dt)
    x_hist.append(copy.deepcopy(x))
    x_d_hist.append(copy.deepcopy(x_d))

    # 현재 속도 저장
    xdot_old = copy.deepcopy(xdot)

    # 결과 시각화
    x_hist_draw = np.array(x_hist)
    x_d_hist_draw = np.array(x_d_hist)

    draw_3d = True
    if draw_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_d_hist_draw[:, 0], x_d_hist_draw[:, 1],
                x_d_hist_draw[:, 2], linewidth=3, color=[0, 1, 0.0],
                label='current reference path')
        ax.plot(x_hist_draw[:, 0], x_hist_draw[:, 1],
                x_hist_draw[:, 2], linewidth=1,
                label='actual robot path')
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'k.',
                markersize=5, label='waypoints')
        ax.legend()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        plt.show()
    else:
        plt.figure()
        plt.plot(x_d_hist_draw[:, 0], x_d_hist_draw[:, 1],
                 linewidth=3, color=[0, 1, 0.0],
                 label='current reference path')
        plt.plot(x_hist_draw[:, 0], x_hist_draw[:, 1],
                 linewidth=1,
                 label='actual robot path')
        plt.plot(path[:, 0], path[:, 1], 'k.', markersize=5,
                 label='waypoints')
        plt.legend()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
