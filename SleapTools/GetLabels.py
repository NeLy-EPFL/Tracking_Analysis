import cv2
import h5py
import math

filename = "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Code/labels.v001.000_MultiMazeBiS_15_Steel_Wax_Female_Starved_noWater_p6-0_80fps_Trimmed_smol_rotated.analysis.h5"
with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

HEAD_INDEX = 0
THORAX_INDEX = 1
ABDO_INDEX = 2
LHINDLEG_INDEX = 3
RHINDLEG_INDEX = 4
RMIDLEG1_INDEX = 5
RMIDLEG2_INDEX = 6
LMIDLEG1_INDEX = 7
LMIDLEG2_INDEX = 8
RFORELEG1_INDEX = 9
RFORELEG2_INDEX = 10
RFORELEG3_INDEX = 11
LFORELEG1_INDEX = 12
LFORELEG2_INDEX = 13
LFORELEG3_INDEX = 14

head_loc = locations[:, HEAD_INDEX, :, :]
thorax_loc = locations[:, THORAX_INDEX, :, :]
abdo_loc = locations[:, ABDO_INDEX, :, :]
lhindleg_loc = locations[:, LHINDLEG_INDEX, :, :]
rhindleg_loc = locations[:, RHINDLEG_INDEX, :, :]
rmidleg1_loc = locations[:, RMIDLEG1_INDEX, :, :]
rmidleg2_loc = locations[:, RMIDLEG2_INDEX, :, :]
lmidleg1_loc = locations[:, LMIDLEG1_INDEX, :, :]
lmidleg2_loc = locations[:, LMIDLEG2_INDEX, :, :]

rforeleg1_loc = locations[:, RFORELEG1_INDEX, :, :]
rforeleg2_loc = locations[:, RFORELEG2_INDEX, :, :]
rforeleg3_loc = locations[:, RFORELEG3_INDEX, :, :]
lforeleg1_loc = locations[:, LFORELEG1_INDEX, :, :]
lforeleg2_loc = locations[:, LFORELEG2_INDEX, :, :]
lforeleg3_loc = locations[:, LFORELEG3_INDEX, :, :]


def draw_head(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = head_loc[frame_number][0][fly]
        y_coord = head_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 2, (0, 0, 255),
                   3)


def draw_thorax(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = thorax_loc[frame_number][0][fly]
        y_coord = thorax_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 2,
                   (0, 255, 0), 3)


def draw_abdo(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = abdo_loc[frame_number][0][fly]
        y_coord = abdo_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 2, (255, 0, 0),
                   3)


def draw_lhindleg(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = lhindleg_loc[frame_number][0][fly]
        y_coord = lhindleg_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (138, 43, 226), 3)


def draw_rhindleg(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = rhindleg_loc[frame_number][0][fly]
        y_coord = rhindleg_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (138, 43, 226), 3)


def draw_rmidleg1(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = rmidleg1_loc[frame_number][0][fly]
        y_coord = rmidleg1_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (151, 255, 255), 3)


def draw_rmidleg2(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = rmidleg2_loc[frame_number][0][fly]
        y_coord = rmidleg2_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (151, 255, 255), 3)


def draw_lmidleg1(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = lmidleg1_loc[frame_number][0][fly]
        y_coord = lmidleg1_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (151, 255, 255), 3)


def draw_lmidleg2(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = lmidleg2_loc[frame_number][0][fly]
        y_coord = lmidleg2_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (151, 255, 255), 3)


def draw_rforeleg1(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = rforeleg1_loc[frame_number][0][fly]
        y_coord = rforeleg1_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (255, 127, 0), 3)


def draw_rforeleg2(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = rforeleg2_loc[frame_number][0][fly]
        y_coord = rforeleg2_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (255, 127, 0), 3)


def draw_rforeleg3(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = rforeleg3_loc[frame_number][0][fly]
        y_coord = rforeleg3_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (255, 127, 0), 3)


def draw_lforeleg1(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = lforeleg1_loc[frame_number][0][fly]
        y_coord = lforeleg1_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (255, 127, 0), 3)


def draw_lforeleg2(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = lforeleg2_loc[frame_number][0][fly]
        y_coord = lforeleg2_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (255, 127, 0), 3)


def draw_lforeleg3(get_frame, frame_number):
    for fly in range(0, 6):
        x_coord = lforeleg3_loc[frame_number][0][fly]
        y_coord = lforeleg3_loc[frame_number][1][fly]
        draw_x_coord = security_check(x_coord)
        draw_y_coord = security_check(y_coord)
        cv2.circle(get_frame, (draw_x_coord, draw_y_coord), 1,
                   (255, 127, 0), 3)


# sometimes, a body part is not detected, in this case the coordinate is set to 0, to avoid a nAn error
def security_check(number):
    if math.isnan(number):
        return 0
    else:
        return round(number)


def draw_entire_skeleton(get_frame, frame_number):
    draw_thorax(get_frame, frame_number)
    draw_head(get_frame, frame_number)
    draw_abdo(get_frame, frame_number)
    draw_lhindleg(get_frame, frame_number)
    draw_rhindleg(get_frame, frame_number)
    draw_rmidleg1(get_frame, frame_number)
    draw_rmidleg2(get_frame, frame_number)
    draw_lmidleg1(get_frame, frame_number)
    draw_lmidleg2(get_frame, frame_number)
    draw_rforeleg1(get_frame, frame_number)
    draw_rforeleg2(get_frame, frame_number)
    draw_rforeleg3(get_frame, frame_number)
    draw_lforeleg1(get_frame, frame_number)
    draw_lforeleg2(get_frame, frame_number)
    draw_lforeleg3(get_frame, frame_number)